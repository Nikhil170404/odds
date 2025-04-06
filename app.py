#!/usr/bin/env python3
"""
Cricket Odds API for BetBhai.io

This API serves cricket odds data scraped from betbhai.io
and provides endpoints for accessing the data with stable IDs.
"""

import os
import re
import time
import json
import logging
import threading
import uvicorn
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    WebDriverException, 
    TimeoutException, 
    NoSuchElementException, 
    StaleElementReferenceException
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cricket_odds_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Determine if running in production environment
IS_PRODUCTION = os.environ.get('RENDER', False)

# Make data directory
DATA_DIR = os.environ.get('DATA_DIR', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Cricket Odds API",
    description="API for real-time cricket odds from betbhai.io",
    version="2.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the domains instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class OddItem(BaseModel):
    position: int
    price: str
    volume: Optional[str] = None

class OddsData(BaseModel):
    back: List[OddItem] = []
    lay: List[OddItem] = []

class Match(BaseModel):
    id: str
    timestamp: str
    team1: Optional[str] = None
    team2: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    in_play: Optional[bool] = False
    score: Optional[List[str]] = None
    odds: Optional[OddsData] = None

class MatchUpdate(BaseModel):
    timestamp: str
    odds_changed: bool = False
    score_changed: bool = False
    status_changed: bool = False

class ScraperStatus(BaseModel):
    status: str
    last_updated: Optional[str] = None
    matches_count: int = 0
    is_running: bool
    error_count: int
    uptime_seconds: int = 0
    changes_since_last_update: int = 0

# Global state
DATA_FILE = os.path.join(DATA_DIR, "cricket_odds_latest.json")
ID_MAPPING_FILE = os.path.join(DATA_DIR, "cricket_match_id_mapping.json")

scraper_state = {
    "data": {"matches": []},
    "status": "idle",
    "last_updated": None,
    "is_running": False,
    "start_time": None,
    "error_count": 0,
    "changes_since_last_update": 0,
    "id_mapping": {},  # Maps legacy IDs to current stable IDs
    "match_history": {},  # Tracks changes for each match
    "lock": threading.Lock()
}

class CricketOddsScraper:
    """Scraper for extracting cricket odds from betbhai.io"""
    
    def __init__(self, url="https://www.betbhai.io/"):
        self.url = url
        self.driver = None
        self.retry_count = 0
        self.max_retries = 5
        self.error_count = 0
        self.max_continuous_errors = 10
        self.force_refresh = False
    
    def setup_driver(self):
        """Set up the Selenium WebDriver with compatibility for Render deployment"""
        try:
            # Close existing driver if any
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
            
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Add user agent to avoid detection
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # Adjust for Render deployment
            if IS_PRODUCTION:
                logger.info("Setting up driver for production environment")
                # Use system chrome binary on Render
                chrome_options.binary_location = "/opt/render/chrome/chrome"
                
                # Initialize the WebDriver without ChromeDriverManager
                service = Service('/opt/render/chromedriver/chromedriver')
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                # For local development, use ChromeDriverManager
                try:
                    from webdriver_manager.chrome import ChromeDriverManager
                    service = Service(ChromeDriverManager().install())
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                except Exception as e:
                    logger.error(f"Error with ChromeDriverManager: {e}")
                    # Fallback to direct ChromeDriver if available
                    service = Service()
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            self.driver.set_page_load_timeout(30)
            logger.info("WebDriver initialized successfully")
            self.retry_count = 0
            return True
        except Exception as e:
            logger.error(f"Error initializing WebDriver: {e}")
            self.retry_count += 1
            self.error_count += 1
            if self.retry_count < self.max_retries:
                logger.info(f"Retrying driver setup (attempt {self.retry_count}/{self.max_retries})...")
                time.sleep(5)
                return self.setup_driver()
            return False
    
    def navigate_to_site(self):
        """Navigate to the website and wait for it to load"""
        try:
            self.driver.get(self.url)
            # Wait for the cricket section to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".inplay-item-list"))
            )
            logger.info("Successfully navigated to the website")
            return True
        except TimeoutException:
            logger.error("Timeout while loading the website")
            self.error_count += 1
            return False
        except WebDriverException as e:
            logger.error(f"WebDriver error while navigating: {e}")
            self.error_count += 1
            return False
        except Exception as e:
            logger.error(f"Unexpected error while navigating: {e}")
            self.error_count += 1
            return False
    
    def _create_stable_id(self, team1: str, team2: str) -> str:
        """Create a stable ID based on team names"""
        if not team1:
            return "unknown_match"
        
        # Sort team names for consistency if both exist
        teams = sorted([team1, team2]) if team2 and team1 != team2 else [team1]
        
        # Normalize team names - remove spaces, special characters, etc.
        normalized = []
        for team in teams:
            # Convert to lowercase and replace non-alphanumeric with underscore
            team = "".join(c.lower() if c.isalnum() else '_' for c in team)
            # Remove consecutive underscores and trim
            team = re.sub(r'_+', '_', team).strip('_')
            normalized.append(team)
        
        # Join team names with vs
        match_key = "__vs__".join(normalized)
        
        return match_key
    
    def extract_cricket_odds(self):
        """Extract cricket odds data from the loaded page"""
        matches = []
        
        try:
            # Find all cricket match items
            cricket_sections = self.driver.find_elements(By.CSS_SELECTOR, 'ion-list.inplay-item-list')
            
            for section in cricket_sections:
                try:
                    # Check if this section is for cricket
                    header_elements = section.find_elements(By.CSS_SELECTOR, '.inplay-item-list__header-logo')
                    
                    is_cricket_section = False
                    for elem in header_elements:
                        try:
                            if 'cricket' in elem.text.lower():
                                is_cricket_section = True
                                break
                            # Also check for cricket icon
                            if elem.find_elements(By.CSS_SELECTOR, '.inplay-content__logo-icon--cricket'):
                                is_cricket_section = True
                                break
                        except (StaleElementReferenceException, NoSuchElementException):
                            continue
                    
                    if not is_cricket_section:
                        continue
                        
                    # Get all match items in this section
                    match_items = section.find_elements(By.CSS_SELECTOR, '.inplay-item')
                    
                    for item in match_items:
                        try:
                            # Extract team names first to create stable ID
                            team1 = ""
                            team2 = ""
                            try:
                                player_elems = item.find_elements(By.CSS_SELECTOR, '.inplay-item__player span')
                                if len(player_elems) >= 1:
                                    team1 = player_elems[0].text
                                    if len(player_elems) > 1:
                                        team2 = player_elems[1].text
                            except (StaleElementReferenceException, NoSuchElementException) as e:
                                logger.warning(f"Error extracting team names: {e}")

                            # Create a stable ID based on team names
                            stable_id = self._create_stable_id(team1, team2)
                            
                            # Initialize match data with stable ID
                            match_data = {
                                'id': f"match_{stable_id}",
                                'timestamp': datetime.now().isoformat(),
                                'team1': team1,
                                'team2': team2
                            }
                            
                            # Extract date and time
                            try:
                                date_elems = item.find_elements(By.CSS_SELECTOR, '.date-content .inPlayDate-content__date')
                                time_elems = item.find_elements(By.CSS_SELECTOR, '.date-content .inPlayDate-content__time')
                                
                                if date_elems and time_elems:
                                    match_data['date'] = date_elems[0].text
                                    match_data['time'] = time_elems[0].text
                            except (StaleElementReferenceException, NoSuchElementException) as e:
                                logger.warning(f"Error extracting date/time: {e}")
                            
                            # Extract current score if available
                            try:
                                score_elem = item.find_elements(By.CSS_SELECTOR, '.score-content:not(.empty)')
                                if score_elem:
                                    score_spans = score_elem[0].find_elements(By.TAG_NAME, 'span')
                                    if score_spans:
                                        match_data['score'] = [span.text for span in score_spans]
                                        match_data['in_play'] = True
                                else:
                                    match_data['in_play'] = False
                            except (StaleElementReferenceException, NoSuchElementException) as e:
                                logger.warning(f"Error extracting score: {e}")
                                match_data['in_play'] = False
                            
                            # Extract odds
                            odds = {'back': [], 'lay': []}
                            
                            try:
                                # Back odds
                                back_buttons = item.find_elements(By.CSS_SELECTOR, '.odd-button.back-color')
                                for i, button in enumerate(back_buttons):
                                    try:
                                        price_elems = button.find_elements(By.CSS_SELECTOR, '.odd-button__price')
                                        volume_elems = button.find_elements(By.CSS_SELECTOR, '.odd-button__volume')
                                        
                                        if price_elems:
                                            price_text = price_elems[0].text
                                            if price_text and price_text != '-':
                                                odd = {
                                                    'position': i,
                                                    'price': price_text,
                                                    'volume': volume_elems[0].text if volume_elems else None
                                                }
                                                odds['back'].append(odd)
                                    except (StaleElementReferenceException, NoSuchElementException):
                                        continue
                                
                                # Lay odds
                                lay_buttons = item.find_elements(By.CSS_SELECTOR, '.odd-button.lay-color')
                                for i, button in enumerate(lay_buttons):
                                    try:
                                        price_elems = button.find_elements(By.CSS_SELECTOR, '.odd-button__price')
                                        volume_elems = button.find_elements(By.CSS_SELECTOR, '.odd-button__volume')
                                        
                                        if price_elems:
                                            price_text = price_elems[0].text
                                            if price_text and price_text != '-':
                                                odd = {
                                                    'position': i,
                                                    'price': price_text,
                                                    'volume': volume_elems[0].text if volume_elems else None
                                                }
                                                odds['lay'].append(odd)
                                    except (StaleElementReferenceException, NoSuchElementException):
                                        continue
                            except (StaleElementReferenceException, NoSuchElementException) as e:
                                logger.warning(f"Error extracting odds: {e}")
                            
                            match_data['odds'] = odds
                            matches.append(match_data)
                        except (StaleElementReferenceException, NoSuchElementException) as e:
                            logger.warning(f"Error processing match item: {e}")
                except (StaleElementReferenceException, NoSuchElementException) as e:
                    logger.warning(f"Error processing cricket section: {e}")
            
            if matches:
                logger.info(f"Extracted {len(matches)} cricket matches")
                # Reset error count on successful extraction
                self.error_count = 0
            else:
                logger.warning("No cricket matches found")
                self.error_count += 1
            
            return matches
            
        except Exception as e:
            logger.error(f"Error extracting cricket odds: {e}")
            self.error_count += 1
            return []
    
    def _match_equal(self, old_match: Dict[str, Any], new_match: Dict[str, Any]) -> bool:
        """Compare two match objects to determine if they are equivalent"""
        # Keys to exclude when comparing (these can change without being considered a "change")
        exclude_keys = {'timestamp', 'id'}
        
        # Helper function to normalize volume strings for comparison
        def normalize_volume(vol_str):
            if not vol_str:
                return None
            # Remove commas and convert to numeric value for comparison
            return vol_str.replace(',', '')
        
        # Helper function to compare odds
        def odds_equal(odds1, odds2):
            if not odds1 and not odds2:
                return True
            if not odds1 or not odds2:
                return False
            
            # Compare back odds
            back1 = sorted(odds1.get('back', []), key=lambda x: x.get('position', 0))
            back2 = sorted(odds2.get('back', []), key=lambda x: x.get('position', 0))
            
            if len(back1) != len(back2):
                return False
                
            for o1, o2 in zip(back1, back2):
                # Compare position and price (most important)
                if o1.get('position') != o2.get('position') or o1.get('price') != o2.get('price'):
                    return False
                
                # Compare normalized volumes
                vol1 = normalize_volume(o1.get('volume'))
                vol2 = normalize_volume(o2.get('volume'))
                if vol1 != vol2:
                    return False
            
            # Compare lay odds
            lay1 = sorted(odds1.get('lay', []), key=lambda x: x.get('position', 0))
            lay2 = sorted(odds2.get('lay', []), key=lambda x: x.get('position', 0))
            
            if len(lay1) != len(lay2):
                return False
                
            for o1, o2 in zip(lay1, lay2):
                # Compare position and price
                if o1.get('position') != o2.get('position') or o1.get('price') != o2.get('price'):
                    return False
                
                # Compare normalized volumes
                vol1 = normalize_volume(o1.get('volume'))
                vol2 = normalize_volume(o2.get('volume'))
                if vol1 != vol2:
                    return False
            
            return True
        
        # Compare all keys except the excluded ones and odds
        for key in set(old_match.keys()) | set(new_match.keys()):
            if key in exclude_keys or key == 'odds':
                continue
            
            if key not in old_match or key not in new_match:
                return False
            
            if old_match[key] != new_match[key]:
                return False
        
        # Compare odds separately
        return odds_equal(old_match.get('odds'), new_match.get('odds'))
    
    def _detect_changes(self, old_match: Dict[str, Any], new_match: Dict[str, Any]) -> Dict[str, bool]:
        """Detect specific changes between two match objects"""
        changes = {
            "odds_changed": False,
            "score_changed": False,
            "status_changed": False
        }
        
        # Check for in_play status change
        if old_match.get('in_play') != new_match.get('in_play'):
            changes["status_changed"] = True
        
        # Check for score changes
        old_score = old_match.get('score')
        new_score = new_match.get('score')
        if (old_score is None and new_score is not None) or \
           (old_score is not None and new_score is None) or \
           (old_score != new_score):
            changes["score_changed"] = True
        
        # Helper function to check if odds have changed
        def odds_changed(old_odds, new_odds):
            if not old_odds and not new_odds:
                return False
                
            if bool(old_odds) != bool(new_odds):
                return True
                
            # Check if back odds changed
            old_back = sorted(old_odds.get('back', []), key=lambda x: x.get('position', 0))
            new_back = sorted(new_odds.get('back', []), key=lambda x: x.get('position', 0))
            
            if len(old_back) != len(new_back):
                return True
                
            for i, (old_item, new_item) in enumerate(zip(old_back, new_back)):
                # Compare prices
                if old_item.get('price') != new_item.get('price'):
                    return True
                
                # Compare volumes (normalize by removing commas)
                old_vol = old_item.get('volume', '').replace(',', '') if old_item.get('volume') else ''
                new_vol = new_item.get('volume', '').replace(',', '') if new_item.get('volume') else ''
                
                if old_vol != new_vol:
                    return True
            
            # Check if lay odds changed
            old_lay = sorted(old_odds.get('lay', []), key=lambda x: x.get('position', 0))
            new_lay = sorted(new_odds.get('lay', []), key=lambda x: x.get('position', 0))
            
            if len(old_lay) != len(new_lay):
                return True
                
            for i, (old_item, new_item) in enumerate(zip(old_lay, new_lay)):
                # Compare prices
                if old_item.get('price') != new_item.get('price'):
                    return True
                
                # Compare volumes
                old_vol = old_item.get('volume', '').replace(',', '') if old_item.get('volume') else ''
                new_vol = new_item.get('volume', '').replace(',', '') if new_item.get('volume') else ''
                
                if old_vol != new_vol:
                    return True
            
            return False
        
        # Check for odds changes
        old_odds = old_match.get('odds', {})
        new_odds = new_match.get('odds', {})
        
        if odds_changed(old_odds, new_odds):
            changes["odds_changed"] = True
        
        return changes
    
    def update_global_state(self, new_matches):
        """Update the global state with new matches data, tracking changes and ID mapping"""
        try:
            changes_made = 0
            current_time = datetime.now().isoformat()
            
            with scraper_state["lock"]:
                # Get current matches and ID mapping
                current_matches = scraper_state["data"].get("matches", [])
                id_mapping = scraper_state.get("id_mapping", {})
                match_history = scraper_state.get("match_history", {})
                
                # Build a dictionary of current matches by ID
                current_matches_by_id = {m.get('id'): m for m in current_matches}
                
                # Create a mapping from team combinations to match IDs
                team_to_id_map = {}
                for match in current_matches:
                    team1 = match.get('team1', '')
                    team2 = match.get('team2', '')
                    if team1 or team2:  # Only map if at least one team is present
                        key = self._create_stable_id(team1, team2)
                        team_to_id_map[key] = match.get('id')
                
                # Process new matches
                updated_matches = []
                processed_ids = set()
                
                for new_match in new_matches:
                    # Extract info for matching
                    team1 = new_match.get('team1', '')
                    team2 = new_match.get('team2', '')
                    match_id = new_match.get('id')
                    stable_key = self._create_stable_id(team1, team2)
                    
                    # Find current match by ID or team combination
                    current_match = None
                    
                    # First try to find by ID
                    if match_id in current_matches_by_id:
                        current_match = current_matches_by_id[match_id]
                    
                    # If not found by ID, try by team combination
                    elif stable_key in team_to_id_map:
                        current_id = team_to_id_map[stable_key]
                        if current_id in current_matches_by_id:
                            current_match = current_matches_by_id[current_id]
                            # Update the ID mapping to point legacy ID to current stable ID
                            id_mapping[current_id] = match_id
                    
                    if current_match:
                        # Check if the match has materially changed
                        if not self._match_equal(current_match, new_match):
                            # Detect specific changes
                            changes = self._detect_changes(current_match, new_match)
                            
                            # Preserve the original ID but update content
                            new_match['id'] = current_match['id']
                            updated_matches.append(new_match)
                            changes_made += 1
                            
                            # Record change history
                            match_history.setdefault(new_match['id'], []).append({
                                'timestamp': current_time,
                                'odds_changed': changes['odds_changed'],
                                'score_changed': changes['score_changed'],
                                'status_changed': changes['status_changed']
                            })
                            
                            logger.debug(f"Updated match: {new_match['id']} - {changes}")
                        else:
                            # No changes, keep current version
                            updated_matches.append(current_match)
                        
                        processed_ids.add(current_match['id'])
                    else:
                        # This is a new match
                        updated_matches.append(new_match)
                        changes_made += 1
                        
                        # Add to match history
                        match_history.setdefault(new_match['id'], []).append({
                            'timestamp': current_time,
                            'odds_changed': True,  # New match always has "new" odds
                            'score_changed': False,
                            'status_changed': False
                        })
                        
                        logger.debug(f"New match added: {new_match['id']}")
                
                # Check for removed matches
                for old_id, old_match in current_matches_by_id.items():
                    if old_id not in processed_ids:
                        # Match was removed
                        changes_made += 1
                        logger.debug(f"Match removed: {old_id}")
                
                # Create output data structure
                output_data = {
                    'timestamp': current_time,
                    'updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'matches': updated_matches
                }
                
                # Update global state
                scraper_state["data"] = output_data
                scraper_state["last_updated"] = current_time
                scraper_state["status"] = "running"
                scraper_state["id_mapping"] = id_mapping
                scraper_state["match_history"] = match_history
                scraper_state["changes_since_last_update"] = changes_made
                
                # Save data to files
                self._save_data_files(output_data, id_mapping)
                
                logger.info(f"Data updated with {changes_made} changes ({len(updated_matches)} matches)")
                return True
        except Exception as e:
            logger.error(f"Error updating global state: {e}")
            self.error_count += 1
            return False
    
    def _save_data_files(self, output_data, id_mapping):
        """Save data to files with error handling"""
        try:
            # Save the main data file
            temp_data_file = f"{DATA_FILE}.tmp"
            with open(temp_data_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename to prevent corruption
            os.replace(temp_data_file, DATA_FILE)
            
            # Save ID mapping file
            temp_id_file = f"{ID_MAPPING_FILE}.tmp"
            with open(temp_id_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'updated': datetime.now().isoformat(),
                    'mapping': id_mapping
                }, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            os.replace(temp_id_file, ID_MAPPING_FILE)
            
            return True
        except Exception as e:
            logger.error(f"Error saving data files: {e}")
            return False
    
    def run(self, interval=2):
        """Run the scraper every 'interval' seconds"""
        # Update scraper state
        with scraper_state["lock"]:
            scraper_state["is_running"] = True
            scraper_state["start_time"] = datetime.now()
            scraper_state["status"] = "starting"
        
        logger.info(f"Starting cricket odds scraper with {interval} second intervals")
        
        if not self.setup_driver():
            logger.error("Failed to set up WebDriver. Exiting.")
            with scraper_state["lock"]:
                scraper_state["is_running"] = False
                scraper_state["status"] = "failed"
            return
        
        try:
            refresh_count = 0
            max_extractions_before_refresh = 15  # Refresh page completely every ~30 seconds (with 2s interval)
            
            # Navigate to the site initially
            if not self.navigate_to_site():
                logger.error("Failed to navigate to the website. Retrying setup...")
                if not self.setup_driver() or not self.navigate_to_site():
                    logger.error("Still failed to navigate. Exiting.")
                    with scraper_state["lock"]:
                        scraper_state["is_running"] = False
                        scraper_state["status"] = "failed"
                    return
            
            # Update status to running
            with scraper_state["lock"]:
                scraper_state["status"] = "running"
            
            while scraper_state["is_running"]:
                start_time = time.time()
                
                # Check if we need to force refresh
                with scraper_state["lock"]:
                    force_refresh = getattr(self, 'force_refresh', False)
                    if force_refresh:
                        self.force_refresh = False
                
                # Check if we've had too many continuous errors
                if self.error_count >= self.max_continuous_errors:
                    logger.error(f"Reached maximum continuous errors ({self.max_continuous_errors}). Resetting driver...")
                    if not self.setup_driver() or not self.navigate_to_site():
                        logger.error("Driver reset failed. Waiting before retrying...")
                        time.sleep(30)  # Wait longer before retrying
                        continue
                    self.error_count = 0
                
                # Check if we need to refresh the page
                if refresh_count >= max_extractions_before_refresh or force_refresh:
                    logger.info("Performing complete page refresh")
                    if not self.navigate_to_site():
                        logger.warning("Page refresh failed, attempting to reset driver")
                        if not self.setup_driver() or not self.navigate_to_site():
                            logger.error("Driver reset failed. Waiting before retrying...")
                            time.sleep(30)  # Wait longer before retrying
                            continue
                    refresh_count = 0
                
                # Extract and update the data
                matches = self.extract_cricket_odds()
                if matches:
                    self.update_global_state(matches)
                
                refresh_count += 1
                
                # Update error count in global state
                with scraper_state["lock"]:
                    scraper_state["error_count"] = self.error_count
                
                # Calculate sleep time to maintain the interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Scraper stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            # Clean up
            try:
                if self.driver:
                    self.driver.quit()
                    logger.info("WebDriver closed")
            except:
                pass
            
            # Update scraper state
            with scraper_state["lock"]:
                scraper_state["is_running"] = False
                scraper_state["status"] = "stopped"

# Create a function to start the scraper in a background thread
def start_scraper_thread():
    if not scraper_state["is_running"]:
        # Create and start the thread
        scraper = CricketOddsScraper()
        thread = threading.Thread(target=scraper.run, args=(2,), daemon=True)
        thread.start()
        logger.info("Scraper thread started")
        return True
    else:
        logger.info("Scraper is already running")
        return False

# Load existing data if available
def load_existing_data():
    try:
        # Load main data file
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                with scraper_state["lock"]:
                    scraper_state["data"] = data
                    scraper_state["last_updated"] = data.get("timestamp", datetime.now().isoformat())
                    logger.info(f"Loaded existing data with {len(data.get('matches', []))} matches")
        
        # Load ID mapping file
        if os.path.exists(ID_MAPPING_FILE):
            with open(ID_MAPPING_FILE, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                with scraper_state["lock"]:
                    scraper_state["id_mapping"] = mapping_data.get("mapping", {})
                    logger.info(f"Loaded ID mapping with {len(scraper_state['id_mapping'])} entries")
    except Exception as e:
        logger.error(f"Error loading existing data: {e}")

# Helper function to find matches by various IDs and handle redirects
def find_match_by_id(match_id: str):
    """Find a match by ID or in the ID mapping"""
    with scraper_state["lock"]:
        matches = scraper_state["data"].get("matches", [])
        id_mapping = scraper_state.get("id_mapping", {})
    
    # First, try direct lookup in current matches
    for match in matches:
        if match.get("id") == match_id:
            return match, None  # Found direct match
    
    # If not found directly, check if it's in the ID mapping
    if match_id in id_mapping:
        new_id = id_mapping[match_id]
        # Check if we can find the match with the new ID
        for match in matches:
            if match.get("id") == new_id:
                return match, new_id  # Found mapped match
    
    # Try to resolve by team name-based matching as last resort
    if match_id.startswith('match_'):
        # Extract any potential dates or teams from the old ID
        parts = match_id.split('_')
        if len(parts) > 2:
            # Try to find matching teams in current matches
            for match in matches:
                team1 = match.get('team1', '')
                team2 = match.get('team2', '')
                if not team1:
                    continue
                    
                if (team1 and team1.lower() in match_id.lower()) or \
                   (team2 and team2.lower() in match_id.lower()):
                    # Potential match found
                    return match, match.get('id')
    
    # Not found at all
    return None, None

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Cricket Odds API",
        "version": "2.0.0",
        "description": "API for real-time cricket odds from betbhai.io",
        "endpoints": [
            {"path": "/matches", "description": "Get all cricket matches"},
            {"path": "/matches/{match_id}", "description": "Get a specific match by ID"},
            {"path": "/status", "description": "Get the scraper status"},
            {"path": "/refresh", "description": "Force a refresh of the data"},
            {"path": "/changes", "description": "Get changes for a specific match"}
        ]
    }

@app.get("/matches", response_model=List[Match], tags=["Matches"])
async def get_matches(
    team: Optional[str] = Query(None, description="Filter by team name"),
    in_play: Optional[bool] = Query(None, description="Filter by in-play status")
):
    """Get all cricket matches with optional filtering"""
    with scraper_state["lock"]:
        matches = scraper_state["data"].get("matches", [])
    
    # Apply filters if provided
    if team:
        team_lower = team.lower()
        matches = [
            m for m in matches 
            if (m.get("team1", "").lower().find(team_lower) != -1 or 
                m.get("team2", "").lower().find(team_lower) != -1)
        ]
    
    if in_play is not None:
        matches = [m for m in matches if m.get("in_play") == in_play]
    
    return matches

@app.get("/matches/{match_id}", tags=["Matches"])
async def get_match(match_id: str, request: Request):
    """Get a specific cricket match by ID with automatic redirection for legacy IDs"""
    match, new_id = find_match_by_id(match_id)
    
    if match:
        # If we found the match using a mapped ID, redirect to the new endpoint
        if new_id and new_id != match_id:
            redirect_url = str(request.url).replace(match_id, new_id)
            return RedirectResponse(url=redirect_url, status_code=status.HTTP_301_MOVED_PERMANENTLY)
        
        # Return the match directly
        return match
    
    # Match not found
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Match with ID {match_id} not found"
    )

@app.get("/status", response_model=ScraperStatus, tags=["System"])
async def get_status():
    """Get the current status of the scraper"""
    with scraper_state["lock"]:
        uptime = (datetime.now() - scraper_state["start_time"]).total_seconds() if scraper_state["start_time"] else 0
        return {
            "status": scraper_state["status"],
            "last_updated": scraper_state["last_updated"],
            "matches_count": len(scraper_state["data"].get("matches", [])),
            "is_running": scraper_state["is_running"],
            "error_count": scraper_state["error_count"],
            "uptime_seconds": int(uptime),
            "changes_since_last_update": scraper_state.get("changes_since_last_update", 0)
        }

@app.get("/changes/{match_id}", tags=["Matches"])
async def get_match_changes(match_id: str):
    """Get the change history for a specific match"""
    # Find the match first to handle redirects
    match, new_id = find_match_by_id(match_id)
    
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match with ID {match_id} not found"
        )
    
    # Use the correct ID to look up history
    lookup_id = new_id if new_id else match_id
    
    with scraper_state["lock"]:
        history = scraper_state.get("match_history", {}).get(lookup_id, [])
    
    return {
        "match_id": lookup_id,
        "team1": match.get("team1"),
        "team2": match.get("team2"),
        "changes": history
    }

@app.post("/refresh", tags=["System"])
async def force_refresh():
    """Force a refresh of the cricket odds data"""
    if not scraper_state["is_running"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Scraper is not running. Start it first."}
        )
    
    # Set the force refresh flag
    with scraper_state["lock"]:
        scraper_state["status"] = "refreshing"
        scraper_state["force_refresh"] = True
    
    return {"message": "Refresh requested successfully"}

@app.post("/start", tags=["System"])
async def start_scraper(background_tasks: BackgroundTasks):
    """Start the cricket odds scraper"""
    if scraper_state["is_running"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Scraper is already running"}
        )
    
    # Start the scraper in a background thread
    background_tasks.add_task(start_scraper_thread)
    
    return {"message": "Scraper starting..."}

@app.post("/stop", tags=["System"])
async def stop_scraper():
    """Stop the cricket odds scraper"""
    if not scraper_state["is_running"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Scraper is not running"}
        )
    
    # Stop the scraper
    with scraper_state["lock"]:
        scraper_state["is_running"] = False
        scraper_state["status"] = "stopping"
    
    return {"message": "Scraper shutdown initiated"}

# On startup
@app.on_event("startup")
async def startup_event():
    # Load existing data
    load_existing_data()
    
    # Initialize scraper state
    scraper_state["start_time"] = datetime.now()
    
    # Start the scraper automatically
    start_scraper_thread()

# On shutdown
@app.on_event("shutdown")
async def shutdown_event():
    # Stop the scraper if running
    with scraper_state["lock"]:
        scraper_state["is_running"] = False
        logger.info("API shutting down, stopping scraper")

if __name__ == "__main__":
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)