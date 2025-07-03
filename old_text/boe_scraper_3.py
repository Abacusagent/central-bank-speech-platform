#!/usr/bin/env python3
"""
Enhanced Bank of England Speech Scraper
Combines multiple data sources into a unified scraping pipeline, and exports a summary sheet.
"""
import io
import json
import logging
import re
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd

# Optional Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Load configuration constants
from config import *

# Ensure output dirs exist and setup logging
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class EnhancedBoEScraper:
    """
    Unified scraper for Bank of England speeches, with reporting sheet export.

    This class provides methods to scrape speech data from various sections of
    the Bank of England and related websites, including sitemaps, news pages,
    quarterly bulletins, and a digital archive requiring Selenium. It extracts
    metadata and full text, and compiles a summary report.

    Attributes:
        delay (float): Delay in seconds between requests to avoid rate limiting.
        speeches_dir (Path): Directory where speech text files will be saved.
        meta_dir (Path): Directory where speech metadata JSON files will be saved.
        speaker_roles (Dict): Database of speaker names and their roles/voting status.
        current_url (Optional[str]): The URL currently being processed, for logging.
    """

    def __init__(self, delay: float = 1.0):
        """
        Initializes the scraper with output directories and speaker database.

        Args:
            delay (float): Time delay in seconds between requests to avoid overwhelming servers.
        """
        self.speeches_dir = OUTPUT_DIR / "speeches"
        self.meta_dir = OUTPUT_DIR / "metadata"
        for d in (self.speeches_dir, self.meta_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.speaker_roles = {}
        self.current_url: Optional[str] = None
        self._initialize_speaker_database_v2() # Initialize the enhanced speaker database

    def _initialize_speaker_database_v2(self):
        """Enhanced speaker database with better name variants and historical coverage."""
        self.speaker_roles = {
            # Current BoE Leadership (2020-present)
            "andrew bailey": {"role": "Governor", "voting_status": "Voting Member", "years": "2020-present"},
            "andrew john bailey": {"role": "Governor", "voting_status": "Voting Member", "years": "2020-present"},
            "bailey": {"role": "Governor", "voting_status": "Voting Member", "years": "2020-present"},
            
            # Current Deputy Governors (2024-2025)
            "clare lombardelli": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2024-present"},
            "lombardelli": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2024-present"},
            
            "dave ramsden": {"role": "Deputy Governor for Markets and Banking", "voting_status": "Voting Member", "years": "2017-present"},
            "david ramsden": {"role": "Deputy Governor for Markets and Banking", "voting_status": "Voting Member", "years": "2017-present"},
            "ramsden": {"role": "Deputy Governor for Markets and Banking", "voting_status": "Voting Member", "years": "2017-present"},
            
            "sarah breeden": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2023-present"},
            "breeden": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2023-present"},
            
            "sarah john": {"role": "Chief Operating Officer", "voting_status": "Non-Voting", "years": "2025-present"},
            
            # Current Chief Economist
            "huw pill": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2021-present"},
            "pill": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2021-present"},
            
            # Current External MPC Members
            "alan taylor": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2024-present"},
            "taylor": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2024-present"},
            
            "catherine l mann": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2021-present"},
            "catherine mann": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2021-present"},
            "mann": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2021-present"},
            
            "jonathan haskel": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2018-present"},
            "haskel": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2018-present"},
            
            "swati dhingra": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2022-present"},
            "dhingra": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2022-present"},
            
            "megan greene": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2023-present"},
            "greene": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2023-present"},
            
            # Past Governors - ENHANCED with more variants
            "mark carney": {"role": "Governor", "voting_status": "Voting Member", "years": "2013-2020"},
            "mark joseph carney": {"role": "Governor", "voting_status": "Voting Member", "years": "2013-2020"},
            "carney": {"role": "Governor", "voting_status": "Voting Member", "years": "2013-2020"},
            
            "mervyn king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            "mervyn allister king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            "king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            "lord king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            "baron king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            "lord king of lothbury": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            
            # ENHANCED Eddie George entries with more variants
            "eddie george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "edward george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "edward alan john george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "steady eddie": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "lord george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "baron george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "sir eddie george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "sir edward george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "e a j george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "e george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            
            # Robin Leigh-Pemberton with ALL variants
            "robin leigh-pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "robin leigh pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "leigh-pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "leigh pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "lord kingsdown": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "baron kingsdown": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "r leigh-pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "robin robert leigh-pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            
            # Gordon Richardson (1973-1983)
            "gordon richardson": {"role": "Governor", "voting_status": "Voting Member", "years": "1973-1983"},
            "gordon william humphreys richardson": {"role": "Governor", "voting_status": "Voting Member", "years": "1973-1983"},
            "richardson": {"role": "Governor", "voting_status": "Voting Member", "years": "1973-1983"},
            "lord richardson": {"role": "Governor", "voting_status": "Voting Member", "years": "1973-1983"},
            "baron richardson": {"role": "Governor", "voting_status": "Voting Member", "years": "1973-1983"},
            "g richardson": {"role": "Governor", "voting_status": "Voting Member", "years": "1973-1983"},
            
            # Leslie O'Brien (1966-1973)
            "leslie o'brien": {"role": "Governor", "voting_status": "Voting Member", "years": "1966-1973"},
            "leslie kenneth o'brien": {"role": "Governor", "voting_status": "Voting Member", "years": "1966-1973"},
            "o'brien": {"role": "Governor", "voting_status": "Voting Member", "years": "1966-1973"},
            "obrien": {"role": "Governor", "voting_status": "Voting Member", "years": "1966-1973"},
            "lord o'brien": {"role": "Governor", "voting_status": "Voting Member", "years": "1966-1973"},
            "l o'brien": {"role": "Governor", "voting_status": "Voting Member", "years": "1966-1973"},
            
            # Past Deputy Governors (Enhanced)
            "ben broadbent": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2011-2024"},
            "broadbent": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2011-2024"},
            
            "jon cunliffe": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2013-2023"},
            "sir jon cunliffe": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2013-2023"},
            "cunliffe": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2013-2023"},
            "jonathan cunliffe": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2013-2023"},
            
            "sam woods": {"role": "Deputy Governor for Prudential Regulation", "voting_status": "Non-Voting", "years": "2016-present"},
            "samuel woods": {"role": "Deputy Governor for Prudential Regulation", "voting_status": "Non-Voting", "years": "2016-present"},
            "woods": {"role": "Deputy Governor for Prudential Regulation", "voting_status": "Non-Voting", "years": "2016-present"},
            
            # Chief Economists (Enhanced)
            "andy haldane": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2014-2021"},
            "andrew haldane": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2014-2021"},
            "andrew g haldane": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2014-2021"},
            "haldane": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2014-2021"},
            
            "spencer dale": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2008-2014"},
            "dale": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2008-2014"},
            
            "charlie bean": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2000-2008"},
            "charles bean": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2000-2008"},
            "charles goodhart bean": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2000-2008"},
            "bean": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2000-2008"},
            
            "john vickers": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "1998-2000"},
            "sir john vickers": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "1998-2000"},
            "vickers": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "1998-2000"},
            
            # Past External MPC Members (Enhanced with more names)
            "silvana tenreyro": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2017-2022"},
            "tenreyro": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2017-2022"},
            
            "gertjan vlieghe": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2015-2021"},
            "vlieghe": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2015-2021"},
            
            "michael saunders": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2016-2021"},
            "saunders": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2016-2021"},
            
            "ian mccafferty": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2012-2018"},
            "mccafferty": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2012-2018"},
            
            "martin weale": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2010-2016"},
            "weale": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2010-2016"},
            
            "david miles": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2009-2015"},
            "miles": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2009-2015"},
            
            "adam posen": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2009-2012"},
            "posen": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2009-2012"},
            
            "andrew sentance": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2006-2011"},
            "sentance": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2006-2011"},
            
            "kate barker": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2001-2010"},
            "barker": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2001-2010"},
            
            "david blanchflower": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2006-2009"},
            "danny blanchflower": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2006-2009"},
            "blanchflower": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2006-2009"},
            
            "stephen nickell": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2000-2006"},
            "nickell": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2000-2006"},
            
            "christopher allsopp": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2000-2006"},
            "allsopp": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2000-2006"},
            
            "sushil wadhwani": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "1999-2002"},
            "wadhwani": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "1999-2002"},
            
            "deanne julius": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "1997-2001"},
            "julius": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "1997-2001"},
            
            "alan budd": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "1997-1999"},
            "budd": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "1997-1999"},
            
            "willem buiter": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "1997-2000"},
            "buiter": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "1997-2000"},
            
            # Executive Directors and Senior Officials (Enhanced)
            "victoria cleland": {"role": "Executive Director for Markets", "voting_status": "Non-Voting", "years": "2014-present"},
            "cleland": {"role": "Executive Director for Markets", "voting_status": "Non-Voting", "years": "2014-present"},
            
            "paul fisher": {"role": "Executive Director for Markets", "voting_status": "Voting Member", "years": "2009-2013"},
            "fisher": {"role": "Executive Director for Markets", "voting_status": "Voting Member", "years": "2009-2013"},
            
            "david rule": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2013-present"},
            "rule": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2013-present"},
            
            # ENHANCED historical officials for pre-1997 period
            "ian plenderleith": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1990s-2000s"},
            "plenderleith": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1990s-2000s"},
            
            "alastair clark": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1990s-2000s"},
            "clark": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1990s-2000s"},
            
            "brian quinn": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1980s-1990s"},
            "quinn": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1980s-1990s"},
            
            "pen kent": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1990s"},
            "kent": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1990s"},
            
            "william cooke": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1980s-1990s"},
            "w p cooke": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1980s-1990s"},
            "cooke": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "1980s-1990s"},
            
            # Historical Deputy Governors 
            "david clementi": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1997-2002"},
            "sir david clementi": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1997-2002"},
            "clementi": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1997-2002"},
            
            "howard davies": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1995-1997"},
            "davies": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1995-1997"},
            
            "rupert pennant-rea": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1993-1995"},
            "pennant-rea": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1993-1995"},
            
            "george blunden": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1976-1990"},
            "blunden": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1976-1990"},
            
            "kit mcmahon": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1980-1986"},
            "christopher mcmahon": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1980-1986"},
            "mcmahon": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1980-1986"},
            
            # Additional historical names from Quarterly Bulletin references
            "marian bell": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2002-2005"},
            "bell": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2002-2005"},
            
            "richard lambert": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2003-2006"},
            "lambert": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2003-2006"},
            
            "paul tucker": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2009-2013"},
            "paul mw tucker": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2009-2013"},
            "tucker": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2009-2013"},
            
            "andrew large": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "2002-2006"},
            "large": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "2002-2006"},
            
            # Additional current officials that were missed
            "jon hall": {"role": "Senior Manager", "voting_status": "Non-Voting", "years": "2018-present"},
            "hall": {"role": "Senior Manager", "voting_status": "Non-Voting", "years": "2018-present"},
            
            "randall kroszner": {"role": "External Board Member", "voting_status": "Non-Voting", "years": "2024-present"},
            "kroszner": {"role": "External Board Member", "voting_status": "Non-Voting", "years": "2024-present"},
        }


    def _fetch_url(self, url: str) -> Optional[str]:
        """
        Fetches URL content with retries and encoding handling.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[str]: The content of the URL as a string, or None if fetching fails.
        """
        self.current_url = url # Set current URL for logging context
        logger.debug(f"Fetching {url}")
        for encoding in ENCODINGS:
            try:
                # Add headers=DEFAULT_HEADERS to the requests.get call
                response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                response.encoding = encoding # Try setting encoding explicitly
                return response.text
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {url} with encoding {encoding}: {e}")
            except UnicodeDecodeError:
                logger.warning(f"UnicodeDecodeError for {url} with encoding {encoding}")
        logger.error(f"Failed to fetch or decode {url} after trying all encodings.")
        self.current_url = None
        return None

    def _get_all_links_from_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Extracts all URLs from a sitemap XML.

        Args:
            sitemap_url (str): The URL of the sitemap XML.

        Returns:
            List[str]: A list of URLs found in the sitemap.
        """
        logger.info(f"Scraping sitemap: {sitemap_url}")
        content = self._fetch_url(sitemap_url)
        if not content:
            logger.error(f"Failed to fetch sitemap: {sitemap_url}")
            return []

        soup = BeautifulSoup(content, 'xml') # Parse as XML
        urls = [loc.text for loc in soup.find_all('loc') if loc.text]
        return urls

    def scrape_sitemap(self, max_items: Optional[int] = None) -> None:
        """
        Scrapes speech URLs from the sitemap.

        Args:
            max_items (Optional[int]): Maximum number of speeches to scrape.
        """
        all_sitemap_urls = self._get_all_links_from_sitemap(SITEMAP_URL)
        processed_count = 0
        for url in all_sitemap_urls:
            if any(selector.replace('a[href*="', '').replace('"]', '') in url for selector in SPEECH_SELECTORS):
                processed_speech = self.process_speech_url(url, 'sitemap')
                if processed_speech:
                    processed_count += 1
                    if max_items and processed_count >= max_items:
                        break
            time.sleep(self.delay)
        logger.info(f"Finished scraping sitemap. Processed {processed_count} new speeches.")

    def scrape_main_page(self, max_items: Optional[int] = None) -> None:
        """
        Scrapes speech URLs from the main speeches listing page.

        Args:
            max_items (Optional[int]): Maximum number of speeches to scrape.
        """
        logger.info(f"Scraping main speeches page: {SPEECHES_URL}")
        content = self._fetch_url(SPEECHES_URL)
        if not content:
            logger.error(f"Failed to fetch main speeches page: {SPEECHES_URL}")
            return

        soup = BeautifulSoup(content, 'html.parser')
        links = []
        for selector in SPEECH_SELECTORS:
            links.extend([a['href'] for a in soup.select(selector) if a.get('href')])
        
        processed_count = 0
        unique_urls = set()
        for link in links:
            full_url = urljoin(SPEECHES_URL, link)
            if full_url not in unique_urls: # Avoid processing duplicate links
                unique_urls.add(full_url)
                processed_speech = self.process_speech_url(full_url, 'main_page')
                if processed_speech:
                    processed_count += 1
                    if max_items and processed_count >= max_items:
                        break
            time.sleep(self.delay)
        logger.info(f"Finished scraping main speeches page. Processed {processed_count} new speeches.")

    def scrape_quarterly_bulletin(self, max_items: Optional[int] = None) -> None:
        """
        Scrapes speech URLs from the Quarterly Bulletin archive page.

        Args:
            max_items (Optional[int]): Maximum number of speeches to scrape.
        """
        logger.info(f"Scraping Quarterly Bulletin: {QUARTERLY_URL}")
        content = self._fetch_url(QUARTERLY_URL)
        if not content:
            logger.error(f"Failed to fetch Quarterly Bulletin page: {QUARTERLY_URL}")
            return

        soup = BeautifulSoup(content, 'html.parser')
        links = []
        # Adjust selector based on actual Quarterly Bulletin page structure if needed
        # Example: if links are within a specific div or class
        # Assuming speech links might be under a generic 'a' tag that contains 'speech' in href
        links.extend([a['href'] for a in soup.select('a[href]') if 'speech' in a.get('href', '').lower()])
        
        processed_count = 0
        unique_urls = set()
        for link in links:
            full_url = urljoin(QUARTERLY_URL, link)
            if full_url not in unique_urls:
                unique_urls.add(full_url)
                processed_speech = self.process_speech_url(full_url, 'quarterly_bulletin')
                if processed_speech:
                    processed_count += 1
                    if max_items and processed_count >= max_items:
                        break
            time.sleep(self.delay)
        logger.info(f"Finished scraping Quarterly Bulletin. Processed {processed_count} new speeches.")

    def scrape_digital_archive(self, max_items: Optional[int] = None) -> None:
        """
        Scrapes speeches from the Digital Archive using Selenium due to dynamic content.
        This method will require a working Selenium setup (e.g., ChromeDriver).

        Args:
            max_items (Optional[int]): Maximum number of speeches to scrape.
        """
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available. Please install selenium and a compatible WebDriver.")
            return

        logger.info(f"Scraping Digital Archive with Selenium: {ARCHIVE_URL}")
        
        options = Options()
        options.add_argument("--headless") # Run in headless mode for server environments
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36')


        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.get(ARCHIVE_URL)
            logger.info(f"WebDriver opened {ARCHIVE_URL}")

            # === IMPROVED WAITING STRATEGY ===
            # Wait for a more general element that indicates the main content area has loaded.
            # Example: A div containing search results, or the main content container.
            # You might need to inspect the page to find a reliable selector here.
            try:
                # Wait for an element that signifies the content is ready, e.g., a results container
                WebDriverWait(driver, 30).until( # Increased timeout
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div.search-results-container, div.archive-content')) 
                )
                logger.info("Main content area of Digital Archive loaded (or search results container).")
            except TimeoutException:
                logger.error(f"Timeout waiting for main archive content at {ARCHIVE_URL}. Page might not have loaded correctly or selector is wrong.")
                return
            except WebDriverException as e:
                logger.error(f"WebDriver error during initial page load/wait for {ARCHIVE_URL}: {e}")
                return

            processed_count = 0
            unique_urls = set()
            
            # Now that the main content is expected to be loaded, try finding links
            links = []
            try:
                # This selector needs to be highly specific to the actual document links on the archive page
                # The previous `a.document-title-link` might be too specific or incorrect for the current site structure.
                # Try a more generic link within a results area if the former fails.
                # Inspect the page source of https://boe.access.preservica.com to get the correct selector.
                # For example, if links are inside a list of search results:
                # links = driver.find_elements(By.CSS_SELECTOR, 'div.search-result-item a')
                links = driver.find_elements(By.CSS_SELECTOR, 'a.document-title-link, div.search-results a[href*="pdf"], div.search-results a[href*="doc"]')
                logger.info(f"Found {len(links)} potential links in Digital Archive.")
            except NoSuchElementException:
                logger.warning(f"Could not find any specific document links on {ARCHIVE_URL}. Review selectors.")
                pass # Continue even if no specific links found yet
            except WebDriverException as e:
                logger.error(f"WebDriver error while finding links on {ARCHIVE_URL}: {e}")
                pass # Continue if element finding itself causes an error

            for link_element in links:
                href = link_element.get_attribute('href')
                if href and href not in unique_urls:
                    full_url = urljoin(ARCHIVE_URL, href)
                    # Filter for actual speech/document URLs if necessary
                    # The archive often links directly to PDFs or documents
                    if ".pdf" in full_url.lower() or ".doc" in full_url.lower() or any(s.replace('a[href*="', '').replace('"]', '') in full_url for s in SPEECH_SELECTORS):
                        unique_urls.add(full_url)
                        processed_speech = self.process_speech_url(full_url, 'digital_archive')
                        if processed_speech:
                            processed_count += 1
                            if max_items and processed_count >= max_items:
                                break
                        time.sleep(self.delay)
            
            logger.info(f"Finished scraping Digital Archive. Processed {processed_count} new speeches.")

        except WebDriverException as e:
            logger.error(f"Selenium error in Digital Archive: {e.msg}")
            logger.error(f"Stacktrace:\n{e.stacktrace}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Selenium scraping: {e}")
        finally:
            if driver:
                driver.quit()
                logger.info("WebDriver closed.")

    def process_speech_url(self, url: str, source: str) -> bool:
        """
        Processes a single speech URL: fetches, extracts metadata/content, and saves.

        Args:
            url (str): The URL of the speech.
            source (str): The source from which the URL was found (e.g., 'sitemap', 'main_page').

        Returns:
            bool: True if the speech was successfully processed and saved, False otherwise.
        """
        # Create a unique key for the speech based on its URL hash
        key = hashlib.md5(url.encode('utf-8')).hexdigest()
        if (self.meta_dir / f"{key}.json").exists():
            logger.info(f"Speech already processed: {url}")
            return False

        logger.info(f"Processing speech: {url} from {source}")
        
        # Assume initial content fetch as text for HTML or binary for PDF
        content_raw = self._fetch_url_raw(url) 
        if not content_raw:
            logger.error(f"Failed to fetch content for {url}")
            return False

        file_type = 'html'
        if 'pdf' in url.lower() or (isinstance(content_raw, bytes) and content_raw.startswith(b'%PDF')):
            file_type = 'pdf'
        
        text_content = ""
        if file_type == 'html':
            text_content = self.extract_text_from_html(content_raw)
        elif file_type == 'pdf':
            text_content = self.extract_text_from_pdf(content_raw)

        if not text_content or len(text_content) < PDF_MIN_LENGTH:
            logger.warning(f"Extracted content too short or empty for {url}. Length: {len(text_content)}")
            return False

        metadata = self.process_speech_metadata(url, text_content, source)
        if not metadata.get('speaker') or not metadata.get('date'):
            logger.warning(f"Missing speaker or date for {url}. Skipping save.")
            return False

        self.save_speech(key, text_content, metadata)
        return True

    def _fetch_url_raw(self, url: str) -> Optional[Union[str, bytes]]:
        """
        Fetches URL content, returning bytes for non-text types (like PDF) or string for HTML.
        Includes User-Agent header for better request handling.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[Union[str, bytes]]: The raw content of the URL (str for HTML, bytes for others), or None.
        """
        self.current_url = url
        logger.debug(f"Fetching raw content for {url}")
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=15, stream=True) # Use stream for potentially large files
            response.raise_for_status()

            # Check content type header to decide how to read
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                for encoding in ENCODINGS: # Try common encodings for HTML
                    try:
                        response.encoding = encoding
                        return response.text
                    except UnicodeDecodeError:
                        continue
                logger.warning(f"Could not decode HTML for {url} with common encodings.")
                return response.text # Return raw text anyway, might be malformed
            elif 'application/pdf' in content_type:
                return response.content # Return raw bytes for PDF
            else:
                # For other types, try to decode as text, otherwise return bytes
                try:
                    return response.text
                except UnicodeDecodeError:
                    return response.content # Fallback to bytes if not text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
        finally:
            self.current_url = None
        return None


    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extracts clean text content from HTML, removing junk elements.

        Args:
            html_content (str): The raw HTML content.

        Returns:
            str: Cleaned text content.
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the main content area using predefined selectors
        main_content_tag = None
        for selector in TEXT_CONTENT_SELECTORS:
            main_content_tag = soup.select_one(selector)
            if main_content_tag:
                break
        
        if not main_content_tag:
            logger.warning(f"Could not find main content div for {self.current_url}. Extracting from body.")
            main_content_tag = soup.find('body') # Fallback to body if no specific content div found

        if not main_content_tag:
            return ""

        # Remove junk elements
        for selector in JUNK_SELECTORS:
            for junk_tag in main_content_tag.select(selector):
                junk_tag.decompose() # Remove the tag and its contents

        # Get text, strip excess whitespace and clean up
        text = main_content_tag.get_text(separator='\n', strip=True)
        text = re.sub(r'\n\s*\n', '\n\n', text) # Replace multiple newlines with at most two
        text = re.sub(r'\s{2,}', ' ', text) # Replace multiple spaces with single space
        return text.strip()

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extracts text content from PDF bytes.

        Args:
            pdf_bytes (bytes): The raw bytes of the PDF file.

        Returns:
            str: Extracted text content from the PDF.
        """
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or "" # Handle potential None from extract_text()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {self.current_url}: {e}")
            return ""

    def process_speech_metadata(self, url: str, content: str, source: str) -> Dict[str, str]:
        """
        Extracts metadata (date, speaker, title, role, voting status) from speech content.

        Args:
            url (str): The URL of the speech.
            content (str): The text content of the speech.
            source (str): The source from which the URL was found.

        Returns:
            Dict[str, str]: A dictionary containing extracted metadata.
        """
        meta = {
            'url': url,
            'date': 'Unknown',
            'speaker': 'Unknown',
            'title': 'Unknown',
            'source': source,
            'role': 'Unknown',
            'voting_status': 'Unknown'
        }

        # Try to extract date
        # Common date patterns: DD Month YYYY, Month DD, YYYY
        date_patterns = [
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b' # YYYY-MM-DD
        ]
        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    # Attempt to parse the date to YYYY-MM-DD format
                    parsed_date = datetime.strptime(match.group(1), '%d %B %Y')
                    meta['date'] = parsed_date.strftime('%Y-%m-%d')
                    break
                except ValueError:
                    try:
                        parsed_date = datetime.strptime(match.group(1), '%B %d, %Y')
                        meta['date'] = parsed_date.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        try:
                            parsed_date = datetime.strptime(match.group(1), '%Y-%m-%d')
                            meta['date'] = parsed_date.strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            pass # Continue to next pattern if parsing fails


        # Try to extract speaker and title from URL if possible
        url_parts = url.split('/')
        if len(url_parts) >= 2:
            # Example: .../speech/speaker-name-date-title
            # Attempt to find speaker and title from URL segments or common patterns
            # This requires knowing URL structure of the BoE
            # For Bank of England speeches, the URL often contains speaker and a simplified title
            # e.g., /speech/andrew-bailey-speech-2023-01-20
            # For now, rely heavily on text content, as URL structure varies.
            pass


        # Extract speaker and title from content, prioritizing speaker from database
        found_speaker_name = 'Unknown'
        found_role = 'Unknown'
        found_voting_status = 'Unknown'

        # Sort speakers by length of their name (longest first) to avoid partial matches
        sorted_speaker_keys = sorted(self.speaker_roles.keys(), key=len, reverse=True)

        for speaker_key in sorted_speaker_keys:
            if speaker_key in content.lower():
                found_speaker_name = speaker_key
                speaker_info = self.speaker_roles[speaker_key]
                found_role = speaker_info.get('role', 'Unknown')
                found_voting_status = speaker_info.get('voting_status', 'Unknown')
                break # Found the longest matching name

        meta['speaker'] = found_speaker_name
        meta['role'] = found_role
        meta['voting_status'] = found_voting_status

        # General title extraction (often appears near the beginning)
        # Look for typical speech title indicators
        title_match = re.search(r'(?:Speech by|Remarks by|Lecture by|Keynote address by|Statement by)\s+.*?\n\s*(.*?)', content, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).split('\n')[0].strip()
            if len(title) > 5: # Basic check for meaningful title
                meta['title'] = title
        else:
            # Fallback: take the first significant line after speaker/date if no explicit title found
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            for line in lines:
                if (found_speaker_name.lower() in line.lower() or meta['date'] in line) or \
                   (not re.match(r'^\d{1,2}\s+\w+\s+\d{4}', line) and not re.match(r'^(By|At|From)\s+.*', line, re.IGNORECASE)):
                    # Skip lines that are just date/speaker or common prefixes
                    continue
                if len(line) > 20 and len(line.split()) > 3: # Heuristic for a meaningful title line
                    meta['title'] = line
                    break


        # If speaker is still 'Unknown' but found in title, use that
        if meta['speaker'] == 'Unknown' and meta['title'] != 'Unknown':
            for speaker_key in sorted_speaker_keys:
                if speaker_key in meta['title'].lower():
                    meta['speaker'] = speaker_key
                    speaker_info = self.speaker_roles[speaker_key]
                    meta['role'] = speaker_info.get('role', 'Unknown')
                    meta['voting_status'] = speaker_info.get('voting_status', 'Unknown')
                    break

        return meta

    def save_speech(self, key: str, content: str, meta: Dict[str, str]) -> None:
        """
        Saves the speech content to a .txt file and metadata to a .json file.
        The text file is named based on speaker, date, and role.

        Args:
            key (str): A unique identifier for the speech (e.g., hash of URL).
            content (str): The extracted text content of the speech.
            meta (Dict[str, str]): A dictionary containing speech metadata.
        """
        # Create a clean filename for the .txt file
        speaker_name_for_filename = meta.get('speaker', 'Unknown').lower().replace(' ', '_').replace('.', '')
        date_for_filename = meta.get('date', 'Unknown_Date')
        role_for_filename = meta.get('role', 'unknown_role').lower().replace(' ', '_').replace('.', '')

        # Basic sanitization for filenames
        speaker_name_for_filename = re.sub(r'[^\w\-_\.]', '', speaker_name_for_filename)
        role_for_filename = re.sub(r'[^\w\-_\.]', '', role_for_filename)

        # Construct the .txt filename
        txt_filename = f"{speaker_name_for_filename}-{date_for_filename}-{role_for_filename}.txt"
        
        # Save content to .txt file
        (self.speeches_dir / txt_filename).write_text(content, encoding='utf-8')
        
        # Save metadata to .json file (named by hash key for uniqueness)
        (self.meta_dir / f"{key}.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')
        
        logger.info(f"Saved speech: {txt_filename} (metadata key: {key})")

    def export_metadata_sheet(self, output_file: Optional[Path] = None) -> None:
        """
        Compile metadata JSON files into a spreadsheet for reporting.

        Args:
            output_file (Optional[Path]): The path to save the Excel report.
                                          Defaults to 'data/boe/report.xlsx'.
        """
        output_file = output_file or (OUTPUT_DIR / 'report.xlsx')
        records = []
        for meta_file in self.meta_dir.glob('*.json'):
            data = json.loads(meta_file.read_text(encoding='utf-8'))
            records.append({
                'url': data.get('url'),
                'date': data.get('date'),
                'speaker': data.get('speaker'),
                'title': data.get('title'),
                'source': data.get('source'),
                'role': data.get('role'), # Add new field
                'voting_status': data.get('voting_status') # Add new field
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            # Reorder columns for better readability
            df = df[['speaker', 'role', 'voting_status', 'date', 'title', 'url', 'source']]
            df.to_excel(output_file, index=False)
            logger.info(f"Exported metadata sheet to {output_file}")
        else:
            logger.warning("No metadata found to export to Excel sheet.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced BoE Speech Scraper with Reporting")
    parser.add_argument('--method', choices=['sitemap','main','quarterly','archive','all'], default='all',
                        help="Scraping method: 'sitemap', 'main', 'quarterly', 'archive', or 'all'.")
    parser.add_argument('--max', type=int, dest='max_items',
                        help="Maximum number of items to scrape per method.")
    parser.add_argument('--delay', type=float, default=1.0,
                        help="Delay in seconds between requests to avoid rate limiting.")
    parser.add_argument('--no-report', action='store_true',
                        help="Do not export the final metadata Excel report.")
    args = parser.parse_args()

    scraper = EnhancedBoEScraper(delay=args.delay)

    if args.method == 'sitemap' or args.method == 'all':
        scraper.scrape_sitemap(max_items=args.max_items)
    if args.method == 'main' or args.method == 'all':
        scraper.scrape_main_page(max_items=args.max_items)
    if args.method == 'quarterly' or args.method == 'all':
        scraper.scrape_quarterly_bulletin(max_items=args.max_items)
    if args.method == 'archive' or args.method == 'all':
        scraper.scrape_digital_archive(max_items=args.max_items) # Note: max_items for Selenium might behave differently

    if not args.no_report:
        scraper.export_metadata_sheet()

    logger.info("Scraping process completed.")