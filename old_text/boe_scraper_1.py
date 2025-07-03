#!/usr/bin/env python3
"""
Enhanced Bank of England Speech Scraper v1 - Comprehensive Speech Collection
Based on Federal Reserve scraper architecture with BoE-specific adaptations.

Key Features:
- Robust date extraction with URL-first priority and content fallback
- Enhanced content extraction with multiple fallback strategies
- Comprehensive BoE officials database (Governors, Deputy Governors, MPC members)
- Strict validation to prevent saving empty/invalid content
- Support for both historical and current website structures
- URL-based speaker recognition with fallback patterns

Author: Central Bank Speech Collector
Date: 2025
Target: Bank of England speeches from 1990s onwards
"""

import requests
from bs4 import BeautifulSoup
import pdfplumber
import hashlib
import json
import os
import logging
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional, Tuple, Set
import time

# Optional Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBoEScraper:
    """
    Enhanced Bank of England speech scraper with robust dating and content extraction.
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.boe_dir = os.path.join(base_dir, "boe")
        self.base_url = "https://www.bankofengland.co.uk"
        self.speeches_url = "https://www.bankofengland.co.uk/news/speeches"
        self.sitemap_url = "https://www.bankofengland.co.uk/sitemap/speeches"  # Add this line

        
        # Ensure directories exist
        os.makedirs(self.boe_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
        
        # Initialize comprehensive speaker database
        self._initialize_speaker_database()
        
        # Initialize URL name patterns for fallback extraction
        self._initialize_url_patterns()
        
        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-GB,en;q=0.9,en-US;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Enhanced statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'url_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0
        }

    def _initialize_speaker_database(self):
        """Initialize comprehensive Bank of England officials database."""
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
            "john": {"role": "Chief Operating Officer", "voting_status": "Non-Voting", "years": "2025-present"},
            
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
            
            # Past Governors
            "mark carney": {"role": "Governor", "voting_status": "Voting Member", "years": "2013-2020"},
            "mark joseph carney": {"role": "Governor", "voting_status": "Voting Member", "years": "2013-2020"},
            "carney": {"role": "Governor", "voting_status": "Voting Member", "years": "2013-2020"},
            
            "mervyn king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            "mervyn allister king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            "king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            "lord king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            "baron king": {"role": "Governor", "voting_status": "Voting Member", "years": "2003-2013"},
            
            "eddie george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "edward george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "edward alan john george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "steady eddie": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "lord george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            "baron george": {"role": "Governor", "voting_status": "Voting Member", "years": "1993-2003"},
            
            "robin leigh-pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "robin leigh pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "leigh-pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "leigh pemberton": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "lord kingsdown": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            "baron kingsdown": {"role": "Governor", "voting_status": "Voting Member", "years": "1983-1993"},
            
            # Past Deputy Governors (Recent)
            "ben broadbent": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2011-2024"},
            "broadbent": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2011-2024"},
            
            "jon cunliffe": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2013-2023"},
            "sir jon cunliffe": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2013-2023"},
            "cunliffe": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2013-2023"},
            
            "sam woods": {"role": "Deputy Governor for Prudential Regulation", "voting_status": "Non-Voting", "years": "2016-present"},
            "samuel woods": {"role": "Deputy Governor for Prudential Regulation", "voting_status": "Non-Voting", "years": "2016-present"},
            "woods": {"role": "Deputy Governor for Prudential Regulation", "voting_status": "Non-Voting", "years": "2016-present"},
            
            "charlotte hogg": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "2017"},
            "hogg": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "2017"},
            
            "nemat shafik": {"role": "Deputy Governor for Markets and Banking", "voting_status": "Voting Member", "years": "2014-2017"},
            "nemat talaat shafik": {"role": "Deputy Governor for Markets and Banking", "voting_status": "Voting Member", "years": "2014-2017"},
            "minouche shafik": {"role": "Deputy Governor for Markets and Banking", "voting_status": "Voting Member", "years": "2014-2017"},
            "shafik": {"role": "Deputy Governor for Markets and Banking", "voting_status": "Voting Member", "years": "2014-2017"},
            
            "paul tucker": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2009-2013"},
            "paul mw tucker": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2009-2013"},
            "tucker": {"role": "Deputy Governor for Financial Stability", "voting_status": "Voting Member", "years": "2009-2013"},
            
            "charlie bean": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2008-2014"},
            "charles bean": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2008-2014"},
            "charles goodhart bean": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2008-2014"},
            "bean": {"role": "Deputy Governor for Monetary Policy", "voting_status": "Voting Member", "years": "2008-2014"},
            
            "john gieve": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "2006-2009"},
            "sir john gieve": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "2006-2009"},
            "gieve": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "2006-2009"},
            
            "rachel lomax": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "2003-2008"},
            "lomax": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "2003-2008"},
            
            "david clementi": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1997-2002"},
            "sir david clementi": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1997-2002"},
            "clementi": {"role": "Deputy Governor", "voting_status": "Voting Member", "years": "1997-2002"},
            
            # Chief Economists (Past and Present)
            "andy haldane": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2014-2021"},
            "andrew haldane": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2014-2021"},
            "andrew g haldane": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2014-2021"},
            "haldane": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2014-2021"},
            
            "spencer dale": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2008-2014"},
            "dale": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2008-2014"},
            
            "charlie bean": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "2000-2008"},
            # (Already listed above as Deputy Governor)
            
            # Past External MPC Members (Recent)
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
            
            # Add to _initialize_speaker_database method:
            "megan greene": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2023-present"},
            "greene": {"role": "External MPC Member", "voting_status": "Voting Member", "years": "2023-present"},

            "afua kyei": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2022-present"},
            "kyei": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2022-present"},

            "victoria cleland": {"role": "Executive Director for Markets", "voting_status": "Non-Voting", "years": "2014-present"},
            "cleland": {"role": "Executive Director for Markets", "voting_status": "Non-Voting", "years": "2014-present"},

            "james benford": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2020-present"},
            "benford": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2020-present"},

            "sasha mills": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2021-present"},
            "mills": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2021-present"},

            "philippe lintern": {"role": "Senior Manager", "voting_status": "Non-Voting", "years": "2018-present"},
            "lintern": {"role": "Senior Manager", "voting_status": "Non-Voting", "years": "2018-present"},

            "gareth truran": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2019-present"},
            "truran": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2019-present"},

            "nathanael benjamin": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2021-present"},
            "benjamin": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2021-present"},

            "shoib khan": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2020-present"},
            "khan": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2020-present"},

            "phil evans": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2019-present"},
            "evans": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2019-present"},

            "victoria saporta": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2018-present"},
            "saporta": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2018-present"},

            "lee foulger": {"role": "Senior Manager", "voting_status": "Non-Voting", "years": "2015-present"},
            "foulger": {"role": "Senior Manager", "voting_status": "Non-Voting", "years": "2015-present"},

            "david bailey": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2020-present"},  # Different from Andrew Bailey
            "randall kroszner": {"role": "External Board Member", "voting_status": "Non-Voting", "years": "2024-present"},
            "kroszner": {"role": "External Board Member", "voting_status": "Non-Voting", "years": "2024-present"},

            "jon hall": {"role": "Senior Manager", "voting_status": "Non-Voting", "years": "2018-present"},
            "hall": {"role": "Senior Manager", "voting_status": "Non-Voting", "years": "2018-present"},

            # Other Senior Officials
            "paul fisher": {"role": "Executive Director for Markets", "voting_status": "Voting Member", "years": "2009-2013"},
            "fisher": {"role": "Executive Director for Markets", "voting_status": "Voting Member", "years": "2009-2013"},
            
            "david rule": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2013-present"},
            "rule": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2013-present"},
            
            "andrew gracie": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2012-2018"},
            "gracie": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2012-2018"},
            
            "chris salmon": {"role": "Executive Director for Markets", "voting_status": "Non-Voting", "years": "2007-2014"},
            "christopher salmon": {"role": "Executive Director for Markets", "voting_status": "Non-Voting", "years": "2007-2014"},
            "salmon": {"role": "Executive Director for Markets", "voting_status": "Non-Voting", "years": "2007-2014"},
            
            "alex brazier": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2014-2021"},
            "alexander brazier": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2014-2021"},
            "brazier": {"role": "Executive Director", "voting_status": "Non-Voting", "years": "2014-2021"},
            
            "john vickers": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "1998-2000"},
            "sir john vickers": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "1998-2000"},
            "vickers": {"role": "Chief Economist", "voting_status": "Voting Member", "years": "1998-2000"},
        }

    def _initialize_url_patterns(self):
        """Initialize URL-based name patterns for fallback extraction."""
        self.url_name_patterns = {}
        
        # Build reverse mapping from speaker database
        for full_name, info in self.speaker_roles.items():
            # Extract last name for URL matching
            if ' ' in full_name:
                last_name = full_name.split()[-1].lower()
                if last_name not in self.url_name_patterns:
                    self.url_name_patterns[last_name] = []
                self.url_name_patterns[last_name].append(full_name)
            
            # Also add full name variants for URL matching
            url_friendly = full_name.replace(' ', '').replace('.', '').replace('-', '').lower()
            if url_friendly not in self.url_name_patterns:
                self.url_name_patterns[url_friendly] = []
            self.url_name_patterns[url_friendly].append(full_name)
        
        # Add common URL patterns manually for better matching
        manual_patterns = {
            'bailey': ['andrew bailey', 'andrew john bailey'],
            'carney': ['mark carney', 'mark joseph carney'],
            'king': ['mervyn king', 'mervyn allister king', 'lord king', 'baron king'],
            'george': ['eddie george', 'edward george', 'edward alan john george', 'steady eddie', 'lord george', 'baron george'],
            'leighpemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'leigh-pemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'broadbent': ['ben broadbent'],
            'cunliffe': ['jon cunliffe', 'sir jon cunliffe'],
            'woods': ['sam woods', 'samuel woods'],
            'ramsden': ['dave ramsden', 'david ramsden'],
            'breeden': ['sarah breeden'],
            'lombardelli': ['clare lombardelli'],
            'pill': ['huw pill'],
            'mann': ['catherine mann', 'catherine l mann'],
            'haskel': ['jonathan haskel'],
            'dhingra': ['swati dhingra'],
            'taylor': ['alan taylor'],
            'haldane': ['andy haldane', 'andrew haldane', 'andrew g haldane'],
            'dale': ['spencer dale'],
            'bean': ['charlie bean', 'charles bean', 'charles goodhart bean'],
            'tucker': ['paul tucker', 'paul mw tucker'],
            'shafik': ['nemat shafik', 'nemat talaat shafik', 'minouche shafik'],
            'hogg': ['charlotte hogg'],
            'gieve': ['john gieve', 'sir john gieve'],
            'lomax': ['rachel lomax'],
            'clementi': ['david clementi', 'sir david clementi'],
            'tenreyro': ['silvana tenreyro'],
            'vlieghe': ['gertjan vlieghe'],
            'saunders': ['michael saunders'],
            'mccafferty': ['ian mccafferty'],
            'weale': ['martin weale'],
            'miles': ['david miles'],
            'posen': ['adam posen'],
            'sentance': ['andrew sentance'],
            'barker': ['kate barker'],
            'blanchflower': ['david blanchflower', 'danny blanchflower'],
            'nickell': ['stephen nickell'],
            'allsopp': ['christopher allsopp'],
            'wadhwani': ['sushil wadhwani'],
            'julius': ['deanne julius'],
            'budd': ['alan budd'],
            'buiter': ['willem buiter'],
            'fisher': ['paul fisher'],
            'rule': ['david rule'],
            'gracie': ['andrew gracie'],
            'salmon': ['chris salmon', 'christopher salmon'],
            'brazier': ['alex brazier', 'alexander brazier'],
            'vickers': ['john vickers', 'sir john vickers'],
            'john': ['sarah john']
        }
        
        # Update with manual patterns
        for pattern, names in manual_patterns.items():
            if pattern not in self.url_name_patterns:
                self.url_name_patterns[pattern] = []
            self.url_name_patterns[pattern].extend(names)
            # Remove duplicates
            self.url_name_patterns[pattern] = list(set(self.url_name_patterns[pattern]))

    def scrape_speeches_from_sitemap(self, max_speeches: Optional[int] = None) -> List[Dict]:
        """
        Scrape speeches from the BoE sitemap - most reliable method.
        """
        logger.info(f"Scraping BoE speeches from sitemap: {self.sitemap_url}")
        all_speeches = []
        
        try:
            response = requests.get(self.sitemap_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all speech links - they follow pattern /speech/YYYY/month/title
            speech_links = soup.find_all('a', href=True)
            
            for link in speech_links:
                href = link.get('href')
                
                # BoE speech URLs follow pattern: /speech/YYYY/month/title-slug
                if href and '/speech/' in href and '/speech/20' in href:
                    full_url = urljoin(self.base_url, href)
                    
                    # Extract date from URL (more reliable than content parsing)
                    url_date = self.extract_date_from_url(full_url)
                    
                    if url_date:
                        speech_info = {
                            'source_url': full_url,
                            'title': link.get_text(strip=True),
                            'date': url_date,
                            'date_source': 'url',
                            'speaker_raw': '',
                            'context_text': link.get_text(strip=True)
                        }
                        all_speeches.append(speech_info)
            
            # Remove duplicates and sort by date (newest first)
            unique_speeches = self._deduplicate_speeches(all_speeches)
            unique_speeches.sort(key=lambda x: x['date'], reverse=True)
            
            if max_speeches:
                unique_speeches = unique_speeches[:max_speeches]
            
            logger.info(f"Found {len(unique_speeches)} speeches from sitemap")
            return unique_speeches
            
        except requests.RequestException as e:
            logger.error(f"Error scraping sitemap: {e}")
            return []

    # ENHANCED DATE EXTRACTION WITH URL PRIORITY
    def extract_date_from_url(self, url: str) -> Optional[str]:
        """
        Extract date from URL with high confidence.
        BoE URLs follow patterns like: /speech/2024/october/title-slug
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            # BoE URL date patterns - Updated for current structure
            date_patterns = [
                # Current BoE pattern: /speech/2024/october/title-slug
                r'/speech/(\d{4})/(\w+)/',
                # Pattern: /speech/2024/10/title-slug  
                r'/speech/(\d{4})/(\d{1,2})/',
                # Legacy patterns
                r'/speeches/(\d{4})/(\w+)/',
                r'/speeches/(\d{4})/(\d{1,2})/',
                # Date in title: /speech/2024/october/speech-2024-10-15
                r'/speech/\d{4}/\w+/.*?(\d{4})-(\d{2})-(\d{2})',
                # Pattern: embedded YYYYMMDD
                r'(\d{8})',
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, path)
                if match:
                    if len(match.groups()) == 2:
                        year_str, month_str = match.groups()
                        year = int(year_str)
                        
                        # Handle month names (common in BoE URLs)
                        month_names = {
                            'january': 1, 'february': 2, 'march': 3, 'april': 4,
                            'may': 5, 'june': 6, 'july': 7, 'august': 8,
                            'september': 9, 'october': 10, 'november': 11, 'december': 12,
                            # Short forms
                            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                        }
                        
                        if month_str.lower() in month_names:
                            month = month_names[month_str.lower()]
                        else:
                            month = int(month_str)
                        
                        # Use first day of month as default
                        if 1990 <= year <= 2030 and 1 <= month <= 12:
                            date_obj = datetime(year, month, 1)
                            formatted_date = date_obj.strftime('%Y-%m-%d')
                            logger.info(f"URL date extraction successful: {url} -> {formatted_date}")
                            return formatted_date
                            
                    elif len(match.groups()) == 3:
                        year_str, month_str, day_str = match.groups()
                        try:
                            year = int(year_str)
                            month = int(month_str)
                            day = int(day_str)
                            
                            if 1990 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                date_obj = datetime(year, month, day)
                                formatted_date = date_obj.strftime('%Y-%m-%d')
                                logger.info(f"URL date extraction successful: {url} -> {formatted_date}")
                                return formatted_date
                        except ValueError:
                            continue
            
            logger.debug(f"No valid date pattern found in URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting date from URL {url}: {e}")
            return None

    def extract_speaker_from_url(self, url: str) -> Optional[str]:
        """
        Extract speaker name from URL as fallback method.
        BoE URLs often follow patterns like:
        - /speeches/2024/december/speech-by-andrew-bailey
        - /speeches/2024/december/bailey-speech-title
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            # Common URL patterns in BoE speeches
            url_patterns = [
                # Pattern: speech-by-firstname-lastname
                r'speech-by-([a-z]+)-([a-z-]+)',
                # Pattern: lastname-speech or lastname-remarks
                r'([a-z]+)-(speech|remarks|address)',
                # Pattern: firstname-lastname-speech
                r'([a-z]+)-([a-z]+)-(speech|remarks|address)',
                # Pattern: /lastname/
                r'/([a-z]+)/',
            ]
            
            for pattern in url_patterns:
                match = re.search(pattern, path)
                if match:
                    if len(match.groups()) >= 2:
                        name_parts = [match.group(1), match.group(2)]
                        if match.group(2) not in ['speech', 'remarks', 'address']:
                            candidate_name = ' '.join(name_parts).replace('-', ' ')
                        else:
                            candidate_name = match.group(1).replace('-', ' ')
                    else:
                        candidate_name = match.group(1).replace('-', ' ')
                    
                    # Clean the candidate name
                    candidate_name = candidate_name.strip()
                    
                    # Check if this matches any of our known patterns
                    if candidate_name in self.url_name_patterns:
                        matched_names = self.url_name_patterns[candidate_name]
                        logger.info(f"URL speaker extraction: '{candidate_name}' -> '{matched_names[0]}'")
                        self.stats['url_fallback_used'] += 1
                        return matched_names[0]
                    
                    # Also try just the last name
                    last_name = candidate_name.split()[-1] if ' ' in candidate_name else candidate_name
                    if last_name in self.url_name_patterns:
                        matched_names = self.url_name_patterns[last_name]
                        logger.info(f"URL speaker extraction (lastname): '{last_name}' -> '{matched_names[0]}'")
                        self.stats['url_fallback_used'] += 1
                        return matched_names[0]
            
            logger.debug(f"No speaker pattern matched for URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting speaker from URL {url}: {e}")
            return None

    def get_speaker_info(self, speaker_name: str, url: str = None) -> Dict[str, str]:
        """
        Get comprehensive speaker information with URL fallback.
        """
        if not speaker_name or speaker_name.strip() == 'Unknown':
            # Try URL fallback if primary extraction failed
            if url:
                url_extracted_name = self.extract_speaker_from_url(url)
                if url_extracted_name:
                    speaker_name = url_extracted_name
                else:
                    self.stats['unknown_speakers'] += 1
                    return {"role": "Unknown", "voting_status": "Unknown", "source": "unknown"}
            else:
                self.stats['unknown_speakers'] += 1
                return {"role": "Unknown", "voting_status": "Unknown", "source": "unknown"}
        
        # Clean and normalize the name for lookup
        normalized_name = self._clean_speaker_name_for_lookup(speaker_name)
        
        # Method 1: Try exact match first
        if normalized_name in self.speaker_roles:
            info = self.speaker_roles[normalized_name].copy()
            info['source'] = 'exact_match'
            logger.debug(f"Exact match found: {speaker_name} -> {normalized_name}")
            return info
        
        # Method 2: Try partial matching for names with different formats
        for known_name, info in self.speaker_roles.items():
            if self._names_match(normalized_name, known_name):
                result = info.copy()
                result['source'] = 'partial_match'
                logger.debug(f"Partial match found: {speaker_name} -> {known_name}")
                return result
        
        # Method 3: Try last name only matching
        last_name = normalized_name.split()[-1] if ' ' in normalized_name else normalized_name
        if last_name in self.speaker_roles:
            result = self.speaker_roles[last_name].copy()
            result['source'] = 'lastname_match'
            logger.debug(f"Last name match found: {speaker_name} -> {last_name}")
            return result
        
        # Method 4: URL fallback as absolute last resort
        if url and (not speaker_name or speaker_name.strip() == 'Unknown'):
            url_extracted_name = self.extract_speaker_from_url(url)
            if url_extracted_name:
                # Recursively call with extracted name (but without URL to avoid infinite loop)
                return self.get_speaker_info(url_extracted_name)
        
        # Final fallback: Unknown speaker
        logger.warning(f"Unknown speaker after all methods: {speaker_name} (normalized: {normalized_name})")
        self.stats['unknown_speakers'] += 1
        return {"role": "Unknown", "voting_status": "Unknown", "source": "unknown"}

    def _clean_speaker_name_for_lookup(self, name: str) -> str:
        """Clean and normalize speaker name for database lookup."""
        if not name:
            return ""
        
        # Remove titles and clean
        name = re.sub(r'\b(?:Governor|Deputy Governor|Chair|President|Dr\.|Mr\.|Ms\.|Mrs\.|Sir|Lord|Baron)\s*', '', name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        name = re.split(r'\s*(?:,|by|remarks|speech)\s*', name, flags=re.IGNORECASE)[0]
        
        # Remove newlines and extra whitespace
        name = ' '.join(name.split())
        
        # Convert to lowercase and remove periods
        name = name.lower().strip().replace('.', '')
        
        return name

    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names likely refer to the same person with enhanced matching."""
        if not name1 or not name2:
            return False
            
        parts1 = set(name1.replace('.', '').replace('-', ' ').split())
        parts2 = set(name2.replace('.', '').replace('-', ' ').split())
        
        # Remove common middle initials and abbreviations
        parts1 = {p for p in parts1 if len(p) > 1}
        parts2 = {p for p in parts2 if len(p) > 1}
        
        common_parts = parts1.intersection(parts2)
        
        # For short names, require full overlap
        if len(parts1) <= 2 and len(parts2) <= 2:
            return len(common_parts) >= min(len(parts1), len(parts2))
        else:
            # For longer names, require at least 2 matching parts
            return len(common_parts) >= 2

    # ENHANCED CONTENT EXTRACTION WITH MULTIPLE STRATEGIES
    def _extract_main_content_enhanced(self, soup: BeautifulSoup) -> str:
        """
        Enhanced content extraction with multiple fallback strategies.
        Addresses the core issue of short/empty content extraction.
        """
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.navigation', 'noscript']):
            element.decompose()
        
        content_candidates = []
        
        # Strategy 1: Try BoE-specific selectors (current website structure)
        boe_selectors = [
            'div.main-content',
            '.speech-content',
            '.article-content',
            '[role="main"]',
            'main',
            'article',
            '.content-area',
            '#main-content',
            '.page-content',
            '.speech-text'
        ]
        
        for selector in boe_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
                if len(text) > 500:  # Must be substantial
                    content_candidates.append(('boe_selector', text, len(text)))
                    logger.debug(f"Found content using selector {selector}: {len(text)} chars")
        
        # Strategy 2: Try to find the largest content block
        # Look for divs with substantial text content
        all_divs = soup.find_all('div')
        for div in all_divs:
            text = div.get_text(separator='\n', strip=True)
            if len(text) > 1000:  # Must be very substantial for this method
                # Check that it's not just navigation or boilerplate
                if not any(skip_text in text.lower() for skip_text in 
                          ['navigation', 'skip to', 'breadcrumb', 'footer', 'sidebar']):
                    content_candidates.append(('largest_div', text, len(text)))
        
        # Strategy 3: Paragraph aggregation (fallback for older pages)
        paragraphs = soup.find_all('p')
        if paragraphs:
            para_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            if len(para_text) > 500:
                content_candidates.append(('paragraphs', para_text, len(para_text)))
        
        # Strategy 4: Body content (last resort)
        body = soup.find('body')
        if body:
            body_text = body.get_text(separator='\n', strip=True)
            if len(body_text) > 800:
                content_candidates.append(('body', body_text, len(body_text)))
        
        # Choose the best candidate
        if content_candidates:
            # Sort by length (prefer longer content) and strategy priority
            content_candidates.sort(key=lambda x: (x[2], x[0] == 'boe_selector'), reverse=True)
            best_strategy, best_content, best_length = content_candidates[0]
            
            logger.info(f"Content extraction strategy: {best_strategy} ({best_length} chars)")
            
            cleaned_content = self._clean_text_content(best_content)
            logger.info(f"After cleaning: {len(cleaned_content)} chars")
            
            return cleaned_content
        
        logger.warning("No substantial content found with any extraction strategy")
        return ""

    def _clean_text_content(self, text: str) -> str:
        """Enhanced text cleaning with BoE-specific patterns."""
        if not text:
            return ""
        
        original_length = len(text)
        
        # Basic whitespace normalization only
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        # Only remove very specific, safe patterns
        safe_patterns = [
            r'Return to top\s*',
            r'Last update:.*?\n',
            r'Skip to main content\s*',
            r'Print this page\s*',
            r'Share this page\s*',
            r'Bank of England\s*',
            r'^\s*Speeches\s*',
        ]
        
        for pattern in safe_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        final_length = len(text)
        
        # Log suspicious cleaning
        if original_length > 1000 and final_length < 200:
            logger.warning(f"Suspicious text cleaning: {original_length} -> {final_length} chars")
        
        return text

    # ENHANCED DATE PARSING WITH BETTER VALIDATION
    def _parse_date_enhanced(self, date_str: str, url: str = None) -> str:
        """
        Enhanced date parsing with URL priority and strict validation.
        """
        # PRIORITY 1: Try URL extraction first (most reliable)
        if url:
            url_date = self.extract_date_from_url(url)
            if url_date:
                return url_date
        
        if not date_str:
            logger.warning(f"No date string provided and URL extraction failed for {url}")
            self.stats['date_extraction_failures'] += 1
            return None
        
        # PRIORITY 2: Parse the provided date string
        formats = [
            '%d %B %Y',    # "15 January 2025"
            '%B %d, %Y',   # "January 15, 2025"
            '%b %d, %Y',   # "Jan 15, 2025"
            '%d %b %Y',    # "15 Jan 2025"
            '%m/%d/%Y',    # "01/15/2025"
            '%d/%m/%Y',    # "15/01/2025"
            '%m-%d-%Y',    # "01-15-2025"
            '%d-%m-%Y',    # "15-01-2025"
            '%Y-%m-%d',    # "2025-01-15"
            '%Y/%m/%d',    # "2025/01/15"
            '%B %d %Y',    # "January 15 2025" (without comma)
            '%b %d %Y',    # "Jan 15 2025" (without comma)
            '%d %B %Y',    # "15 January 2025"
            '%Y%m%d',      # "20250115" (YYYYMMDD)
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                # Validate reasonable date range
                if 1990 <= dt.year <= 2030:
                    formatted_date = dt.strftime('%Y-%m-%d')
                    logger.debug(f"Successfully parsed date: {date_str} -> {formatted_date}")
                    return formatted_date
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str} for URL: {url}")
        self.stats['date_extraction_failures'] += 1
        return None

    # COMPREHENSIVE VALIDATION SYSTEM
    def _validate_speech_data(self, speech_data: Dict) -> bool:
        """
        Comprehensive validation to prevent saving invalid speeches.
        """
        if not speech_data or 'metadata' not in speech_data or 'content' not in speech_data:
            logger.warning("Speech data missing required components")
            self.stats['validation_failures'] += 1
            return False
        
        metadata = speech_data['metadata']
        content = speech_data['content']
        
        # Content validation
        if not content or len(content.strip()) < 50:
            logger.warning(f"Content too short: {len(content.strip()) if content else 0} chars")
            self.stats['content_too_short'] += 1
            return False
        
        # Check for placeholder content
        placeholder_indicators = [
            'lorem ipsum',
            'placeholder',
            'test content',
            'coming soon',
            'under construction',
            'page not found'
        ]
        
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in placeholder_indicators):
            logger.warning("Content appears to be placeholder text")
            self.stats['validation_failures'] += 1
            return False
        
        # Metadata validation
        required_fields = ['title', 'speaker', 'date', 'source_url']
        for field in required_fields:
            if not metadata.get(field):
                logger.warning(f"Missing required metadata field: {field}")
                self.stats['validation_failures'] += 1
                return False
        
        # Date validation (don't allow None or current year defaults)
        date_str = metadata.get('date')
        if not date_str or date_str.startswith('2025-01-01'):
            logger.warning(f"Invalid or default date: {date_str}")
            self.stats['validation_failures'] += 1
            return False
        
        # Speaker validation
        speaker = metadata.get('speaker', '').lower()
        if speaker in ['unknown', 'unknown speaker', '']:
            logger.warning(f"Unknown speaker: {metadata.get('speaker')}")
            self.stats['validation_failures'] += 1
            return False
        
        # Title validation
        title = metadata.get('title', '')
        if len(title) < 10 or title.lower() in ['untitled speech', 'untitled', 'speech']:
            logger.warning(f"Invalid title: {title}")
            self.stats['validation_failures'] += 1
            return False
        
        return True

    # ENHANCED SCRAPING WITH BETTER ERROR HANDLING
    def scrape_speeches_by_year_range(self, start_year: int = 1996, end_year: int = None) -> List[Dict]:
        """
        Enhanced year-by-year scraping with better error handling.
        Note: BoE may not have year-specific pages like Fed, so this will try common patterns.
        """
        if end_year is None:
            end_year = datetime.now().year
        
        logger.info(f"Scraping BoE speeches from {start_year} to {end_year}")
        all_speeches = []
        
        # Try different URL patterns for yearly speeches
        year_patterns = [
            "https://www.bankofengland.co.uk/news/speeches/{year}",
            "https://www.bankofengland.co.uk/speeches/{year}",
            "https://www.bankofengland.co.uk/news/speeches?year={year}",
        ]
        
        for year in range(start_year, end_year + 1):
            year_found = False
            
            for pattern in year_patterns:
                year_url = pattern.format(year=year)
                logger.info(f"Trying year {year}: {year_url}")
                
                try:
                    response = requests.get(year_url, headers=self.headers, timeout=30)
                    
                    if response.status_code == 404:
                        logger.debug(f"No speeches page found for {year} at {year_url} (404)")
                        continue
                    elif response.status_code != 200:
                        logger.warning(f"Failed to fetch {year} speeches: HTTP {response.status_code}")
                        continue
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    year_speeches = self._extract_speeches_from_year_page(soup, year)
                    
                    if year_speeches:
                        logger.info(f"Found {len(year_speeches)} speeches for {year}")
                        all_speeches.extend(year_speeches)
                        year_found = True
                        break
                    
                except requests.RequestException as e:
                    logger.error(f"Network error scraping {year}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error scraping {year}: {e}")
                    continue
            
            if not year_found:
                logger.info(f"No speeches found for {year} using any pattern")
            
            # Respectful delay
            time.sleep(1)
        
        logger.info(f"Total speeches found across all years: {len(all_speeches)}")
        return all_speeches

    def scrape_speeches_main_page(self, max_speeches: int = 50) -> List[Dict]:
        """
        Scrape speeches from the main speeches page.
        """
        logger.info(f"Scraping BoE main speeches page: {self.speeches_url}")
        all_speeches = []
        
        try:
            response = requests.get(self.speeches_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            speeches = self._extract_speeches_from_main_page(soup)
            
            if max_speeches:
                speeches = speeches[:max_speeches]
            
            logger.info(f"Found {len(speeches)} speeches on main page")
            all_speeches.extend(speeches)
            
        except requests.RequestException as e:
            logger.error(f"Error scraping main speeches page: {e}")
        except Exception as e:
            logger.error(f"Unexpected error scraping main speeches page: {e}")
        
        return all_speeches

    def _extract_speeches_from_main_page(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract speeches from the main speeches page with updated selectors."""
        speeches = []
        
        # Updated selectors for current BoE website structure
        speech_selectors = [
            'a[href*="/speech/"]',  # Current BoE structure
            'a[href*="/news/speeches/"]',  # Legacy structure
            '.card a[href*="/speech/"]',  # Card-based layout
            '.listing-item a[href*="/speech/"]',  # List layout
            'article a[href*="/speech/"]',  # Article links
            '[data-type="speech"] a',  # Data attribute based
        ]
        
        for selector in speech_selectors:
            links = soup.select(selector)
            
            for link in links:
                href = link.get('href')
                
                if href and '/speech/' in href:
                    # Skip navigation and filter links
                    if any(skip in href for skip in ['?', '#', 'page=']):
                        continue
                    
                    full_url = urljoin(self.base_url, href)
                    
                    # Extract metadata from link context
                    speech_info = self._extract_speech_metadata_from_context(link, None, full_url)
                    if speech_info:
                        speech_info['source_url'] = full_url
                        speeches.append(speech_info)
        
        # Remove duplicates by URL
        return self._deduplicate_speeches(speeches)

    def _extract_speeches_from_year_page(self, soup: BeautifulSoup, year: int) -> List[Dict]:
        """Enhanced speech extraction from yearly pages."""
        speeches = []
        
        # Find all speech links with better pattern matching
        speech_links = soup.find_all('a', href=True)
        
        for link in speech_links:
            href = link.get('href')
            
            # Enhanced BoE speech URL pattern matching
            if href and ('/news/speeches/' in href or '/speeches/' in href):
                # Skip year navigation links
                if href.endswith('/speeches') or f'/{year}' not in href:
                    continue
                
                full_url = urljoin(self.base_url, href)
                
                # Extract metadata from link context
                speech_info = self._extract_speech_metadata_from_context(link, year, full_url)
                if speech_info:
                    speech_info['source_url'] = full_url
                    speech_info['year'] = year
                    speeches.append(speech_info)
        
        # Also extract from structured content (tables, lists)
        speeches.extend(self._extract_from_structured_content(soup, year))
        
        # Remove duplicates by URL
        return self._deduplicate_speeches(speeches)

    def _extract_speech_metadata_from_context(self, link, year: int, url: str) -> Optional[Dict]:
        """Enhanced metadata extraction with URL priority for dating."""
        try:
            link_text = link.get_text(strip=True)
            
            # Skip if not substantial enough to be a speech title
            if not link_text or len(link_text) < 5:
                return None
            
            # PRIORITY 1: Extract date from URL (most reliable)
            url_date = self.extract_date_from_url(url)
            
            # Look for metadata in parent elements (up to 3 levels)
            parent = link.parent
            context_text = ""
            date_from_context = None
            speaker_from_context = None
            
            for level in range(3):
                if parent is None:
                    break
                
                parent_text = parent.get_text()
                context_text += parent_text + " "
                
                # Look for date patterns (only if URL date extraction failed)
                if not url_date:
                    date_patterns = [
                        r'\b(\d{1,2}\s+\w+\s+\d{4})\b',  # "15 January 2025"
                        r'\b(\w+\s+\d{1,2},\s+\d{4})\b',  # "January 15, 2025"
                        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'  # "15/01/2025"
                    ]
                    
                    for pattern in date_patterns:
                        date_match = re.search(pattern, parent_text)
                        if date_match:
                            date_from_context = date_match.group(1)
                            break
                
                # Look for speaker patterns
                speaker_patterns = [
                    r'\b(?:Governor|Deputy Governor|Chair|President)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                    r'\b(?:By|Remarks by|Speech by)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair|President)'
                ]
                
                for pattern in speaker_patterns:
                    speaker_match = re.search(pattern, parent_text)
                    if speaker_match:
                        speaker_from_context = speaker_match.group(1)
                        break
                
                parent = parent.parent
            
            # Determine the best date
            final_date = None
            if url_date:
                final_date = url_date
                date_source = 'url'
            elif date_from_context:
                parsed_date = self._parse_date_enhanced(date_from_context, url)
                if parsed_date:
                    final_date = parsed_date
                    date_source = 'context'
            
            # Don't return speech info if we can't get a reliable date
            if not final_date:
                logger.warning(f"Could not extract reliable date for speech: {url}")
                return None
            
            return {
                'title': link_text,
                'date_raw': date_from_context or '',
                'date': final_date,
                'date_source': date_source,
                'speaker_raw': speaker_from_context or '',
                'context_text': context_text.strip()
            }
            
        except Exception as e:
            logger.debug(f"Error extracting speech metadata: {e}")
            return None

    def _extract_from_structured_content(self, soup: BeautifulSoup, year: int) -> List[Dict]:
        """Extract speeches from structured content like tables."""
        speeches = []
        
        # Look for tables with speech listings
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                links = row.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    if href and ('/news/speeches/' in href or '/speeches/' in href):
                        full_url = urljoin(self.base_url, href)
                        speech_info = self._extract_speech_metadata_from_table_row(row, link, year, full_url)
                        if speech_info:
                            speech_info['source_url'] = full_url
                            speeches.append(speech_info)
        
        return speeches

    def _extract_speech_metadata_from_table_row(self, row, link, year: int, url: str) -> Optional[Dict]:
        """Enhanced table row metadata extraction."""
        try:
            link_text = link.get_text(strip=True)
            row_text = row.get_text()
            
            # PRIORITY 1: Extract date from URL
            url_date = self.extract_date_from_url(url)
            
            # Enhanced date extraction from row text (fallback)
            date_from_row = None
            if not url_date:
                date_patterns = [
                    r'\b(\d{1,2}\s+\w+\s+\d{4})\b',
                    r'\b(\w+\s+\d{1,2},\s+\d{4})\b',
                    r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
                    r'\b(\d{4}-\d{2}-\d{2})\b'
                ]
                
                for pattern in date_patterns:
                    date_match = re.search(pattern, row_text)
                    if date_match:
                        date_from_row = date_match.group(1)
                        break
            
            # Enhanced speaker extraction
            speaker_patterns = [
                r'\b(?:Governor|Deputy Governor|Chair|President)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair|President)'
            ]
            
            speaker_match = None
            for pattern in speaker_patterns:
                speaker_match = re.search(pattern, row_text)
                if speaker_match:
                    break
            
            # Determine final date
            final_date = None
            if url_date:
                final_date = url_date
                date_source = 'url'
            elif date_from_row:
                parsed_date = self._parse_date_enhanced(date_from_row, url)
                if parsed_date:
                    final_date = parsed_date
                    date_source = 'table_row'
            
            # Only return if we have a reliable date
            if link_text and final_date:
                return {
                    'title': link_text,
                    'date_raw': date_from_row or '',
                    'date': final_date,
                    'date_source': date_source,
                    'speaker_raw': speaker_match.group(1) if speaker_match else '',
                    'context_text': row_text
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting from table row: {e}")
            return None

    def _deduplicate_speeches(self, speeches: List[Dict]) -> List[Dict]:
        """Remove duplicate speeches based on URL."""
        unique_speeches = []
        seen_urls = set()
        
        for speech in speeches:
            url = speech.get('source_url')
            if url and url not in seen_urls:
                unique_speeches.append(speech)
                seen_urls.add(url)
        
        return unique_speeches

    # ENHANCED CONTENT PROCESSING
    def scrape_speech_content(self, speech_info: Dict) -> Optional[Dict]:
        """Enhanced speech content scraping with strict validation."""
        url = speech_info['source_url']
        logger.info(f"Scraping content from: {url}")
        
        self.stats['total_processed'] += 1
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                result = self._extract_pdf_content_enhanced(response.content, speech_info)
            else:
                result = self._extract_html_content_enhanced(response.text, speech_info, url)
            
            # CRITICAL: Validate before returning
            if result and self._validate_speech_data(result):
                self.stats['successful_extractions'] += 1
                return result
            else:
                logger.warning(f"Speech failed validation: {url}")
                self.stats['content_extraction_failures'] += 1
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            self.stats['content_extraction_failures'] += 1
            return None

    def _extract_html_content_enhanced(self, html_content: str, speech_info: Dict, url: str) -> Optional[Dict]:
        """Enhanced HTML content extraction with strict validation."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract components with enhanced methods
            title = self._extract_title_enhanced(soup, speech_info.get('title', ''))
            speaker_info = self._extract_speaker_enhanced(soup, speech_info.get('speaker_raw', ''), url)
            
            # Use enhanced date parsing with URL priority
            date = self._parse_date_enhanced(speech_info.get('date_raw', ''), url)
            if not date:
                # If we still don't have a date, this speech is not valid
                logger.warning(f"No valid date found for {url}")
                return None
            
            location = self._extract_location_enhanced(soup)
            content = self._extract_main_content_enhanced(soup)
            
            # Early validation - content must be substantial
            if not content or len(content.strip()) < 200:
                logger.warning(f"Content too short or empty for {url}: {len(content.strip()) if content else 0} chars")
                return None
            
            # Get speaker information (with URL fallback)
            speaker_name = speaker_info.get('name', 'Unknown')
            role_info = self.get_speaker_info(speaker_name, url)
            
            # Build comprehensive metadata
            metadata = {
                'title': title,
                'speaker': speaker_name,
                'role': role_info.get('role', 'Unknown'),
                'institution': 'Bank of England',
                'country': 'UK',
                'date': date,
                'location': location,
                'language': 'en',
                'source_url': url,
                'source_type': 'HTML',
                'voting_status': role_info.get('voting_status', 'Unknown'),
                'recognition_source': role_info.get('source', 'unknown'),
                'date_source': speech_info.get('date_source', 'unknown'),
                'tags': self._extract_content_tags(content),
                'scrape_timestamp': datetime.now().isoformat()
            }
            
            # Add service years if available
            if 'years' in role_info:
                metadata['service_years'] = role_info['years']
            
            return {
                'metadata': metadata,
                'content': content
            }
            
        except Exception as e:
            logger.error(f"Error extracting HTML content from {url}: {e}")
            return None

    def _extract_speaker_enhanced(self, soup: BeautifulSoup, fallback_speaker: str, url: str) -> Dict[str, str]:
        """Enhanced speaker extraction with multiple methods."""
        # Method 1: Try specific speaker selectors
        speaker_selectors = [
            '.speech-author',
            '.article-author',
            '.byline',
            '.speaker-name',
            '.author-name',
            '.speech-by',
            '[class*="author"]',
            '[class*="speaker"]'
        ]
        
        for selector in speaker_selectors:
            element = soup.select_one(selector)
            if element:
                speaker_text = element.get_text(strip=True)
                name = self._clean_speaker_name(speaker_text)
                if name and name != 'Unknown':
                    return {'name': name, 'raw_text': speaker_text, 'method': 'css_selector'}
        
        # Method 2: Search in article content with enhanced patterns
        article = soup.find('main') or soup.find('article') or soup
        if article:
            text = article.get_text()
            
            # Enhanced speaker patterns
            patterns = [
                r'Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Deputy Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Chair\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'(?:By|Remarks by|Speech by)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    name = self._clean_speaker_name(matches[0])
                    if name != 'Unknown':
                        return {'name': name, 'raw_text': text[:200], 'method': 'content_pattern'}
        
        # Method 3: Use fallback speaker if provided
        if fallback_speaker:
            cleaned = self._clean_speaker_name(fallback_speaker)
            if cleaned != 'Unknown':
                return {'name': cleaned, 'raw_text': fallback_speaker, 'method': 'fallback'}
        
        # Method 4: URL extraction will be handled by get_speaker_info()
        return {'name': 'Unknown', 'raw_text': '', 'method': 'failed'}

    def _clean_speaker_name(self, raw_name: str) -> str:
        """Enhanced speaker name cleaning."""
        if not raw_name:
            return 'Unknown'
        
        # Remove newlines and normalize whitespace
        raw_name = ' '.join(raw_name.split())
        
        # Remove titles
        name = re.sub(r'\b(?:Governor|Deputy Governor|Chair|President|Dr\.|Mr\.|Ms\.|Mrs\.|Sir|Lord|Baron)\s*', '', raw_name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        name = re.split(r'\s*(?:,|by|remarks|speech)\s*', name, flags=re.IGNORECASE)[0]
        
        # Clean and validate
        name = ' '.join(name.split()).strip()
        
        # Return Unknown if name is too short or contains problematic characters
        if len(name) < 2 or any(char in name for char in ['\n', '\t', '|', '<', '>']):
            return 'Unknown'
        
        return name if name else 'Unknown'

    def _extract_title_enhanced(self, soup: BeautifulSoup, fallback_title: str) -> str:
        """Enhanced title extraction with multiple strategies."""
        selectors = [
            'h1.article-title',
            'h1',
            '.speech-title',
            '.article-header h1',
            'h2.title',
            '.page-header h1',
            '.main-title',
            '[class*="title"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                # Enhanced validation
                if (title and len(title) > 10 and len(title) < 300 and 
                    'Bank of England' not in title and 
                    not title.lower().startswith('speeches')):
                    return title
        
        # Try meta tags
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            title = meta_title['content']
            if 'Bank of England' not in title and len(title) > 10:
                return title
        
        # Clean up fallback title
        if fallback_title and len(fallback_title) > 10:
            return fallback_title
        
        return 'Untitled Speech'

    def _extract_location_enhanced(self, soup: BeautifulSoup) -> str:
        """Enhanced location extraction."""
        location_selectors = [
            '.speech-location',
            '.event-location',
            '.venue',
            '[class*="location"]',
            '[class*="venue"]'
        ]
        
        for selector in location_selectors:
            element = soup.select_one(selector)
            if element:
                location = element.get_text(strip=True)
                if location and len(location) < 100:
                    return location
        
        # Search in content for location patterns
        article = soup.find('main') or soup.find('article') or soup
        if article:
            text = article.get_text()
            patterns = [
                r'(?:at|in)\s+([A-Z][a-zA-Z\s,]+(?:University|College|Institute|Center|Centre|Hotel|Club|Conference|School))',
                r'(?:London|Birmingham|Manchester|Edinburgh|Cardiff|Belfast|Liverpool|Bristol),?\s*([A-Z]{2}|\w+)',
                r'(?:delivered at|speaking at|remarks at)\s+([A-Z][a-zA-Z\s,]{5,50})'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    location = matches[0] if isinstance(matches[0], str) else ', '.join(matches[0])
                    if len(location) < 100:
                        return location
        
        return 'London, UK'

    def _extract_content_tags(self, content: str) -> List[str]:
        """Enhanced content tag extraction."""
        tags = []
        content_lower = content.lower()
        
        # Enhanced keyword mapping for monetary policy topics
        keywords = {
            'inflation': ['inflation', 'price stability', 'cpi', 'rpi', 'deflation', 'disinflation', 'price level', 'core inflation'],
            'interest_rates': ['interest rate', 'bank rate', 'monetary policy', 'policy rate', 'rate rise', 'rate cut', 'rate increase', 'rate decrease', 'base rate'],
            'employment': ['employment', 'unemployment', 'labour market', 'labor market', 'jobs', 'jobless', 'payroll', 'employment data', 'job growth', 'labour force'],
            'financial_stability': ['financial stability', 'banking', 'supervision', 'regulation', 'systemic risk', 'stress test', 'capital requirements', 'prudential'],
            'economic_outlook': ['economic outlook', 'forecast', 'projection', 'growth', 'recession', 'expansion', 'economic conditions', 'gdp'],
            'monetary_policy': ['monetary policy', 'mpc', 'monetary policy committee', 'quantitative easing', 'qe', 'gilt purchases', 'asset purchases'],
            'banking': ['bank', 'banking', 'credit', 'lending', 'deposits', 'financial institutions', 'commercial banks'],
            'markets': ['market', 'financial markets', 'capital markets', 'bond market', 'stock market', 'equity markets', 'gilt'],
            'crisis': ['crisis', 'pandemic', 'covid', 'financial crisis', 'economic crisis', 'emergency', 'coronavirus'],
            'brexit': ['brexit', 'european union', 'eu', 'single market', 'customs union', 'trade deal'],
            'international': ['international', 'global', 'foreign', 'trade', 'exchange rate', 'emerging markets', 'global economy']
        }
        
        for tag, terms in keywords.items():
            if any(term in content_lower for term in terms):
                tags.append(tag)
        
        return tags

    def _extract_pdf_content_enhanced(self, pdf_content: bytes, speech_info: Dict) -> Optional[Dict]:
        """Enhanced PDF content extraction."""
        try:
            import io
            
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if not text or len(text.strip()) < 200:
                    logger.warning("PDF content too short or empty")
                    return None
                
                # Extract metadata from PDF header
                lines = text.split('\n')[:20]
                header_text = '\n'.join(lines)
                
                # Extract components
                speaker_name = self._extract_speaker_from_text(header_text, speech_info.get('speaker_raw', ''))
                title = self._extract_title_from_text(header_text, speech_info.get('title', ''))
                
                # Use enhanced date parsing with URL priority
                date = self._parse_date_enhanced(speech_info.get('date_raw', ''), speech_info['source_url'])
                if not date:
                    logger.warning(f"No valid date found for PDF: {speech_info['source_url']}")
                    return None
                
                location = self._extract_location_from_text(header_text)
                
                # Get speaker information (with URL fallback)
                role_info = self.get_speaker_info(speaker_name, speech_info['source_url'])
                
                # Build metadata
                metadata = {
                    'title': title,
                    'speaker': speaker_name,
                    'role': role_info.get('role', 'Unknown'),
                    'institution': 'Bank of England',
                    'country': 'UK',
                    'date': date,
                    'location': location,
                    'language': 'en',
                    'source_url': speech_info['source_url'],
                    'source_type': 'PDF',
                    'voting_status': role_info.get('voting_status', 'Unknown'),
                    'recognition_source': role_info.get('source', 'unknown'),
                    'date_source': speech_info.get('date_source', 'unknown'),
                    'tags': self._extract_content_tags(text),
                    'scrape_timestamp': datetime.now().isoformat()
                }
                
                if 'years' in role_info:
                    metadata['service_years'] = role_info['years']
                
                return {
                    'metadata': metadata,
                    'content': self._clean_text_content(text)
                }
                
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return None

    def _extract_speaker_from_text(self, text: str, fallback_speaker: str = '') -> str:
        """Extract speaker name from text content."""
        patterns = [
            r'(?:Governor|Deputy Governor|Chair|President)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
            r'By\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
            r'Remarks by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
            r'Speech by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair|President)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                name = self._clean_speaker_name(matches[0])
                if name != 'Unknown':
                    return name
        
        # Use fallback
        if fallback_speaker:
            cleaned = self._clean_speaker_name(fallback_speaker)
            if cleaned != 'Unknown':
                return cleaned
        
        return 'Unknown'

    def _extract_title_from_text(self, text: str, fallback_title: str = '') -> str:
        """Extract title from text content."""
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if (len(line) > 10 and len(line) < 200 and 
                not re.match(r'^\d', line) and 
                not re.match(r'^(?:Governor|Deputy Governor|Chair|President)', line) and
                not line.lower().startswith('bank of england')):
                return line
        
        return fallback_title or 'Untitled Speech'

    def _extract_location_from_text(self, text: str) -> str:
        """Extract location from text content."""
        patterns = [
            r'(?:at|in)\s+([A-Z][a-zA-Z\s,]+(?:University|College|Institute|Center|Centre|Hotel|Club|Conference))',
            r'(?:London|Birmingham|Manchester|Edinburgh|Cardiff|Belfast),?\s*([A-Z]{2}|\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0] if isinstance(matches[0], str) else ', '.join(matches[0])
        
        return 'London, UK'

    # ENHANCED FILE HANDLING WITH BETTER SANITIZATION
    def save_speech(self, speech_data: Dict) -> Optional[str]:
        """Enhanced speech saving with better filename handling."""
        try:
            metadata = speech_data['metadata']
            content = speech_data['content']
            
            # Generate content hash for uniqueness
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:10]
            
            # Enhanced speaker name sanitization
            speaker_name = metadata['speaker']
            if speaker_name and speaker_name != 'Unknown':
                # More robust filename sanitization
                clean_speaker = re.sub(r'[^\w\s-]', '', speaker_name.lower())
                clean_speaker = re.sub(r'\s+', '-', clean_speaker)
                clean_speaker = clean_speaker.strip('-')
                # Ensure it's not empty after cleaning
                if not clean_speaker:
                    clean_speaker = 'unknown-speaker'
            else:
                clean_speaker = 'unknown-speaker'
            
            # Use the actual date from metadata (already validated)
            date_str = metadata['date']
            base_filename = f"{date_str}_{clean_speaker}-{content_hash}"
            
            # Final filename sanitization
            base_filename = re.sub(r'[^\w\-.]', '', base_filename)
            
            # Create directory structure
            speech_dir = os.path.join(self.boe_dir, base_filename)
            os.makedirs(speech_dir, exist_ok=True)
            
            # Save metadata as JSON
            json_path = os.path.join(speech_dir, f"{base_filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Save content as text
            txt_path = os.path.join(speech_dir, f"{base_filename}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Saved speech: {base_filename}")
            return base_filename
            
        except Exception as e:
            logger.error(f"Error saving speech: {e}")
            return None

    # SELENIUM METHODS (ENHANCED)
    def scrape_speeches_selenium(self, max_speeches: int = 100) -> List[Dict]:
        """Enhanced Selenium-based scraping."""
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available for dynamic scraping")
            return []
        
        logger.info("Starting Selenium-based scraping...")
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        speeches = []
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.speeches_url)
            
            wait = WebDriverWait(driver, 15)
            
            # Multiple selectors for speech links
            selectors = [
                "a[href*='/news/speeches/']",
                "a[href*='/speeches/']",
                ".speech-list a",
                "table a[href*='speech']",
                "ul li a[href*='speech']"
            ]
            
            for selector in selectors:
                try:
                    elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))
                    
                    if elements:
                        logger.info(f"Found {len(elements)} elements with selector: {selector}")
                        
                        for element in elements[:max_speeches]:
                            try:
                                href = element.get_attribute("href")
                                text = element.text.strip()
                                
                                if href and ('/news/speeches/' in href or '/speeches/' in href):
                                    # Try to extract date from URL
                                    url_date = self.extract_date_from_url(href)
                                    
                                    # Only add if we can get a valid date
                                    if url_date:
                                        speeches.append({
                                            'source_url': href,
                                            'title': text,
                                            'date': url_date,
                                            'date_source': 'url',
                                            'context_text': text,
                                            'speaker_raw': ''
                                        })
                                    
                            except Exception as e:
                                logger.debug(f"Error processing element: {e}")
                                continue
                        
                        if speeches:
                            break
                            
                except (TimeoutException, NoSuchElementException):
                    logger.debug(f"Selector failed: {selector}")
                    continue
            
        except Exception as e:
            logger.error(f"Selenium scraping error: {e}")
        finally:
            try:
                driver.quit()
            except:
                pass
        
        logger.info(f"Selenium found {len(speeches)} speeches with valid dates")
        return speeches

        # MAIN EXECUTION METHODS
    def run_comprehensive_scraping(self, method: str = "all", start_year: int = 1996, 
                                max_speeches: Optional[int] = None) -> Dict[str, int]:
        """
        Enhanced comprehensive BoE speech scraping with strict validation.
        """
        logger.info(f"Starting enhanced BoE speech scraping")
        logger.info(f"Method: {method}, Start year: {start_year}, Max speeches: {max_speeches}")
        
        # Reset statistics
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'url_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0
        }
        
        all_speeches = []
        
        # Approach 1: Sitemap scraping (most reliable - ADD THIS AS PRIMARY METHOD)
        if method in ["sitemap", "all"]:
            logger.info("Running sitemap scraping...")
            sitemap_speeches = self.scrape_speeches_from_sitemap(max_speeches or 100)
            all_speeches.extend(sitemap_speeches)
            logger.info(f"Sitemap method found {len(sitemap_speeches)} speeches")
        
        # Approach 2: Main speeches page scraping (keep as backup)
        if method in ["main", "all"] and len(all_speeches) < 10:  # Only if sitemap failed
            logger.info("Running main speeches page scraping...")
            main_speeches = self.scrape_speeches_main_page(max_speeches or 50)
            all_speeches.extend(main_speeches)
            logger.info(f"Main page method found {len(main_speeches)} speeches")
        
        # Approach 3: Selenium dynamic scraping (supplementary)
        if method in ["selenium", "all"] and SELENIUM_AVAILABLE:
            logger.info("Running Selenium dynamic scraping...")
            selenium_speeches = self.scrape_speeches_selenium(max_speeches or 100)
            all_speeches.extend(selenium_speeches)
            logger.info(f"Selenium method found {len(selenium_speeches)} speeches")
        
        # Remove duplicates
        unique_speeches = self._deduplicate_speeches(all_speeches)
        logger.info(f"Total unique speeches found: {len(unique_speeches)}")
        
        if not unique_speeches:
            logger.warning("No speeches found!")
            return self.stats
        
        # Additional filtering: only process speeches with valid dates
        valid_speeches = []
        for speech in unique_speeches:
            if speech.get('date') and not speech['date'].startswith('2025-01-01'):
                valid_speeches.append(speech)
            else:
                logger.warning(f"Skipping speech with invalid date: {speech.get('source_url')}")
        
        logger.info(f"Speeches with valid dates: {len(valid_speeches)}")
        
        # Limit speeches if requested
        if max_speeches:
            valid_speeches = valid_speeches[:max_speeches]
            logger.info(f"Limited to {max_speeches} speeches")
        
        # Process each speech
        logger.info(f"Processing {len(valid_speeches)} speeches...")
        
        for i, speech_info in enumerate(valid_speeches, 1):
            logger.info(f"Processing speech {i}/{len(valid_speeches)}: {speech_info['source_url']}")
            
            try:
                # Extract content and metadata
                speech_data = self.scrape_speech_content(speech_info)
                
                if speech_data:
                    # Save speech (already validated in scrape_speech_content)
                    saved_filename = self.save_speech(speech_data)
                    if saved_filename:
                        self.stats['saved_speeches'] += 1
                        
                        # Log speaker recognition details
                        metadata = speech_data['metadata']
                        logger.info(f"Successfully saved: {saved_filename}")
                        logger.info(f"  Speaker: {metadata['speaker']} ({metadata['recognition_source']})")
                        logger.info(f"  Role: {metadata['role']}")
                        logger.info(f"  Date: {metadata['date']} ({metadata['date_source']})")
                    else:
                        logger.error(f"Failed to save speech from {speech_info['source_url']}")
                else:
                    logger.error(f"Failed to extract or validate content from {speech_info['source_url']}")
                
            except Exception as e:
                logger.error(f"Unexpected error processing {speech_info['source_url']}: {e}")
            
            # Respectful delay
            time.sleep(1)
        
        # Final statistics
        logger.info("=== ENHANCED BoE SCRAPING COMPLETE ===")
        logger.info(f"Total speeches processed: {self.stats['total_processed']}")
        logger.info(f"Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"Saved speeches: {self.stats['saved_speeches']}")
        logger.info(f"URL fallback used: {self.stats['url_fallback_used']}")
        logger.info(f"Unknown speakers: {self.stats['unknown_speakers']}")
        logger.info(f"Date extraction failures: {self.stats['date_extraction_failures']}")
        logger.info(f"Content too short: {self.stats['content_too_short']}")
        logger.info(f"Validation failures: {self.stats['validation_failures']}")
        logger.info(f"Content extraction failures: {self.stats['content_extraction_failures']}")
        
        return self.stats

    def print_speaker_database_info(self):
        """Print information about the speaker database for debugging."""
        logger.info("=== BoE SPEAKER DATABASE INFO ===")
        logger.info(f"Total speaker entries: {len(self.speaker_roles)}")
        
        # Count by role
        role_counts = {}
        for info in self.speaker_roles.values():
            role = info.get('role', 'Unknown')
            role_counts[role] = role_counts.get(role, 0) + 1
        
        logger.info("Speakers by role:")
        for role, count in sorted(role_counts.items()):
            logger.info(f"  {role}: {count}")
        
        logger.info(f"URL pattern entries: {len(self.url_name_patterns)}")

    # LEGACY COMPATIBILITY
    def scrape_all_speeches(self, limit: Optional[int] = None) -> int:
        """Legacy method for backward compatibility."""
        stats = self.run_comprehensive_scraping(method="all", max_speeches=limit)
        return stats['saved_speeches']


def main():
    """Main function to run the enhanced BoE speech scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Bank of England Speech Scraper v1")
    parser.add_argument("--method", choices=["main", "historical", "selenium", "all"], 
                       default="all", help="Scraping method to use")
    parser.add_argument("--start-year", type=int, default=1996, 
                       help="Starting year for historical scraping")
    parser.add_argument("--max-speeches", type=int, 
                       help="Maximum number of speeches to scrape")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--show-db-info", action="store_true", 
                       help="Show speaker database information")
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = EnhancedBoEScraper(base_dir=args.output_dir)
    
    # Show database info if requested
    if args.show_db_info:
        scraper.print_speaker_database_info()
        return
    
    # Run scraping
    stats = scraper.run_comprehensive_scraping(
        method=args.method,
        start_year=args.start_year,
        max_speeches=args.max_speeches
    )
    
    # Print final results
    print(f"\n=== ENHANCED BoE SCRAPING RESULTS ===")
    print(f"Method used: {args.method}")
    print(f"Output directory: {scraper.boe_dir}")
    print(f"")
    print(f"📊 STATISTICS:")
    print(f"  Total speeches processed: {stats['total_processed']}")
    print(f"  Successfully saved: {stats['saved_speeches']}")
    print(f"  Content extraction failures: {stats['content_extraction_failures']}")
    print(f"  Date extraction failures: {stats['date_extraction_failures']}")
    print(f"  Content too short: {stats['content_too_short']}")
    print(f"  Validation failures: {stats['validation_failures']}")
    print(f"")
    print(f"🎯 SPEAKER RECOGNITION:")
    print(f"  URL fallback used: {stats['url_fallback_used']}")
    print(f"  Unknown speakers: {stats['unknown_speakers']}")
    print(f"")
    print(f"✅ SUCCESS RATE: {(stats['saved_speeches']/max(stats['total_processed'], 1)*100):.1f}%")
    
    if stats['url_fallback_used'] > 0:
        print(f"\n🔄 URL FALLBACK EFFECTIVENESS:")
        print(f"  Speeches rescued by URL extraction: {stats['url_fallback_used']}")


if __name__ == "__main__":
    main()
            