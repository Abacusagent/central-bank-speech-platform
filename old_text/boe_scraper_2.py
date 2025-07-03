    def _log_final_statistics(self):
        """Log comprehensive final statistics."""
        logger.info("=== ENHANCED BoE SCRAPING COMPLETE ===")
        logger.info(f"Total speeches processed: {self.stats['total_processed']}")
        logger.info(f"Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"Saved speeches: {self.stats['saved_speeches']}")
        logger.info(f"Historical speeches found: {self.stats['historical_speeches_found']}")
        logger.info(f"  - Quarterly Bulletin: {self.stats['quarterly_bulletin_speeches']}")
        logger.info(f"  - Digital Archive: {self.stats['digital_archive_speeches']}")
        logger.info(f"URL fallback used: {self.stats['url_fallback_used']}")
        logger.info(f"Content fallback used: {self.stats['content_fallback_used']}")
        logger.info(f"Unknown speakers: {self.stats['unknown_speakers']}")
        logger.info(f"Date extraction failures: {self.stats['date_extraction_failures']}")
        logger.info(f"Content too short: {self.stats['content_too_short']}")
        logger.info(f"Validation failures: {self.stats['validation_failures']}")
        logger.info(f"Content extraction failures: {self.stats['content_extraction_failures']}")
        
        # Calculate success rate
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['saved_speeches'] / self.stats['total_processed']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")

    # METHODS FROM ORIGINAL SCRAPER (Enhanced versions)

    def scrape_speeches_from_sitemap(self, max_speeches: Optional[int] = None) -> List[Dict]:
        """Enhanced sitemap scraping with better error handling."""
        logger.info(f"Scraping BoE speeches from sitemap: {self.sitemap_url}")
        all_speeches = []
        
        try:
            response = requests.get(self.sitemap_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all speech links - enhanced pattern matching
            speech_links = soup.find_all('a', href=True)
            
            for link in speech_links:
                href = link.get('href')
                
                # Enhanced BoE speech URL patterns
                if href and ('/speech/' in href or '/news/speeches/' in href):
                    # Skip navigation and filter links
                    if any(skip in href for skip in ['?', '#', 'page=', 'filter=', 'search=']):
                        continue
                    
                    full_url = urljoin(self.base_url, href)
                    
                    # Extract date from URL (more reliable than content parsing)
                    url_date = self.extract_date_from_url_enhanced(full_url)
                    
                    if url_date:
                        # Extract speaker from URL if possible
                        speaker_from_url = self.extract_speaker_from_url_enhanced(full_url)
                        
                        speech_info = {
                            'source_url': full_url,
                            'title': link.get_text(strip=True),
                            'date': url_date,
                            'date_source': 'url',
                            'speaker_raw': speaker_from_url or '',
                            'context_text': link.get_text(strip=True)
                        }
                        all_speeches.append(speech_info)
            
            # Remove duplicates and sort by date (newest first)
            unique_speeches = self._deduplicate_speeches_enhanced(all_speeches)
            unique_speeches.sort(key=lambda x: x.get('date', ''), reverse=True)
            
            if max_speeches:
                unique_speeches = unique_speeches[:max_speeches]
            
            logger.info(f"Found {len(unique_speeches)} speeches from sitemap")
            return unique_speeches
            
        except requests.RequestException as e:
            logger.error(f"Error scraping sitemap: {e}")
            return []

    def scrape_speeches_main_page(self, max_speeches: int = 50) -> List[Dict]:
        """Enhanced main page scraping."""
        logger.info(f"Scraping BoE main speeches page: {self.speeches_url}")
        all_speeches = []
        
        try:
            response = requests.get(self.speeches_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            speeches = self._extract_speeches_from_main_page_enhanced(soup)
            
            if max_speeches:
                speeches = speeches[:max_speeches]
            
            logger.info(f"Found {len(speeches)} speeches on main page")
            all_speeches.extend(speeches)
            
        except requests.RequestException as e:
            logger.error(f"Error scraping main speeches page: {e}")
        except Exception as e:
            logger.error(f"Unexpected error scraping main speeches page: {e}")
        
        return all_speeches

    def _extract_speeches_from_main_page_enhanced(self, soup: BeautifulSoup) -> List[Dict]:
        """Enhanced speech extraction from main page."""
        speeches = []
        
        # Enhanced selectors for current BoE website structure
        speech_selectors = [
            'a[href*="/speech/"]',
            'a[href*="/news/speeches/"]',
            '.card a[href*="/speech/"]',
            '.listing-item a[href*="/speech/"]',
            'article a[href*="/speech/"]',
            '[data-type="speech"] a',
            '.speech-link',
            '.news-item a[href*="speech"]'
        ]
        
        for selector in speech_selectors:
            links = soup.select(selector)
            
            for link in links:
                href = link.get('href')
                
                if href and '/speech/' in href:
                    # Skip navigation and filter links
                    if any(skip in href for skip in ['?', '#', 'page=', 'filter=', 'search=']):
                        continue
                    
                    full_url = urljoin(self.base_url, href)
                    
                    # Enhanced metadata extraction
                    speech_info = self._extract_speech_metadata_from_context_enhanced(link, None, full_url)
                    if speech_info:
                        speech_info['source_url'] = full_url
                        speeches.append(speech_info)
        
        # Remove duplicates by URL
        return self._deduplicate_speeches_enhanced(speeches)

    def _extract_speech_metadata_from_context_enhanced(self, link, year: int, url: str) -> Optional[Dict]:
        """Enhanced metadata extraction with better speaker recognition."""
        try:
            link_text = link.get_text(strip=True)
            
            # Skip if not substantial enough to be a speech title
            if not link_text or len(link_text) < 5:
                return None
            
            # Enhanced date extraction - URL first
            url_date = self.extract_date_from_url_enhanced(url)
            
            # Look for metadata in parent elements (up to 4 levels)
            parent = link.parent
            context_text = ""
            date_from_context = None
            speaker_from_context = None
            
            for level in range(4):
                if parent is None:
                    break
                
                parent_text = parent.get_text()
                context_text += parent_text + " "
                
                # Enhanced date patterns (only if URL date extraction failed)
                if not url_date:
                    date_patterns = [
                        r'\b(\d{1,2}\s+\w+\s+\d{4})\b',  # "15 January 2025"
                        r'\b(\w+\s+\d{1,2},?\s+\d{4})\b',  # "January 15, 2025"
                        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',  # "15/01/2025"
                        r'\b(\d{4}-\d{2}-\d{2})\b'  # "2025-01-15"
                    ]
                    
                    for pattern in date_patterns:
                        date_match = re.search(pattern, parent_text)
                        if date_match:
                            date_from_context = date_match.group(1)
                            break
                
                # Enhanced speaker patterns
                speaker_patterns = [
                    r'\b(?:Governor|Deputy Governor|Chair|President|Chief Economist)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                    r'\b(?:By|Remarks by|Speech by|Address by|Lecture by)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair|President|Chief Economist)',
                    r'\b(?:Lord|Sir)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*[-–]\s*(?:speech|remarks|address)'
                ]
                
                for pattern in speaker_patterns:
                    speaker_match = re.search(pattern, parent_text)
                    if speaker_match:
                        candidate_speaker = speaker_match.group(1)
                        # Validate it's not a generic term
                        if candidate_speaker.lower() not in ['governor', 'deputy', 'chair', 'president', 'chief']:
                            speaker_from_context = candidate_speaker
                            break
                
                parent = parent.parent
            
            # Enhanced speaker extraction from URL if not found in context
            if not speaker_from_context:
                speaker_from_context = self.extract_speaker_from_url_enhanced(url)
            
            # Determine the best date
            final_date = None
            date_source = 'unknown'
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
                logger.debug(f"Could not extract reliable date for speech: {url}")
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
            logger.debug(f"Error extracting enhanced speech metadata: {e}")
            return None

    def _parse_date_enhanced(self, date_str: str, url: str = None) -> str:
        """Enhanced date parsing with better validation."""
        # Priority 1: Try URL extraction first (most reliable)
        if url:
            url_date = self.extract_date_from_url_enhanced(url)
            if url_date:
                return url_date
        
        if not date_str:
            logger.warning(f"No date string provided and URL extraction failed for {url}")
            self.stats['date_extraction_failures'] += 1
            return None
        
        # Enhanced date formats
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
            '%Y%m%d',      # "20250115" (YYYYMMDD)
            '%d/%m/%y',    # "15/01/25"
            '%m/%d/%y',    # "01/15/25"
        ]
        
        # Clean the date string
        date_str = date_str.strip()
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # Validate reasonable date range (extended for historical coverage)
                if 1960 <= dt.year <= 2030:
                    formatted_date = dt.strftime('%Y-%m-%d')
                    logger.debug(f"Successfully parsed date: {date_str} -> {formatted_date}")
                    return formatted_date
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str} for URL: {url}")
        self.stats['date_extraction_failures'] += 1
        return None

    def scrape_speeches_selenium(self, max_speeches: int = 100) -> List[Dict]:
        """Enhanced Selenium-based scraping with better error handling."""
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available for dynamic scraping")
            return []
        
        logger.info("Starting enhanced Selenium-based scraping...")
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        
        speeches = []
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.speeches_url)
            
            wait = WebDriverWait(driver, 20)
            
            # Enhanced selectors for speech links
            selectors = [
                "a[href*='/speech/']",
                "a[href*='/news/speeches/']",
                ".speech-list a",
                ".news-list a[href*='speech']",
                "table a[href*='speech']",
                "ul li a[href*='speech']",
                ".card a[href*='speech']"
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
                                
                                if (href and 
                                    ('/speech/' in href or '/news/speeches/' in href) and
                                    text and len(text) > 5):
                                    
                                    # Extract date from URL
                                    url_date = self.extract_date_from_url_enhanced(href)
                                    
                                    # Extract speaker from URL
                                    url_speaker = self.extract_speaker_from_url_enhanced(href)
                                    
                                    # Only add if we can get a valid date or substantial content
                                    if url_date or len(text) > 20:
                                        speeches.append({
                                            'source_url': href,
                                            'title': text,
                                            'date': url_date or 'unknown',
                                            'date_source': 'url' if url_date else 'unknown',
                                            'context_text': text,
                                            'speaker_raw': url_speaker or ''
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
            logger.error(f"Enhanced Selenium scraping error: {e}")
        finally:
            try:
                driver.quit()
            except:
                pass
        
        logger.info(f"Enhanced Selenium found {len(speeches)} speeches")
        return speeches

    # ENHANCED PDF EXTRACTION

    def _extract_pdf_content_enhanced(self, pdf_content: bytes, speech_info: Dict) -> Optional[Dict]:
        """Enhanced PDF content extraction with better error handling."""
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if not text or len(text.strip()) < 200:
                    logger.warning("PDF content too short or empty")
                    return None
                
                # Extract metadata from PDF header/content
                lines = text.split('\n')[:30]  # Look at more lines for better extraction
                header_text = '\n'.join(lines)
                
                # Enhanced speaker extraction for PDFs
                speaker_name = self._extract_speaker_from_text_enhanced(header_text, speech_info.get('speaker_raw', ''))
                
                # Enhanced title extraction
                title = self._extract_title_from_text_enhanced(header_text, speech_info.get('title', ''))
                
                # Enhanced date parsing with URL priority
                date = self.extract_date_from_url_enhanced(speech_info['source_url'])
                if not date:
                    date = self._parse_date_enhanced(speech_info.get('date_raw', ''), speech_info['source_url'])
                
                if not date:
                    logger.warning(f"No valid date found for PDF: {speech_info['source_url']}")
                    return None
                
                location = self._extract_location_from_text_enhanced(header_text)
                
                # Get enhanced speaker information
                role_info = self.get_speaker_info_enhanced(speaker_name, speech_info['source_url'], text)
                
                # Build metadata
                metadata = {
                    'title': title,
                    'speaker': role_info.get('matched_name', speaker_name) if role_info.get('source') != 'unknown' else speaker_name,
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
                    'date_source': speech_info.get('date_source', 'url'),
                    'tags': self._extract_content_tags_enhanced(text),
                    'scrape_timestamp': datetime.now().isoformat(),
                    'content_length': len(text)
                }
                
                if 'years' in role_info:
                    metadata['service_years'] = role_info['years']
                
                return {
                    'metadata': metadata,
                    'content': self._clean_text_content_enhanced(text)
                }
                
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return None

    def _extract_speaker_from_text_enhanced(self, text: str, fallback_speaker: str = '') -> str:
        """Enhanced speaker extraction from text content."""
        patterns = [
            r'(?:Governor|Deputy Governor|Chair|President|Chief Economist)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
            r'(?:By|Remarks by|Speech by|Address by|Lecture by)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair|President|Chief Economist)',
            r'(?:Lord|Sir)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)+)\s*#!/usr/bin/env python3
"""
Enhanced Bank of England Speech Scraper v2 - Historical & Current Coverage
Major improvements based on analysis of scraping issues and BoE website structure.

Key Issues Fixed:
1. Speaker recognition failures - Enhanced URL extraction and database lookups
2. Content validation failures - Improved content extraction and validation
3. Historical coverage gaps - Added Quarterly Bulletin and Digital Archive scraping
4. Success rate (38.3% -> target 80%+)

New Features:
- Historical speeches from 1990s via BoE Digital Archive
- Quarterly Bulletin speech extraction (1960s-2006)
- Enhanced speaker extraction from content and URLs
- Better content validation and metadata extraction
- Fallback mechanisms for difficult extractions

Author: Enhanced Central Bank Speech Collector
Date: 2025
Target: Comprehensive BoE speech coverage from 1960s onwards
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
from urllib.parse import urljoin, urlparse, quote
from typing import Dict, List, Optional, Tuple, Set
import time
import io

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

class EnhancedBoEScraperV2:
    """
    Enhanced Bank of England speech scraper with historical coverage and improved recognition.
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.boe_dir = os.path.join(base_dir, "boe")
        self.base_url = "https://www.bankofengland.co.uk"
        self.speeches_url = "https://www.bankofengland.co.uk/news/speeches"
        self.sitemap_url = "https://www.bankofengland.co.uk/sitemap/speeches"
        self.digital_archive_url = "https://boe.access.preservica.com"
        self.quarterly_bulletin_url = "https://www.escoe.ac.uk/research/historical-data/publist/beqb/"
        
        # Ensure directories exist
        os.makedirs(self.boe_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
        
        # Initialize comprehensive speaker database (ENHANCED)
        self._initialize_speaker_database_v2()
        
        # Initialize URL name patterns for fallback extraction
        self._initialize_url_patterns_v2()
        
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
            'content_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0,
            'historical_speeches_found': 0,
            'quarterly_bulletin_speeches': 0,
            'digital_archive_speeches': 0
        }

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

    def _initialize_url_patterns_v2(self):
        """Enhanced URL patterns for better speaker extraction."""
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
        
        # Enhanced manual patterns with historical names
        manual_patterns = {
            # Current officials
            'bailey': ['andrew bailey', 'andrew john bailey'],
            'lombardelli': ['clare lombardelli'],
            'ramsden': ['dave ramsden', 'david ramsden'],
            'breeden': ['sarah breeden'],
            'pill': ['huw pill'],
            'haskel': ['jonathan haskel'],
            'dhingra': ['swati dhingra'],
            'mann': ['catherine mann', 'catherine l mann'],
            'taylor': ['alan taylor'],
            'greene': ['megan greene'],
            'hall': ['jon hall'],
            'kroszner': ['randall kroszner'],
            
            # Past Governors - Enhanced
            'carney': ['mark carney', 'mark joseph carney'],
            'king': ['mervyn king', 'mervyn allister king', 'lord king', 'baron king', 'lord king of lothbury'],
            'george': ['eddie george', 'edward george', 'edward alan john george', 'steady eddie', 'lord george', 'baron george', 'sir eddie george', 'sir edward george'],
            'leighpemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'leigh-pemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'richardson': ['gordon richardson', 'gordon william humphreys richardson', 'lord richardson', 'baron richardson'],
            'obrien': ["leslie o'brien", "leslie kenneth o'brien", "lord o'brien"],
            "o'brien": ["leslie o'brien", "leslie kenneth o'brien", "lord o'brien"],
            
            # Deputy Governors
            'broadbent': ['ben broadbent'],
            'cunliffe': ['jon cunliffe', 'sir jon cunliffe', 'jonathan cunliffe'],
            'woods': ['sam woods', 'samuel woods'],
            'clementi': ['david clementi', 'sir david clementi'],
            'davies': ['howard davies'],
            'pennant-rea': ['rupert pennant-rea'],
            'blunden': ['george blunden'],
            'mcmahon': ['kit mcmahon', 'christopher mcmahon'],
            
            # Chief Economists
            'haldane': ['andy haldane', 'andrew haldane', 'andrew g haldane'],
            'dale': ['spencer dale'],
            'bean': ['charlie bean', 'charles bean', 'charles goodhart bean'],
            'vickers': ['john vickers', 'sir john vickers'],
            
            # External MPC Members
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
            'bell': ['marian bell'],
            'lambert': ['richard lambert'],
            
            # Executive Directors and Officials
            'tucker': ['paul tucker', 'paul mw tucker'],
            'large': ['andrew large'],
            'fisher': ['paul fisher'],
            'rule': ['david rule'],
            'cleland': ['victoria cleland'],
            'plenderleith': ['ian plenderleith'],
            'clark': ['alastair clark'],
            'quinn': ['brian quinn'],
            'kent': ['pen kent'],
            'cooke': ['william cooke', 'w p cooke'],
        }
        
        # Update with manual patterns
        for pattern, names in manual_patterns.items():
            if pattern not in self.url_name_patterns:
                self.url_name_patterns[pattern] = []
            self.url_name_patterns[pattern].extend(names)
            # Remove duplicates
            self.url_name_patterns[pattern] = list(set(self.url_name_patterns[pattern]))

    def extract_speaker_from_url_enhanced(self, url: str) -> Optional[str]:
        """
        ENHANCED speaker extraction from URL with better pattern matching.
        This addresses the main issue causing validation failures.
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            # Enhanced URL patterns for BoE speeches
            url_patterns = [
                # Pattern: speech-by-firstname-lastname
                r'speech-by-([a-z-]+)-([a-z-]+)',
                # Pattern: lastname-speech or lastname-remarks
                r'([a-z-]+)-(speech|remarks|address)',
                # Pattern: firstname-lastname-speech
                r'([a-z-]+)-([a-z-]+)-(speech|remarks|address)',
                # Pattern: remarks-given-by-firstname-lastname
                r'remarks-given-by-([a-z-]+)-([a-z-]+)',
                # Pattern: /lastname/ in path
                r'/([a-z-]+)/',
                # Pattern: just lastname before file extension
                r'([a-z-]+)\.(pdf|html?) ,
            ]
            
            for pattern in url_patterns:
                match = re.search(pattern, path)
                if match:
                    groups = match.groups()
                    
                    # Extract name parts, excluding keywords
                    name_parts = []
                    for group in groups:
                        if group not in ['speech', 'remarks', 'address', 'pdf', 'html', 'htm']:
                            name_parts.append(group.replace('-', ' '))
                    
                    if name_parts:
                        candidate_name = ' '.join(name_parts).strip()
                        
                        # Direct lookup in our patterns
                        if candidate_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[candidate_name]
                            logger.info(f"URL speaker extraction: '{candidate_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
                        
                        # Try just the last word (surname)
                        last_name = candidate_name.split()[-1] if ' ' in candidate_name else candidate_name
                        if last_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[last_name]
                            logger.info(f"URL speaker extraction (lastname): '{last_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
                        
                        # Try without hyphens/spaces
                        clean_name = candidate_name.replace(' ', '').replace('-', '')
                        if clean_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[clean_name]
                            logger.info(f"URL speaker extraction (clean): '{clean_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
            
            logger.debug(f"No speaker pattern matched for URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting speaker from URL {url}: {e}")
            return None

    def extract_speaker_from_content_enhanced(self, soup: BeautifulSoup, url: str = None) -> Optional[str]:
        """
        ENHANCED speaker extraction from content with multiple strategies.
        This is a key improvement to reduce validation failures.
        """
        # Strategy 1: Try specific speaker selectors
        speaker_selectors = [
            '.speech-author',
            '.article-author', 
            '.byline',
            '.speaker-name',
            '.author-name',
            '.speech-by',
            '[class*="author"]',
            '[class*="speaker"]',
            'h1 + p',  # Often speaker info is in paragraph after title
            '.meta-author',
            '.speech-meta'
        ]
        
        for selector in speaker_selectors:
            elements = soup.select(selector)
            for element in elements:
                speaker_text = element.get_text(strip=True)
                name = self._clean_and_validate_speaker_name(speaker_text)
                if name and name != 'Unknown':
                    logger.debug(f"Speaker found via CSS selector '{selector}': {name}")
                    return name
        
        # Strategy 2: Search in structured content with enhanced patterns
        content_areas = [
            soup.find('main'),
            soup.find('article'), 
            soup.find('div', class_='content'),
            soup.find('div', class_='speech'),
            soup
        ]
        
        for area in content_areas:
            if not area:
                continue
                
            text = area.get_text()
            
            # Enhanced speaker patterns for BoE content
            patterns = [
                # Standard titles with names
                r'Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Deputy Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Chair\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Chief Economist\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                
                # "By" patterns
                r'(?:^|\n)\s*By\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'(?:Remarks|Speech|Address)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Given by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                
                # Name followed by title
                r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair|Chief Economist)',
                
                # Lord/Sir titles
                r'(?:Lord|Sir)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                
                # Pattern: Name at start of speech
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*\n',
                
                # MPC member pattern
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*),?\s+(?:External )?MPC [Mm]ember',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE)
                if matches:
                    for match in matches:
                        name = self._clean_and_validate_speaker_name(match)
                        if name and name != 'Unknown':
                            logger.debug(f"Speaker found via content pattern: {name}")
                            return name
        
        # Strategy 3: URL fallback
        if url:
            url_name = self.extract_speaker_from_url_enhanced(url)
            if url_name:
                return url_name
        
        # Strategy 4: Title extraction fallback
        title_text = soup.find('title')
        if title_text:
            title = title_text.get_text()
            # Look for "speech by NAME" in title
            title_pattern = r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)'
            match = re.search(title_pattern, title, re.IGNORECASE)
            if match:
                name = self._clean_and_validate_speaker_name(match.group(1))
                if name and name != 'Unknown':
                    logger.debug(f"Speaker found in title: {name}")
                    return name
        
        logger.debug("No speaker found with any extraction method")
        return None

    def _clean_and_validate_speaker_name(self, raw_name: str) -> str:
        """
        Enhanced speaker name cleaning and validation.
        This is crucial for reducing validation failures.
        """
        if not raw_name:
            return 'Unknown'
        
        # Remove newlines and normalize whitespace
        raw_name = ' '.join(raw_name.split())
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            r'\b(?:The\s+)?(?:Rt\s+)?(?:Hon\s+)?(?:Sir|Lord|Baron|Dame|Dr|Mr|Ms|Mrs)\s+',
            r'\b(?:Governor|Deputy Governor|Chair|President|Chief Economist)\s+',
            r'\b(?:External\s+)?MPC\s+Member\s+',
        ]
        
        for prefix in prefixes_to_remove:
            raw_name = re.sub(prefix, '', raw_name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        raw_name = re.split(r'\s*(?:,|by|remarks|speech|address|gave|given)\s*', raw_name, flags=re.IGNORECASE)[0]
        
        # Clean and validate
        name = ' '.join(raw_name.split()).strip()
        
        # Remove periods and normalize
        name = name.replace('.', '').strip()
        
        # Validate: must be reasonable length and format
        if len(name) < 2 or len(name) > 50:
            return 'Unknown'
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', name):
            return 'Unknown'
        
        # Check against known problematic patterns
        problematic_patterns = [
            r'^\d+ ,  # Just numbers
            r'^[^\w\s]+ ,  # Just punctuation
            r'unknown|speaker|author|governor|deputy|chair|president',  # Generic terms
        ]
        
        name_lower = name.lower()
        for pattern in problematic_patterns:
            if re.match(pattern, name_lower):
                return 'Unknown'
        
        return name if name else 'Unknown'

    def get_speaker_info_enhanced(self, speaker_name: str, url: str = None, content: str = None) -> Dict[str, str]:
        """
        Enhanced speaker information lookup with multiple fallback strategies.
        """
        # Strategy 1: Try extraction from content first if available
        if content and (not speaker_name or speaker_name == 'Unknown'):
            soup_content = BeautifulSoup(content, 'html.parser')
            extracted_name = self.extract_speaker_from_content_enhanced(soup_content, url)
            if extracted_name and extracted_name != 'Unknown':
                speaker_name = extracted_name
                self.stats['content_fallback_used'] += 1
        
        # Strategy 2: Try URL extraction if still unknown
        if (not speaker_name or speaker_name == 'Unknown') and url:
            url_extracted_name = self.extract_speaker_from_url_enhanced(url)
            if url_extracted_name:
                speaker_name = url_extracted_name
        
        if not speaker_name or speaker_name.strip() == 'Unknown':
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
            if self._names_match_enhanced(normalized_name, known_name):
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
        
        # Method 4: Try fuzzy matching on first/last name combinations
        name_parts = normalized_name.split()
        if len(name_parts) >= 2:
            first_last = f"{name_parts[0]} {name_parts[-1]}"
            if first_last in self.speaker_roles:
                result = self.speaker_roles[first_last].copy()
                result['source'] = 'first_last_match'
                logger.debug(f"First-last match found: {speaker_name} -> {first_last}")
                return result
        
        # Final fallback: Unknown speaker
        logger.warning(f"Unknown speaker after all methods: {speaker_name} (normalized: {normalized_name})")
        self.stats['unknown_speakers'] += 1
        return {"role": "Unknown", "voting_status": "Unknown", "source": "unknown"}

    def _names_match_enhanced(self, name1: str, name2: str) -> bool:
        """Enhanced name matching with better fuzzy logic."""
        if not name1 or not name2:
            return False
            
        parts1 = set(name1.replace('.', '').replace('-', ' ').split())
        parts2 = set(name2.replace('.', '').replace('-', ' ').split())
        
        # Remove common middle initials and abbreviations
        parts1 = {p for p in parts1 if len(p) > 1}
        parts2 = {p for p in parts2 if len(p) > 1}
        
        common_parts = parts1.intersection(parts2)
        
        # For short names (2 parts or less), require full overlap
        if len(parts1) <= 2 and len(parts2) <= 2:
            return len(common_parts) >= min(len(parts1), len(parts2))
        else:
            # For longer names, require at least 2 matching parts
            return len(common_parts) >= 2

    def _clean_speaker_name_for_lookup(self, name: str) -> str:
        """Enhanced speaker name cleaning for database lookup."""
        if not name:
            return ""
        
        # Remove titles and clean more thoroughly
        name = re.sub(r'\b(?:The\s+)?(?:Rt\s+)?(?:Hon\s+)?(?:Governor|Deputy Governor|Chair|President|Dr\.|Mr\.|Ms\.|Mrs\.|Sir|Lord|Baron|Dame)\s*', '', name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        name = re.split(r'\s*(?:,|by|remarks|speech|address|gave|given)\s*', name, flags=re.IGNORECASE)[0]
        
        # Remove newlines and extra whitespace
        name = ' '.join(name.split())
        
        # Convert to lowercase and remove periods
        name = name.lower().strip().replace('.', '')
        
        return name

    def extract_date_from_url_enhanced(self, url: str) -> Optional[str]:
        """
        Enhanced date extraction with more patterns and better validation.
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            # Enhanced BoE URL date patterns
            date_patterns = [
                # Current BoE pattern: /speech/2024/october/title-slug
                r'/speech/(\d{4})/(\w+)/',
                # Pattern: /speech/2024/10/title-slug  
                r'/speech/(\d{4})/(\d{1,2})/',
                # Legacy patterns
                r'/speeches/(\d{4})/(\w+)/',
                r'/speeches/(\d{4})/(\d{1,2})/',
                # Media files pattern: /files/speech/2024/july/
                r'/files/speech/(\d{4})/(\w+)/',
                # Date in filename: speech-2024-10-15
                r'speech-(\d{4})-(\d{2})-(\d{2})',
                # Pattern: embedded YYYYMMDD
                r'(\d{8})',
                # Year-only pattern for historical speeches
                r'/(\d{4})/',
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, path)
                if match:
                    groups = match.groups()
                    
                    if len(groups) == 1:
                        # Could be YYYYMMDD or just year
                        date_str = groups[0]
                        if len(date_str) == 8:  # YYYYMMDD
                            try:
                                year = int(date_str[:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                if 1960 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                    date_obj = datetime(year, month, day)
                                    return date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                continue
                        elif len(date_str) == 4:  # Just year
                            try:
                                year = int(date_str)
                                if 1960 <= year <= 2030:
                                    # Use January 1st as default
                                    return f"{year}-01-01"
                            except ValueError:
                                continue
                    
                    elif len(groups) == 2:
                        year_str, month_str = groups
                        try:
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
                            if 1960 <= year <= 2030 and 1 <= month <= 12:
                                date_obj = datetime(year, month, 1)
                                formatted_date = date_obj.strftime('%Y-%m-%d')
                                logger.debug(f"URL date extraction successful: {url} -> {formatted_date}")
                                return formatted_date
                        except ValueError:
                            continue
                            
                    elif len(groups) == 3:
                        year_str, month_str, day_str = groups
                        try:
                            year = int(year_str)
                            month = int(month_str)
                            day = int(day_str)
                            
                            if 1960 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                date_obj = datetime(year, month, day)
                                formatted_date = date_obj.strftime('%Y-%m-%d')
                                logger.debug(f"URL date extraction successful: {url} -> {formatted_date}")
                                return formatted_date
                        except ValueError:
                            continue
            
            logger.debug(f"No valid date pattern found in URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting date from URL {url}: {e}")
            return None

    # NEW METHODS FOR HISTORICAL COVERAGE

    def scrape_quarterly_bulletin_speeches(self, start_year: int = 1960, end_year: int = 2006) -> List[Dict]:
        """
        Scrape historical speeches from Bank of England Quarterly Bulletins (1960-2006).
        This provides access to speeches from the pre-digital era.
        """
        logger.info(f"Scraping BoE Quarterly Bulletin speeches ({start_year}-{end_year})")
        
        all_speeches = []
        
        try:
            response = requests.get(self.quarterly_bulletin_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all speech references in the quarterly bulletin listings
            # Pattern: "speech by [Name]" or "Governor's speech"
            speech_patterns = [
                r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*[-–]\s*speech',
                r"Governor's speech",
                r'remarks by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                r'lecture by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            ]
            
            # Look for speech references in the bulletin content
            text = soup.get_text()
            lines = text.split('\n')
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['speech', 'remarks', 'lecture', 'address']):
                    # Extract year from line if present
                    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', line)
                    if year_match:
                        year = int(year_match.group(1))
                        if start_year <= year <= end_year:
                            
                            # Try to extract speaker name
                            speaker = None
                            for pattern in speech_patterns:
                                match = re.search(pattern, line, re.IGNORECASE)
                                if match and match.groups():
                                    speaker = match.group(1)
                                    break
                            
                            if not speaker and "governor's speech" in line.lower():
                                # Determine governor based on year
                                if 1966 <= year <= 1973:
                                    speaker = "leslie o'brien"
                                elif 1973 <= year <= 1983:
                                    speaker = "gordon richardson"
                                elif 1983 <= year <= 1993:
                                    speaker = "robin leigh-pemberton"
                                elif 1993 <= year <= 2003:
                                    speaker = "eddie george"
                                elif 2003 <= year <= 2006:
                                    speaker = "mervyn king"
                            
                            if speaker:
                                speech_info = {
                                    'title': line.strip(),
                                    'speaker_raw': speaker,
                                    'date': f"{year}-01-01",  # Approximate date
                                    'date_source': 'quarterly_bulletin',
                                    'source_url': f"{self.quarterly_bulletin_url}#{year}",
                                    'context_text': line.strip(),
                                    'source_type': 'Quarterly Bulletin'
                                }
                                all_speeches.append(speech_info)
                                self.stats['quarterly_bulletin_speeches'] += 1
            
            logger.info(f"Found {len(all_speeches)} speeches in Quarterly Bulletins")
            
        except Exception as e:
            logger.error(f"Error scraping Quarterly Bulletins: {e}")
        
        return all_speeches

    def scrape_digital_archive_speeches(self, start_year: int = 1990, end_year: int = 2020) -> List[Dict]:
        """
        Scrape speeches from the BoE Digital Archive.
        This covers the gap between Quarterly Bulletins and modern website.
        """
        logger.info(f"Scraping BoE Digital Archive speeches ({start_year}-{end_year})")
        
        all_speeches = []
        
        # The digital archive is organized by year folders
        for year in range(start_year, end_year + 1):
            try:
                # Try to access the speeches folder for this year
                archive_url = f"{self.digital_archive_url}/?name=SPEECHES_{year}"
                
                response = requests.get(archive_url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for speech links in the archive
                    speech_links = soup.find_all('a', href=True)
                    
                    for link in speech_links:
                        href = link.get('href')
                        text = link.get_text(strip=True)
                        
                        # Check if this looks like a speech
                        if (href and text and 
                            any(keyword in text.lower() for keyword in ['speech', 'remarks', 'address', 'lecture']) and
                            len(text) > 10):
                            
                            full_url = urljoin(self.digital_archive_url, href)
                            
                            # Extract speaker from title/context
                            speaker = self._extract_speaker_from_title(text)
                            
                            speech_info = {
                                'title': text,
                                'speaker_raw': speaker or '',
                                'date': f"{year}-01-01",  # Approximate date
                                'date_source': 'digital_archive',
                                'source_url': full_url,
                                'context_text': text,
                                'source_type': 'Digital Archive'
                            }
                            all_speeches.append(speech_info)
                            self.stats['digital_archive_speeches'] += 1
                
                time.sleep(1)  # Be respectful to the archive
                
            except Exception as e:
                logger.debug(f"Could not access digital archive for {year}: {e}")
                continue
        
        logger.info(f"Found {len(all_speeches)} speeches in Digital Archive")
        return all_speeches

    def _extract_speaker_from_title(self, title: str) -> Optional[str]:
        """Extract speaker name from speech title."""
        if not title:
            return None
        
        # Common patterns in speech titles
        patterns = [
            r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'remarks by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'address by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'lecture by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*[-–]\s*speech',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*:',  # Name at start followed by colon
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                name = self._clean_and_validate_speaker_name(match.group(1))
                if name and name != 'Unknown':
                    return name
        
        return None

    # ENHANCED CONTENT EXTRACTION

    def _extract_main_content_enhanced_v2(self, soup: BeautifulSoup, url: str = None) -> str:
        """
        Enhanced content extraction with multiple fallback strategies.
        This addresses the core issue of short/empty content extraction.
        """
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.navigation', 'noscript', 'aside']):
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
            '.speech-text',
            '.text-content',
            '.body-content',
            '#content',
            '.entry-content',
            '.post-content'
        ]
        
        for selector in boe_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
                if len(text) > 300:  # Must be substantial
                    content_candidates.append(('boe_selector', text, len(text), selector))
                    logger.debug(f"Found content using selector {selector}: {len(text)} chars")
        
        # Strategy 2: Try to find the largest meaningful content block
        all_divs = soup.find_all(['div', 'section', 'article'])
        for div in all_divs:
            text = div.get_text(separator='\n', strip=True)
            if len(text) > 800:  # Must be very substantial for this method
                # Check that it's not just navigation or boilerplate
                skip_indicators = [
                    'navigation', 'skip to', 'breadcrumb', 'footer', 'sidebar', 
                    'menu', 'search', 'cookie', 'privacy', 'terms'
                ]
                if not any(skip_text in text.lower() for skip_text in skip_indicators):
                    content_candidates.append(('largest_div', text, len(text), 'div_search'))
        
        # Strategy 3: Paragraph aggregation with content filtering
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Filter out navigation and short paragraphs
            meaningful_paras = []
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if (len(p_text) > 20 and 
                    not any(skip in p_text.lower() for skip in ['cookie', 'javascript', 'skip to', 'navigation'])):
                    meaningful_paras.append(p_text)
            
            if meaningful_paras:
                para_text = '\n\n'.join(meaningful_paras)
                if len(para_text) > 500:
                    content_candidates.append(('paragraphs', para_text, len(para_text), 'paragraph_agg'))
        
        # Strategy 4: Look for content in specific BoE content patterns
        content_patterns = [
            # Look for div containing multiple paragraphs (speech content)
            lambda: soup.find('div', string=re.compile(r'speech|remarks|address', re.I)),
            # Look for container with substantial text
            lambda: soup.find('div', attrs={'class': re.compile(r'content|speech|text|body', re.I)}),
            # Look for main content area
            lambda: soup.find(attrs={'id': re.compile(r'main|content|speech', re.I)}),
        ]
        
        for pattern_func in content_patterns:
            try:
                element = pattern_func()
                if element:
                    # Get parent or the element itself
                    content_area = element.parent if element.parent else element
                    text = content_area.get_text(separator='\n', strip=True)
                    if len(text) > 400:
                        content_candidates.append(('pattern_match', text, len(text), 'content_pattern'))
            except:
                continue
        
        # Strategy 5: Body content (last resort) with better filtering
        body = soup.find('body')
        if body:
            body_text = body.get_text(separator='\n', strip=True)
            if len(body_text) > 1000:
                # Try to remove header/footer/navigation from body text
                lines = body_text.split('\n')
                filtered_lines = []
                for line in lines:
                    line = line.strip()
                    if (len(line) > 10 and 
                        not any(skip in line.lower() for skip in [
                            'bank of england', 'speeches', 'navigation', 'search', 
                            'menu', 'home', 'about', 'contact', 'privacy', 'cookies'
                        ])):
                        filtered_lines.append(line)
                
                filtered_text = '\n'.join(filtered_lines)
                if len(filtered_text) > 600:
                    content_candidates.append(('body_filtered', filtered_text, len(filtered_text), 'body'))
        
        # Choose the best candidate based on length and strategy priority
        if content_candidates:
            # Sort by strategy priority and length
            strategy_priority = {
                'boe_selector': 4,
                'pattern_match': 3,
                'largest_div': 2,
                'paragraphs': 1,
                'body_filtered': 0
            }
            
            content_candidates.sort(key=lambda x: (strategy_priority.get(x[0], 0), x[2]), reverse=True)
            best_strategy, best_content, best_length, selector = content_candidates[0]
            
            logger.info(f"Content extraction strategy: {best_strategy} via {selector} ({best_length} chars)")
            
            # Additional validation: ensure content looks like a speech
            if self._validate_speech_content(best_content):
                cleaned_content = self._clean_text_content_enhanced(best_content)
                logger.info(f"After cleaning: {len(cleaned_content)} chars")
                return cleaned_content
            else:
                logger.warning(f"Content failed speech validation, trying next candidate")
                # Try next best candidate
                if len(content_candidates) > 1:
                    second_strategy, second_content, second_length, second_selector = content_candidates[1]
                    if self._validate_speech_content(second_content):
                        cleaned_content = self._clean_text_content_enhanced(second_content)
                        logger.info(f"Using second candidate: {second_strategy} via {second_selector} ({len(cleaned_content)} chars)")
                        return cleaned_content
        
        logger.warning("No substantial valid content found with any extraction strategy")
        return ""

    def _validate_speech_content(self, content: str) -> bool:
        """Validate that content looks like an actual speech."""
        if not content or len(content) < 200:
            return False
        
        # Check for speech indicators
        speech_indicators = [
            'thank', 'pleased', 'good morning', 'good afternoon', 'good evening',
            'ladies and gentlemen', 'chair', 'chairman', 'colleagues',
            'today', 'economic', 'policy', 'bank', 'financial', 'market'
        ]
        
        content_lower = content.lower()
        indicator_count = sum(1 for indicator in speech_indicators if indicator in content_lower)
        
        # Must have at least 3 speech indicators
        if indicator_count < 3:
            return False
        
        # Check it's not just boilerplate
        boilerplate_indicators = [
            'cookies', 'javascript', 'browser', 'website', 'homepage',
            'navigation', 'search results', 'no results found'
        ]
        
        boilerplate_count = sum(1 for indicator in boilerplate_indicators if indicator in content_lower)
        
        # Reject if too much boilerplate
        if boilerplate_count > 2:
            return False
        
        return True

    def _clean_text_content_enhanced(self, text: str) -> str:
        """Enhanced text cleaning with better preservation of speech content."""
        if not text:
            return ""
        
        original_length = len(text)
        
        # Split into lines for better processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and very short lines that are likely navigation
            if len(line) < 3:
                continue
            
            # Skip lines that are clearly navigation/boilerplate
            skip_patterns = [
                r'^(Home|About|Contact|Search|Menu|Navigation) ,
                r'^(Print this page|Share this page|Last update).*',
                r'^(Skip to|Return to|Back to).*',
                r'^(Copyright|Terms|Privacy|Cookie).*',
                r'^\s*\d+\s* ,  # Just numbers
                r'^[^\w]* ,     # Just punctuation
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            cleaned_lines.append(line)
        
        # Rejoin with proper spacing
        text = '\n'.join(cleaned_lines)
        
        # Basic cleanup
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        final_length = len(text)
        
        # Log suspicious cleaning only if major content loss
        if original_length > 2000 and final_length < original_length * 0.3:
            logger.warning(f"Significant content reduction during cleaning: {original_length} -> {final_length} chars")
        
        return text

    # ENHANCED VALIDATION SYSTEM

    def _validate_speech_data_enhanced(self, speech_data: Dict) -> bool:
        """
        Enhanced validation to prevent saving invalid speeches.
        Addresses the main cause of validation failures.
        """
        if not speech_data or 'metadata' not in speech_data or 'content' not in speech_data:
            logger.warning("Speech data missing required components")
            self.stats['validation_failures'] += 1
            return False
        
        metadata = speech_data['metadata']
        content = speech_data['content']
        
        # Enhanced content validation
        if not content or len(content.strip()) < 100:  # Reduced threshold but still meaningful
            logger.warning(f"Content too short: {len(content.strip()) if content else 0} chars")
            self.stats['content_too_short'] += 1
            return False
        
        # Check for placeholder/error content
        placeholder_indicators = [
            'lorem ipsum', 'placeholder', 'test content', 'coming soon',
            'under construction', 'page not found', 'error 404', '404 not found',
            'no content available', 'content not available'
        ]
        
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in placeholder_indicators):
            logger.warning("Content appears to be placeholder or error text")
            self.stats['validation_failures'] += 1
            return False
        
        # Enhanced metadata validation
        required_fields = ['title', 'speaker', 'date', 'source_url']
        for field in required_fields:
            if not metadata.get(field):
                logger.warning(f"Missing required metadata field: {field}")
                self.stats['validation_failures'] += 1
                return False
        
        # Enhanced date validation
        date_str = metadata.get('date')
        if not date_str:
            logger.warning("No date provided")
            self.stats['validation_failures'] += 1
            return False
        
        # Allow reasonable default dates for historical speeches
        if date_str.endswith('-01-01'):
            # This is okay for historical speeches where we only have year
            pass
        elif date_str == '2025-01-01':
            # This suggests a parsing failure
            logger.warning(f"Invalid default date: {date_str}")
            self.stats['validation_failures'] += 1
            return False
        
        # Enhanced speaker validation with more permissive rules
        speaker = metadata.get('speaker', '').lower()
        if speaker in ['unknown', 'unknown speaker', '']:
            # For historical speeches, we might not always know the speaker
            # Allow if content is substantial and looks like a speech
            if len(content) > 1000 and self._validate_speech_content(content):
                logger.info("Allowing speech with unknown speaker due to substantial content")
            else:
                logger.warning(f"Unknown speaker with insufficient content: {metadata.get('speaker')}")
                self.stats['validation_failures'] += 1
                return False
        
        # Enhanced title validation
        title = metadata.get('title', '')
        if len(title) < 5:  # Reduced threshold
            logger.warning(f"Title too short: {title}")
            self.stats['validation_failures'] += 1
            return False
        
        # Check title isn't just generic
        generic_titles = ['untitled speech', 'untitled', 'speech', 'remarks', 'address']
        if title.lower() in generic_titles:
            logger.warning(f"Generic title: {title}")
            self.stats['validation_failures'] += 1
            return False
        
        return True

    # COMPREHENSIVE SCRAPING WITH HISTORICAL COVERAGE

    def run_comprehensive_scraping_v2(self, method: str = "all", start_year: int = 1960, 
                                    max_speeches: Optional[int] = None, 
                                    include_historical: bool = True) -> Dict[str, int]:
        """
        Enhanced comprehensive BoE speech scraping with historical coverage.
        """
        logger.info(f"Starting enhanced BoE speech scraping v2")
        logger.info(f"Method: {method}, Start year: {start_year}, Max speeches: {max_speeches}")
        logger.info(f"Include historical: {include_historical}")
        
        # Reset statistics
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'url_fallback_used': 0,
            'content_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0,
            'historical_speeches_found': 0,
            'quarterly_bulletin_speeches': 0,
            'digital_archive_speeches': 0
        }
        
        all_speeches = []
        
        # Phase 1: Historical speeches (if requested)
        if include_historical and start_year < 1997:
            logger.info("=== PHASE 1: HISTORICAL SPEECH COLLECTION ===")
            
            # Quarterly Bulletin speeches (1960-2006)
            if start_year <= 2006:
                logger.info("Collecting speeches from Quarterly Bulletins...")
                qb_speeches = self.scrape_quarterly_bulletin_speeches(start_year, min(2006, 2025))
                all_speeches.extend(qb_speeches)
                self.stats['historical_speeches_found'] += len(qb_speeches)
            
            # Digital Archive speeches (1990-2020)
            if start_year <= 2020:
                logger.info("Collecting speeches from Digital Archive...")
                da_speeches = self.scrape_digital_archive_speeches(max(1990, start_year), min(2020, 2025))
                all_speeches.extend(da_speeches)
                self.stats['historical_speeches_found'] += len(da_speeches)
        
        # Phase 2: Modern website scraping
        logger.info("=== PHASE 2: MODERN WEBSITE SCRAPING ===")
        
        # Approach 1: Sitemap scraping (most reliable)
        if method in ["sitemap", "all"]:
            logger.info("Running sitemap scraping...")
            sitemap_speeches = self.scrape_speeches_from_sitemap(max_speeches or 100)
            all_speeches.extend(sitemap_speeches)
            logger.info(f"Sitemap method found {len(sitemap_speeches)} speeches")
        
        # Approach 2: Main speeches page scraping
        if method in ["main", "all"] and len([s for s in all_speeches if s.get('date', '').startswith('202')]) < 10:
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
        unique_speeches = self._deduplicate_speeches_enhanced(all_speeches)
        logger.info(f"Total unique speeches found: {len(unique_speeches)}")
        
        if not unique_speeches:
            logger.warning("No speeches found!")
            return self.stats
        
        # Filter by year if requested
        if start_year:
            filtered_speeches = []
            for speech in unique_speeches:
                speech_date = speech.get('date', '')
                if speech_date:
                    try:
                        speech_year = int(speech_date[:4])
                        if speech_year >= start_year:
                            filtered_speeches.append(speech)
                    except (ValueError, IndexError):
                        # Include if we can't parse the date
                        filtered_speeches.append(speech)
                else:
                    filtered_speeches.append(speech)
            
            unique_speeches = filtered_speeches
            logger.info(f"After year filtering ({start_year}+): {len(unique_speeches)} speeches")
        
        # Limit speeches if requested
        if max_speeches:
            unique_speeches = unique_speeches[:max_speeches]
            logger.info(f"Limited to {max_speeches} speeches")
        
        # Phase 3: Process each speech
        logger.info(f"=== PHASE 3: PROCESSING {len(unique_speeches)} SPEECHES ===")
        
        for i, speech_info in enumerate(unique_speeches, 1):
            logger.info(f"Processing speech {i}/{len(unique_speeches)}: {speech_info['source_url']}")
            
            try:
                # Extract content and metadata with enhanced methods
                speech_data = self.scrape_speech_content_enhanced(speech_info)
                
                if speech_data:
                    # Save speech (already validated in scrape_speech_content_enhanced)
                    saved_filename = self.save_speech_enhanced(speech_data)
                    if saved_filename:
                        self.stats['saved_speeches'] += 1
                        
                        # Log speaker recognition details
                        metadata = speech_data['metadata']
                        logger.info(f"✓ Successfully saved: {saved_filename}")
                        logger.info(f"  Speaker: {metadata['speaker']} ({metadata.get('recognition_source', 'unknown')})")
                        logger.info(f"  Role: {metadata['role']}")
                        logger.info(f"  Date: {metadata['date']} ({metadata.get('date_source', 'unknown')})")
                    else:
                        logger.error(f"✗ Failed to save speech from {speech_info['source_url']}")
                else:
                    logger.error(f"✗ Failed to extract or validate content from {speech_info['source_url']}")
                
            except Exception as e:
                logger.error(f"✗ Unexpected error processing {speech_info['source_url']}: {e}")
            
            # Respectful delay
            time.sleep(0.5)
        
        # Final statistics
        self._log_final_statistics()
        
        return self.stats

    def scrape_speech_content_enhanced(self, speech_info: Dict) -> Optional[Dict]:
        """Enhanced speech content scraping with better validation and fallbacks."""
        url = speech_info['source_url']
        logger.debug(f"Scraping content from: {url}")
        
        self.stats['total_processed'] += 1
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                result = self._extract_pdf_content_enhanced(response.content, speech_info)
            else:
                result = self._extract_html_content_enhanced_v2(response.text, speech_info, url)
            
            # Enhanced validation
            if result and self._validate_speech_data_enhanced(result):
                self.stats['successful_extractions'] += 1
                return result
            else:
                logger.warning(f"Speech failed enhanced validation: {url}")
                self.stats['content_extraction_failures'] += 1
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            self.stats['content_extraction_failures'] += 1
            return None

    def _extract_html_content_enhanced_v2(self, html_content: str, speech_info: Dict, url: str) -> Optional[Dict]:
        """Enhanced HTML content extraction with better speaker recognition."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Enhanced content extraction
            content = self._extract_main_content_enhanced_v2(soup, url)
            
            # Early exit if no substantial content
            if not content or len(content.strip()) < 100:
                logger.warning(f"Insufficient content extracted from {url}: {len(content.strip()) if content else 0} chars")
                return None
            
            # Enhanced component extraction
            title = self._extract_title_enhanced(soup, speech_info.get('title', ''))
            
            # Enhanced speaker extraction with content awareness
            speaker_name = self.extract_speaker_from_content_enhanced(soup, url)
            if not speaker_name or speaker_name == 'Unknown':
                speaker_name = speech_info.get('speaker_raw', '')
            
            # Enhanced date extraction
            date = self.extract_date_from_url_enhanced(url)
            if not date:
                date = speech_info.get('date', '')
            
            location = self._extract_location_enhanced(soup)
            
            # Get enhanced speaker information
            role_info = self.get_speaker_info_enhanced(speaker_name, url, content)
            
            # Build comprehensive metadata
            metadata = {
                'title': title,
                'speaker': role_info.get('matched_name', speaker_name) if role_info.get('source') != 'unknown' else speaker_name,
                'role': role_info.get('role', 'Unknown'),
                'institution': 'Bank of England',
                'country': 'UK',
                'date': date,
                'location': location,
                'language': 'en',
                'source_url': url,
                'source_type': speech_info.get('source_type', 'HTML'),
                'voting_status': role_info.get('voting_status', 'Unknown'),
                'recognition_source': role_info.get('source', 'unknown'),
                'date_source': speech_info.get('date_source', 'url'),
                'tags': self._extract_content_tags_enhanced(content),
                'scrape_timestamp': datetime.now().isoformat(),
                'content_length': len(content)
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

    def _extract_content_tags_enhanced(self, content: str) -> List[str]:
        """Enhanced content tag extraction with more comprehensive keywords."""
        tags = []
        content_lower = content.lower()
        
        # Enhanced keyword mapping
        keywords = {
            'inflation': ['inflation', 'price stability', 'cpi', 'rpi', 'deflation', 'disinflation', 'price level', 'core inflation', 'headline inflation'],
            'interest_rates': ['interest rate', 'bank rate', 'monetary policy', 'policy rate', 'rate rise', 'rate cut', 'rate increase', 'rate decrease', 'base rate', 'official rate'],
            'employment': ['employment', 'unemployment', 'labour market', 'labor market', 'jobs', 'jobless', 'payroll', 'employment data', 'job growth', 'labour force', 'wage'],
            'financial_stability': ['financial stability', 'banking', 'supervision', 'regulation', 'systemic risk', 'stress test', 'capital requirements', 'prudential', 'financial system'],
            'economic_outlook': ['economic outlook', 'forecast', 'projection', 'growth', 'recession', 'expansion', 'economic conditions', 'gdp', 'economic recovery'],
            'monetary_policy': ['monetary policy', 'mpc', 'monetary policy committee', 'quantitative easing', 'qe', 'gilt purchases', 'asset purchases', 'forward guidance'],
            'banking': ['bank', 'banking', 'credit', 'lending', 'deposits', 'financial institutions', 'commercial banks', 'banking sector'],
            'markets': ['market', 'financial markets', 'capital markets', 'bond market', 'stock market', 'equity markets', 'gilt', 'currency'],
            'crisis': ['crisis', 'pandemic', 'covid', 'financial crisis', 'economic crisis', 'emergency', 'coronavirus', '2008 crisis'],
            'brexit': ['brexit', 'european union', 'eu', 'single market', 'customs union', 'trade deal', 'referendum'],
            'international': ['international', 'global', 'foreign', 'trade', 'exchange rate', 'emerging markets', 'global economy', 'international cooperation'],
            'technology': ['technology', 'digital', 'fintech', 'innovation', 'artificial intelligence', 'ai', 'blockchain', 'cryptocurrency'],
            'climate': ['climate', 'environmental', 'green', 'sustainability', 'carbon', 'net zero', 'climate change']
        }
        
        for tag, terms in keywords.items():
            if any(term in content_lower for term in terms):
                tags.append(tag)
        
        return tags

    def _deduplicate_speeches_enhanced(self, speeches: List[Dict]) -> List[Dict]:
        """Enhanced deduplication with better matching."""
        unique_speeches = []
        seen_urls = set()
        seen_combinations = set()
        
        for speech in speeches:
            url = speech.get('source_url', '')
            title = speech.get('title', '').lower().strip()
            date = speech.get('date', '')
            
            # Primary deduplication by URL
            if url and url not in seen_urls:
                # Secondary deduplication by title+date combination
                combination = f"{title}_{date}"
                if combination not in seen_combinations:
                    unique_speeches.append(speech)
                    seen_urls.add(url)
                    seen_combinations.add(combination)
        
        return unique_speeches

    def save_speech_enhanced(self, speech_data: Dict) -> Optional[str]:
        """Enhanced speech saving with better error handling."""
        try:
            metadata = speech_data['metadata']
            content = speech_data['content']
            
            # Generate content hash for uniqueness
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:10]
            
            # Enhanced speaker name sanitization
            speaker_name = metadata.get('speaker', 'unknown')
            if speaker_name and speaker_name != 'Unknown':
                clean_speaker = re.sub(r'[^\w\s-]', '', speaker_name.lower())
                clean_speaker = re.sub(r'\s+', '-', clean_speaker)
                clean_speaker = clean_speaker.strip('-')[:20]  # Limit length
                if not clean_speaker:
                    clean_speaker = 'unknown-speaker'
            else:
                clean_speaker = 'unknown-speaker'
            
            # Use the date from metadata
            date_str = metadata.get('date', 'unknown-date')
            if date_str and date_str != 'unknown-date':
                date_part = date_str[:10]  # YYYY-MM-DD
            else:
                date_part = 'unknown-date'
            
            base_filename = f"{date_part}_{clean_speaker}-{content_hash}"
            
            # Final filename sanitization
            base_filename = re.sub(r'[^\w\-.]', '', base_filename)[:100]  # Limit total length
            
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
            
            logger.debug(f"Saved speech: {base_filename}")
            return base_filename
            
        except Exception as e:
            logger.error(f"Error saving speech: {e}")
            return None

    def _log_final_statistics(self):
        """Log comprehensive final statistics."""
        logger.info("=== ENHANCED BoE SCRAPING COMPLETE ===")
        logger.info(f"Total speeches processed: {self.stats['total_processed']}")
        logger.info(f"Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"Saved speeches: {self.stats['saved_speeches']}")
        logger.info(f"Historical speeches found: {self.stats['historical_speeches_found']}")
        logger.info(f"  - Quarterly Bulletin: {self.stats['quarterly_bulletin_speeches']}")
        logger.info(f"  - Digital Archive: {self.stats['digital_archive_speeches']}")
        logger.info(f"URL fallback used: {self.stats['url_fallback_used']}")
        logger.info(f"Content fallback used: {self.stats['content_fallback_used']}")
        logger.info(f"Unknown speakers: {self.stats['unknown_speakers']}")
        logger.info(f"Date extraction failures: {self.stats['date ,  # Name on its own line
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                for match in matches:
                    name = self._clean_and_validate_speaker_name(match)
                    if name and name != 'Unknown':
                        return name
        
        # Use fallback
        if fallback_speaker:
            cleaned = self._clean_and_validate_speaker_name(fallback_speaker)
            if cleaned != 'Unknown':
                return cleaned
        
        return 'Unknown'

    def _extract_title_from_text_enhanced(self, text: str, fallback_title: str = '') -> str:
        """Enhanced title extraction from text content."""
        lines = text.split('\n')
        
        # Look for title in first several lines
        for line in lines[:15]:
            line = line.strip()
            if (len(line) > 15 and len(line) < 200 and 
                not re.match(r'^\d', line) and 
                not re.match(r'^(?:Governor|Deputy Governor|Chair|President)', line) and
                not line.lower().startswith('bank of england') and
                not line.lower().startswith('speech by') and
                ':' not in line[:20] and  # Avoid metadata lines
                not re.match(r'^\w+\s+\d{1,2},?\s+\d{4}', line)):  # Avoid date lines
                
                # Check if it looks like a title (has meaningful words)
                words = line.split()
                if len(words) >= 3 and not all(word.isupper() for word in words):
                    return line
        
        return fallback_title or 'Untitled Speech'

    def _extract_location_from_text_enhanced(self, text: str) -> str:
        """Enhanced location extraction from text content."""
        patterns = [
            r'(?:at|in)\s+([A-Z][a-zA-Z\s,&]+(?:University|College|Institute|Center|Centre|Hotel|Club|Conference|School|Hall|House))',
            r'(?:delivered at|speaking at|remarks at|given at)\s+([A-Z][a-zA-Z\s,&]{5,50})',
            r'(?:London|Birmingham|Manchester|Edinburgh|Cardiff|Belfast|Liverpool|Bristol|Leeds|Sheffield),?\s*(?:UK|England|Scotland|Wales)',
            r'([A-Z][a-zA-Z\s]+),\s+(?:London|UK|England)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                location = matches[0] if isinstance(matches[0], str) else ', '.join(matches[0])
                if len(location) < 100:
                    return location.strip()
        
        return 'London, UK'

    # ENHANCED UTILITY METHODS

    def _extract_title_enhanced(self, soup: BeautifulSoup, fallback_title: str) -> str:
        """Enhanced title extraction with more strategies."""
        selectors = [
            'h1.article-title',
            'h1.speech-title', 
            'h1',
            '.speech-title',
            '.article-header h1',
            'h2.title',
            '.page-header h1',
            '.main-title',
            '[class*="title"]',
            '.entry-title',
            '.post-title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                # Enhanced validation
                if (title and len(title) > 10 and len(title) < 300 and 
                    'Bank of England' not in title and 
                    not title.lower().startswith('speeches') and
                    not title.lower().startswith('news')):
                    return title
        
        # Try meta tags
        meta_tags = ['og:title', 'twitter:title', 'title']
        for tag in meta_tags:
            meta_title = soup.find('meta', property=tag) or soup.find('meta', attrs={'name': tag})
            if meta_title and meta_title.get('content'):
                title = meta_title['content']
                if 'Bank of England' not in title and len(title) > 10 and len(title) < 300:
                    return title
        
        # Try page title
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Clean up common suffixes
            title = re.sub(r'\s*\|\s*Bank of England.*#!/usr/bin/env python3
"""
Enhanced Bank of England Speech Scraper v2 - Historical & Current Coverage
Major improvements based on analysis of scraping issues and BoE website structure.

Key Issues Fixed:
1. Speaker recognition failures - Enhanced URL extraction and database lookups
2. Content validation failures - Improved content extraction and validation
3. Historical coverage gaps - Added Quarterly Bulletin and Digital Archive scraping
4. Success rate (38.3% -> target 80%+)

New Features:
- Historical speeches from 1990s via BoE Digital Archive
- Quarterly Bulletin speech extraction (1960s-2006)
- Enhanced speaker extraction from content and URLs
- Better content validation and metadata extraction
- Fallback mechanisms for difficult extractions

Author: Enhanced Central Bank Speech Collector
Date: 2025
Target: Comprehensive BoE speech coverage from 1960s onwards
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
from urllib.parse import urljoin, urlparse, quote
from typing import Dict, List, Optional, Tuple, Set
import time
import io

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

class EnhancedBoEScraperV2:
    """
    Enhanced Bank of England speech scraper with historical coverage and improved recognition.
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.boe_dir = os.path.join(base_dir, "boe")
        self.base_url = "https://www.bankofengland.co.uk"
        self.speeches_url = "https://www.bankofengland.co.uk/news/speeches"
        self.sitemap_url = "https://www.bankofengland.co.uk/sitemap/speeches"
        self.digital_archive_url = "https://boe.access.preservica.com"
        self.quarterly_bulletin_url = "https://www.escoe.ac.uk/research/historical-data/publist/beqb/"
        
        # Ensure directories exist
        os.makedirs(self.boe_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
        
        # Initialize comprehensive speaker database (ENHANCED)
        self._initialize_speaker_database_v2()
        
        # Initialize URL name patterns for fallback extraction
        self._initialize_url_patterns_v2()
        
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
            'content_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0,
            'historical_speeches_found': 0,
            'quarterly_bulletin_speeches': 0,
            'digital_archive_speeches': 0
        }

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

    def _initialize_url_patterns_v2(self):
        """Enhanced URL patterns for better speaker extraction."""
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
        
        # Enhanced manual patterns with historical names
        manual_patterns = {
            # Current officials
            'bailey': ['andrew bailey', 'andrew john bailey'],
            'lombardelli': ['clare lombardelli'],
            'ramsden': ['dave ramsden', 'david ramsden'],
            'breeden': ['sarah breeden'],
            'pill': ['huw pill'],
            'haskel': ['jonathan haskel'],
            'dhingra': ['swati dhingra'],
            'mann': ['catherine mann', 'catherine l mann'],
            'taylor': ['alan taylor'],
            'greene': ['megan greene'],
            'hall': ['jon hall'],
            'kroszner': ['randall kroszner'],
            
            # Past Governors - Enhanced
            'carney': ['mark carney', 'mark joseph carney'],
            'king': ['mervyn king', 'mervyn allister king', 'lord king', 'baron king', 'lord king of lothbury'],
            'george': ['eddie george', 'edward george', 'edward alan john george', 'steady eddie', 'lord george', 'baron george', 'sir eddie george', 'sir edward george'],
            'leighpemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'leigh-pemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'richardson': ['gordon richardson', 'gordon william humphreys richardson', 'lord richardson', 'baron richardson'],
            'obrien': ["leslie o'brien", "leslie kenneth o'brien", "lord o'brien"],
            "o'brien": ["leslie o'brien", "leslie kenneth o'brien", "lord o'brien"],
            
            # Deputy Governors
            'broadbent': ['ben broadbent'],
            'cunliffe': ['jon cunliffe', 'sir jon cunliffe', 'jonathan cunliffe'],
            'woods': ['sam woods', 'samuel woods'],
            'clementi': ['david clementi', 'sir david clementi'],
            'davies': ['howard davies'],
            'pennant-rea': ['rupert pennant-rea'],
            'blunden': ['george blunden'],
            'mcmahon': ['kit mcmahon', 'christopher mcmahon'],
            
            # Chief Economists
            'haldane': ['andy haldane', 'andrew haldane', 'andrew g haldane'],
            'dale': ['spencer dale'],
            'bean': ['charlie bean', 'charles bean', 'charles goodhart bean'],
            'vickers': ['john vickers', 'sir john vickers'],
            
            # External MPC Members
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
            'bell': ['marian bell'],
            'lambert': ['richard lambert'],
            
            # Executive Directors and Officials
            'tucker': ['paul tucker', 'paul mw tucker'],
            'large': ['andrew large'],
            'fisher': ['paul fisher'],
            'rule': ['david rule'],
            'cleland': ['victoria cleland'],
            'plenderleith': ['ian plenderleith'],
            'clark': ['alastair clark'],
            'quinn': ['brian quinn'],
            'kent': ['pen kent'],
            'cooke': ['william cooke', 'w p cooke'],
        }
        
        # Update with manual patterns
        for pattern, names in manual_patterns.items():
            if pattern not in self.url_name_patterns:
                self.url_name_patterns[pattern] = []
            self.url_name_patterns[pattern].extend(names)
            # Remove duplicates
            self.url_name_patterns[pattern] = list(set(self.url_name_patterns[pattern]))

    def extract_speaker_from_url_enhanced(self, url: str) -> Optional[str]:
        """
        ENHANCED speaker extraction from URL with better pattern matching.
        This addresses the main issue causing validation failures.
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            # Enhanced URL patterns for BoE speeches
            url_patterns = [
                # Pattern: speech-by-firstname-lastname
                r'speech-by-([a-z-]+)-([a-z-]+)',
                # Pattern: lastname-speech or lastname-remarks
                r'([a-z-]+)-(speech|remarks|address)',
                # Pattern: firstname-lastname-speech
                r'([a-z-]+)-([a-z-]+)-(speech|remarks|address)',
                # Pattern: remarks-given-by-firstname-lastname
                r'remarks-given-by-([a-z-]+)-([a-z-]+)',
                # Pattern: /lastname/ in path
                r'/([a-z-]+)/',
                # Pattern: just lastname before file extension
                r'([a-z-]+)\.(pdf|html?) ,
            ]
            
            for pattern in url_patterns:
                match = re.search(pattern, path)
                if match:
                    groups = match.groups()
                    
                    # Extract name parts, excluding keywords
                    name_parts = []
                    for group in groups:
                        if group not in ['speech', 'remarks', 'address', 'pdf', 'html', 'htm']:
                            name_parts.append(group.replace('-', ' '))
                    
                    if name_parts:
                        candidate_name = ' '.join(name_parts).strip()
                        
                        # Direct lookup in our patterns
                        if candidate_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[candidate_name]
                            logger.info(f"URL speaker extraction: '{candidate_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
                        
                        # Try just the last word (surname)
                        last_name = candidate_name.split()[-1] if ' ' in candidate_name else candidate_name
                        if last_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[last_name]
                            logger.info(f"URL speaker extraction (lastname): '{last_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
                        
                        # Try without hyphens/spaces
                        clean_name = candidate_name.replace(' ', '').replace('-', '')
                        if clean_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[clean_name]
                            logger.info(f"URL speaker extraction (clean): '{clean_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
            
            logger.debug(f"No speaker pattern matched for URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting speaker from URL {url}: {e}")
            return None

    def extract_speaker_from_content_enhanced(self, soup: BeautifulSoup, url: str = None) -> Optional[str]:
        """
        ENHANCED speaker extraction from content with multiple strategies.
        This is a key improvement to reduce validation failures.
        """
        # Strategy 1: Try specific speaker selectors
        speaker_selectors = [
            '.speech-author',
            '.article-author', 
            '.byline',
            '.speaker-name',
            '.author-name',
            '.speech-by',
            '[class*="author"]',
            '[class*="speaker"]',
            'h1 + p',  # Often speaker info is in paragraph after title
            '.meta-author',
            '.speech-meta'
        ]
        
        for selector in speaker_selectors:
            elements = soup.select(selector)
            for element in elements:
                speaker_text = element.get_text(strip=True)
                name = self._clean_and_validate_speaker_name(speaker_text)
                if name and name != 'Unknown':
                    logger.debug(f"Speaker found via CSS selector '{selector}': {name}")
                    return name
        
        # Strategy 2: Search in structured content with enhanced patterns
        content_areas = [
            soup.find('main'),
            soup.find('article'), 
            soup.find('div', class_='content'),
            soup.find('div', class_='speech'),
            soup
        ]
        
        for area in content_areas:
            if not area:
                continue
                
            text = area.get_text()
            
            # Enhanced speaker patterns for BoE content
            patterns = [
                # Standard titles with names
                r'Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Deputy Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Chair\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Chief Economist\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                
                # "By" patterns
                r'(?:^|\n)\s*By\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'(?:Remarks|Speech|Address)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Given by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                
                # Name followed by title
                r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair|Chief Economist)',
                
                # Lord/Sir titles
                r'(?:Lord|Sir)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                
                # Pattern: Name at start of speech
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*\n',
                
                # MPC member pattern
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*),?\s+(?:External )?MPC [Mm]ember',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE)
                if matches:
                    for match in matches:
                        name = self._clean_and_validate_speaker_name(match)
                        if name and name != 'Unknown':
                            logger.debug(f"Speaker found via content pattern: {name}")
                            return name
        
        # Strategy 3: URL fallback
        if url:
            url_name = self.extract_speaker_from_url_enhanced(url)
            if url_name:
                return url_name
        
        # Strategy 4: Title extraction fallback
        title_text = soup.find('title')
        if title_text:
            title = title_text.get_text()
            # Look for "speech by NAME" in title
            title_pattern = r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)'
            match = re.search(title_pattern, title, re.IGNORECASE)
            if match:
                name = self._clean_and_validate_speaker_name(match.group(1))
                if name and name != 'Unknown':
                    logger.debug(f"Speaker found in title: {name}")
                    return name
        
        logger.debug("No speaker found with any extraction method")
        return None

    def _clean_and_validate_speaker_name(self, raw_name: str) -> str:
        """
        Enhanced speaker name cleaning and validation.
        This is crucial for reducing validation failures.
        """
        if not raw_name:
            return 'Unknown'
        
        # Remove newlines and normalize whitespace
        raw_name = ' '.join(raw_name.split())
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            r'\b(?:The\s+)?(?:Rt\s+)?(?:Hon\s+)?(?:Sir|Lord|Baron|Dame|Dr|Mr|Ms|Mrs)\s+',
            r'\b(?:Governor|Deputy Governor|Chair|President|Chief Economist)\s+',
            r'\b(?:External\s+)?MPC\s+Member\s+',
        ]
        
        for prefix in prefixes_to_remove:
            raw_name = re.sub(prefix, '', raw_name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        raw_name = re.split(r'\s*(?:,|by|remarks|speech|address|gave|given)\s*', raw_name, flags=re.IGNORECASE)[0]
        
        # Clean and validate
        name = ' '.join(raw_name.split()).strip()
        
        # Remove periods and normalize
        name = name.replace('.', '').strip()
        
        # Validate: must be reasonable length and format
        if len(name) < 2 or len(name) > 50:
            return 'Unknown'
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', name):
            return 'Unknown'
        
        # Check against known problematic patterns
        problematic_patterns = [
            r'^\d+ ,  # Just numbers
            r'^[^\w\s]+ ,  # Just punctuation
            r'unknown|speaker|author|governor|deputy|chair|president',  # Generic terms
        ]
        
        name_lower = name.lower()
        for pattern in problematic_patterns:
            if re.match(pattern, name_lower):
                return 'Unknown'
        
        return name if name else 'Unknown'

    def get_speaker_info_enhanced(self, speaker_name: str, url: str = None, content: str = None) -> Dict[str, str]:
        """
        Enhanced speaker information lookup with multiple fallback strategies.
        """
        # Strategy 1: Try extraction from content first if available
        if content and (not speaker_name or speaker_name == 'Unknown'):
            soup_content = BeautifulSoup(content, 'html.parser')
            extracted_name = self.extract_speaker_from_content_enhanced(soup_content, url)
            if extracted_name and extracted_name != 'Unknown':
                speaker_name = extracted_name
                self.stats['content_fallback_used'] += 1
        
        # Strategy 2: Try URL extraction if still unknown
        if (not speaker_name or speaker_name == 'Unknown') and url:
            url_extracted_name = self.extract_speaker_from_url_enhanced(url)
            if url_extracted_name:
                speaker_name = url_extracted_name
        
        if not speaker_name or speaker_name.strip() == 'Unknown':
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
            if self._names_match_enhanced(normalized_name, known_name):
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
        
        # Method 4: Try fuzzy matching on first/last name combinations
        name_parts = normalized_name.split()
        if len(name_parts) >= 2:
            first_last = f"{name_parts[0]} {name_parts[-1]}"
            if first_last in self.speaker_roles:
                result = self.speaker_roles[first_last].copy()
                result['source'] = 'first_last_match'
                logger.debug(f"First-last match found: {speaker_name} -> {first_last}")
                return result
        
        # Final fallback: Unknown speaker
        logger.warning(f"Unknown speaker after all methods: {speaker_name} (normalized: {normalized_name})")
        self.stats['unknown_speakers'] += 1
        return {"role": "Unknown", "voting_status": "Unknown", "source": "unknown"}

    def _names_match_enhanced(self, name1: str, name2: str) -> bool:
        """Enhanced name matching with better fuzzy logic."""
        if not name1 or not name2:
            return False
            
        parts1 = set(name1.replace('.', '').replace('-', ' ').split())
        parts2 = set(name2.replace('.', '').replace('-', ' ').split())
        
        # Remove common middle initials and abbreviations
        parts1 = {p for p in parts1 if len(p) > 1}
        parts2 = {p for p in parts2 if len(p) > 1}
        
        common_parts = parts1.intersection(parts2)
        
        # For short names (2 parts or less), require full overlap
        if len(parts1) <= 2 and len(parts2) <= 2:
            return len(common_parts) >= min(len(parts1), len(parts2))
        else:
            # For longer names, require at least 2 matching parts
            return len(common_parts) >= 2

    def _clean_speaker_name_for_lookup(self, name: str) -> str:
        """Enhanced speaker name cleaning for database lookup."""
        if not name:
            return ""
        
        # Remove titles and clean more thoroughly
        name = re.sub(r'\b(?:The\s+)?(?:Rt\s+)?(?:Hon\s+)?(?:Governor|Deputy Governor|Chair|President|Dr\.|Mr\.|Ms\.|Mrs\.|Sir|Lord|Baron|Dame)\s*', '', name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        name = re.split(r'\s*(?:,|by|remarks|speech|address|gave|given)\s*', name, flags=re.IGNORECASE)[0]
        
        # Remove newlines and extra whitespace
        name = ' '.join(name.split())
        
        # Convert to lowercase and remove periods
        name = name.lower().strip().replace('.', '')
        
        return name

    def extract_date_from_url_enhanced(self, url: str) -> Optional[str]:
        """
        Enhanced date extraction with more patterns and better validation.
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            # Enhanced BoE URL date patterns
            date_patterns = [
                # Current BoE pattern: /speech/2024/october/title-slug
                r'/speech/(\d{4})/(\w+)/',
                # Pattern: /speech/2024/10/title-slug  
                r'/speech/(\d{4})/(\d{1,2})/',
                # Legacy patterns
                r'/speeches/(\d{4})/(\w+)/',
                r'/speeches/(\d{4})/(\d{1,2})/',
                # Media files pattern: /files/speech/2024/july/
                r'/files/speech/(\d{4})/(\w+)/',
                # Date in filename: speech-2024-10-15
                r'speech-(\d{4})-(\d{2})-(\d{2})',
                # Pattern: embedded YYYYMMDD
                r'(\d{8})',
                # Year-only pattern for historical speeches
                r'/(\d{4})/',
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, path)
                if match:
                    groups = match.groups()
                    
                    if len(groups) == 1:
                        # Could be YYYYMMDD or just year
                        date_str = groups[0]
                        if len(date_str) == 8:  # YYYYMMDD
                            try:
                                year = int(date_str[:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                if 1960 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                    date_obj = datetime(year, month, day)
                                    return date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                continue
                        elif len(date_str) == 4:  # Just year
                            try:
                                year = int(date_str)
                                if 1960 <= year <= 2030:
                                    # Use January 1st as default
                                    return f"{year}-01-01"
                            except ValueError:
                                continue
                    
                    elif len(groups) == 2:
                        year_str, month_str = groups
                        try:
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
                            if 1960 <= year <= 2030 and 1 <= month <= 12:
                                date_obj = datetime(year, month, 1)
                                formatted_date = date_obj.strftime('%Y-%m-%d')
                                logger.debug(f"URL date extraction successful: {url} -> {formatted_date}")
                                return formatted_date
                        except ValueError:
                            continue
                            
                    elif len(groups) == 3:
                        year_str, month_str, day_str = groups
                        try:
                            year = int(year_str)
                            month = int(month_str)
                            day = int(day_str)
                            
                            if 1960 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                date_obj = datetime(year, month, day)
                                formatted_date = date_obj.strftime('%Y-%m-%d')
                                logger.debug(f"URL date extraction successful: {url} -> {formatted_date}")
                                return formatted_date
                        except ValueError:
                            continue
            
            logger.debug(f"No valid date pattern found in URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting date from URL {url}: {e}")
            return None

    # NEW METHODS FOR HISTORICAL COVERAGE

    def scrape_quarterly_bulletin_speeches(self, start_year: int = 1960, end_year: int = 2006) -> List[Dict]:
        """
        Scrape historical speeches from Bank of England Quarterly Bulletins (1960-2006).
        This provides access to speeches from the pre-digital era.
        """
        logger.info(f"Scraping BoE Quarterly Bulletin speeches ({start_year}-{end_year})")
        
        all_speeches = []
        
        try:
            response = requests.get(self.quarterly_bulletin_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all speech references in the quarterly bulletin listings
            # Pattern: "speech by [Name]" or "Governor's speech"
            speech_patterns = [
                r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*[-–]\s*speech',
                r"Governor's speech",
                r'remarks by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                r'lecture by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            ]
            
            # Look for speech references in the bulletin content
            text = soup.get_text()
            lines = text.split('\n')
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['speech', 'remarks', 'lecture', 'address']):
                    # Extract year from line if present
                    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', line)
                    if year_match:
                        year = int(year_match.group(1))
                        if start_year <= year <= end_year:
                            
                            # Try to extract speaker name
                            speaker = None
                            for pattern in speech_patterns:
                                match = re.search(pattern, line, re.IGNORECASE)
                                if match and match.groups():
                                    speaker = match.group(1)
                                    break
                            
                            if not speaker and "governor's speech" in line.lower():
                                # Determine governor based on year
                                if 1966 <= year <= 1973:
                                    speaker = "leslie o'brien"
                                elif 1973 <= year <= 1983:
                                    speaker = "gordon richardson"
                                elif 1983 <= year <= 1993:
                                    speaker = "robin leigh-pemberton"
                                elif 1993 <= year <= 2003:
                                    speaker = "eddie george"
                                elif 2003 <= year <= 2006:
                                    speaker = "mervyn king"
                            
                            if speaker:
                                speech_info = {
                                    'title': line.strip(),
                                    'speaker_raw': speaker,
                                    'date': f"{year}-01-01",  # Approximate date
                                    'date_source': 'quarterly_bulletin',
                                    'source_url': f"{self.quarterly_bulletin_url}#{year}",
                                    'context_text': line.strip(),
                                    'source_type': 'Quarterly Bulletin'
                                }
                                all_speeches.append(speech_info)
                                self.stats['quarterly_bulletin_speeches'] += 1
            
            logger.info(f"Found {len(all_speeches)} speeches in Quarterly Bulletins")
            
        except Exception as e:
            logger.error(f"Error scraping Quarterly Bulletins: {e}")
        
        return all_speeches

    def scrape_digital_archive_speeches(self, start_year: int = 1990, end_year: int = 2020) -> List[Dict]:
        """
        Scrape speeches from the BoE Digital Archive.
        This covers the gap between Quarterly Bulletins and modern website.
        """
        logger.info(f"Scraping BoE Digital Archive speeches ({start_year}-{end_year})")
        
        all_speeches = []
        
        # The digital archive is organized by year folders
        for year in range(start_year, end_year + 1):
            try:
                # Try to access the speeches folder for this year
                archive_url = f"{self.digital_archive_url}/?name=SPEECHES_{year}"
                
                response = requests.get(archive_url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for speech links in the archive
                    speech_links = soup.find_all('a', href=True)
                    
                    for link in speech_links:
                        href = link.get('href')
                        text = link.get_text(strip=True)
                        
                        # Check if this looks like a speech
                        if (href and text and 
                            any(keyword in text.lower() for keyword in ['speech', 'remarks', 'address', 'lecture']) and
                            len(text) > 10):
                            
                            full_url = urljoin(self.digital_archive_url, href)
                            
                            # Extract speaker from title/context
                            speaker = self._extract_speaker_from_title(text)
                            
                            speech_info = {
                                'title': text,
                                'speaker_raw': speaker or '',
                                'date': f"{year}-01-01",  # Approximate date
                                'date_source': 'digital_archive',
                                'source_url': full_url,
                                'context_text': text,
                                'source_type': 'Digital Archive'
                            }
                            all_speeches.append(speech_info)
                            self.stats['digital_archive_speeches'] += 1
                
                time.sleep(1)  # Be respectful to the archive
                
            except Exception as e:
                logger.debug(f"Could not access digital archive for {year}: {e}")
                continue
        
        logger.info(f"Found {len(all_speeches)} speeches in Digital Archive")
        return all_speeches

    def _extract_speaker_from_title(self, title: str) -> Optional[str]:
        """Extract speaker name from speech title."""
        if not title:
            return None
        
        # Common patterns in speech titles
        patterns = [
            r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'remarks by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'address by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'lecture by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*[-–]\s*speech',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*:',  # Name at start followed by colon
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                name = self._clean_and_validate_speaker_name(match.group(1))
                if name and name != 'Unknown':
                    return name
        
        return None

    # ENHANCED CONTENT EXTRACTION

    def _extract_main_content_enhanced_v2(self, soup: BeautifulSoup, url: str = None) -> str:
        """
        Enhanced content extraction with multiple fallback strategies.
        This addresses the core issue of short/empty content extraction.
        """
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.navigation', 'noscript', 'aside']):
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
            '.speech-text',
            '.text-content',
            '.body-content',
            '#content',
            '.entry-content',
            '.post-content'
        ]
        
        for selector in boe_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
                if len(text) > 300:  # Must be substantial
                    content_candidates.append(('boe_selector', text, len(text), selector))
                    logger.debug(f"Found content using selector {selector}: {len(text)} chars")
        
        # Strategy 2: Try to find the largest meaningful content block
        all_divs = soup.find_all(['div', 'section', 'article'])
        for div in all_divs:
            text = div.get_text(separator='\n', strip=True)
            if len(text) > 800:  # Must be very substantial for this method
                # Check that it's not just navigation or boilerplate
                skip_indicators = [
                    'navigation', 'skip to', 'breadcrumb', 'footer', 'sidebar', 
                    'menu', 'search', 'cookie', 'privacy', 'terms'
                ]
                if not any(skip_text in text.lower() for skip_text in skip_indicators):
                    content_candidates.append(('largest_div', text, len(text), 'div_search'))
        
        # Strategy 3: Paragraph aggregation with content filtering
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Filter out navigation and short paragraphs
            meaningful_paras = []
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if (len(p_text) > 20 and 
                    not any(skip in p_text.lower() for skip in ['cookie', 'javascript', 'skip to', 'navigation'])):
                    meaningful_paras.append(p_text)
            
            if meaningful_paras:
                para_text = '\n\n'.join(meaningful_paras)
                if len(para_text) > 500:
                    content_candidates.append(('paragraphs', para_text, len(para_text), 'paragraph_agg'))
        
        # Strategy 4: Look for content in specific BoE content patterns
        content_patterns = [
            # Look for div containing multiple paragraphs (speech content)
            lambda: soup.find('div', string=re.compile(r'speech|remarks|address', re.I)),
            # Look for container with substantial text
            lambda: soup.find('div', attrs={'class': re.compile(r'content|speech|text|body', re.I)}),
            # Look for main content area
            lambda: soup.find(attrs={'id': re.compile(r'main|content|speech', re.I)}),
        ]
        
        for pattern_func in content_patterns:
            try:
                element = pattern_func()
                if element:
                    # Get parent or the element itself
                    content_area = element.parent if element.parent else element
                    text = content_area.get_text(separator='\n', strip=True)
                    if len(text) > 400:
                        content_candidates.append(('pattern_match', text, len(text), 'content_pattern'))
            except:
                continue
        
        # Strategy 5: Body content (last resort) with better filtering
        body = soup.find('body')
        if body:
            body_text = body.get_text(separator='\n', strip=True)
            if len(body_text) > 1000:
                # Try to remove header/footer/navigation from body text
                lines = body_text.split('\n')
                filtered_lines = []
                for line in lines:
                    line = line.strip()
                    if (len(line) > 10 and 
                        not any(skip in line.lower() for skip in [
                            'bank of england', 'speeches', 'navigation', 'search', 
                            'menu', 'home', 'about', 'contact', 'privacy', 'cookies'
                        ])):
                        filtered_lines.append(line)
                
                filtered_text = '\n'.join(filtered_lines)
                if len(filtered_text) > 600:
                    content_candidates.append(('body_filtered', filtered_text, len(filtered_text), 'body'))
        
        # Choose the best candidate based on length and strategy priority
        if content_candidates:
            # Sort by strategy priority and length
            strategy_priority = {
                'boe_selector': 4,
                'pattern_match': 3,
                'largest_div': 2,
                'paragraphs': 1,
                'body_filtered': 0
            }
            
            content_candidates.sort(key=lambda x: (strategy_priority.get(x[0], 0), x[2]), reverse=True)
            best_strategy, best_content, best_length, selector = content_candidates[0]
            
            logger.info(f"Content extraction strategy: {best_strategy} via {selector} ({best_length} chars)")
            
            # Additional validation: ensure content looks like a speech
            if self._validate_speech_content(best_content):
                cleaned_content = self._clean_text_content_enhanced(best_content)
                logger.info(f"After cleaning: {len(cleaned_content)} chars")
                return cleaned_content
            else:
                logger.warning(f"Content failed speech validation, trying next candidate")
                # Try next best candidate
                if len(content_candidates) > 1:
                    second_strategy, second_content, second_length, second_selector = content_candidates[1]
                    if self._validate_speech_content(second_content):
                        cleaned_content = self._clean_text_content_enhanced(second_content)
                        logger.info(f"Using second candidate: {second_strategy} via {second_selector} ({len(cleaned_content)} chars)")
                        return cleaned_content
        
        logger.warning("No substantial valid content found with any extraction strategy")
        return ""

    def _validate_speech_content(self, content: str) -> bool:
        """Validate that content looks like an actual speech."""
        if not content or len(content) < 200:
            return False
        
        # Check for speech indicators
        speech_indicators = [
            'thank', 'pleased', 'good morning', 'good afternoon', 'good evening',
            'ladies and gentlemen', 'chair', 'chairman', 'colleagues',
            'today', 'economic', 'policy', 'bank', 'financial', 'market'
        ]
        
        content_lower = content.lower()
        indicator_count = sum(1 for indicator in speech_indicators if indicator in content_lower)
        
        # Must have at least 3 speech indicators
        if indicator_count < 3:
            return False
        
        # Check it's not just boilerplate
        boilerplate_indicators = [
            'cookies', 'javascript', 'browser', 'website', 'homepage',
            'navigation', 'search results', 'no results found'
        ]
        
        boilerplate_count = sum(1 for indicator in boilerplate_indicators if indicator in content_lower)
        
        # Reject if too much boilerplate
        if boilerplate_count > 2:
            return False
        
        return True

    def _clean_text_content_enhanced(self, text: str) -> str:
        """Enhanced text cleaning with better preservation of speech content."""
        if not text:
            return ""
        
        original_length = len(text)
        
        # Split into lines for better processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and very short lines that are likely navigation
            if len(line) < 3:
                continue
            
            # Skip lines that are clearly navigation/boilerplate
            skip_patterns = [
                r'^(Home|About|Contact|Search|Menu|Navigation) ,
                r'^(Print this page|Share this page|Last update).*',
                r'^(Skip to|Return to|Back to).*',
                r'^(Copyright|Terms|Privacy|Cookie).*',
                r'^\s*\d+\s* ,  # Just numbers
                r'^[^\w]* ,     # Just punctuation
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            cleaned_lines.append(line)
        
        # Rejoin with proper spacing
        text = '\n'.join(cleaned_lines)
        
        # Basic cleanup
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        final_length = len(text)
        
        # Log suspicious cleaning only if major content loss
        if original_length > 2000 and final_length < original_length * 0.3:
            logger.warning(f"Significant content reduction during cleaning: {original_length} -> {final_length} chars")
        
        return text

    # ENHANCED VALIDATION SYSTEM

    def _validate_speech_data_enhanced(self, speech_data: Dict) -> bool:
        """
        Enhanced validation to prevent saving invalid speeches.
        Addresses the main cause of validation failures.
        """
        if not speech_data or 'metadata' not in speech_data or 'content' not in speech_data:
            logger.warning("Speech data missing required components")
            self.stats['validation_failures'] += 1
            return False
        
        metadata = speech_data['metadata']
        content = speech_data['content']
        
        # Enhanced content validation
        if not content or len(content.strip()) < 100:  # Reduced threshold but still meaningful
            logger.warning(f"Content too short: {len(content.strip()) if content else 0} chars")
            self.stats['content_too_short'] += 1
            return False
        
        # Check for placeholder/error content
        placeholder_indicators = [
            'lorem ipsum', 'placeholder', 'test content', 'coming soon',
            'under construction', 'page not found', 'error 404', '404 not found',
            'no content available', 'content not available'
        ]
        
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in placeholder_indicators):
            logger.warning("Content appears to be placeholder or error text")
            self.stats['validation_failures'] += 1
            return False
        
        # Enhanced metadata validation
        required_fields = ['title', 'speaker', 'date', 'source_url']
        for field in required_fields:
            if not metadata.get(field):
                logger.warning(f"Missing required metadata field: {field}")
                self.stats['validation_failures'] += 1
                return False
        
        # Enhanced date validation
        date_str = metadata.get('date')
        if not date_str:
            logger.warning("No date provided")
            self.stats['validation_failures'] += 1
            return False
        
        # Allow reasonable default dates for historical speeches
        if date_str.endswith('-01-01'):
            # This is okay for historical speeches where we only have year
            pass
        elif date_str == '2025-01-01':
            # This suggests a parsing failure
            logger.warning(f"Invalid default date: {date_str}")
            self.stats['validation_failures'] += 1
            return False
        
        # Enhanced speaker validation with more permissive rules
        speaker = metadata.get('speaker', '').lower()
        if speaker in ['unknown', 'unknown speaker', '']:
            # For historical speeches, we might not always know the speaker
            # Allow if content is substantial and looks like a speech
            if len(content) > 1000 and self._validate_speech_content(content):
                logger.info("Allowing speech with unknown speaker due to substantial content")
            else:
                logger.warning(f"Unknown speaker with insufficient content: {metadata.get('speaker')}")
                self.stats['validation_failures'] += 1
                return False
        
        # Enhanced title validation
        title = metadata.get('title', '')
        if len(title) < 5:  # Reduced threshold
            logger.warning(f"Title too short: {title}")
            self.stats['validation_failures'] += 1
            return False
        
        # Check title isn't just generic
        generic_titles = ['untitled speech', 'untitled', 'speech', 'remarks', 'address']
        if title.lower() in generic_titles:
            logger.warning(f"Generic title: {title}")
            self.stats['validation_failures'] += 1
            return False
        
        return True

    # COMPREHENSIVE SCRAPING WITH HISTORICAL COVERAGE

    def run_comprehensive_scraping_v2(self, method: str = "all", start_year: int = 1960, 
                                    max_speeches: Optional[int] = None, 
                                    include_historical: bool = True) -> Dict[str, int]:
        """
        Enhanced comprehensive BoE speech scraping with historical coverage.
        """
        logger.info(f"Starting enhanced BoE speech scraping v2")
        logger.info(f"Method: {method}, Start year: {start_year}, Max speeches: {max_speeches}")
        logger.info(f"Include historical: {include_historical}")
        
        # Reset statistics
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'url_fallback_used': 0,
            'content_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0,
            'historical_speeches_found': 0,
            'quarterly_bulletin_speeches': 0,
            'digital_archive_speeches': 0
        }
        
        all_speeches = []
        
        # Phase 1: Historical speeches (if requested)
        if include_historical and start_year < 1997:
            logger.info("=== PHASE 1: HISTORICAL SPEECH COLLECTION ===")
            
            # Quarterly Bulletin speeches (1960-2006)
            if start_year <= 2006:
                logger.info("Collecting speeches from Quarterly Bulletins...")
                qb_speeches = self.scrape_quarterly_bulletin_speeches(start_year, min(2006, 2025))
                all_speeches.extend(qb_speeches)
                self.stats['historical_speeches_found'] += len(qb_speeches)
            
            # Digital Archive speeches (1990-2020)
            if start_year <= 2020:
                logger.info("Collecting speeches from Digital Archive...")
                da_speeches = self.scrape_digital_archive_speeches(max(1990, start_year), min(2020, 2025))
                all_speeches.extend(da_speeches)
                self.stats['historical_speeches_found'] += len(da_speeches)
        
        # Phase 2: Modern website scraping
        logger.info("=== PHASE 2: MODERN WEBSITE SCRAPING ===")
        
        # Approach 1: Sitemap scraping (most reliable)
        if method in ["sitemap", "all"]:
            logger.info("Running sitemap scraping...")
            sitemap_speeches = self.scrape_speeches_from_sitemap(max_speeches or 100)
            all_speeches.extend(sitemap_speeches)
            logger.info(f"Sitemap method found {len(sitemap_speeches)} speeches")
        
        # Approach 2: Main speeches page scraping
        if method in ["main", "all"] and len([s for s in all_speeches if s.get('date', '').startswith('202')]) < 10:
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
        unique_speeches = self._deduplicate_speeches_enhanced(all_speeches)
        logger.info(f"Total unique speeches found: {len(unique_speeches)}")
        
        if not unique_speeches:
            logger.warning("No speeches found!")
            return self.stats
        
        # Filter by year if requested
        if start_year:
            filtered_speeches = []
            for speech in unique_speeches:
                speech_date = speech.get('date', '')
                if speech_date:
                    try:
                        speech_year = int(speech_date[:4])
                        if speech_year >= start_year:
                            filtered_speeches.append(speech)
                    except (ValueError, IndexError):
                        # Include if we can't parse the date
                        filtered_speeches.append(speech)
                else:
                    filtered_speeches.append(speech)
            
            unique_speeches = filtered_speeches
            logger.info(f"After year filtering ({start_year}+): {len(unique_speeches)} speeches")
        
        # Limit speeches if requested
        if max_speeches:
            unique_speeches = unique_speeches[:max_speeches]
            logger.info(f"Limited to {max_speeches} speeches")
        
        # Phase 3: Process each speech
        logger.info(f"=== PHASE 3: PROCESSING {len(unique_speeches)} SPEECHES ===")
        
        for i, speech_info in enumerate(unique_speeches, 1):
            logger.info(f"Processing speech {i}/{len(unique_speeches)}: {speech_info['source_url']}")
            
            try:
                # Extract content and metadata with enhanced methods
                speech_data = self.scrape_speech_content_enhanced(speech_info)
                
                if speech_data:
                    # Save speech (already validated in scrape_speech_content_enhanced)
                    saved_filename = self.save_speech_enhanced(speech_data)
                    if saved_filename:
                        self.stats['saved_speeches'] += 1
                        
                        # Log speaker recognition details
                        metadata = speech_data['metadata']
                        logger.info(f"✓ Successfully saved: {saved_filename}")
                        logger.info(f"  Speaker: {metadata['speaker']} ({metadata.get('recognition_source', 'unknown')})")
                        logger.info(f"  Role: {metadata['role']}")
                        logger.info(f"  Date: {metadata['date']} ({metadata.get('date_source', 'unknown')})")
                    else:
                        logger.error(f"✗ Failed to save speech from {speech_info['source_url']}")
                else:
                    logger.error(f"✗ Failed to extract or validate content from {speech_info['source_url']}")
                
            except Exception as e:
                logger.error(f"✗ Unexpected error processing {speech_info['source_url']}: {e}")
            
            # Respectful delay
            time.sleep(0.5)
        
        # Final statistics
        self._log_final_statistics()
        
        return self.stats

    def scrape_speech_content_enhanced(self, speech_info: Dict) -> Optional[Dict]:
        """Enhanced speech content scraping with better validation and fallbacks."""
        url = speech_info['source_url']
        logger.debug(f"Scraping content from: {url}")
        
        self.stats['total_processed'] += 1
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                result = self._extract_pdf_content_enhanced(response.content, speech_info)
            else:
                result = self._extract_html_content_enhanced_v2(response.text, speech_info, url)
            
            # Enhanced validation
            if result and self._validate_speech_data_enhanced(result):
                self.stats['successful_extractions'] += 1
                return result
            else:
                logger.warning(f"Speech failed enhanced validation: {url}")
                self.stats['content_extraction_failures'] += 1
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            self.stats['content_extraction_failures'] += 1
            return None

    def _extract_html_content_enhanced_v2(self, html_content: str, speech_info: Dict, url: str) -> Optional[Dict]:
        """Enhanced HTML content extraction with better speaker recognition."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Enhanced content extraction
            content = self._extract_main_content_enhanced_v2(soup, url)
            
            # Early exit if no substantial content
            if not content or len(content.strip()) < 100:
                logger.warning(f"Insufficient content extracted from {url}: {len(content.strip()) if content else 0} chars")
                return None
            
            # Enhanced component extraction
            title = self._extract_title_enhanced(soup, speech_info.get('title', ''))
            
            # Enhanced speaker extraction with content awareness
            speaker_name = self.extract_speaker_from_content_enhanced(soup, url)
            if not speaker_name or speaker_name == 'Unknown':
                speaker_name = speech_info.get('speaker_raw', '')
            
            # Enhanced date extraction
            date = self.extract_date_from_url_enhanced(url)
            if not date:
                date = speech_info.get('date', '')
            
            location = self._extract_location_enhanced(soup)
            
            # Get enhanced speaker information
            role_info = self.get_speaker_info_enhanced(speaker_name, url, content)
            
            # Build comprehensive metadata
            metadata = {
                'title': title,
                'speaker': role_info.get('matched_name', speaker_name) if role_info.get('source') != 'unknown' else speaker_name,
                'role': role_info.get('role', 'Unknown'),
                'institution': 'Bank of England',
                'country': 'UK',
                'date': date,
                'location': location,
                'language': 'en',
                'source_url': url,
                'source_type': speech_info.get('source_type', 'HTML'),
                'voting_status': role_info.get('voting_status', 'Unknown'),
                'recognition_source': role_info.get('source', 'unknown'),
                'date_source': speech_info.get('date_source', 'url'),
                'tags': self._extract_content_tags_enhanced(content),
                'scrape_timestamp': datetime.now().isoformat(),
                'content_length': len(content)
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

    def _extract_content_tags_enhanced(self, content: str) -> List[str]:
        """Enhanced content tag extraction with more comprehensive keywords."""
        tags = []
        content_lower = content.lower()
        
        # Enhanced keyword mapping
        keywords = {
            'inflation': ['inflation', 'price stability', 'cpi', 'rpi', 'deflation', 'disinflation', 'price level', 'core inflation', 'headline inflation'],
            'interest_rates': ['interest rate', 'bank rate', 'monetary policy', 'policy rate', 'rate rise', 'rate cut', 'rate increase', 'rate decrease', 'base rate', 'official rate'],
            'employment': ['employment', 'unemployment', 'labour market', 'labor market', 'jobs', 'jobless', 'payroll', 'employment data', 'job growth', 'labour force', 'wage'],
            'financial_stability': ['financial stability', 'banking', 'supervision', 'regulation', 'systemic risk', 'stress test', 'capital requirements', 'prudential', 'financial system'],
            'economic_outlook': ['economic outlook', 'forecast', 'projection', 'growth', 'recession', 'expansion', 'economic conditions', 'gdp', 'economic recovery'],
            'monetary_policy': ['monetary policy', 'mpc', 'monetary policy committee', 'quantitative easing', 'qe', 'gilt purchases', 'asset purchases', 'forward guidance'],
            'banking': ['bank', 'banking', 'credit', 'lending', 'deposits', 'financial institutions', 'commercial banks', 'banking sector'],
            'markets': ['market', 'financial markets', 'capital markets', 'bond market', 'stock market', 'equity markets', 'gilt', 'currency'],
            'crisis': ['crisis', 'pandemic', 'covid', 'financial crisis', 'economic crisis', 'emergency', 'coronavirus', '2008 crisis'],
            'brexit': ['brexit', 'european union', 'eu', 'single market', 'customs union', 'trade deal', 'referendum'],
            'international': ['international', 'global', 'foreign', 'trade', 'exchange rate', 'emerging markets', 'global economy', 'international cooperation'],
            'technology': ['technology', 'digital', 'fintech', 'innovation', 'artificial intelligence', 'ai', 'blockchain', 'cryptocurrency'],
            'climate': ['climate', 'environmental', 'green', 'sustainability', 'carbon', 'net zero', 'climate change']
        }
        
        for tag, terms in keywords.items():
            if any(term in content_lower for term in terms):
                tags.append(tag)
        
        return tags

    def _deduplicate_speeches_enhanced(self, speeches: List[Dict]) -> List[Dict]:
        """Enhanced deduplication with better matching."""
        unique_speeches = []
        seen_urls = set()
        seen_combinations = set()
        
        for speech in speeches:
            url = speech.get('source_url', '')
            title = speech.get('title', '').lower().strip()
            date = speech.get('date', '')
            
            # Primary deduplication by URL
            if url and url not in seen_urls:
                # Secondary deduplication by title+date combination
                combination = f"{title}_{date}"
                if combination not in seen_combinations:
                    unique_speeches.append(speech)
                    seen_urls.add(url)
                    seen_combinations.add(combination)
        
        return unique_speeches

    def save_speech_enhanced(self, speech_data: Dict) -> Optional[str]:
        """Enhanced speech saving with better error handling."""
        try:
            metadata = speech_data['metadata']
            content = speech_data['content']
            
            # Generate content hash for uniqueness
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:10]
            
            # Enhanced speaker name sanitization
            speaker_name = metadata.get('speaker', 'unknown')
            if speaker_name and speaker_name != 'Unknown':
                clean_speaker = re.sub(r'[^\w\s-]', '', speaker_name.lower())
                clean_speaker = re.sub(r'\s+', '-', clean_speaker)
                clean_speaker = clean_speaker.strip('-')[:20]  # Limit length
                if not clean_speaker:
                    clean_speaker = 'unknown-speaker'
            else:
                clean_speaker = 'unknown-speaker'
            
            # Use the date from metadata
            date_str = metadata.get('date', 'unknown-date')
            if date_str and date_str != 'unknown-date':
                date_part = date_str[:10]  # YYYY-MM-DD
            else:
                date_part = 'unknown-date'
            
            base_filename = f"{date_part}_{clean_speaker}-{content_hash}"
            
            # Final filename sanitization
            base_filename = re.sub(r'[^\w\-.]', '', base_filename)[:100]  # Limit total length
            
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
            
            logger.debug(f"Saved speech: {base_filename}")
            return base_filename
            
        except Exception as e:
            logger.error(f"Error saving speech: {e}")
            return None

    def _log_final_statistics(self):
        """Log comprehensive final statistics."""
        logger.info("=== ENHANCED BoE SCRAPING COMPLETE ===")
        logger.info(f"Total speeches processed: {self.stats['total_processed']}")
        logger.info(f"Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"Saved speeches: {self.stats['saved_speeches']}")
        logger.info(f"Historical speeches found: {self.stats['historical_speeches_found']}")
        logger.info(f"  - Quarterly Bulletin: {self.stats['quarterly_bulletin_speeches']}")
        logger.info(f"  - Digital Archive: {self.stats['digital_archive_speeches']}")
        logger.info(f"URL fallback used: {self.stats['url_fallback_used']}")
        logger.info(f"Content fallback used: {self.stats['content_fallback_used']}")
        logger.info(f"Unknown speakers: {self.stats['unknown_speakers']}")
        logger.info(f"Date extraction failures: {self.stats['date , '', title)
            title = re.sub(r'\s*-\s*Bank of England.*#!/usr/bin/env python3
"""
Enhanced Bank of England Speech Scraper v2 - Historical & Current Coverage
Major improvements based on analysis of scraping issues and BoE website structure.

Key Issues Fixed:
1. Speaker recognition failures - Enhanced URL extraction and database lookups
2. Content validation failures - Improved content extraction and validation
3. Historical coverage gaps - Added Quarterly Bulletin and Digital Archive scraping
4. Success rate (38.3% -> target 80%+)

New Features:
- Historical speeches from 1990s via BoE Digital Archive
- Quarterly Bulletin speech extraction (1960s-2006)
- Enhanced speaker extraction from content and URLs
- Better content validation and metadata extraction
- Fallback mechanisms for difficult extractions

Author: Enhanced Central Bank Speech Collector
Date: 2025
Target: Comprehensive BoE speech coverage from 1960s onwards
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
from urllib.parse import urljoin, urlparse, quote
from typing import Dict, List, Optional, Tuple, Set
import time
import io

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

class EnhancedBoEScraperV2:
    """
    Enhanced Bank of England speech scraper with historical coverage and improved recognition.
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.boe_dir = os.path.join(base_dir, "boe")
        self.base_url = "https://www.bankofengland.co.uk"
        self.speeches_url = "https://www.bankofengland.co.uk/news/speeches"
        self.sitemap_url = "https://www.bankofengland.co.uk/sitemap/speeches"
        self.digital_archive_url = "https://boe.access.preservica.com"
        self.quarterly_bulletin_url = "https://www.escoe.ac.uk/research/historical-data/publist/beqb/"
        
        # Ensure directories exist
        os.makedirs(self.boe_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
        
        # Initialize comprehensive speaker database (ENHANCED)
        self._initialize_speaker_database_v2()
        
        # Initialize URL name patterns for fallback extraction
        self._initialize_url_patterns_v2()
        
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
            'content_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0,
            'historical_speeches_found': 0,
            'quarterly_bulletin_speeches': 0,
            'digital_archive_speeches': 0
        }

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

    def _initialize_url_patterns_v2(self):
        """Enhanced URL patterns for better speaker extraction."""
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
        
        # Enhanced manual patterns with historical names
        manual_patterns = {
            # Current officials
            'bailey': ['andrew bailey', 'andrew john bailey'],
            'lombardelli': ['clare lombardelli'],
            'ramsden': ['dave ramsden', 'david ramsden'],
            'breeden': ['sarah breeden'],
            'pill': ['huw pill'],
            'haskel': ['jonathan haskel'],
            'dhingra': ['swati dhingra'],
            'mann': ['catherine mann', 'catherine l mann'],
            'taylor': ['alan taylor'],
            'greene': ['megan greene'],
            'hall': ['jon hall'],
            'kroszner': ['randall kroszner'],
            
            # Past Governors - Enhanced
            'carney': ['mark carney', 'mark joseph carney'],
            'king': ['mervyn king', 'mervyn allister king', 'lord king', 'baron king', 'lord king of lothbury'],
            'george': ['eddie george', 'edward george', 'edward alan john george', 'steady eddie', 'lord george', 'baron george', 'sir eddie george', 'sir edward george'],
            'leighpemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'leigh-pemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'richardson': ['gordon richardson', 'gordon william humphreys richardson', 'lord richardson', 'baron richardson'],
            'obrien': ["leslie o'brien", "leslie kenneth o'brien", "lord o'brien"],
            "o'brien": ["leslie o'brien", "leslie kenneth o'brien", "lord o'brien"],
            
            # Deputy Governors
            'broadbent': ['ben broadbent'],
            'cunliffe': ['jon cunliffe', 'sir jon cunliffe', 'jonathan cunliffe'],
            'woods': ['sam woods', 'samuel woods'],
            'clementi': ['david clementi', 'sir david clementi'],
            'davies': ['howard davies'],
            'pennant-rea': ['rupert pennant-rea'],
            'blunden': ['george blunden'],
            'mcmahon': ['kit mcmahon', 'christopher mcmahon'],
            
            # Chief Economists
            'haldane': ['andy haldane', 'andrew haldane', 'andrew g haldane'],
            'dale': ['spencer dale'],
            'bean': ['charlie bean', 'charles bean', 'charles goodhart bean'],
            'vickers': ['john vickers', 'sir john vickers'],
            
            # External MPC Members
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
            'bell': ['marian bell'],
            'lambert': ['richard lambert'],
            
            # Executive Directors and Officials
            'tucker': ['paul tucker', 'paul mw tucker'],
            'large': ['andrew large'],
            'fisher': ['paul fisher'],
            'rule': ['david rule'],
            'cleland': ['victoria cleland'],
            'plenderleith': ['ian plenderleith'],
            'clark': ['alastair clark'],
            'quinn': ['brian quinn'],
            'kent': ['pen kent'],
            'cooke': ['william cooke', 'w p cooke'],
        }
        
        # Update with manual patterns
        for pattern, names in manual_patterns.items():
            if pattern not in self.url_name_patterns:
                self.url_name_patterns[pattern] = []
            self.url_name_patterns[pattern].extend(names)
            # Remove duplicates
            self.url_name_patterns[pattern] = list(set(self.url_name_patterns[pattern]))

    def extract_speaker_from_url_enhanced(self, url: str) -> Optional[str]:
        """
        ENHANCED speaker extraction from URL with better pattern matching.
        This addresses the main issue causing validation failures.
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            # Enhanced URL patterns for BoE speeches
            url_patterns = [
                # Pattern: speech-by-firstname-lastname
                r'speech-by-([a-z-]+)-([a-z-]+)',
                # Pattern: lastname-speech or lastname-remarks
                r'([a-z-]+)-(speech|remarks|address)',
                # Pattern: firstname-lastname-speech
                r'([a-z-]+)-([a-z-]+)-(speech|remarks|address)',
                # Pattern: remarks-given-by-firstname-lastname
                r'remarks-given-by-([a-z-]+)-([a-z-]+)',
                # Pattern: /lastname/ in path
                r'/([a-z-]+)/',
                # Pattern: just lastname before file extension
                r'([a-z-]+)\.(pdf|html?) ,
            ]
            
            for pattern in url_patterns:
                match = re.search(pattern, path)
                if match:
                    groups = match.groups()
                    
                    # Extract name parts, excluding keywords
                    name_parts = []
                    for group in groups:
                        if group not in ['speech', 'remarks', 'address', 'pdf', 'html', 'htm']:
                            name_parts.append(group.replace('-', ' '))
                    
                    if name_parts:
                        candidate_name = ' '.join(name_parts).strip()
                        
                        # Direct lookup in our patterns
                        if candidate_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[candidate_name]
                            logger.info(f"URL speaker extraction: '{candidate_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
                        
                        # Try just the last word (surname)
                        last_name = candidate_name.split()[-1] if ' ' in candidate_name else candidate_name
                        if last_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[last_name]
                            logger.info(f"URL speaker extraction (lastname): '{last_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
                        
                        # Try without hyphens/spaces
                        clean_name = candidate_name.replace(' ', '').replace('-', '')
                        if clean_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[clean_name]
                            logger.info(f"URL speaker extraction (clean): '{clean_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
            
            logger.debug(f"No speaker pattern matched for URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting speaker from URL {url}: {e}")
            return None

    def extract_speaker_from_content_enhanced(self, soup: BeautifulSoup, url: str = None) -> Optional[str]:
        """
        ENHANCED speaker extraction from content with multiple strategies.
        This is a key improvement to reduce validation failures.
        """
        # Strategy 1: Try specific speaker selectors
        speaker_selectors = [
            '.speech-author',
            '.article-author', 
            '.byline',
            '.speaker-name',
            '.author-name',
            '.speech-by',
            '[class*="author"]',
            '[class*="speaker"]',
            'h1 + p',  # Often speaker info is in paragraph after title
            '.meta-author',
            '.speech-meta'
        ]
        
        for selector in speaker_selectors:
            elements = soup.select(selector)
            for element in elements:
                speaker_text = element.get_text(strip=True)
                name = self._clean_and_validate_speaker_name(speaker_text)
                if name and name != 'Unknown':
                    logger.debug(f"Speaker found via CSS selector '{selector}': {name}")
                    return name
        
        # Strategy 2: Search in structured content with enhanced patterns
        content_areas = [
            soup.find('main'),
            soup.find('article'), 
            soup.find('div', class_='content'),
            soup.find('div', class_='speech'),
            soup
        ]
        
        for area in content_areas:
            if not area:
                continue
                
            text = area.get_text()
            
            # Enhanced speaker patterns for BoE content
            patterns = [
                # Standard titles with names
                r'Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Deputy Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Chair\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Chief Economist\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                
                # "By" patterns
                r'(?:^|\n)\s*By\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'(?:Remarks|Speech|Address)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Given by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                
                # Name followed by title
                r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair|Chief Economist)',
                
                # Lord/Sir titles
                r'(?:Lord|Sir)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                
                # Pattern: Name at start of speech
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*\n',
                
                # MPC member pattern
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*),?\s+(?:External )?MPC [Mm]ember',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE)
                if matches:
                    for match in matches:
                        name = self._clean_and_validate_speaker_name(match)
                        if name and name != 'Unknown':
                            logger.debug(f"Speaker found via content pattern: {name}")
                            return name
        
        # Strategy 3: URL fallback
        if url:
            url_name = self.extract_speaker_from_url_enhanced(url)
            if url_name:
                return url_name
        
        # Strategy 4: Title extraction fallback
        title_text = soup.find('title')
        if title_text:
            title = title_text.get_text()
            # Look for "speech by NAME" in title
            title_pattern = r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)'
            match = re.search(title_pattern, title, re.IGNORECASE)
            if match:
                name = self._clean_and_validate_speaker_name(match.group(1))
                if name and name != 'Unknown':
                    logger.debug(f"Speaker found in title: {name}")
                    return name
        
        logger.debug("No speaker found with any extraction method")
        return None

    def _clean_and_validate_speaker_name(self, raw_name: str) -> str:
        """
        Enhanced speaker name cleaning and validation.
        This is crucial for reducing validation failures.
        """
        if not raw_name:
            return 'Unknown'
        
        # Remove newlines and normalize whitespace
        raw_name = ' '.join(raw_name.split())
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            r'\b(?:The\s+)?(?:Rt\s+)?(?:Hon\s+)?(?:Sir|Lord|Baron|Dame|Dr|Mr|Ms|Mrs)\s+',
            r'\b(?:Governor|Deputy Governor|Chair|President|Chief Economist)\s+',
            r'\b(?:External\s+)?MPC\s+Member\s+',
        ]
        
        for prefix in prefixes_to_remove:
            raw_name = re.sub(prefix, '', raw_name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        raw_name = re.split(r'\s*(?:,|by|remarks|speech|address|gave|given)\s*', raw_name, flags=re.IGNORECASE)[0]
        
        # Clean and validate
        name = ' '.join(raw_name.split()).strip()
        
        # Remove periods and normalize
        name = name.replace('.', '').strip()
        
        # Validate: must be reasonable length and format
        if len(name) < 2 or len(name) > 50:
            return 'Unknown'
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', name):
            return 'Unknown'
        
        # Check against known problematic patterns
        problematic_patterns = [
            r'^\d+ ,  # Just numbers
            r'^[^\w\s]+ ,  # Just punctuation
            r'unknown|speaker|author|governor|deputy|chair|president',  # Generic terms
        ]
        
        name_lower = name.lower()
        for pattern in problematic_patterns:
            if re.match(pattern, name_lower):
                return 'Unknown'
        
        return name if name else 'Unknown'

    def get_speaker_info_enhanced(self, speaker_name: str, url: str = None, content: str = None) -> Dict[str, str]:
        """
        Enhanced speaker information lookup with multiple fallback strategies.
        """
        # Strategy 1: Try extraction from content first if available
        if content and (not speaker_name or speaker_name == 'Unknown'):
            soup_content = BeautifulSoup(content, 'html.parser')
            extracted_name = self.extract_speaker_from_content_enhanced(soup_content, url)
            if extracted_name and extracted_name != 'Unknown':
                speaker_name = extracted_name
                self.stats['content_fallback_used'] += 1
        
        # Strategy 2: Try URL extraction if still unknown
        if (not speaker_name or speaker_name == 'Unknown') and url:
            url_extracted_name = self.extract_speaker_from_url_enhanced(url)
            if url_extracted_name:
                speaker_name = url_extracted_name
        
        if not speaker_name or speaker_name.strip() == 'Unknown':
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
            if self._names_match_enhanced(normalized_name, known_name):
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
        
        # Method 4: Try fuzzy matching on first/last name combinations
        name_parts = normalized_name.split()
        if len(name_parts) >= 2:
            first_last = f"{name_parts[0]} {name_parts[-1]}"
            if first_last in self.speaker_roles:
                result = self.speaker_roles[first_last].copy()
                result['source'] = 'first_last_match'
                logger.debug(f"First-last match found: {speaker_name} -> {first_last}")
                return result
        
        # Final fallback: Unknown speaker
        logger.warning(f"Unknown speaker after all methods: {speaker_name} (normalized: {normalized_name})")
        self.stats['unknown_speakers'] += 1
        return {"role": "Unknown", "voting_status": "Unknown", "source": "unknown"}

    def _names_match_enhanced(self, name1: str, name2: str) -> bool:
        """Enhanced name matching with better fuzzy logic."""
        if not name1 or not name2:
            return False
            
        parts1 = set(name1.replace('.', '').replace('-', ' ').split())
        parts2 = set(name2.replace('.', '').replace('-', ' ').split())
        
        # Remove common middle initials and abbreviations
        parts1 = {p for p in parts1 if len(p) > 1}
        parts2 = {p for p in parts2 if len(p) > 1}
        
        common_parts = parts1.intersection(parts2)
        
        # For short names (2 parts or less), require full overlap
        if len(parts1) <= 2 and len(parts2) <= 2:
            return len(common_parts) >= min(len(parts1), len(parts2))
        else:
            # For longer names, require at least 2 matching parts
            return len(common_parts) >= 2

    def _clean_speaker_name_for_lookup(self, name: str) -> str:
        """Enhanced speaker name cleaning for database lookup."""
        if not name:
            return ""
        
        # Remove titles and clean more thoroughly
        name = re.sub(r'\b(?:The\s+)?(?:Rt\s+)?(?:Hon\s+)?(?:Governor|Deputy Governor|Chair|President|Dr\.|Mr\.|Ms\.|Mrs\.|Sir|Lord|Baron|Dame)\s*', '', name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        name = re.split(r'\s*(?:,|by|remarks|speech|address|gave|given)\s*', name, flags=re.IGNORECASE)[0]
        
        # Remove newlines and extra whitespace
        name = ' '.join(name.split())
        
        # Convert to lowercase and remove periods
        name = name.lower().strip().replace('.', '')
        
        return name

    def extract_date_from_url_enhanced(self, url: str) -> Optional[str]:
        """
        Enhanced date extraction with more patterns and better validation.
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            # Enhanced BoE URL date patterns
            date_patterns = [
                # Current BoE pattern: /speech/2024/october/title-slug
                r'/speech/(\d{4})/(\w+)/',
                # Pattern: /speech/2024/10/title-slug  
                r'/speech/(\d{4})/(\d{1,2})/',
                # Legacy patterns
                r'/speeches/(\d{4})/(\w+)/',
                r'/speeches/(\d{4})/(\d{1,2})/',
                # Media files pattern: /files/speech/2024/july/
                r'/files/speech/(\d{4})/(\w+)/',
                # Date in filename: speech-2024-10-15
                r'speech-(\d{4})-(\d{2})-(\d{2})',
                # Pattern: embedded YYYYMMDD
                r'(\d{8})',
                # Year-only pattern for historical speeches
                r'/(\d{4})/',
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, path)
                if match:
                    groups = match.groups()
                    
                    if len(groups) == 1:
                        # Could be YYYYMMDD or just year
                        date_str = groups[0]
                        if len(date_str) == 8:  # YYYYMMDD
                            try:
                                year = int(date_str[:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                if 1960 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                    date_obj = datetime(year, month, day)
                                    return date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                continue
                        elif len(date_str) == 4:  # Just year
                            try:
                                year = int(date_str)
                                if 1960 <= year <= 2030:
                                    # Use January 1st as default
                                    return f"{year}-01-01"
                            except ValueError:
                                continue
                    
                    elif len(groups) == 2:
                        year_str, month_str = groups
                        try:
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
                            if 1960 <= year <= 2030 and 1 <= month <= 12:
                                date_obj = datetime(year, month, 1)
                                formatted_date = date_obj.strftime('%Y-%m-%d')
                                logger.debug(f"URL date extraction successful: {url} -> {formatted_date}")
                                return formatted_date
                        except ValueError:
                            continue
                            
                    elif len(groups) == 3:
                        year_str, month_str, day_str = groups
                        try:
                            year = int(year_str)
                            month = int(month_str)
                            day = int(day_str)
                            
                            if 1960 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                date_obj = datetime(year, month, day)
                                formatted_date = date_obj.strftime('%Y-%m-%d')
                                logger.debug(f"URL date extraction successful: {url} -> {formatted_date}")
                                return formatted_date
                        except ValueError:
                            continue
            
            logger.debug(f"No valid date pattern found in URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting date from URL {url}: {e}")
            return None

    # NEW METHODS FOR HISTORICAL COVERAGE

    def scrape_quarterly_bulletin_speeches(self, start_year: int = 1960, end_year: int = 2006) -> List[Dict]:
        """
        Scrape historical speeches from Bank of England Quarterly Bulletins (1960-2006).
        This provides access to speeches from the pre-digital era.
        """
        logger.info(f"Scraping BoE Quarterly Bulletin speeches ({start_year}-{end_year})")
        
        all_speeches = []
        
        try:
            response = requests.get(self.quarterly_bulletin_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all speech references in the quarterly bulletin listings
            # Pattern: "speech by [Name]" or "Governor's speech"
            speech_patterns = [
                r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*[-–]\s*speech',
                r"Governor's speech",
                r'remarks by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                r'lecture by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            ]
            
            # Look for speech references in the bulletin content
            text = soup.get_text()
            lines = text.split('\n')
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['speech', 'remarks', 'lecture', 'address']):
                    # Extract year from line if present
                    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', line)
                    if year_match:
                        year = int(year_match.group(1))
                        if start_year <= year <= end_year:
                            
                            # Try to extract speaker name
                            speaker = None
                            for pattern in speech_patterns:
                                match = re.search(pattern, line, re.IGNORECASE)
                                if match and match.groups():
                                    speaker = match.group(1)
                                    break
                            
                            if not speaker and "governor's speech" in line.lower():
                                # Determine governor based on year
                                if 1966 <= year <= 1973:
                                    speaker = "leslie o'brien"
                                elif 1973 <= year <= 1983:
                                    speaker = "gordon richardson"
                                elif 1983 <= year <= 1993:
                                    speaker = "robin leigh-pemberton"
                                elif 1993 <= year <= 2003:
                                    speaker = "eddie george"
                                elif 2003 <= year <= 2006:
                                    speaker = "mervyn king"
                            
                            if speaker:
                                speech_info = {
                                    'title': line.strip(),
                                    'speaker_raw': speaker,
                                    'date': f"{year}-01-01",  # Approximate date
                                    'date_source': 'quarterly_bulletin',
                                    'source_url': f"{self.quarterly_bulletin_url}#{year}",
                                    'context_text': line.strip(),
                                    'source_type': 'Quarterly Bulletin'
                                }
                                all_speeches.append(speech_info)
                                self.stats['quarterly_bulletin_speeches'] += 1
            
            logger.info(f"Found {len(all_speeches)} speeches in Quarterly Bulletins")
            
        except Exception as e:
            logger.error(f"Error scraping Quarterly Bulletins: {e}")
        
        return all_speeches

    def scrape_digital_archive_speeches(self, start_year: int = 1990, end_year: int = 2020) -> List[Dict]:
        """
        Scrape speeches from the BoE Digital Archive.
        This covers the gap between Quarterly Bulletins and modern website.
        """
        logger.info(f"Scraping BoE Digital Archive speeches ({start_year}-{end_year})")
        
        all_speeches = []
        
        # The digital archive is organized by year folders
        for year in range(start_year, end_year + 1):
            try:
                # Try to access the speeches folder for this year
                archive_url = f"{self.digital_archive_url}/?name=SPEECHES_{year}"
                
                response = requests.get(archive_url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for speech links in the archive
                    speech_links = soup.find_all('a', href=True)
                    
                    for link in speech_links:
                        href = link.get('href')
                        text = link.get_text(strip=True)
                        
                        # Check if this looks like a speech
                        if (href and text and 
                            any(keyword in text.lower() for keyword in ['speech', 'remarks', 'address', 'lecture']) and
                            len(text) > 10):
                            
                            full_url = urljoin(self.digital_archive_url, href)
                            
                            # Extract speaker from title/context
                            speaker = self._extract_speaker_from_title(text)
                            
                            speech_info = {
                                'title': text,
                                'speaker_raw': speaker or '',
                                'date': f"{year}-01-01",  # Approximate date
                                'date_source': 'digital_archive',
                                'source_url': full_url,
                                'context_text': text,
                                'source_type': 'Digital Archive'
                            }
                            all_speeches.append(speech_info)
                            self.stats['digital_archive_speeches'] += 1
                
                time.sleep(1)  # Be respectful to the archive
                
            except Exception as e:
                logger.debug(f"Could not access digital archive for {year}: {e}")
                continue
        
        logger.info(f"Found {len(all_speeches)} speeches in Digital Archive")
        return all_speeches

    def _extract_speaker_from_title(self, title: str) -> Optional[str]:
        """Extract speaker name from speech title."""
        if not title:
            return None
        
        # Common patterns in speech titles
        patterns = [
            r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'remarks by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'address by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'lecture by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*[-–]\s*speech',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*:',  # Name at start followed by colon
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                name = self._clean_and_validate_speaker_name(match.group(1))
                if name and name != 'Unknown':
                    return name
        
        return None

    # ENHANCED CONTENT EXTRACTION

    def _extract_main_content_enhanced_v2(self, soup: BeautifulSoup, url: str = None) -> str:
        """
        Enhanced content extraction with multiple fallback strategies.
        This addresses the core issue of short/empty content extraction.
        """
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.navigation', 'noscript', 'aside']):
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
            '.speech-text',
            '.text-content',
            '.body-content',
            '#content',
            '.entry-content',
            '.post-content'
        ]
        
        for selector in boe_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
                if len(text) > 300:  # Must be substantial
                    content_candidates.append(('boe_selector', text, len(text), selector))
                    logger.debug(f"Found content using selector {selector}: {len(text)} chars")
        
        # Strategy 2: Try to find the largest meaningful content block
        all_divs = soup.find_all(['div', 'section', 'article'])
        for div in all_divs:
            text = div.get_text(separator='\n', strip=True)
            if len(text) > 800:  # Must be very substantial for this method
                # Check that it's not just navigation or boilerplate
                skip_indicators = [
                    'navigation', 'skip to', 'breadcrumb', 'footer', 'sidebar', 
                    'menu', 'search', 'cookie', 'privacy', 'terms'
                ]
                if not any(skip_text in text.lower() for skip_text in skip_indicators):
                    content_candidates.append(('largest_div', text, len(text), 'div_search'))
        
        # Strategy 3: Paragraph aggregation with content filtering
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Filter out navigation and short paragraphs
            meaningful_paras = []
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if (len(p_text) > 20 and 
                    not any(skip in p_text.lower() for skip in ['cookie', 'javascript', 'skip to', 'navigation'])):
                    meaningful_paras.append(p_text)
            
            if meaningful_paras:
                para_text = '\n\n'.join(meaningful_paras)
                if len(para_text) > 500:
                    content_candidates.append(('paragraphs', para_text, len(para_text), 'paragraph_agg'))
        
        # Strategy 4: Look for content in specific BoE content patterns
        content_patterns = [
            # Look for div containing multiple paragraphs (speech content)
            lambda: soup.find('div', string=re.compile(r'speech|remarks|address', re.I)),
            # Look for container with substantial text
            lambda: soup.find('div', attrs={'class': re.compile(r'content|speech|text|body', re.I)}),
            # Look for main content area
            lambda: soup.find(attrs={'id': re.compile(r'main|content|speech', re.I)}),
        ]
        
        for pattern_func in content_patterns:
            try:
                element = pattern_func()
                if element:
                    # Get parent or the element itself
                    content_area = element.parent if element.parent else element
                    text = content_area.get_text(separator='\n', strip=True)
                    if len(text) > 400:
                        content_candidates.append(('pattern_match', text, len(text), 'content_pattern'))
            except:
                continue
        
        # Strategy 5: Body content (last resort) with better filtering
        body = soup.find('body')
        if body:
            body_text = body.get_text(separator='\n', strip=True)
            if len(body_text) > 1000:
                # Try to remove header/footer/navigation from body text
                lines = body_text.split('\n')
                filtered_lines = []
                for line in lines:
                    line = line.strip()
                    if (len(line) > 10 and 
                        not any(skip in line.lower() for skip in [
                            'bank of england', 'speeches', 'navigation', 'search', 
                            'menu', 'home', 'about', 'contact', 'privacy', 'cookies'
                        ])):
                        filtered_lines.append(line)
                
                filtered_text = '\n'.join(filtered_lines)
                if len(filtered_text) > 600:
                    content_candidates.append(('body_filtered', filtered_text, len(filtered_text), 'body'))
        
        # Choose the best candidate based on length and strategy priority
        if content_candidates:
            # Sort by strategy priority and length
            strategy_priority = {
                'boe_selector': 4,
                'pattern_match': 3,
                'largest_div': 2,
                'paragraphs': 1,
                'body_filtered': 0
            }
            
            content_candidates.sort(key=lambda x: (strategy_priority.get(x[0], 0), x[2]), reverse=True)
            best_strategy, best_content, best_length, selector = content_candidates[0]
            
            logger.info(f"Content extraction strategy: {best_strategy} via {selector} ({best_length} chars)")
            
            # Additional validation: ensure content looks like a speech
            if self._validate_speech_content(best_content):
                cleaned_content = self._clean_text_content_enhanced(best_content)
                logger.info(f"After cleaning: {len(cleaned_content)} chars")
                return cleaned_content
            else:
                logger.warning(f"Content failed speech validation, trying next candidate")
                # Try next best candidate
                if len(content_candidates) > 1:
                    second_strategy, second_content, second_length, second_selector = content_candidates[1]
                    if self._validate_speech_content(second_content):
                        cleaned_content = self._clean_text_content_enhanced(second_content)
                        logger.info(f"Using second candidate: {second_strategy} via {second_selector} ({len(cleaned_content)} chars)")
                        return cleaned_content
        
        logger.warning("No substantial valid content found with any extraction strategy")
        return ""

    def _validate_speech_content(self, content: str) -> bool:
        """Validate that content looks like an actual speech."""
        if not content or len(content) < 200:
            return False
        
        # Check for speech indicators
        speech_indicators = [
            'thank', 'pleased', 'good morning', 'good afternoon', 'good evening',
            'ladies and gentlemen', 'chair', 'chairman', 'colleagues',
            'today', 'economic', 'policy', 'bank', 'financial', 'market'
        ]
        
        content_lower = content.lower()
        indicator_count = sum(1 for indicator in speech_indicators if indicator in content_lower)
        
        # Must have at least 3 speech indicators
        if indicator_count < 3:
            return False
        
        # Check it's not just boilerplate
        boilerplate_indicators = [
            'cookies', 'javascript', 'browser', 'website', 'homepage',
            'navigation', 'search results', 'no results found'
        ]
        
        boilerplate_count = sum(1 for indicator in boilerplate_indicators if indicator in content_lower)
        
        # Reject if too much boilerplate
        if boilerplate_count > 2:
            return False
        
        return True

    def _clean_text_content_enhanced(self, text: str) -> str:
        """Enhanced text cleaning with better preservation of speech content."""
        if not text:
            return ""
        
        original_length = len(text)
        
        # Split into lines for better processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and very short lines that are likely navigation
            if len(line) < 3:
                continue
            
            # Skip lines that are clearly navigation/boilerplate
            skip_patterns = [
                r'^(Home|About|Contact|Search|Menu|Navigation) ,
                r'^(Print this page|Share this page|Last update).*',
                r'^(Skip to|Return to|Back to).*',
                r'^(Copyright|Terms|Privacy|Cookie).*',
                r'^\s*\d+\s* ,  # Just numbers
                r'^[^\w]* ,     # Just punctuation
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            cleaned_lines.append(line)
        
        # Rejoin with proper spacing
        text = '\n'.join(cleaned_lines)
        
        # Basic cleanup
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        final_length = len(text)
        
        # Log suspicious cleaning only if major content loss
        if original_length > 2000 and final_length < original_length * 0.3:
            logger.warning(f"Significant content reduction during cleaning: {original_length} -> {final_length} chars")
        
        return text

    # ENHANCED VALIDATION SYSTEM

    def _validate_speech_data_enhanced(self, speech_data: Dict) -> bool:
        """
        Enhanced validation to prevent saving invalid speeches.
        Addresses the main cause of validation failures.
        """
        if not speech_data or 'metadata' not in speech_data or 'content' not in speech_data:
            logger.warning("Speech data missing required components")
            self.stats['validation_failures'] += 1
            return False
        
        metadata = speech_data['metadata']
        content = speech_data['content']
        
        # Enhanced content validation
        if not content or len(content.strip()) < 100:  # Reduced threshold but still meaningful
            logger.warning(f"Content too short: {len(content.strip()) if content else 0} chars")
            self.stats['content_too_short'] += 1
            return False
        
        # Check for placeholder/error content
        placeholder_indicators = [
            'lorem ipsum', 'placeholder', 'test content', 'coming soon',
            'under construction', 'page not found', 'error 404', '404 not found',
            'no content available', 'content not available'
        ]
        
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in placeholder_indicators):
            logger.warning("Content appears to be placeholder or error text")
            self.stats['validation_failures'] += 1
            return False
        
        # Enhanced metadata validation
        required_fields = ['title', 'speaker', 'date', 'source_url']
        for field in required_fields:
            if not metadata.get(field):
                logger.warning(f"Missing required metadata field: {field}")
                self.stats['validation_failures'] += 1
                return False
        
        # Enhanced date validation
        date_str = metadata.get('date')
        if not date_str:
            logger.warning("No date provided")
            self.stats['validation_failures'] += 1
            return False
        
        # Allow reasonable default dates for historical speeches
        if date_str.endswith('-01-01'):
            # This is okay for historical speeches where we only have year
            pass
        elif date_str == '2025-01-01':
            # This suggests a parsing failure
            logger.warning(f"Invalid default date: {date_str}")
            self.stats['validation_failures'] += 1
            return False
        
        # Enhanced speaker validation with more permissive rules
        speaker = metadata.get('speaker', '').lower()
        if speaker in ['unknown', 'unknown speaker', '']:
            # For historical speeches, we might not always know the speaker
            # Allow if content is substantial and looks like a speech
            if len(content) > 1000 and self._validate_speech_content(content):
                logger.info("Allowing speech with unknown speaker due to substantial content")
            else:
                logger.warning(f"Unknown speaker with insufficient content: {metadata.get('speaker')}")
                self.stats['validation_failures'] += 1
                return False
        
        # Enhanced title validation
        title = metadata.get('title', '')
        if len(title) < 5:  # Reduced threshold
            logger.warning(f"Title too short: {title}")
            self.stats['validation_failures'] += 1
            return False
        
        # Check title isn't just generic
        generic_titles = ['untitled speech', 'untitled', 'speech', 'remarks', 'address']
        if title.lower() in generic_titles:
            logger.warning(f"Generic title: {title}")
            self.stats['validation_failures'] += 1
            return False
        
        return True

    # COMPREHENSIVE SCRAPING WITH HISTORICAL COVERAGE

    def run_comprehensive_scraping_v2(self, method: str = "all", start_year: int = 1960, 
                                    max_speeches: Optional[int] = None, 
                                    include_historical: bool = True) -> Dict[str, int]:
        """
        Enhanced comprehensive BoE speech scraping with historical coverage.
        """
        logger.info(f"Starting enhanced BoE speech scraping v2")
        logger.info(f"Method: {method}, Start year: {start_year}, Max speeches: {max_speeches}")
        logger.info(f"Include historical: {include_historical}")
        
        # Reset statistics
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'url_fallback_used': 0,
            'content_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0,
            'historical_speeches_found': 0,
            'quarterly_bulletin_speeches': 0,
            'digital_archive_speeches': 0
        }
        
        all_speeches = []
        
        # Phase 1: Historical speeches (if requested)
        if include_historical and start_year < 1997:
            logger.info("=== PHASE 1: HISTORICAL SPEECH COLLECTION ===")
            
            # Quarterly Bulletin speeches (1960-2006)
            if start_year <= 2006:
                logger.info("Collecting speeches from Quarterly Bulletins...")
                qb_speeches = self.scrape_quarterly_bulletin_speeches(start_year, min(2006, 2025))
                all_speeches.extend(qb_speeches)
                self.stats['historical_speeches_found'] += len(qb_speeches)
            
            # Digital Archive speeches (1990-2020)
            if start_year <= 2020:
                logger.info("Collecting speeches from Digital Archive...")
                da_speeches = self.scrape_digital_archive_speeches(max(1990, start_year), min(2020, 2025))
                all_speeches.extend(da_speeches)
                self.stats['historical_speeches_found'] += len(da_speeches)
        
        # Phase 2: Modern website scraping
        logger.info("=== PHASE 2: MODERN WEBSITE SCRAPING ===")
        
        # Approach 1: Sitemap scraping (most reliable)
        if method in ["sitemap", "all"]:
            logger.info("Running sitemap scraping...")
            sitemap_speeches = self.scrape_speeches_from_sitemap(max_speeches or 100)
            all_speeches.extend(sitemap_speeches)
            logger.info(f"Sitemap method found {len(sitemap_speeches)} speeches")
        
        # Approach 2: Main speeches page scraping
        if method in ["main", "all"] and len([s for s in all_speeches if s.get('date', '').startswith('202')]) < 10:
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
        unique_speeches = self._deduplicate_speeches_enhanced(all_speeches)
        logger.info(f"Total unique speeches found: {len(unique_speeches)}")
        
        if not unique_speeches:
            logger.warning("No speeches found!")
            return self.stats
        
        # Filter by year if requested
        if start_year:
            filtered_speeches = []
            for speech in unique_speeches:
                speech_date = speech.get('date', '')
                if speech_date:
                    try:
                        speech_year = int(speech_date[:4])
                        if speech_year >= start_year:
                            filtered_speeches.append(speech)
                    except (ValueError, IndexError):
                        # Include if we can't parse the date
                        filtered_speeches.append(speech)
                else:
                    filtered_speeches.append(speech)
            
            unique_speeches = filtered_speeches
            logger.info(f"After year filtering ({start_year}+): {len(unique_speeches)} speeches")
        
        # Limit speeches if requested
        if max_speeches:
            unique_speeches = unique_speeches[:max_speeches]
            logger.info(f"Limited to {max_speeches} speeches")
        
        # Phase 3: Process each speech
        logger.info(f"=== PHASE 3: PROCESSING {len(unique_speeches)} SPEECHES ===")
        
        for i, speech_info in enumerate(unique_speeches, 1):
            logger.info(f"Processing speech {i}/{len(unique_speeches)}: {speech_info['source_url']}")
            
            try:
                # Extract content and metadata with enhanced methods
                speech_data = self.scrape_speech_content_enhanced(speech_info)
                
                if speech_data:
                    # Save speech (already validated in scrape_speech_content_enhanced)
                    saved_filename = self.save_speech_enhanced(speech_data)
                    if saved_filename:
                        self.stats['saved_speeches'] += 1
                        
                        # Log speaker recognition details
                        metadata = speech_data['metadata']
                        logger.info(f"✓ Successfully saved: {saved_filename}")
                        logger.info(f"  Speaker: {metadata['speaker']} ({metadata.get('recognition_source', 'unknown')})")
                        logger.info(f"  Role: {metadata['role']}")
                        logger.info(f"  Date: {metadata['date']} ({metadata.get('date_source', 'unknown')})")
                    else:
                        logger.error(f"✗ Failed to save speech from {speech_info['source_url']}")
                else:
                    logger.error(f"✗ Failed to extract or validate content from {speech_info['source_url']}")
                
            except Exception as e:
                logger.error(f"✗ Unexpected error processing {speech_info['source_url']}: {e}")
            
            # Respectful delay
            time.sleep(0.5)
        
        # Final statistics
        self._log_final_statistics()
        
        return self.stats

    def scrape_speech_content_enhanced(self, speech_info: Dict) -> Optional[Dict]:
        """Enhanced speech content scraping with better validation and fallbacks."""
        url = speech_info['source_url']
        logger.debug(f"Scraping content from: {url}")
        
        self.stats['total_processed'] += 1
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                result = self._extract_pdf_content_enhanced(response.content, speech_info)
            else:
                result = self._extract_html_content_enhanced_v2(response.text, speech_info, url)
            
            # Enhanced validation
            if result and self._validate_speech_data_enhanced(result):
                self.stats['successful_extractions'] += 1
                return result
            else:
                logger.warning(f"Speech failed enhanced validation: {url}")
                self.stats['content_extraction_failures'] += 1
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            self.stats['content_extraction_failures'] += 1
            return None

    def _extract_html_content_enhanced_v2(self, html_content: str, speech_info: Dict, url: str) -> Optional[Dict]:
        """Enhanced HTML content extraction with better speaker recognition."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Enhanced content extraction
            content = self._extract_main_content_enhanced_v2(soup, url)
            
            # Early exit if no substantial content
            if not content or len(content.strip()) < 100:
                logger.warning(f"Insufficient content extracted from {url}: {len(content.strip()) if content else 0} chars")
                return None
            
            # Enhanced component extraction
            title = self._extract_title_enhanced(soup, speech_info.get('title', ''))
            
            # Enhanced speaker extraction with content awareness
            speaker_name = self.extract_speaker_from_content_enhanced(soup, url)
            if not speaker_name or speaker_name == 'Unknown':
                speaker_name = speech_info.get('speaker_raw', '')
            
            # Enhanced date extraction
            date = self.extract_date_from_url_enhanced(url)
            if not date:
                date = speech_info.get('date', '')
            
            location = self._extract_location_enhanced(soup)
            
            # Get enhanced speaker information
            role_info = self.get_speaker_info_enhanced(speaker_name, url, content)
            
            # Build comprehensive metadata
            metadata = {
                'title': title,
                'speaker': role_info.get('matched_name', speaker_name) if role_info.get('source') != 'unknown' else speaker_name,
                'role': role_info.get('role', 'Unknown'),
                'institution': 'Bank of England',
                'country': 'UK',
                'date': date,
                'location': location,
                'language': 'en',
                'source_url': url,
                'source_type': speech_info.get('source_type', 'HTML'),
                'voting_status': role_info.get('voting_status', 'Unknown'),
                'recognition_source': role_info.get('source', 'unknown'),
                'date_source': speech_info.get('date_source', 'url'),
                'tags': self._extract_content_tags_enhanced(content),
                'scrape_timestamp': datetime.now().isoformat(),
                'content_length': len(content)
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

    def _extract_content_tags_enhanced(self, content: str) -> List[str]:
        """Enhanced content tag extraction with more comprehensive keywords."""
        tags = []
        content_lower = content.lower()
        
        # Enhanced keyword mapping
        keywords = {
            'inflation': ['inflation', 'price stability', 'cpi', 'rpi', 'deflation', 'disinflation', 'price level', 'core inflation', 'headline inflation'],
            'interest_rates': ['interest rate', 'bank rate', 'monetary policy', 'policy rate', 'rate rise', 'rate cut', 'rate increase', 'rate decrease', 'base rate', 'official rate'],
            'employment': ['employment', 'unemployment', 'labour market', 'labor market', 'jobs', 'jobless', 'payroll', 'employment data', 'job growth', 'labour force', 'wage'],
            'financial_stability': ['financial stability', 'banking', 'supervision', 'regulation', 'systemic risk', 'stress test', 'capital requirements', 'prudential', 'financial system'],
            'economic_outlook': ['economic outlook', 'forecast', 'projection', 'growth', 'recession', 'expansion', 'economic conditions', 'gdp', 'economic recovery'],
            'monetary_policy': ['monetary policy', 'mpc', 'monetary policy committee', 'quantitative easing', 'qe', 'gilt purchases', 'asset purchases', 'forward guidance'],
            'banking': ['bank', 'banking', 'credit', 'lending', 'deposits', 'financial institutions', 'commercial banks', 'banking sector'],
            'markets': ['market', 'financial markets', 'capital markets', 'bond market', 'stock market', 'equity markets', 'gilt', 'currency'],
            'crisis': ['crisis', 'pandemic', 'covid', 'financial crisis', 'economic crisis', 'emergency', 'coronavirus', '2008 crisis'],
            'brexit': ['brexit', 'european union', 'eu', 'single market', 'customs union', 'trade deal', 'referendum'],
            'international': ['international', 'global', 'foreign', 'trade', 'exchange rate', 'emerging markets', 'global economy', 'international cooperation'],
            'technology': ['technology', 'digital', 'fintech', 'innovation', 'artificial intelligence', 'ai', 'blockchain', 'cryptocurrency'],
            'climate': ['climate', 'environmental', 'green', 'sustainability', 'carbon', 'net zero', 'climate change']
        }
        
        for tag, terms in keywords.items():
            if any(term in content_lower for term in terms):
                tags.append(tag)
        
        return tags

    def _deduplicate_speeches_enhanced(self, speeches: List[Dict]) -> List[Dict]:
        """Enhanced deduplication with better matching."""
        unique_speeches = []
        seen_urls = set()
        seen_combinations = set()
        
        for speech in speeches:
            url = speech.get('source_url', '')
            title = speech.get('title', '').lower().strip()
            date = speech.get('date', '')
            
            # Primary deduplication by URL
            if url and url not in seen_urls:
                # Secondary deduplication by title+date combination
                combination = f"{title}_{date}"
                if combination not in seen_combinations:
                    unique_speeches.append(speech)
                    seen_urls.add(url)
                    seen_combinations.add(combination)
        
        return unique_speeches

    def save_speech_enhanced(self, speech_data: Dict) -> Optional[str]:
        """Enhanced speech saving with better error handling."""
        try:
            metadata = speech_data['metadata']
            content = speech_data['content']
            
            # Generate content hash for uniqueness
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:10]
            
            # Enhanced speaker name sanitization
            speaker_name = metadata.get('speaker', 'unknown')
            if speaker_name and speaker_name != 'Unknown':
                clean_speaker = re.sub(r'[^\w\s-]', '', speaker_name.lower())
                clean_speaker = re.sub(r'\s+', '-', clean_speaker)
                clean_speaker = clean_speaker.strip('-')[:20]  # Limit length
                if not clean_speaker:
                    clean_speaker = 'unknown-speaker'
            else:
                clean_speaker = 'unknown-speaker'
            
            # Use the date from metadata
            date_str = metadata.get('date', 'unknown-date')
            if date_str and date_str != 'unknown-date':
                date_part = date_str[:10]  # YYYY-MM-DD
            else:
                date_part = 'unknown-date'
            
            base_filename = f"{date_part}_{clean_speaker}-{content_hash}"
            
            # Final filename sanitization
            base_filename = re.sub(r'[^\w\-.]', '', base_filename)[:100]  # Limit total length
            
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
            
            logger.debug(f"Saved speech: {base_filename}")
            return base_filename
            
        except Exception as e:
            logger.error(f"Error saving speech: {e}")
            return None

    def _log_final_statistics(self):
        """Log comprehensive final statistics."""
        logger.info("=== ENHANCED BoE SCRAPING COMPLETE ===")
        logger.info(f"Total speeches processed: {self.stats['total_processed']}")
        logger.info(f"Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"Saved speeches: {self.stats['saved_speeches']}")
        logger.info(f"Historical speeches found: {self.stats['historical_speeches_found']}")
        logger.info(f"  - Quarterly Bulletin: {self.stats['quarterly_bulletin_speeches']}")
        logger.info(f"  - Digital Archive: {self.stats['digital_archive_speeches']}")
        logger.info(f"URL fallback used: {self.stats['url_fallback_used']}")
        logger.info(f"Content fallback used: {self.stats['content_fallback_used']}")
        logger.info(f"Unknown speakers: {self.stats['unknown_speakers']}")
        logger.info(f"Date extraction failures: {self.stats['date , '', title)
            if len(title) > 10:
                return title
        
        # Clean up fallback title
        if fallback_title and len(fallback_title) > 10:
            return fallback_title
        
        return 'Untitled Speech'

    def _extract_location_enhanced(self, soup: BeautifulSoup) -> str:
        """Enhanced location extraction with more patterns."""
        location_selectors = [
            '.speech-location',
            '.event-location',
            '.venue',
            '.location',
            '[class*="location"]',
            '[class*="venue"]',
            '.event-details',
            '.speech-meta .location'
        ]
        
        for selector in location_selectors:
            element = soup.select_one(selector)
            if element:
                location = element.get_text(strip=True)
                if location and 5 < len(location) < 100:
                    return location
        
        # Enhanced search in content for location patterns
        content_areas = [soup.find('main'), soup.find('article'), soup]
        
        for area in content_areas:
            if not area:
                continue
                
            text = area.get_text()
            patterns = [
                r'(?:delivered at|speaking at|remarks at|given at|held at)\s+([A-Z][a-zA-Z\s,&]+(?:University|College|Institute|Center|Centre|Hotel|Club|Conference|School|Hall|House))',
                r'(?:at|in)\s+([A-Z][a-zA-Z\s,&]+(?:University|College|Institute|Center|Centre|Hotel|Club|Conference|School))',
                r'(?:London|Birmingham|Manchester|Edinburgh|Cardiff|Belfast|Liverpool|Bristol),?\s*([A-Z]{2}|\w+)',
                r'([A-Z][a-zA-Z\s]+),\s+(?:London|UK|England)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    location = matches[0] if isinstance(matches[0], str) else ', '.join(matches[0])
                    if len(location) < 100:
                        return location
        
        return 'London, UK'

    def print_speaker_database_info_enhanced(self):
        """Enhanced speaker database information display."""
        logger.info("=== ENHANCED BoE SPEAKER DATABASE INFO ===")
        logger.info(f"Total speaker entries: {len(self.speaker_roles)}")
        
        # Count by role
        role_counts = {}
        for info in self.speaker_roles.values():
            role = info.get('role', 'Unknown')
            role_counts[role] = role_counts.get(role, 0) + 1
        
        logger.info("Speakers by role:")
        for role, count in sorted(role_counts.items()):
            logger.info(f"  {role}: {count}")
        
        # Count by era
        era_counts = {'Current (2020+)': 0, 'Recent (2000-2020)': 0, 'Historical (pre-2000)': 0}
        for info in self.speaker_roles.values():
            years = info.get('years', '')
            if 'present' in years or '2020' in years:
                era_counts['Current (2020+)'] += 1
            elif any(year in years for year in ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']):
                era_counts['Recent (2000-2020)'] += 1
            else:
                era_counts['Historical (pre-2000)'] += 1
        
        logger.info("Speakers by era:")
        for era, count in era_counts.items():
            logger.info(f"  {era}: {count}")
        
        logger.info(f"URL pattern entries: {len(self.url_name_patterns)}")

    # LEGACY COMPATIBILITY

    def scrape_all_speeches(self, limit: Optional[int] = None, include_historical: bool = True) -> int:
        """Enhanced legacy method with historical coverage."""
        stats = self.run_comprehensive_scraping_v2(
            method="all", 
            max_speeches=limit, 
            include_historical=include_historical
        )
        return stats['saved_speeches']


def main():
    """Enhanced main function with historical coverage options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Bank of England Speech Scraper v2 - Historical & Current Coverage")
    parser.add_argument("--method", choices=["main", "sitemap", "selenium", "all"], 
                       default="all", help="Scraping method to use")
    parser.add_argument("--start-year", type=int, default=1960, 
                       help="Starting year for scraping (supports back to 1960)")
    parser.add_argument("--max-speeches", type=int, 
                       help="Maximum number of speeches to scrape")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--show-db-#!/usr/bin/env python3
"""
Enhanced Bank of England Speech Scraper v2 - Historical & Current Coverage
Major improvements based on analysis of scraping issues and BoE website structure.

Key Issues Fixed:
1. Speaker recognition failures - Enhanced URL extraction and database lookups
2. Content validation failures - Improved content extraction and validation
3. Historical coverage gaps - Added Quarterly Bulletin and Digital Archive scraping
4. Success rate (38.3% -> target 80%+)

New Features:
- Historical speeches from 1990s via BoE Digital Archive
- Quarterly Bulletin speech extraction (1960s-2006)
- Enhanced speaker extraction from content and URLs
- Better content validation and metadata extraction
- Fallback mechanisms for difficult extractions

Author: Enhanced Central Bank Speech Collector
Date: 2025
Target: Comprehensive BoE speech coverage from 1960s onwards
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
from urllib.parse import urljoin, urlparse, quote
from typing import Dict, List, Optional, Tuple, Set
import time
import io

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

class EnhancedBoEScraperV2:
    """
    Enhanced Bank of England speech scraper with historical coverage and improved recognition.
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.boe_dir = os.path.join(base_dir, "boe")
        self.base_url = "https://www.bankofengland.co.uk"
        self.speeches_url = "https://www.bankofengland.co.uk/news/speeches"
        self.sitemap_url = "https://www.bankofengland.co.uk/sitemap/speeches"
        self.digital_archive_url = "https://boe.access.preservica.com"
        self.quarterly_bulletin_url = "https://www.escoe.ac.uk/research/historical-data/publist/beqb/"
        
        # Ensure directories exist
        os.makedirs(self.boe_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
        
        # Initialize comprehensive speaker database (ENHANCED)
        self._initialize_speaker_database_v2()
        
        # Initialize URL name patterns for fallback extraction
        self._initialize_url_patterns_v2()
        
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
            'content_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0,
            'historical_speeches_found': 0,
            'quarterly_bulletin_speeches': 0,
            'digital_archive_speeches': 0
        }

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

    def _initialize_url_patterns_v2(self):
        """Enhanced URL patterns for better speaker extraction."""
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
        
        # Enhanced manual patterns with historical names
        manual_patterns = {
            # Current officials
            'bailey': ['andrew bailey', 'andrew john bailey'],
            'lombardelli': ['clare lombardelli'],
            'ramsden': ['dave ramsden', 'david ramsden'],
            'breeden': ['sarah breeden'],
            'pill': ['huw pill'],
            'haskel': ['jonathan haskel'],
            'dhingra': ['swati dhingra'],
            'mann': ['catherine mann', 'catherine l mann'],
            'taylor': ['alan taylor'],
            'greene': ['megan greene'],
            'hall': ['jon hall'],
            'kroszner': ['randall kroszner'],
            
            # Past Governors - Enhanced
            'carney': ['mark carney', 'mark joseph carney'],
            'king': ['mervyn king', 'mervyn allister king', 'lord king', 'baron king', 'lord king of lothbury'],
            'george': ['eddie george', 'edward george', 'edward alan john george', 'steady eddie', 'lord george', 'baron george', 'sir eddie george', 'sir edward george'],
            'leighpemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'leigh-pemberton': ['robin leigh-pemberton', 'robin leigh pemberton', 'lord kingsdown', 'baron kingsdown'],
            'richardson': ['gordon richardson', 'gordon william humphreys richardson', 'lord richardson', 'baron richardson'],
            'obrien': ["leslie o'brien", "leslie kenneth o'brien", "lord o'brien"],
            "o'brien": ["leslie o'brien", "leslie kenneth o'brien", "lord o'brien"],
            
            # Deputy Governors
            'broadbent': ['ben broadbent'],
            'cunliffe': ['jon cunliffe', 'sir jon cunliffe', 'jonathan cunliffe'],
            'woods': ['sam woods', 'samuel woods'],
            'clementi': ['david clementi', 'sir david clementi'],
            'davies': ['howard davies'],
            'pennant-rea': ['rupert pennant-rea'],
            'blunden': ['george blunden'],
            'mcmahon': ['kit mcmahon', 'christopher mcmahon'],
            
            # Chief Economists
            'haldane': ['andy haldane', 'andrew haldane', 'andrew g haldane'],
            'dale': ['spencer dale'],
            'bean': ['charlie bean', 'charles bean', 'charles goodhart bean'],
            'vickers': ['john vickers', 'sir john vickers'],
            
            # External MPC Members
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
            'bell': ['marian bell'],
            'lambert': ['richard lambert'],
            
            # Executive Directors and Officials
            'tucker': ['paul tucker', 'paul mw tucker'],
            'large': ['andrew large'],
            'fisher': ['paul fisher'],
            'rule': ['david rule'],
            'cleland': ['victoria cleland'],
            'plenderleith': ['ian plenderleith'],
            'clark': ['alastair clark'],
            'quinn': ['brian quinn'],
            'kent': ['pen kent'],
            'cooke': ['william cooke', 'w p cooke'],
        }
        
        # Update with manual patterns
        for pattern, names in manual_patterns.items():
            if pattern not in self.url_name_patterns:
                self.url_name_patterns[pattern] = []
            self.url_name_patterns[pattern].extend(names)
            # Remove duplicates
            self.url_name_patterns[pattern] = list(set(self.url_name_patterns[pattern]))

    def extract_speaker_from_url_enhanced(self, url: str) -> Optional[str]:
        """
        ENHANCED speaker extraction from URL with better pattern matching.
        This addresses the main issue causing validation failures.
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            # Enhanced URL patterns for BoE speeches
            url_patterns = [
                # Pattern: speech-by-firstname-lastname
                r'speech-by-([a-z-]+)-([a-z-]+)',
                # Pattern: lastname-speech or lastname-remarks
                r'([a-z-]+)-(speech|remarks|address)',
                # Pattern: firstname-lastname-speech
                r'([a-z-]+)-([a-z-]+)-(speech|remarks|address)',
                # Pattern: remarks-given-by-firstname-lastname
                r'remarks-given-by-([a-z-]+)-([a-z-]+)',
                # Pattern: /lastname/ in path
                r'/([a-z-]+)/',
                # Pattern: just lastname before file extension
                r'([a-z-]+)\.(pdf|html?) ,
            ]
            
            for pattern in url_patterns:
                match = re.search(pattern, path)
                if match:
                    groups = match.groups()
                    
                    # Extract name parts, excluding keywords
                    name_parts = []
                    for group in groups:
                        if group not in ['speech', 'remarks', 'address', 'pdf', 'html', 'htm']:
                            name_parts.append(group.replace('-', ' '))
                    
                    if name_parts:
                        candidate_name = ' '.join(name_parts).strip()
                        
                        # Direct lookup in our patterns
                        if candidate_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[candidate_name]
                            logger.info(f"URL speaker extraction: '{candidate_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
                        
                        # Try just the last word (surname)
                        last_name = candidate_name.split()[-1] if ' ' in candidate_name else candidate_name
                        if last_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[last_name]
                            logger.info(f"URL speaker extraction (lastname): '{last_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
                        
                        # Try without hyphens/spaces
                        clean_name = candidate_name.replace(' ', '').replace('-', '')
                        if clean_name in self.url_name_patterns:
                            matched_names = self.url_name_patterns[clean_name]
                            logger.info(f"URL speaker extraction (clean): '{clean_name}' -> '{matched_names[0]}'")
                            self.stats['url_fallback_used'] += 1
                            return matched_names[0]
            
            logger.debug(f"No speaker pattern matched for URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting speaker from URL {url}: {e}")
            return None

    def extract_speaker_from_content_enhanced(self, soup: BeautifulSoup, url: str = None) -> Optional[str]:
        """
        ENHANCED speaker extraction from content with multiple strategies.
        This is a key improvement to reduce validation failures.
        """
        # Strategy 1: Try specific speaker selectors
        speaker_selectors = [
            '.speech-author',
            '.article-author', 
            '.byline',
            '.speaker-name',
            '.author-name',
            '.speech-by',
            '[class*="author"]',
            '[class*="speaker"]',
            'h1 + p',  # Often speaker info is in paragraph after title
            '.meta-author',
            '.speech-meta'
        ]
        
        for selector in speaker_selectors:
            elements = soup.select(selector)
            for element in elements:
                speaker_text = element.get_text(strip=True)
                name = self._clean_and_validate_speaker_name(speaker_text)
                if name and name != 'Unknown':
                    logger.debug(f"Speaker found via CSS selector '{selector}': {name}")
                    return name
        
        # Strategy 2: Search in structured content with enhanced patterns
        content_areas = [
            soup.find('main'),
            soup.find('article'), 
            soup.find('div', class_='content'),
            soup.find('div', class_='speech'),
            soup
        ]
        
        for area in content_areas:
            if not area:
                continue
                
            text = area.get_text()
            
            # Enhanced speaker patterns for BoE content
            patterns = [
                # Standard titles with names
                r'Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Deputy Governor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Chair\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Chief Economist\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                
                # "By" patterns
                r'(?:^|\n)\s*By\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'(?:Remarks|Speech|Address)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                r'Given by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                
                # Name followed by title
                r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Governor|Deputy Governor|Chair|Chief Economist)',
                
                # Lord/Sir titles
                r'(?:Lord|Sir)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                
                # Pattern: Name at start of speech
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*\n',
                
                # MPC member pattern
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*),?\s+(?:External )?MPC [Mm]ember',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE)
                if matches:
                    for match in matches:
                        name = self._clean_and_validate_speaker_name(match)
                        if name and name != 'Unknown':
                            logger.debug(f"Speaker found via content pattern: {name}")
                            return name
        
        # Strategy 3: URL fallback
        if url:
            url_name = self.extract_speaker_from_url_enhanced(url)
            if url_name:
                return url_name
        
        # Strategy 4: Title extraction fallback
        title_text = soup.find('title')
        if title_text:
            title = title_text.get_text()
            # Look for "speech by NAME" in title
            title_pattern = r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)'
            match = re.search(title_pattern, title, re.IGNORECASE)
            if match:
                name = self._clean_and_validate_speaker_name(match.group(1))
                if name and name != 'Unknown':
                    logger.debug(f"Speaker found in title: {name}")
                    return name
        
        logger.debug("No speaker found with any extraction method")
        return None

    def _clean_and_validate_speaker_name(self, raw_name: str) -> str:
        """
        Enhanced speaker name cleaning and validation.
        This is crucial for reducing validation failures.
        """
        if not raw_name:
            return 'Unknown'
        
        # Remove newlines and normalize whitespace
        raw_name = ' '.join(raw_name.split())
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            r'\b(?:The\s+)?(?:Rt\s+)?(?:Hon\s+)?(?:Sir|Lord|Baron|Dame|Dr|Mr|Ms|Mrs)\s+',
            r'\b(?:Governor|Deputy Governor|Chair|President|Chief Economist)\s+',
            r'\b(?:External\s+)?MPC\s+Member\s+',
        ]
        
        for prefix in prefixes_to_remove:
            raw_name = re.sub(prefix, '', raw_name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        raw_name = re.split(r'\s*(?:,|by|remarks|speech|address|gave|given)\s*', raw_name, flags=re.IGNORECASE)[0]
        
        # Clean and validate
        name = ' '.join(raw_name.split()).strip()
        
        # Remove periods and normalize
        name = name.replace('.', '').strip()
        
        # Validate: must be reasonable length and format
        if len(name) < 2 or len(name) > 50:
            return 'Unknown'
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', name):
            return 'Unknown'
        
        # Check against known problematic patterns
        problematic_patterns = [
            r'^\d+ ,  # Just numbers
            r'^[^\w\s]+ ,  # Just punctuation
            r'unknown|speaker|author|governor|deputy|chair|president',  # Generic terms
        ]
        
        name_lower = name.lower()
        for pattern in problematic_patterns:
            if re.match(pattern, name_lower):
                return 'Unknown'
        
        return name if name else 'Unknown'

    def get_speaker_info_enhanced(self, speaker_name: str, url: str = None, content: str = None) -> Dict[str, str]:
        """
        Enhanced speaker information lookup with multiple fallback strategies.
        """
        # Strategy 1: Try extraction from content first if available
        if content and (not speaker_name or speaker_name == 'Unknown'):
            soup_content = BeautifulSoup(content, 'html.parser')
            extracted_name = self.extract_speaker_from_content_enhanced(soup_content, url)
            if extracted_name and extracted_name != 'Unknown':
                speaker_name = extracted_name
                self.stats['content_fallback_used'] += 1
        
        # Strategy 2: Try URL extraction if still unknown
        if (not speaker_name or speaker_name == 'Unknown') and url:
            url_extracted_name = self.extract_speaker_from_url_enhanced(url)
            if url_extracted_name:
                speaker_name = url_extracted_name
        
        if not speaker_name or speaker_name.strip() == 'Unknown':
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
            if self._names_match_enhanced(normalized_name, known_name):
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
        
        # Method 4: Try fuzzy matching on first/last name combinations
        name_parts = normalized_name.split()
        if len(name_parts) >= 2:
            first_last = f"{name_parts[0]} {name_parts[-1]}"
            if first_last in self.speaker_roles:
                result = self.speaker_roles[first_last].copy()
                result['source'] = 'first_last_match'
                logger.debug(f"First-last match found: {speaker_name} -> {first_last}")
                return result
        
        # Final fallback: Unknown speaker
        logger.warning(f"Unknown speaker after all methods: {speaker_name} (normalized: {normalized_name})")
        self.stats['unknown_speakers'] += 1
        return {"role": "Unknown", "voting_status": "Unknown", "source": "unknown"}

    def _names_match_enhanced(self, name1: str, name2: str) -> bool:
        """Enhanced name matching with better fuzzy logic."""
        if not name1 or not name2:
            return False
            
        parts1 = set(name1.replace('.', '').replace('-', ' ').split())
        parts2 = set(name2.replace('.', '').replace('-', ' ').split())
        
        # Remove common middle initials and abbreviations
        parts1 = {p for p in parts1 if len(p) > 1}
        parts2 = {p for p in parts2 if len(p) > 1}
        
        common_parts = parts1.intersection(parts2)
        
        # For short names (2 parts or less), require full overlap
        if len(parts1) <= 2 and len(parts2) <= 2:
            return len(common_parts) >= min(len(parts1), len(parts2))
        else:
            # For longer names, require at least 2 matching parts
            return len(common_parts) >= 2

    def _clean_speaker_name_for_lookup(self, name: str) -> str:
        """Enhanced speaker name cleaning for database lookup."""
        if not name:
            return ""
        
        # Remove titles and clean more thoroughly
        name = re.sub(r'\b(?:The\s+)?(?:Rt\s+)?(?:Hon\s+)?(?:Governor|Deputy Governor|Chair|President|Dr\.|Mr\.|Ms\.|Mrs\.|Sir|Lord|Baron|Dame)\s*', '', name, flags=re.IGNORECASE)
        
        # Remove everything after comma or other delimiters
        name = re.split(r'\s*(?:,|by|remarks|speech|address|gave|given)\s*', name, flags=re.IGNORECASE)[0]
        
        # Remove newlines and extra whitespace
        name = ' '.join(name.split())
        
        # Convert to lowercase and remove periods
        name = name.lower().strip().replace('.', '')
        
        return name

    def extract_date_from_url_enhanced(self, url: str) -> Optional[str]:
        """
        Enhanced date extraction with more patterns and better validation.
        """
        if not url:
            return None
            
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            # Enhanced BoE URL date patterns
            date_patterns = [
                # Current BoE pattern: /speech/2024/october/title-slug
                r'/speech/(\d{4})/(\w+)/',
                # Pattern: /speech/2024/10/title-slug  
                r'/speech/(\d{4})/(\d{1,2})/',
                # Legacy patterns
                r'/speeches/(\d{4})/(\w+)/',
                r'/speeches/(\d{4})/(\d{1,2})/',
                # Media files pattern: /files/speech/2024/july/
                r'/files/speech/(\d{4})/(\w+)/',
                # Date in filename: speech-2024-10-15
                r'speech-(\d{4})-(\d{2})-(\d{2})',
                # Pattern: embedded YYYYMMDD
                r'(\d{8})',
                # Year-only pattern for historical speeches
                r'/(\d{4})/',
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, path)
                if match:
                    groups = match.groups()
                    
                    if len(groups) == 1:
                        # Could be YYYYMMDD or just year
                        date_str = groups[0]
                        if len(date_str) == 8:  # YYYYMMDD
                            try:
                                year = int(date_str[:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                if 1960 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                    date_obj = datetime(year, month, day)
                                    return date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                continue
                        elif len(date_str) == 4:  # Just year
                            try:
                                year = int(date_str)
                                if 1960 <= year <= 2030:
                                    # Use January 1st as default
                                    return f"{year}-01-01"
                            except ValueError:
                                continue
                    
                    elif len(groups) == 2:
                        year_str, month_str = groups
                        try:
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
                            if 1960 <= year <= 2030 and 1 <= month <= 12:
                                date_obj = datetime(year, month, 1)
                                formatted_date = date_obj.strftime('%Y-%m-%d')
                                logger.debug(f"URL date extraction successful: {url} -> {formatted_date}")
                                return formatted_date
                        except ValueError:
                            continue
                            
                    elif len(groups) == 3:
                        year_str, month_str, day_str = groups
                        try:
                            year = int(year_str)
                            month = int(month_str)
                            day = int(day_str)
                            
                            if 1960 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                date_obj = datetime(year, month, day)
                                formatted_date = date_obj.strftime('%Y-%m-%d')
                                logger.debug(f"URL date extraction successful: {url} -> {formatted_date}")
                                return formatted_date
                        except ValueError:
                            continue
            
            logger.debug(f"No valid date pattern found in URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting date from URL {url}: {e}")
            return None

    # NEW METHODS FOR HISTORICAL COVERAGE

    def scrape_quarterly_bulletin_speeches(self, start_year: int = 1960, end_year: int = 2006) -> List[Dict]:
        """
        Scrape historical speeches from Bank of England Quarterly Bulletins (1960-2006).
        This provides access to speeches from the pre-digital era.
        """
        logger.info(f"Scraping BoE Quarterly Bulletin speeches ({start_year}-{end_year})")
        
        all_speeches = []
        
        try:
            response = requests.get(self.quarterly_bulletin_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all speech references in the quarterly bulletin listings
            # Pattern: "speech by [Name]" or "Governor's speech"
            speech_patterns = [
                r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*[-–]\s*speech',
                r"Governor's speech",
                r'remarks by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                r'lecture by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            ]
            
            # Look for speech references in the bulletin content
            text = soup.get_text()
            lines = text.split('\n')
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['speech', 'remarks', 'lecture', 'address']):
                    # Extract year from line if present
                    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', line)
                    if year_match:
                        year = int(year_match.group(1))
                        if start_year <= year <= end_year:
                            
                            # Try to extract speaker name
                            speaker = None
                            for pattern in speech_patterns:
                                match = re.search(pattern, line, re.IGNORECASE)
                                if match and match.groups():
                                    speaker = match.group(1)
                                    break
                            
                            if not speaker and "governor's speech" in line.lower():
                                # Determine governor based on year
                                if 1966 <= year <= 1973:
                                    speaker = "leslie o'brien"
                                elif 1973 <= year <= 1983:
                                    speaker = "gordon richardson"
                                elif 1983 <= year <= 1993:
                                    speaker = "robin leigh-pemberton"
                                elif 1993 <= year <= 2003:
                                    speaker = "eddie george"
                                elif 2003 <= year <= 2006:
                                    speaker = "mervyn king"
                            
                            if speaker:
                                speech_info = {
                                    'title': line.strip(),
                                    'speaker_raw': speaker,
                                    'date': f"{year}-01-01",  # Approximate date
                                    'date_source': 'quarterly_bulletin',
                                    'source_url': f"{self.quarterly_bulletin_url}#{year}",
                                    'context_text': line.strip(),
                                    'source_type': 'Quarterly Bulletin'
                                }
                                all_speeches.append(speech_info)
                                self.stats['quarterly_bulletin_speeches'] += 1
            
            logger.info(f"Found {len(all_speeches)} speeches in Quarterly Bulletins")
            
        except Exception as e:
            logger.error(f"Error scraping Quarterly Bulletins: {e}")
        
        return all_speeches

    def scrape_digital_archive_speeches(self, start_year: int = 1990, end_year: int = 2020) -> List[Dict]:
        """
        Scrape speeches from the BoE Digital Archive.
        This covers the gap between Quarterly Bulletins and modern website.
        """
        logger.info(f"Scraping BoE Digital Archive speeches ({start_year}-{end_year})")
        
        all_speeches = []
        
        # The digital archive is organized by year folders
        for year in range(start_year, end_year + 1):
            try:
                # Try to access the speeches folder for this year
                archive_url = f"{self.digital_archive_url}/?name=SPEECHES_{year}"
                
                response = requests.get(archive_url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for speech links in the archive
                    speech_links = soup.find_all('a', href=True)
                    
                    for link in speech_links:
                        href = link.get('href')
                        text = link.get_text(strip=True)
                        
                        # Check if this looks like a speech
                        if (href and text and 
                            any(keyword in text.lower() for keyword in ['speech', 'remarks', 'address', 'lecture']) and
                            len(text) > 10):
                            
                            full_url = urljoin(self.digital_archive_url, href)
                            
                            # Extract speaker from title/context
                            speaker = self._extract_speaker_from_title(text)
                            
                            speech_info = {
                                'title': text,
                                'speaker_raw': speaker or '',
                                'date': f"{year}-01-01",  # Approximate date
                                'date_source': 'digital_archive',
                                'source_url': full_url,
                                'context_text': text,
                                'source_type': 'Digital Archive'
                            }
                            all_speeches.append(speech_info)
                            self.stats['digital_archive_speeches'] += 1
                
                time.sleep(1)  # Be respectful to the archive
                
            except Exception as e:
                logger.debug(f"Could not access digital archive for {year}: {e}")
                continue
        
        logger.info(f"Found {len(all_speeches)} speeches in Digital Archive")
        return all_speeches

    def _extract_speaker_from_title(self, title: str) -> Optional[str]:
        """Extract speaker name from speech title."""
        if not title:
            return None
        
        # Common patterns in speech titles
        patterns = [
            r'speech by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'remarks by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'address by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'lecture by ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*[-–]\s*speech',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*:',  # Name at start followed by colon
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                name = self._clean_and_validate_speaker_name(match.group(1))
                if name and name != 'Unknown':
                    return name
        
        return None

    # ENHANCED CONTENT EXTRACTION

    def _extract_main_content_enhanced_v2(self, soup: BeautifulSoup, url: str = None) -> str:
        """
        Enhanced content extraction with multiple fallback strategies.
        This addresses the core issue of short/empty content extraction.
        """
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.navigation', 'noscript', 'aside']):
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
            '.speech-text',
            '.text-content',
            '.body-content',
            '#content',
            '.entry-content',
            '.post-content'
        ]
        
        for selector in boe_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
                if len(text) > 300:  # Must be substantial
                    content_candidates.append(('boe_selector', text, len(text), selector))
                    logger.debug(f"Found content using selector {selector}: {len(text)} chars")
        
        # Strategy 2: Try to find the largest meaningful content block
        all_divs = soup.find_all(['div', 'section', 'article'])
        for div in all_divs:
            text = div.get_text(separator='\n', strip=True)
            if len(text) > 800:  # Must be very substantial for this method
                # Check that it's not just navigation or boilerplate
                skip_indicators = [
                    'navigation', 'skip to', 'breadcrumb', 'footer', 'sidebar', 
                    'menu', 'search', 'cookie', 'privacy', 'terms'
                ]
                if not any(skip_text in text.lower() for skip_text in skip_indicators):
                    content_candidates.append(('largest_div', text, len(text), 'div_search'))
        
        # Strategy 3: Paragraph aggregation with content filtering
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Filter out navigation and short paragraphs
            meaningful_paras = []
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if (len(p_text) > 20 and 
                    not any(skip in p_text.lower() for skip in ['cookie', 'javascript', 'skip to', 'navigation'])):
                    meaningful_paras.append(p_text)
            
            if meaningful_paras:
                para_text = '\n\n'.join(meaningful_paras)
                if len(para_text) > 500:
                    content_candidates.append(('paragraphs', para_text, len(para_text), 'paragraph_agg'))
        
        # Strategy 4: Look for content in specific BoE content patterns
        content_patterns = [
            # Look for div containing multiple paragraphs (speech content)
            lambda: soup.find('div', string=re.compile(r'speech|remarks|address', re.I)),
            # Look for container with substantial text
            lambda: soup.find('div', attrs={'class': re.compile(r'content|speech|text|body', re.I)}),
            # Look for main content area
            lambda: soup.find(attrs={'id': re.compile(r'main|content|speech', re.I)}),
        ]
        
        for pattern_func in content_patterns:
            try:
                element = pattern_func()
                if element:
                    # Get parent or the element itself
                    content_area = element.parent if element.parent else element
                    text = content_area.get_text(separator='\n', strip=True)
                    if len(text) > 400:
                        content_candidates.append(('pattern_match', text, len(text), 'content_pattern'))
            except:
                continue
        
        # Strategy 5: Body content (last resort) with better filtering
        body = soup.find('body')
        if body:
            body_text = body.get_text(separator='\n', strip=True)
            if len(body_text) > 1000:
                # Try to remove header/footer/navigation from body text
                lines = body_text.split('\n')
                filtered_lines = []
                for line in lines:
                    line = line.strip()
                    if (len(line) > 10 and 
                        not any(skip in line.lower() for skip in [
                            'bank of england', 'speeches', 'navigation', 'search', 
                            'menu', 'home', 'about', 'contact', 'privacy', 'cookies'
                        ])):
                        filtered_lines.append(line)
                
                filtered_text = '\n'.join(filtered_lines)
                if len(filtered_text) > 600:
                    content_candidates.append(('body_filtered', filtered_text, len(filtered_text), 'body'))
        
        # Choose the best candidate based on length and strategy priority
        if content_candidates:
            # Sort by strategy priority and length
            strategy_priority = {
                'boe_selector': 4,
                'pattern_match': 3,
                'largest_div': 2,
                'paragraphs': 1,
                'body_filtered': 0
            }
            
            content_candidates.sort(key=lambda x: (strategy_priority.get(x[0], 0), x[2]), reverse=True)
            best_strategy, best_content, best_length, selector = content_candidates[0]
            
            logger.info(f"Content extraction strategy: {best_strategy} via {selector} ({best_length} chars)")
            
            # Additional validation: ensure content looks like a speech
            if self._validate_speech_content(best_content):
                cleaned_content = self._clean_text_content_enhanced(best_content)
                logger.info(f"After cleaning: {len(cleaned_content)} chars")
                return cleaned_content
            else:
                logger.warning(f"Content failed speech validation, trying next candidate")
                # Try next best candidate
                if len(content_candidates) > 1:
                    second_strategy, second_content, second_length, second_selector = content_candidates[1]
                    if self._validate_speech_content(second_content):
                        cleaned_content = self._clean_text_content_enhanced(second_content)
                        logger.info(f"Using second candidate: {second_strategy} via {second_selector} ({len(cleaned_content)} chars)")
                        return cleaned_content
        
        logger.warning("No substantial valid content found with any extraction strategy")
        return ""

    def _validate_speech_content(self, content: str) -> bool:
        """Validate that content looks like an actual speech."""
        if not content or len(content) < 200:
            return False
        
        # Check for speech indicators
        speech_indicators = [
            'thank', 'pleased', 'good morning', 'good afternoon', 'good evening',
            'ladies and gentlemen', 'chair', 'chairman', 'colleagues',
            'today', 'economic', 'policy', 'bank', 'financial', 'market'
        ]
        
        content_lower = content.lower()
        indicator_count = sum(1 for indicator in speech_indicators if indicator in content_lower)
        
        # Must have at least 3 speech indicators
        if indicator_count < 3:
            return False
        
        # Check it's not just boilerplate
        boilerplate_indicators = [
            'cookies', 'javascript', 'browser', 'website', 'homepage',
            'navigation', 'search results', 'no results found'
        ]
        
        boilerplate_count = sum(1 for indicator in boilerplate_indicators if indicator in content_lower)
        
        # Reject if too much boilerplate
        if boilerplate_count > 2:
            return False
        
        return True

    def _clean_text_content_enhanced(self, text: str) -> str:
        """Enhanced text cleaning with better preservation of speech content."""
        if not text:
            return ""
        
        original_length = len(text)
        
        # Split into lines for better processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and very short lines that are likely navigation
            if len(line) < 3:
                continue
            
            # Skip lines that are clearly navigation/boilerplate
            skip_patterns = [
                r'^(Home|About|Contact|Search|Menu|Navigation) ,
                r'^(Print this page|Share this page|Last update).*',
                r'^(Skip to|Return to|Back to).*',
                r'^(Copyright|Terms|Privacy|Cookie).*',
                r'^\s*\d+\s* ,  # Just numbers
                r'^[^\w]* ,     # Just punctuation
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            cleaned_lines.append(line)
        
        # Rejoin with proper spacing
        text = '\n'.join(cleaned_lines)
        
        # Basic cleanup
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        final_length = len(text)
        
        # Log suspicious cleaning only if major content loss
        if original_length > 2000 and final_length < original_length * 0.3:
            logger.warning(f"Significant content reduction during cleaning: {original_length} -> {final_length} chars")
        
        return text

    # ENHANCED VALIDATION SYSTEM

    def _validate_speech_data_enhanced(self, speech_data: Dict) -> bool:
        """
        Enhanced validation to prevent saving invalid speeches.
        Addresses the main cause of validation failures.
        """
        if not speech_data or 'metadata' not in speech_data or 'content' not in speech_data:
            logger.warning("Speech data missing required components")
            self.stats['validation_failures'] += 1
            return False
        
        metadata = speech_data['metadata']
        content = speech_data['content']
        
        # Enhanced content validation
        if not content or len(content.strip()) < 100:  # Reduced threshold but still meaningful
            logger.warning(f"Content too short: {len(content.strip()) if content else 0} chars")
            self.stats['content_too_short'] += 1
            return False
        
        # Check for placeholder/error content
        placeholder_indicators = [
            'lorem ipsum', 'placeholder', 'test content', 'coming soon',
            'under construction', 'page not found', 'error 404', '404 not found',
            'no content available', 'content not available'
        ]
        
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in placeholder_indicators):
            logger.warning("Content appears to be placeholder or error text")
            self.stats['validation_failures'] += 1
            return False
        
        # Enhanced metadata validation
        required_fields = ['title', 'speaker', 'date', 'source_url']
        for field in required_fields:
            if not metadata.get(field):
                logger.warning(f"Missing required metadata field: {field}")
                self.stats['validation_failures'] += 1
                return False
        
        # Enhanced date validation
        date_str = metadata.get('date')
        if not date_str:
            logger.warning("No date provided")
            self.stats['validation_failures'] += 1
            return False
        
        # Allow reasonable default dates for historical speeches
        if date_str.endswith('-01-01'):
            # This is okay for historical speeches where we only have year
            pass
        elif date_str == '2025-01-01':
            # This suggests a parsing failure
            logger.warning(f"Invalid default date: {date_str}")
            self.stats['validation_failures'] += 1
            return False
        
        # Enhanced speaker validation with more permissive rules
        speaker = metadata.get('speaker', '').lower()
        if speaker in ['unknown', 'unknown speaker', '']:
            # For historical speeches, we might not always know the speaker
            # Allow if content is substantial and looks like a speech
            if len(content) > 1000 and self._validate_speech_content(content):
                logger.info("Allowing speech with unknown speaker due to substantial content")
            else:
                logger.warning(f"Unknown speaker with insufficient content: {metadata.get('speaker')}")
                self.stats['validation_failures'] += 1
                return False
        
        # Enhanced title validation
        title = metadata.get('title', '')
        if len(title) < 5:  # Reduced threshold
            logger.warning(f"Title too short: {title}")
            self.stats['validation_failures'] += 1
            return False
        
        # Check title isn't just generic
        generic_titles = ['untitled speech', 'untitled', 'speech', 'remarks', 'address']
        if title.lower() in generic_titles:
            logger.warning(f"Generic title: {title}")
            self.stats['validation_failures'] += 1
            return False
        
        return True

    # COMPREHENSIVE SCRAPING WITH HISTORICAL COVERAGE

    def run_comprehensive_scraping_v2(self, method: str = "all", start_year: int = 1960, 
                                    max_speeches: Optional[int] = None, 
                                    include_historical: bool = True) -> Dict[str, int]:
        """
        Enhanced comprehensive BoE speech scraping with historical coverage.
        """
        logger.info(f"Starting enhanced BoE speech scraping v2")
        logger.info(f"Method: {method}, Start year: {start_year}, Max speeches: {max_speeches}")
        logger.info(f"Include historical: {include_historical}")
        
        # Reset statistics
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'url_fallback_used': 0,
            'content_fallback_used': 0,
            'unknown_speakers': 0,
            'content_extraction_failures': 0,
            'date_extraction_failures': 0,
            'content_too_short': 0,
            'validation_failures': 0,
            'saved_speeches': 0,
            'historical_speeches_found': 0,
            'quarterly_bulletin_speeches': 0,
            'digital_archive_speeches': 0
        }
        
        all_speeches = []
        
        # Phase 1: Historical speeches (if requested)
        if include_historical and start_year < 1997:
            logger.info("=== PHASE 1: HISTORICAL SPEECH COLLECTION ===")
            
            # Quarterly Bulletin speeches (1960-2006)
            if start_year <= 2006:
                logger.info("Collecting speeches from Quarterly Bulletins...")
                qb_speeches = self.scrape_quarterly_bulletin_speeches(start_year, min(2006, 2025))
                all_speeches.extend(qb_speeches)
                self.stats['historical_speeches_found'] += len(qb_speeches)
            
            # Digital Archive speeches (1990-2020)
            if start_year <= 2020:
                logger.info("Collecting speeches from Digital Archive...")
                da_speeches = self.scrape_digital_archive_speeches(max(1990, start_year), min(2020, 2025))
                all_speeches.extend(da_speeches)
                self.stats['historical_speeches_found'] += len(da_speeches)
        
        # Phase 2: Modern website scraping
        logger.info("=== PHASE 2: MODERN WEBSITE SCRAPING ===")
        
        # Approach 1: Sitemap scraping (most reliable)
        if method in ["sitemap", "all"]:
            logger.info("Running sitemap scraping...")
            sitemap_speeches = self.scrape_speeches_from_sitemap(max_speeches or 100)
            all_speeches.extend(sitemap_speeches)
            logger.info(f"Sitemap method found {len(sitemap_speeches)} speeches")
        
        # Approach 2: Main speeches page scraping
        if method in ["main", "all"] and len([s for s in all_speeches if s.get('date', '').startswith('202')]) < 10:
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
        unique_speeches = self._deduplicate_speeches_enhanced(all_speeches)
        logger.info(f"Total unique speeches found: {len(unique_speeches)}")
        
        if not unique_speeches:
            logger.warning("No speeches found!")
            return self.stats
        
        # Filter by year if requested
        if start_year:
            filtered_speeches = []
            for speech in unique_speeches:
                speech_date = speech.get('date', '')
                if speech_date:
                    try:
                        speech_year = int(speech_date[:4])
                        if speech_year >= start_year:
                            filtered_speeches.append(speech)
                    except (ValueError, IndexError):
                        # Include if we can't parse the date
                        filtered_speeches.append(speech)
                else:
                    filtered_speeches.append(speech)
            
            unique_speeches = filtered_speeches
            logger.info(f"After year filtering ({start_year}+): {len(unique_speeches)} speeches")
        
        # Limit speeches if requested
        if max_speeches:
            unique_speeches = unique_speeches[:max_speeches]
            logger.info(f"Limited to {max_speeches} speeches")
        
        # Phase 3: Process each speech
        logger.info(f"=== PHASE 3: PROCESSING {len(unique_speeches)} SPEECHES ===")
        
        for i, speech_info in enumerate(unique_speeches, 1):
            logger.info(f"Processing speech {i}/{len(unique_speeches)}: {speech_info['source_url']}")
            
            try:
                # Extract content and metadata with enhanced methods
                speech_data = self.scrape_speech_content_enhanced(speech_info)
                
                if speech_data:
                    # Save speech (already validated in scrape_speech_content_enhanced)
                    saved_filename = self.save_speech_enhanced(speech_data)
                    if saved_filename:
                        self.stats['saved_speeches'] += 1
                        
                        # Log speaker recognition details
                        metadata = speech_data['metadata']
                        logger.info(f"✓ Successfully saved: {saved_filename}")
                        logger.info(f"  Speaker: {metadata['speaker']} ({metadata.get('recognition_source', 'unknown')})")
                        logger.info(f"  Role: {metadata['role']}")
                        logger.info(f"  Date: {metadata['date']} ({metadata.get('date_source', 'unknown')})")
                    else:
                        logger.error(f"✗ Failed to save speech from {speech_info['source_url']}")
                else:
                    logger.error(f"✗ Failed to extract or validate content from {speech_info['source_url']}")
                
            except Exception as e:
                logger.error(f"✗ Unexpected error processing {speech_info['source_url']}: {e}")
            
            # Respectful delay
            time.sleep(0.5)
        
        # Final statistics
        self._log_final_statistics()
        
        return self.stats

    def scrape_speech_content_enhanced(self, speech_info: Dict) -> Optional[Dict]:
        """Enhanced speech content scraping with better validation and fallbacks."""
        url = speech_info['source_url']
        logger.debug(f"Scraping content from: {url}")
        
        self.stats['total_processed'] += 1
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                result = self._extract_pdf_content_enhanced(response.content, speech_info)
            else:
                result = self._extract_html_content_enhanced_v2(response.text, speech_info, url)
            
            # Enhanced validation
            if result and self._validate_speech_data_enhanced(result):
                self.stats['successful_extractions'] += 1
                return result
            else:
                logger.warning(f"Speech failed enhanced validation: {url}")
                self.stats['content_extraction_failures'] += 1
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            self.stats['content_extraction_failures'] += 1
            return None

    def _extract_html_content_enhanced_v2(self, html_content: str, speech_info: Dict, url: str) -> Optional[Dict]:
        """Enhanced HTML content extraction with better speaker recognition."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Enhanced content extraction
            content = self._extract_main_content_enhanced_v2(soup, url)
            
            # Early exit if no substantial content
            if not content or len(content.strip()) < 100:
                logger.warning(f"Insufficient content extracted from {url}: {len(content.strip()) if content else 0} chars")
                return None
            
            # Enhanced component extraction
            title = self._extract_title_enhanced(soup, speech_info.get('title', ''))
            
            # Enhanced speaker extraction with content awareness
            speaker_name = self.extract_speaker_from_content_enhanced(soup, url)
            if not speaker_name or speaker_name == 'Unknown':
                speaker_name = speech_info.get('speaker_raw', '')
            
            # Enhanced date extraction
            date = self.extract_date_from_url_enhanced(url)
            if not date:
                date = speech_info.get('date', '')
            
            location = self._extract_location_enhanced(soup)
            
            # Get enhanced speaker information
            role_info = self.get_speaker_info_enhanced(speaker_name, url, content)
            
            # Build comprehensive metadata
            metadata = {
                'title': title,
                'speaker': role_info.get('matched_name', speaker_name) if role_info.get('source') != 'unknown' else speaker_name,
                'role': role_info.get('role', 'Unknown'),
                'institution': 'Bank of England',
                'country': 'UK',
                'date': date,
                'location': location,
                'language': 'en',
                'source_url': url,
                'source_type': speech_info.get('source_type', 'HTML'),
                'voting_status': role_info.get('voting_status', 'Unknown'),
                'recognition_source': role_info.get('source', 'unknown'),
                'date_source': speech_info.get('date_source', 'url'),
                'tags': self._extract_content_tags_enhanced(content),
                'scrape_timestamp': datetime.now().isoformat(),
                'content_length': len(content)
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

    def _extract_content_tags_enhanced(self, content: str) -> List[str]:
        """Enhanced content tag extraction with more comprehensive keywords."""
        tags = []
        content_lower = content.lower()
        
        # Enhanced keyword mapping
        keywords = {
            'inflation': ['inflation', 'price stability', 'cpi', 'rpi', 'deflation', 'disinflation', 'price level', 'core inflation', 'headline inflation'],
            'interest_rates': ['interest rate', 'bank rate', 'monetary policy', 'policy rate', 'rate rise', 'rate cut', 'rate increase', 'rate decrease', 'base rate', 'official rate'],
            'employment': ['employment', 'unemployment', 'labour market', 'labor market', 'jobs', 'jobless', 'payroll', 'employment data', 'job growth', 'labour force', 'wage'],
            'financial_stability': ['financial stability', 'banking', 'supervision', 'regulation', 'systemic risk', 'stress test', 'capital requirements', 'prudential', 'financial system'],
            'economic_outlook': ['economic outlook', 'forecast', 'projection', 'growth', 'recession', 'expansion', 'economic conditions', 'gdp', 'economic recovery'],
            'monetary_policy': ['monetary policy', 'mpc', 'monetary policy committee', 'quantitative easing', 'qe', 'gilt purchases', 'asset purchases', 'forward guidance'],
            'banking': ['bank', 'banking', 'credit', 'lending', 'deposits', 'financial institutions', 'commercial banks', 'banking sector'],
            'markets': ['market', 'financial markets', 'capital markets', 'bond market', 'stock market', 'equity markets', 'gilt', 'currency'],
            'crisis': ['crisis', 'pandemic', 'covid', 'financial crisis', 'economic crisis', 'emergency', 'coronavirus', '2008 crisis'],
            'brexit': ['brexit', 'european union', 'eu', 'single market', 'customs union', 'trade deal', 'referendum'],
            'international': ['international', 'global', 'foreign', 'trade', 'exchange rate', 'emerging markets', 'global economy', 'international cooperation'],
            'technology': ['technology', 'digital', 'fintech', 'innovation', 'artificial intelligence', 'ai', 'blockchain', 'cryptocurrency'],
            'climate': ['climate', 'environmental', 'green', 'sustainability', 'carbon', 'net zero', 'climate change']
        }
        
        for tag, terms in keywords.items():
            if any(term in content_lower for term in terms):
                tags.append(tag)
        
        return tags

    def _deduplicate_speeches_enhanced(self, speeches: List[Dict]) -> List[Dict]:
        """Enhanced deduplication with better matching."""
        unique_speeches = []
        seen_urls = set()
        seen_combinations = set()
        
        for speech in speeches:
            url = speech.get('source_url', '')
            title = speech.get('title', '').lower().strip()
            date = speech.get('date', '')
            
            # Primary deduplication by URL
            if url and url not in seen_urls:
                # Secondary deduplication by title+date combination
                combination = f"{title}_{date}"
                if combination not in seen_combinations:
                    unique_speeches.append(speech)
                    seen_urls.add(url)
                    seen_combinations.add(combination)
        
        return unique_speeches

    def save_speech_enhanced(self, speech_data: Dict) -> Optional[str]:
        """Enhanced speech saving with better error handling."""
        try:
            metadata = speech_data['metadata']
            content = speech_data['content']
            
            # Generate content hash for uniqueness
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:10]
            
            # Enhanced speaker name sanitization
            speaker_name = metadata.get('speaker', 'unknown')
            if speaker_name and speaker_name != 'Unknown':
                clean_speaker = re.sub(r'[^\w\s-]', '', speaker_name.lower())
                clean_speaker = re.sub(r'\s+', '-', clean_speaker)
                clean_speaker = clean_speaker.strip('-')[:20]  # Limit length
                if not clean_speaker:
                    clean_speaker = 'unknown-speaker'
            else:
                clean_speaker = 'unknown-speaker'
            
            # Use the date from metadata
            date_str = metadata.get('date', 'unknown-date')
            if date_str and date_str != 'unknown-date':
                date_part = date_str[:10]  # YYYY-MM-DD
            else:
                date_part = 'unknown-date'
            
            base_filename = f"{date_part}_{clean_speaker}-{content_hash}"
            
            # Final filename sanitization
            base_filename = re.sub(r'[^\w\-.]', '', base_filename)[:100]  # Limit total length
            
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
            
            logger.debug(f"Saved speech: {base_filename}")
            return base_filename
            
        except Exception as e:
            logger.error(f"Error saving speech: {e}")
            return None

    def _log_final_statistics(self):
        """Log comprehensive final statistics."""
        logger.info("=== ENHANCED BoE SCRAPING COMPLETE ===")
        logger.info(f"Total speeches processed: {self.stats['total_processed']}")
        logger.info(f"Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"Saved speeches: {self.stats['saved_speeches']}")
        logger.info(f"Historical speeches found: {self.stats['historical_speeches_found']}")
        logger.info(f"  - Quarterly Bulletin: {self.stats['quarterly_bulletin_speeches']}")
        logger.info(f"  - Digital Archive: {self.stats['digital_archive_speeches']}")
        logger.info(f"URL fallback used: {self.stats['url_fallback_used']}")
        logger.info(f"Content fallback used: {self.stats['content_fallback_used']}")
        logger.info(f"Unknown speakers: {self.stats['unknown_speakers']}")
        logger.info(f"Date extraction failures: {self.stats['date 