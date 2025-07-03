#!/usr/bin/env python3
"""
Enhanced Central Bank Speech Collector - BIS Scraper Module

This module provides multiple approaches to scrape central bank speeches from BIS:
1. Direct API/bulk download approach (recommended)
2. Selenium-based dynamic scraping
3. Alternative endpoint scraping

Author: Central Bank Speech Collector
Date: 2025
"""

import os
import re
import json
import time
import hashlib
import logging
import requests
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional, Tuple
import zipfile
import io

from bs4 import BeautifulSoup
import pdfplumber

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
    print("Selenium not available. Install with: pip install selenium")


class EnhancedBISScraper:
    """Enhanced scraper for BIS Central Bank Speeches with multiple approaches."""
    
    def __init__(self, base_url: str = "https://www.bis.org/cbspeeches/index.htm",
                 output_dir: str = "data", delay: float = 1.0):
        """
        Initialize the enhanced BIS scraper.
        
        Args:
            base_url: BIS speeches index page URL
            output_dir: Base directory for outputs
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.base_domain = "https://www.bis.org"
        self.output_dir = output_dir
        self.delay = delay
        
        # BIS bulk download URLs - FIXED: Use complete historical dataset
        self.bulk_download_url = "https://www.bis.org/cbspeeches/download.htm"
        self.bulk_data_url = "https://www.bis.org/speeches/speeches.zip"  # Complete dataset (1996-present)
        
        # Create output directories
        self.speeches_dir = os.path.join(output_dir, "speeches")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.bulk_dir = os.path.join(output_dir, "bulk")
        os.makedirs(self.speeches_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.bulk_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Setup session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, "enhanced_bis_scraper.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content for filename."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def polite_get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Make a polite HTTP GET request with delay and error handling."""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None
    
    # APPROACH 1: BULK DOWNLOAD (RECOMMENDED) - ENHANCED VERSION
    def download_bulk_speeches(self) -> bool:
        """
        Download the complete BIS speeches dataset as a bulk file.
        This is the most efficient and reliable method.
        """
        self.logger.info("Attempting to download BIS bulk speeches dataset...")
        
        try:
            # ENHANCED: Try multiple known URLs for complete dataset
            complete_urls = [
                "https://www.bis.org/speeches/speeches.zip",  # Primary complete dataset URL
                "https://www.bis.org/cbspeeches/speeches.zip"  # Alternative URL pattern
            ]
            
            for bulk_url in complete_urls:
                self.logger.info(f"Trying complete dataset URL: {bulk_url}")
                response = self.polite_get(bulk_url)
                
                if response and response.status_code == 200:
                    file_size = len(response.content) / (1024 * 1024)  # Size in MB
                    self.logger.info(f"Successfully downloaded dataset: {file_size:.1f}MB from {bulk_url}")
                    
                    # Save and extract the zip file
                    zip_path = os.path.join(self.bulk_dir, "cbspeeches_complete.zip")
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Extract the zip file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        extracted_files = zip_ref.namelist()
                        self.logger.info(f"Extracting {len(extracted_files)} files from ZIP")
                        zip_ref.extractall(self.bulk_dir)
                        
                        # Log what files were extracted
                        for file in extracted_files[:10]:  # Show first 10 files
                            self.logger.info(f"Extracted: {file}")
                        if len(extracted_files) > 10:
                            self.logger.info(f"... and {len(extracted_files) - 10} more files")
                    
                    self.logger.info(f"Successfully downloaded and extracted complete speeches dataset to {self.bulk_dir}")
                    return True
                else:
                    self.logger.warning(f"Failed to download from {bulk_url}")
            
            # If direct URLs fail, try parsing the download page
            self.logger.warning("Direct download failed, trying to parse download page...")
            return self._parse_download_page_for_complete_dataset()
            
        except Exception as e:
            self.logger.error(f"Error downloading bulk speeches: {e}")
            return False
    
    def _parse_download_page_for_complete_dataset(self) -> bool:
        """Parse the BIS download page to find the complete dataset link."""
        try:
            download_page = self.polite_get(self.bulk_download_url)
            if not download_page:
                return False
            
            soup = BeautifulSoup(download_page.content, 'html.parser')
            
            # Look for the "Download all speeches" link
            download_links = soup.find_all('a', href=True)
            for link in download_links:
                href = link.get('href')
                link_text = link.get_text().lower()
                
                # Prioritize "all speeches" or complete dataset links
                if (href and href.endswith('.zip') and 
                    ('all' in link_text or 'complete' in link_text or 
                     'speeches.zip' in href)):
                    
                    full_url = urljoin(self.base_domain, href)
                    self.logger.info(f"Found complete dataset link: {full_url}")
                    
                    response = self.polite_get(full_url)
                    if response and response.status_code == 200:
                        file_size = len(response.content) / (1024 * 1024)
                        self.logger.info(f"Downloaded complete dataset: {file_size:.1f}MB")
                        
                        # Save and extract
                        zip_path = os.path.join(self.bulk_dir, "cbspeeches_complete.zip")
                        with open(zip_path, 'wb') as f:
                            f.write(response.content)
                        
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(self.bulk_dir)
                        
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error parsing download page: {e}")
            return False
    
    def process_bulk_speeches(self) -> Dict[str, int]:
        """Process the downloaded bulk speeches - ENHANCED VERSION."""
        stats = {'total_found': 0, 'successfully_processed': 0, 'failed': 0, 'skipped': 0}
        
        try:
            self.logger.info(f"Processing bulk speeches from directory: {self.bulk_dir}")
            
            # ENHANCED: Walk through all subdirectories to find files
            processed_files = []
            for root, dirs, files in os.walk(self.bulk_dir):
                for file_name in files:
                    if file_name.endswith(('.csv', '.txt', '.json')):
                        full_path = os.path.join(root, file_name)
                        processed_files.append(full_path)
                        self.logger.info(f"Found bulk file: {full_path}")
            
            if not processed_files:
                self.logger.warning("No CSV, TXT, or JSON files found in bulk directory")
                # List all files to help debug
                all_files = []
                for root, dirs, files in os.walk(self.bulk_dir):
                    for file_name in files:
                        all_files.append(os.path.join(root, file_name))
                
                self.logger.info(f"All files in bulk directory: {all_files}")
                return stats
            
            # Process each file
            for file_path in processed_files:
                self.logger.info(f"Processing bulk file: {file_path}")
                
                if file_path.endswith('.csv'):
                    file_stats = self._process_bulk_csv(file_path)
                elif file_path.endswith('.txt'):
                    file_stats = self._process_bulk_text(file_path)
                elif file_path.endswith('.json'):
                    file_stats = self._process_bulk_json(file_path)
                else:
                    continue
                
                # Aggregate stats
                for key in stats:
                    if key in file_stats:
                        stats[key] += file_stats[key]
                
                self.logger.info(f"File stats for {file_path}: {file_stats}")
            
            self.logger.info(f"Total bulk processing stats: {stats}")
            
        except Exception as e:
            self.logger.error(f"Error processing bulk speeches: {e}")
        
        return stats
    
    def _process_bulk_csv(self, csv_path: str) -> Dict[str, int]:
        """Process bulk CSV file - ENHANCED VERSION."""
        try:
            import pandas as pd
        except ImportError:
            self.logger.error("pandas not available for CSV processing")
            return {'total_found': 0, 'successfully_processed': 0, 'failed': 0}
        
        stats = {'total_found': 0, 'successfully_processed': 0, 'failed': 0}
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    self.logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                self.logger.error(f"Could not read CSV file {csv_path} with any encoding")
                return stats
            
            stats['total_found'] = len(df)
            self.logger.info(f"Found {len(df)} rows in CSV file")
            self.logger.info(f"CSV columns: {list(df.columns)}")
            
            for idx, row in df.iterrows():
                try:
                    # ENHANCED: More flexible field mapping
                    title = str(row.get('title', row.get('Title', row.get('speech_title', ''))))
                    speaker = str(row.get('speaker', row.get('author', row.get('Speaker', row.get('Author', '')))))
                    date = str(row.get('date', row.get('Date', row.get('speech_date', ''))))
                    content = str(row.get('text', row.get('content', row.get('Text', row.get('Content', row.get('speech_text', ''))))))
                    url = str(row.get('url', row.get('source_url', row.get('URL', ''))))
                    
                    # Extract metadata
                    metadata = {
                        'title': title,
                        'speaker': speaker,
                        'date': date,
                        'institution': 'BIS',
                        'source_url': url,
                        'source_type': 'bulk_csv',
                        'scrape_timestamp': datetime.now().isoformat(),
                        'file_source': csv_path
                    }
                    
                    if content and len(content.strip()) > 100:
                        self.save_speech_and_metadata(content, metadata)
                        stats['successfully_processed'] += 1
                        
                        if stats['successfully_processed'] % 100 == 0:
                            self.logger.info(f"Processed {stats['successfully_processed']} speeches from CSV")
                    else:
                        stats['failed'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing CSV row {idx}: {e}")
                    stats['failed'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error reading CSV file {csv_path}: {e}")
        
        return stats
    
    def _process_bulk_text(self, txt_path: str) -> Dict[str, int]:
        """Process bulk text file - ENHANCED VERSION."""
        stats = {'total_found': 0, 'successfully_processed': 0, 'failed': 0}
        
        try:
            # Try different encodings
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(txt_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    self.logger.info(f"Successfully read TXT with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if not content:
                self.logger.error(f"Could not read TXT file {txt_path}")
                return stats
            
            # If it's a single large text file, treat it as one speech
            if len(content.strip()) > 100:
                metadata = {
                    'title': f"BIS Speech from {os.path.basename(txt_path)}",
                    'speaker': 'Unknown',
                    'date': '',
                    'institution': 'BIS',
                    'source_url': '',
                    'source_type': 'bulk_txt',
                    'scrape_timestamp': datetime.now().isoformat(),
                    'file_source': txt_path
                }
                
                self.save_speech_and_metadata(content, metadata)
                stats['total_found'] = 1
                stats['successfully_processed'] = 1
            
        except Exception as e:
            self.logger.error(f"Error processing TXT file {txt_path}: {e}")
        
        return stats
    
    def _process_bulk_json(self, json_path: str) -> Dict[str, int]:
        """Process bulk JSON file - ENHANCED VERSION."""
        stats = {'total_found': 0, 'successfully_processed': 0, 'failed': 0}
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                speeches = data
            elif isinstance(data, dict):
                if 'speeches' in data:
                    speeches = data['speeches']
                elif 'data' in data:
                    speeches = data['data']
                else:
                    speeches = [data]
            else:
                speeches = [data]
            
            stats['total_found'] = len(speeches)
            self.logger.info(f"Found {len(speeches)} speeches in JSON file")
            
            for speech in speeches:
                try:
                    # ENHANCED: More flexible field mapping
                    if isinstance(speech, dict):
                        title = speech.get('title', speech.get('Title', ''))
                        speaker = speech.get('speaker', speech.get('author', speech.get('Speaker', speech.get('Author', ''))))
                        date = speech.get('date', speech.get('Date', ''))
                        content = speech.get('text', speech.get('content', speech.get('Text', speech.get('Content', ''))))
                        url = speech.get('url', speech.get('source_url', ''))
                    else:
                        # If speech is just text
                        content = str(speech)
                        title = 'BIS Speech'
                        speaker = 'Unknown'
                        date = ''
                        url = ''
                    
                    metadata = {
                        'title': title,
                        'speaker': speaker,
                        'date': date,
                        'institution': 'BIS',
                        'source_url': url,
                        'source_type': 'bulk_json',
                        'scrape_timestamp': datetime.now().isoformat(),
                        'file_source': json_path
                    }
                    
                    if content and len(content.strip()) > 100:
                        self.save_speech_and_metadata(content, metadata)
                        stats['successfully_processed'] += 1
                        
                        if stats['successfully_processed'] % 100 == 0:
                            self.logger.info(f"Processed {stats['successfully_processed']} speeches from JSON")
                    else:
                        stats['failed'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing JSON speech: {e}")
                    stats['failed'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error reading JSON file {json_path}: {e}")
        
        return stats
    
    # REUSE ALL OTHER METHODS FROM ORIGINAL (unchanged)
    def extract_speech_links_selenium(self, max_speeches: int = 100) -> List[Dict[str, str]]:
        """Extract speech links using Selenium for dynamic content."""
        if not SELENIUM_AVAILABLE:
            self.logger.error("Selenium not available. Cannot use dynamic scraping.")
            return []
        
        self.logger.info("Starting Selenium-based speech link extraction...")
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        speech_links = []
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.base_url)
            
            wait = WebDriverWait(driver, 10)
            
            speech_selectors = [
                "table tr",
                ".speech-list tr",
                ".speeches tr",
                "ul li a[href*='speech']",
                "a[href*='cbspeeches']"
            ]
            
            for selector in speech_selectors:
                try:
                    elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))
                    
                    if elements:
                        self.logger.info(f"Found elements with selector: {selector}")
                        
                        for element in elements[:max_speeches]:
                            try:
                                link = element.find_element(By.TAG_NAME, "a") if element.tag_name != "a" else element
                                href = link.get_attribute("href")
                                text = element.text.strip()
                                
                                if href and self.is_speech_link(href):
                                    speech_info = {
                                        'url': href,
                                        'title': self.clean_text(link.text),
                                        'raw_text': self.clean_text(text)
                                    }
                                    speech_links.append(speech_info)
                                    
                            except Exception as e:
                                self.logger.debug(f"Error processing element: {e}")
                                continue
                        
                        if speech_links:
                            break
                            
                except TimeoutException:
                    self.logger.debug(f"Timeout waiting for selector: {selector}")
                    continue
                except NoSuchElementException:
                    self.logger.debug(f"No elements found for selector: {selector}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error with Selenium scraping: {e}")
        finally:
            try:
                driver.quit()
            except:
                pass
        
        self.logger.info(f"Found {len(speech_links)} speech links using Selenium")
        return speech_links
    
    def _scrape_paginated_speeches_selenium(self, driver, remaining_count: int) -> List[Dict[str, str]]:
        """Scrape additional pages using Selenium."""
        additional_speeches = []
        
        try:
            pagination_selectors = [
                "a[href*='page']",
                ".pagination a",
                ".next",
                "a:contains('Next')"
            ]
            
            for page_num in range(2, 10):
                if len(additional_speeches) >= remaining_count:
                    break
                
                next_clicked = False
                for selector in pagination_selectors:
                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, selector)
                        if next_button.is_displayed() and next_button.is_enabled():
                            driver.execute_script("arguments[0].click();", next_button)
                            time.sleep(2)
                            next_clicked = True
                            break
                    except:
                        continue
                
                if not next_clicked:
                    break
                
        except Exception as e:
            self.logger.error(f"Error with pagination: {e}")
        
        return additional_speeches
    
    def extract_speech_links_alternative(self) -> List[Dict[str, str]]:
        """Try alternative BIS endpoints that might have static content."""
        self.logger.info("Trying alternative BIS endpoints...")
        
        alternative_urls = [
            "https://www.bis.org/cbspeeches/index.htm?page=1",
            "https://www.bis.org/cbspeeches/recent.htm",
            "https://www.bis.org/list/cbspeeches/index.htm",
            "https://www.bis.org/speeches/",
            "https://www.bis.org/review/"
        ]
        
        speech_links = []
        
        for url in alternative_urls:
            self.logger.info(f"Trying alternative URL: {url}")
            response = self.polite_get(url)
            
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link.get('href')
                    if href and self.is_speech_link(href):
                        full_url = urljoin(self.base_domain, href)
                        
                        speech_info = {
                            'url': full_url,
                            'title': self.clean_text(link.get_text()),
                            'raw_text': self.clean_text(link.parent.get_text() if link.parent else link.get_text())
                        }
                        speech_links.append(speech_info)
                
                if speech_links:
                    self.logger.info(f"Found {len(speech_links)} speeches from {url}")
                    break
        
        return speech_links
    
    def is_speech_link(self, href: str) -> bool:
        """Check if a link appears to be a speech link."""
        speech_indicators = [
            '/cbspeeches/', '/review/', '/speech/', '.pdf',
            'r' + re.search(r'\d{6}', href).group() if re.search(r'\d{6}', href) else ''
        ]
        return any(indicator in href.lower() for indicator in speech_indicators if indicator)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        return ' '.join(text.strip().split())
    
    def extract_metadata_from_listing(self, raw_text: str, url: str) -> Dict[str, str]:
        """Extract metadata from the speech listing text."""
        metadata = {
            'title': '',
            'speaker': '',
            'role': '',
            'institution': 'BIS',
            'country': '',
            'date': '',
            'location': '',
            'language': 'en',
            'source_url': url,
            'source_type': 'pdf' if url.lower().endswith('.pdf') else 'html',
            'tags': [],
            'scrape_timestamp': datetime.now().isoformat()
        }
        
        lines = raw_text.split('\n')
        for line in lines:
            line = line.strip()
            
            date_match = re.search(r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b', line, re.IGNORECASE)
            if date_match and not metadata['date']:
                metadata['date'] = f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}"
            
            if 'by ' in line.lower() and not metadata['speaker']:
                speaker_match = re.search(r'by\s+([^,]+)', line, re.IGNORECASE)
                if speaker_match:
                    metadata['speaker'] = self.clean_text(speaker_match.group(1))
        
        return metadata
    
    def download_and_parse_speech(self, speech_info: Dict[str, str]) -> Optional[Tuple[str, Dict[str, str]]]:
        """Download and parse a single speech."""
        url = speech_info['url']
        self.logger.info(f"Processing speech: {url}")
        
        metadata = self.extract_metadata_from_listing(speech_info['raw_text'], url)
        metadata['title'] = speech_info['title']
        
        response = self.polite_get(url)
        if not response:
            return None
        
        content_type = response.headers.get('content-type', '').lower()
        
        try:
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                content = self.extract_pdf_content(response.content)
                metadata['source_type'] = 'pdf'
            else:
                content, enhanced_metadata = self.extract_html_content(response.text, url)
                metadata.update(enhanced_metadata)
                metadata['source_type'] = 'html'
            
            if not content or len(content.strip()) < 100:
                self.logger.warning(f"Extracted content too short for {url}")
                return None
            
            return content, metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing speech {url}: {e}")
            return None
    
    def extract_pdf_content(self, pdf_bytes: bytes) -> str:
        """Extract text content from PDF bytes."""
        try:
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                content = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
                return self.clean_text(content)
        except Exception as e:
            self.logger.error(f"Error extracting PDF content: {e}")
            return ""
    
    def extract_html_content(self, html: str, url: str) -> Tuple[str, Dict[str, str]]:
        """Extract text content and enhanced metadata from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        content_selectors = [
            'main', 'article', '.content', '#content', 
            '.speech-content', '.text-content', '.body-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        content = self.clean_text(main_content.get_text())
        
        enhanced_metadata = {}
        
        title_elem = soup.find('meta', property='og:title') or soup.find('title') or soup.find('h1')
        if title_elem:
            if title_elem.name == 'meta':
                enhanced_metadata['title'] = self.clean_text(title_elem.get('content', ''))
            else:
                enhanced_metadata['title'] = self.clean_text(title_elem.get_text())
        
        return content, enhanced_metadata
    
    def save_speech_and_metadata(self, content: str, metadata: Dict[str, str]) -> str:
        """Save speech content and metadata using content hash as filename."""
        content_hash = self.get_content_hash(content)
        
        speech_file = os.path.join(self.speeches_dir, f"{content_hash}.txt")
        with open(speech_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        metadata_file = os.path.join(self.metadata_dir, f"{content_hash}.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved speech and metadata with hash: {content_hash}")
        return content_hash
    
    def run(self, method: str = "bulk", max_speeches: Optional[int] = None) -> Dict[str, int]:
        """
        Run the scraping process using the specified method.
        
        Args:
            method: "bulk", "selenium", "alternative", or "all"
            max_speeches: Maximum number of speeches to process
            
        Returns:
            Dictionary with processing statistics
        """
        self.logger.info(f"Starting BIS speech scraping with method: {method}")
        
        stats = {'total_found': 0, 'successfully_processed': 0, 'failed': 0, 'skipped': 0}
        
        if method == "bulk":
            # Try bulk download first
            if self.download_bulk_speeches():
                return self.process_bulk_speeches()
            else:
                self.logger.warning("Bulk download failed, falling back to alternative methods")
                method = "alternative"
        
        if method == "selenium":
            speech_links = self.extract_speech_links_selenium(max_speeches or 100)
        elif method == "alternative":
            speech_links = self.extract_speech_links_alternative()
        elif method == "all":
            # Try all methods
            speech_links = []
            speech_links.extend(self.extract_speech_links_selenium(max_speeches or 50))
            speech_links.extend(self.extract_speech_links_alternative())
            # Remove duplicates
            seen_urls = set()
            unique_links = []
            for link in speech_links:
                if link['url'] not in seen_urls:
                    unique_links.append(link)
                    seen_urls.add(link['url'])
            speech_links = unique_links
        else:
            self.logger.error(f"Unknown method: {method}")
            return stats
        
        stats['total_found'] = len(speech_links)
        
        if not speech_links:
            self.logger.warning("No speech links found")
            return stats
        
        if max_speeches:
            speech_links = speech_links[:max_speeches]
        
        # Process each speech
        for i, speech_info in enumerate(speech_links, 1):
            self.logger.info(f"Processing speech {i}/{len(speech_links)}")
            
            try:
                result = self.download_and_parse_speech(speech_info)
                if result:
                    content, metadata = result
                    content_hash = self.save_speech_and_metadata(content, metadata)
                    stats['successfully_processed'] += 1
                    self.logger.info(f"Successfully processed speech: {content_hash}")
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Unexpected error processing speech {speech_info['url']}: {e}")
                stats['failed'] += 1
        
        self.logger.info(f"Scraping complete. Stats: {stats}")
        return stats


def main():
    """Main function to run the enhanced BIS scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced BIS Speech Scraper")
    parser.add_argument("--method", choices=["bulk", "selenium", "alternative", "all"], 
                       default="bulk", help="Scraping method to use")
    parser.add_argument("--max-speeches", type=int, help="Maximum number of speeches to scrape")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests")
    
    args = parser.parse_args()
    
    scraper = EnhancedBISScraper(
        output_dir=args.output_dir,
        delay=args.delay
    )
    
    stats = scraper.run(method=args.method, max_speeches=args.max_speeches)
    
    print(f"Enhanced BIS scraping completed using method '{args.method}':")
    print(f"  Total found: {stats['total_found']}")
    print(f"  Successfully processed: {stats['successfully_processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()