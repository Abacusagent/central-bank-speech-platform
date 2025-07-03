#!/usr/bin/env python3
"""
Federal Reserve Plugin for Central Bank Speech Analysis Platform

This plugin implements the CentralBankScraperPlugin interface for the Federal Reserve,
following Domain-Driven Design principles and the platform's plugin architecture.

Key Features:
- Complete Federal Reserve speaker database with historical coverage
- Robust date extraction with URL-first priority
- Enhanced content extraction with multiple fallback strategies
- Comprehensive validation to ensure data quality
- Full compliance with platform plugin interface

Author: Central Bank Speech Collector
Date: 2025
Institution: Federal Reserve System (United States)
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import requests
from bs4 import BeautifulSoup
import pdfplumber

# Domain imports
from domain.entities import CentralBankSpeaker, Institution, InstitutionType
from domain.value_objects import DateRange, Url

# Interface imports
from interfaces.plugin_interfaces import (
    CentralBankScraperPlugin, SpeechMetadata, SpeechContent, Speaker,
    ValidationResult, ValidationStatus, SpeechType, SpeakerDatabase
)

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

logger = logging.getLogger(__name__)


class FederalReservePlugin(CentralBankScraperPlugin):
    """
    Federal Reserve plugin implementing the central bank scraper interface.
    
    This plugin handles the discovery, extraction, and validation of speeches
    from the Federal Reserve System, including the Board of Governors and
    the 12 Federal Reserve Banks.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Federal Reserve plugin.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.base_url = "https://www.federalreserve.gov"
        self.speeches_url = "https://www.federalreserve.gov/newsevents/speeches.htm"
        
        # Request headers for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Initialize components
        self._initialize_speaker_database()
        self._initialize_url_patterns()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'url_fallback_used': 0,
            'unknown_speakers': 0,
            'validation_failures': 0,
        }
        
        logger.info("Federal Reserve plugin initialized")
    
    def get_institution_code(self) -> str:
        """Get the institution code for the Federal Reserve."""
        return "FED"
    
    def get_institution(self) -> Institution:
        """Get the Federal Reserve institution entity."""
        return Institution(
            code="FED",
            name="Federal Reserve System",
            country="United States",
            institution_type=InstitutionType.CENTRAL_BANK,
            established_date=date(1913, 12, 23),
            website_url="https://www.federalreserve.gov",
            description="The central banking system of the United States"
        )
    
    def discover_speeches(self, date_range: DateRange) -> List[SpeechMetadata]:
        """
        Discover Federal Reserve speeches within the specified date range.
        
        Args:
            date_range: Date range to search for speeches
            
        Returns:
            List of speech metadata for discovered speeches
        """
        logger.info(f"Discovering Federal Reserve speeches for {date_range}")
        
        all_speeches = []
        
        # Method 1: Historical year-by-year scraping
        start_year = date_range.start_date.year
        end_year = date_range.end_date.year
        
        for year in range(start_year, end_year + 1):
            year_speeches = self._scrape_speeches_by_year(year)
            
            # Filter by date range
            filtered_speeches = [
                speech for speech in year_speeches
                if speech.date and date_range.contains(speech.date)
            ]
            
            all_speeches.extend(filtered_speeches)
            logger.info(f"Found {len(filtered_speeches)} speeches for {year}")
            
            # Respectful delay
            time.sleep(1)
        
        # Method 2: Current speeches page (for recent speeches)
        if date_range.end_date >= date.today():
            current_speeches = self._scrape_current_speeches_page()
            current_filtered = [
                speech for speech in current_speeches
                if speech.date and date_range.contains(speech.date)
            ]
            all_speeches.extend(current_filtered)
        
        # Remove duplicates
        unique_speeches = self._deduplicate_speeches(all_speeches)
        
        logger.info(f"Discovered {len(unique_speeches)} unique Federal Reserve speeches")
        return unique_speeches
    
    def extract_speech_content(self, speech_metadata: SpeechMetadata) -> SpeechContent:
        """
        Extract content from a Federal Reserve speech.
        
        Args:
            speech_metadata: Metadata for the speech to extract
            
        Returns:
            Extracted speech content
            
        Raises:
            Exception: If content extraction fails
        """
        logger.debug(f"Extracting content from: {speech_metadata.url}")
        
        try:
            response = requests.get(speech_metadata.url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or speech_metadata.url.lower().endswith('.pdf'):
                return self._extract_pdf_content(response.content, speech_metadata)
            else:
                return self._extract_html_content(response.text, speech_metadata)
                
        except Exception as e:
            logger.error(f"Failed to extract content from {speech_metadata.url}: {e}")
            raise
    
    def get_speaker_database(self) -> SpeakerDatabase:
        """
        Get the Federal Reserve speaker database.
        
        Returns:
            Database containing all known Federal Reserve speakers
        """
        speakers = []
        
        for speaker_data in self.speaker_roles.values():
            # Skip duplicate entries (we use the most complete name as primary)
            if speaker_data.get('is_primary', True):
                speaker = CentralBankSpeaker(
                    id=uuid4(),
                    name=speaker_data['name'],
                    role=speaker_data['role'],
                    institution=self.get_institution(),
                    voting_member=speaker_data.get('voting_status') == 'Voting Member',
                    biographical_notes=f"Service years: {speaker_data.get('years', 'Unknown')}"
                )
                
                # Add alternate names
                for alt_name in speaker_data.get('alternate_names', []):
                    speaker.add_alternate_name(alt_name)
                
                speakers.append(speaker)
        
        return SpeakerDatabase(speakers=speakers)
    
    def validate_speech_authenticity(self, speech_metadata: SpeechMetadata, 
                                   content: SpeechContent) -> ValidationResult:
        """
        Validate that a speech is authentic Federal Reserve content.
        
        Args:
            speech_metadata: Speech metadata to validate
            content: Speech content to validate
            
        Returns:
            Validation result with confidence score and any issues
        """
        issues = []
        confidence = 1.0
        
        # Validate URL domain
        if not speech_metadata.url.startswith('https://www.federalreserve.gov'):
            issues.append("URL is not from federalreserve.gov domain")
            confidence -= 0.3
        
        # Validate content length
        if len(content.cleaned_text.strip()) < 200:
            issues.append("Content is too short to be a substantial speech")
            confidence -= 0.4
        
        # Check for placeholder content
        placeholder_indicators = [
            'lorem ipsum', 'placeholder', 'test content', 'coming soon',
            'under construction', 'page not found'
        ]
        
        content_lower = content.cleaned_text.lower()
        if any(indicator in content_lower for indicator in placeholder_indicators):
            issues.append("Content appears to be placeholder text")
            confidence -= 0.5
        
        # Validate speaker recognition
        speaker_info = self._get_speaker_info(speech_metadata.speaker_name, speech_metadata.url)
        if speaker_info.get('source') == 'unknown':
            issues.append("Speaker could not be identified in Federal Reserve database")
            confidence -= 0.2
        
        # Validate date reasonableness
        if speech_metadata.date:
            if speech_metadata.date < date(1913, 1, 1):  # Before Fed establishment
                issues.append("Speech date is before Federal Reserve establishment")
                confidence -= 0.3
            elif speech_metadata.date > date.today():
                issues.append("Speech date is in the future")
                confidence -= 0.2
        
        # Check for Federal Reserve terminology
        fed_terms = [
            'federal reserve', 'monetary policy', 'interest rate', 'inflation',
            'employment', 'fomc', 'federal open market committee', 'central bank'
        ]
        
        if not any(term in content_lower for term in fed_terms):
            issues.append("Content lacks typical Federal Reserve terminology")
            confidence -= 0.1
        
        # Determine validation status
        if confidence >= 0.7:
            status = ValidationStatus.VALID
        elif confidence >= 0.4:
            status = ValidationStatus.QUESTIONABLE
        else:
            status = ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            confidence=max(0.0, confidence),
            issues=issues,
            metadata={
                'plugin': 'federal_reserve',
                'validation_timestamp': datetime.now().isoformat(),
                'speaker_recognition_source': speaker_info.get('source', 'unknown')
            }
        )
    
    # Internal methods for speech discovery and extraction
    
    def _scrape_speeches_by_year(self, year: int) -> List[SpeechMetadata]:
        """Scrape speeches from a specific year's page."""
        year_url = f"https://www.federalreserve.gov/newsevents/speech/{year}-speeches.htm"
        
        try:
            response = requests.get(year_url, headers=self.headers, timeout=30)
            
            if response.status_code == 404:
                logger.debug(f"No speeches page found for {year}")
                return []
            
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            return self._extract_speeches_from_page(soup, year)
            
        except Exception as e:
            logger.error(f"Error scraping speeches for {year}: {e}")
            return []
    
    def _scrape_current_speeches_page(self) -> List[SpeechMetadata]:
        """Scrape the current speeches page for recent speeches."""
        try:
            response = requests.get(self.speeches_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return self._extract_speeches_from_page(soup)
            
        except Exception as e:
            logger.error(f"Error scraping current speeches page: {e}")
            return []
    
    def _extract_speeches_from_page(self, soup: BeautifulSoup, year: Optional[int] = None) -> List[SpeechMetadata]:
        """Extract speech metadata from a BeautifulSoup page."""
        speeches = []
        
        # Find all speech links
        speech_links = soup.find_all('a', href=True)
        
        for link in speech_links:
            href = link.get('href')
            
            # Check if this is a speech link
            if not (href and '/newsevents/speech/' in href and href.endswith('.htm')):
                continue
            
            # Skip year navigation links
            if '-speeches.htm' in href:
                continue
            
            full_url = urljoin(self.base_url, href)
            
            # Extract metadata from link context
            metadata = self._extract_metadata_from_link_context(link, full_url, year)
            if metadata:
                speeches.append(metadata)
        
        return speeches
    
    def _extract_metadata_from_link_context(self, link, url: str, year: Optional[int] = None) -> Optional[SpeechMetadata]:
        """Extract speech metadata from link and its context."""
        try:
            title = link.get_text(strip=True)
            
            if not title or len(title) < 5:
                return None
            
            # Extract date from URL (most reliable method)
            speech_date = self._extract_date_from_url(url)
            
            # Look for speaker and date in parent elements
            speaker_name = ""
            date_from_context = None
            
            parent = link.parent
            for level in range(3):
                if parent is None:
                    break
                
                parent_text = parent.get_text()
                
                # Look for speaker patterns
                if not speaker_name:
                    speaker_patterns = [
                        r'\b(?:Chair|Governor|President|Vice Chair)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                        r'\b(?:By|Remarks by|Speech by)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*)',
                        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]*)*),?\s+(?:Chair|Governor|President|Vice Chair)'
                    ]
                    
                    for pattern in speaker_patterns:
                        speaker_match = re.search(pattern, parent_text)
                        if speaker_match:
                            speaker_name = self._clean_speaker_name(speaker_match.group(1))
                            break
                
                # Look for date patterns (if URL date extraction failed)
                if not speech_date:
                    date_match = re.search(r'\b(\w+\s+\d{1,2},\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', parent_text)
                    if date_match:
                        date_from_context = date_match.group(1)
                
                parent = parent.parent
            
            # Use URL date if available, otherwise parse context date
            if not speech_date and date_from_context:
                speech_date = self._parse_date_string(date_from_context)
            
            # Try URL speaker extraction if no speaker found in context
            if not speaker_name:
                speaker_name = self._extract_speaker_from_url(url) or ""
            
            # Only return metadata if we have a valid date
            if not speech_date:
                logger.warning(f"Could not extract valid date for speech: {url}")
                return None
            
            return SpeechMetadata(
                url=url,
                title=title,
                speaker_name=speaker_name,
                date=speech_date,
                institution_code="FED",
                speech_type=SpeechType.FORMAL_SPEECH,
                language="en"
            )
            
        except Exception as e:
            logger.debug(f"Error extracting metadata from link: {e}")
            return None
    
    def _extract_html_content(self, html: str, metadata: SpeechMetadata) -> SpeechContent:
        """Extract content from HTML speech page."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.navigation', 'noscript']):
            element.decompose()
        
        # Try multiple content extraction strategies
        content_candidates = []
        
        # Strategy 1: Fed-specific selectors
        fed_selectors = [
            'div#article',
            '.article-content',
            '.speech-content',
            '[role="main"]',
            'main',
            'article',
            '.col-md-8',
            '.content-area',
            '#main-content'
        ]
        
        for selector in fed_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
                if len(text) > 500:
                    content_candidates.append(('fed_selector', text, len(text)))
        
        # Strategy 2: Paragraph aggregation
        paragraphs = soup.find_all('p')
        if paragraphs:
            para_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            if len(para_text) > 500:
                content_candidates.append(('paragraphs', para_text, len(para_text)))
        
        # Choose the best candidate
        if content_candidates:
            content_candidates.sort(key=lambda x: x[2], reverse=True)
            raw_text = content_candidates[0][1]
        else:
            # Fallback to body text
            body = soup.find('body')
            raw_text = body.get_text(separator='\n', strip=True) if body else ""
        
        # Clean the text
        cleaned_text = self._clean_text_content(raw_text)
        
        # Calculate word count
        word_count = len(cleaned_text.split()) if cleaned_text else 0
        
        return SpeechContent(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            extraction_method="html_multi_strategy",
            confidence_score=0.9 if content_candidates else 0.6,
            word_count=word_count,
            extraction_timestamp=datetime.now()
        )
    
    def _extract_pdf_content(self, pdf_bytes: bytes, metadata: SpeechMetadata) -> SpeechContent:
        """Extract content from PDF speech."""
        try:
            import io
            
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if not text:
                    raise Exception("No text extracted from PDF")
                
                cleaned_text = self._clean_text_content(text)
                word_count = len(cleaned_text.split()) if cleaned_text else 0
                
                return SpeechContent(
                    raw_text=text,
                    cleaned_text=cleaned_text,
                    extraction_method="pdf_pdfplumber",
                    confidence_score=0.8,
                    word_count=word_count,
                    extraction_timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise
    
    def _clean_text_content(self, text: str) -> str:
        """Clean and normalize extracted text content."""
        if not text:
            return ""
        
        # Basic whitespace normalization
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        # Remove specific Fed boilerplate patterns
        boilerplate_patterns = [
            r'Return to top\s*',
            r'Last update:.*?\n',
            r'Skip to main content\s*',
            r'Print this page\s*',
            r'Board of Governors of the Federal Reserve System.*?(?=\n|$)',
            r'Federal Reserve Bank of.*?(?=\n|$)',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Final cleanup
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        return text.strip()
    
    # Date and speaker extraction utilities
    
    def _extract_date_from_url(self, url: str) -> Optional[date]:
        """Extract date from Fed URL with high confidence."""
        if not url:
            return None
        
        try:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            name_part = os.path.splitext(filename)[0]
            
            # Fed URL date patterns (YYYYMMDD format)
            date_patterns = [
                r'[a-zA-Z]+(\d{8})[a-z]?$',
                r'[a-zA-Z]+[-_](\d{8})[a-z]?$',
                r'^(\d{8})[a-z]?$',
                r'[a-zA-Z]+(\d{6})[a-z]?$'  # YYMMDD format
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, name_part)
                if match:
                    date_str = match.group(1)
                    
                    # Handle 6-digit dates (YYMMDD -> YYYYMMDD)
                    if len(date_str) == 6:
                        year = int(date_str[:2])
                        if year <= 30:
                            date_str = "20" + date_str
                        else:
                            date_str = "19" + date_str
                    
                    # Validate and create date
                    if len(date_str) == 8:
                        try:
                            year = int(date_str[:4])
                            month = int(date_str[4:6])
                            day = int(date_str[6:8])
                            
                            if 1990 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                                return date(year, month, day)
                        except ValueError:
                            continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting date from URL {url}: {e}")
            return None
    
    def _extract_speaker_from_url(self, url: str) -> Optional[str]:
        """Extract speaker name from Fed URL."""
        if not url:
            return None
        
        try:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            name_part = os.path.splitext(filename)[0]
            
            # Fed URL speaker patterns
            patterns = [
                r'^([a-zA-Z]+)[-_]?\d{8}',
                r'^([a-zA-Z]+)[-_]?\d{6}',
                r'^([a-zA-Z]+)[-_]?',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, name_part, re.IGNORECASE)
                if match:
                    candidate_name = match.group(1).lower()
                    
                    if candidate_name in self.url_name_patterns:
                        matched_names = self.url_name_patterns[candidate_name]
                        self.stats['url_fallback_used'] += 1
                        return matched_names[0]
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting speaker from URL {url}: {e}")
            return None
    
    def _parse_date_string(self, date_str: str) -> Optional[date]:
        """Parse a date string into a date object."""
        if not date_str:
            return None
        
        formats = [
            '%B %d, %Y',   # "January 15, 2025"
            '%b %d, %Y',   # "Jan 15, 2025"
            '%m/%d/%Y',    # "01/15/2025"
            '%m-%d-%Y',    # "01-15-2025"
            '%Y-%m-%d',    # "2025-01-15"
            '%B %d %Y',    # "January 15 2025"
            '%b %d %Y',    # "Jan 15 2025"
            '%Y%m%d',      # "20250115"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                if 1990 <= dt.year <= 2030:
                    return dt.date()
            except ValueError:
                continue
        
        return None
    
    def _clean_speaker_name(self, raw_name: str) -> str:
        """Clean and normalize speaker name."""
        if not raw_name:
            return ""
        
        # Remove titles and clean
        name = re.sub(r'\b(?:Governor|Chair|President|Vice Chair|Dr\.|Mr\.|Ms\.|Mrs\.)\s*', '', raw_name, flags=re.IGNORECASE)
        name = re.split(r'\s*(?:,|by|remarks|speech)\s*', name, flags=re.IGNORECASE)[0]
        name = ' '.join(name.split()).strip()
        
        return name if len(name) >= 2 else ""
    
    def _get_speaker_info(self, speaker_name: str, url: str = None) -> Dict[str, str]:
        """Get speaker information with fallback mechanisms."""
        if not speaker_name:
            if url:
                speaker_name = self._extract_speaker_from_url(url) or ""
            if not speaker_name:
                self.stats['unknown_speakers'] += 1
                return {"role": "Unknown", "voting_status": "Unknown", "source": "unknown"}
        
        # Normalize name for lookup
        normalized_name = speaker_name.lower().strip().replace('.', '')
        
        # Try exact match
        if normalized_name in self.speaker_roles:
            info = self.speaker_roles[normalized_name].copy()
            info['source'] = 'exact_match'
            return info
        
        # Try partial matching
        for known_name, info in self.speaker_roles.items():
            if self._names_match(normalized_name, known_name):
                result = info.copy()
                result['source'] = 'partial_match'
                return result
        
        # Try last name matching
        last_name = normalized_name.split()[-1] if ' ' in normalized_name else normalized_name
        if last_name in self.speaker_roles:
            result = self.speaker_roles[last_name].copy()
            result['source'] = 'lastname_match'
            return result
        
        self.stats['unknown_speakers'] += 1
        return {"role": "Unknown", "voting_status": "Unknown", "source": "unknown"}
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names likely refer to the same person."""
        if not name1 or not name2:
            return False
        
        parts1 = set(name1.replace('.', '').split())
        parts2 = set(name2.replace('.', '').split())
        
        # Remove short parts (initials)
        parts1 = {p for p in parts1 if len(p) > 1}
        parts2 = {p for p in parts2 if len(p) > 1}
        
        common_parts = parts1.intersection(parts2)
        
        if len(parts1) <= 2 and len(parts2) <= 2:
            return len(common_parts) >= min(len(parts1), len(parts2))
        else:
            return len(common_parts) >= 2
    
    def _deduplicate_speeches(self, speeches: List[SpeechMetadata]) -> List[SpeechMetadata]:
        """Remove duplicate speeches based on URL."""
        unique_speeches = []
        seen_urls = set()
        
        for speech in speeches:
            if speech.url not in seen_urls:
                unique_speeches.append(speech)
                seen_urls.add(speech.url)
        
        return unique_speeches
    
    # Speaker database initialization
    
    def _initialize_speaker_database(self):
        """Initialize comprehensive Federal Reserve officials database."""
        # Complete speaker database from the original implementation
        self.speaker_roles = {
            # Current Fed Leadership (2024-2025)
            "jerome powell": {"name": "Jerome Powell", "role": "Chair", "voting_status": "Voting Member", "years": "2018-present", "is_primary": True},
            "jerome h. powell": {"name": "Jerome Powell", "role": "Chair", "voting_status": "Voting Member", "years": "2018-present", "is_primary": False},
            "jerome h powell": {"name": "Jerome Powell", "role": "Chair", "voting_status": "Voting Member", "years": "2018-present", "is_primary": False},
            "powell": {"name": "Jerome Powell", "role": "Chair", "voting_status": "Voting Member", "years": "2018-present", "is_primary": False},
            
            "philip jefferson": {"name": "Philip Jefferson", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2022-present", "is_primary": True},
            "philip n. jefferson": {"name": "Philip Jefferson", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2022-present", "is_primary": False},
            "philip n jefferson": {"name": "Philip Jefferson", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2022-present", "is_primary": False},
            "jefferson": {"name": "Philip Jefferson", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2022-present", "is_primary": False},
            
            "michael barr": {"name": "Michael Barr", "role": "Vice Chair for Supervision", "voting_status": "Voting Member", "years": "2022-present", "is_primary": True},
            "michael s. barr": {"name": "Michael Barr", "role": "Vice Chair for Supervision", "voting_status": "Voting Member", "years": "2022-present", "is_primary": False},
            "michael s barr": {"name": "Michael Barr", "role": "Vice Chair for Supervision", "voting_status": "Voting Member", "years": "2022-present", "is_primary": False},
            "barr": {"name": "Michael Barr", "role": "Vice Chair for Supervision", "voting_status": "Voting Member", "years": "2022-present", "is_primary": False},
            
            # Current Board of Governors
            "lisa cook": {"name": "Lisa Cook", "role": "Governor", "voting_status": "Voting Member", "years": "2022-present", "is_primary": True},
            "lisa d. cook": {"name": "Lisa Cook", "role": "Governor", "voting_status": "Voting Member", "years": "2022-present", "is_primary": False},
            "lisa d cook": {"name": "Lisa Cook", "role": "Governor", "voting_status": "Voting Member", "years": "2022-present", "is_primary": False},
            "cook": {"name": "Lisa Cook", "role": "Governor", "voting_status": "Voting Member", "years": "2022-present", "is_primary": False},
            
            "adriana kugler": {"name": "Adriana Kugler", "role": "Governor", "voting_status": "Voting Member", "years": "2023-present", "is_primary": True},
            "adriana d. kugler": {"name": "Adriana Kugler", "role": "Governor", "voting_status": "Voting Member", "years": "2023-present", "is_primary": False},
            "adriana d kugler": {"name": "Adriana Kugler", "role": "Governor", "voting_status": "Voting Member", "years": "2023-present", "is_primary": False},
            "kugler": {"name": "Adriana Kugler", "role": "Governor", "voting_status": "Voting Member", "years": "2023-present", "is_primary": False},
            
            "christopher waller": {"name": "Christopher Waller", "role": "Governor", "voting_status": "Voting Member", "years": "2020-present", "is_primary": True},
            "christopher j. waller": {"name": "Christopher Waller", "role": "Governor", "voting_status": "Voting Member", "years": "2020-present", "is_primary": False},
            "christopher j waller": {"name": "Christopher Waller", "role": "Governor", "voting_status": "Voting Member", "years": "2020-present", "is_primary": False},
            "waller": {"name": "Christopher Waller", "role": "Governor", "voting_status": "Voting Member", "years": "2020-present", "is_primary": False},
            
            "michelle bowman": {"name": "Michelle Bowman", "role": "Governor", "voting_status": "Voting Member", "years": "2018-present", "is_primary": True},
            "michelle w. bowman": {"name": "Michelle Bowman", "role": "Governor", "voting_status": "Voting Member", "years": "2018-present", "is_primary": False},
            "michelle w bowman": {"name": "Michelle Bowman", "role": "Governor", "voting_status": "Voting Member", "years": "2018-present", "is_primary": False},
            "bowman": {"name": "Michelle Bowman", "role": "Governor", "voting_status": "Voting Member", "years": "2018-present", "is_primary": False},
            
            # Past Fed Chairs
            "janet yellen": {"name": "Janet Yellen", "role": "Chair", "voting_status": "Voting Member", "years": "2014-2018", "is_primary": True},
            "janet l. yellen": {"name": "Janet Yellen", "role": "Chair", "voting_status": "Voting Member", "years": "2014-2018", "is_primary": False},
            "janet l yellen": {"name": "Janet Yellen", "role": "Chair", "voting_status": "Voting Member", "years": "2014-2018", "is_primary": False},
            "yellen": {"name": "Janet Yellen", "role": "Chair", "voting_status": "Voting Member", "years": "2014-2018", "is_primary": False},
            
            "ben bernanke": {"name": "Ben Bernanke", "role": "Chair", "voting_status": "Voting Member", "years": "2006-2014", "is_primary": True},
            "ben s. bernanke": {"name": "Ben Bernanke", "role": "Chair", "voting_status": "Voting Member", "years": "2006-2014", "is_primary": False},
            "ben s bernanke": {"name": "Ben Bernanke", "role": "Chair", "voting_status": "Voting Member", "years": "2006-2014", "is_primary": False},
            "bernanke": {"name": "Ben Bernanke", "role": "Chair", "voting_status": "Voting Member", "years": "2006-2014", "is_primary": False},
            
            "alan greenspan": {"name": "Alan Greenspan", "role": "Chair", "voting_status": "Voting Member", "years": "1987-2006", "is_primary": True},
            "greenspan": {"name": "Alan Greenspan", "role": "Chair", "voting_status": "Voting Member", "years": "1987-2006", "is_primary": False},
            
            "paul volcker": {"name": "Paul Volcker", "role": "Chair", "voting_status": "Voting Member", "years": "1979-1987", "is_primary": True},
            "volcker": {"name": "Paul Volcker", "role": "Chair", "voting_status": "Voting Member", "years": "1979-1987", "is_primary": False},
            
            # Past Vice Chairs
            "lael brainard": {"name": "Lael Brainard", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2022-2023", "is_primary": True},
            "brainard": {"name": "Lael Brainard", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2022-2023", "is_primary": False},
            
            "richard clarida": {"name": "Richard Clarida", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2018-2022", "is_primary": True},
            "richard h. clarida": {"name": "Richard Clarida", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2018-2022", "is_primary": False},
            "richard h clarida": {"name": "Richard Clarida", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2018-2022", "is_primary": False},
            "clarida": {"name": "Richard Clarida", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2018-2022", "is_primary": False},
            
            "randal quarles": {"name": "Randal Quarles", "role": "Vice Chair for Supervision", "voting_status": "Voting Member", "years": "2017-2021", "is_primary": True},
            "randal k. quarles": {"name": "Randal Quarles", "role": "Vice Chair for Supervision", "voting_status": "Voting Member", "years": "2017-2021", "is_primary": False},
            "randal k quarles": {"name": "Randal Quarles", "role": "Vice Chair for Supervision", "voting_status": "Voting Member", "years": "2017-2021", "is_primary": False},
            "quarles": {"name": "Randal Quarles", "role": "Vice Chair for Supervision", "voting_status": "Voting Member", "years": "2017-2021", "is_primary": False},
            
            "stanley fischer": {"name": "Stanley Fischer", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2014-2017", "is_primary": True},
            "fischer": {"name": "Stanley Fischer", "role": "Vice Chair", "voting_status": "Voting Member", "years": "2014-2017", "is_primary": False},
            
            # Current Federal Reserve Bank Presidents
            # New York Fed (always voting)
            "john williams": {"name": "John Williams", "role": "President", "voting_status": "Always Voting", "bank": "New York", "years": "2018-present", "is_primary": True},
            "john c. williams": {"name": "John Williams", "role": "President", "voting_status": "Always Voting", "bank": "New York", "years": "2018-present", "is_primary": False},
            "john c williams": {"name": "John Williams", "role": "President", "voting_status": "Always Voting", "bank": "New York", "years": "2018-present", "is_primary": False},
            "williams": {"name": "John Williams", "role": "President", "voting_status": "Always Voting", "bank": "New York", "years": "2018-present", "is_primary": False},
            
            # Regional Fed Presidents (rotating voting)
            "austan goolsbee": {"name": "Austan Goolsbee", "role": "President", "voting_status": "Rotating Voter", "bank": "Chicago", "years": "2023-present", "is_primary": True},
            "austan d. goolsbee": {"name": "Austan Goolsbee", "role": "President", "voting_status": "Rotating Voter", "bank": "Chicago", "years": "2023-present", "is_primary": False},
            "goolsbee": {"name": "Austan Goolsbee", "role": "President", "voting_status": "Rotating Voter", "bank": "Chicago", "years": "2023-present", "is_primary": False},
            
            "neel kashkari": {"name": "Neel Kashkari", "role": "President", "voting_status": "Rotating Voter", "bank": "Minneapolis", "years": "2016-present", "is_primary": True},
            "kashkari": {"name": "Neel Kashkari", "role": "President", "voting_status": "Rotating Voter", "bank": "Minneapolis", "years": "2016-present", "is_primary": False},
            
            "raphael bostic": {"name": "Raphael Bostic", "role": "President", "voting_status": "Rotating Voter", "bank": "Atlanta", "years": "2017-present", "is_primary": True},
            "bostic": {"name": "Raphael Bostic", "role": "President", "voting_status": "Rotating Voter", "bank": "Atlanta", "years": "2017-present", "is_primary": False},
            
            "mary daly": {"name": "Mary Daly", "role": "President", "voting_status": "Rotating Voter", "bank": "San Francisco", "years": "2018-present", "is_primary": True},
            "mary c. daly": {"name": "Mary Daly", "role": "President", "voting_status": "Rotating Voter", "bank": "San Francisco", "years": "2018-present", "is_primary": False},
            "daly": {"name": "Mary Daly", "role": "President", "voting_status": "Rotating Voter", "bank": "San Francisco", "years": "2018-present", "is_primary": False},
            
            "beth hammack": {"name": "Beth Hammack", "role": "President", "voting_status": "Rotating Voter", "bank": "Cleveland", "years": "2024-present", "is_primary": True},
            "beth m. hammack": {"name": "Beth Hammack", "role": "President", "voting_status": "Rotating Voter", "bank": "Cleveland", "years": "2024-present", "is_primary": False},
            "hammack": {"name": "Beth Hammack", "role": "President", "voting_status": "Rotating Voter", "bank": "Cleveland", "years": "2024-present", "is_primary": False},
            
            "thomas barkin": {"name": "Thomas Barkin", "role": "President", "voting_status": "Rotating Voter", "bank": "Richmond", "years": "2018-present", "is_primary": True},
            "thomas i. barkin": {"name": "Thomas Barkin", "role": "President", "voting_status": "Rotating Voter", "bank": "Richmond", "years": "2018-present", "is_primary": False},
            "barkin": {"name": "Thomas Barkin", "role": "President", "voting_status": "Rotating Voter", "bank": "Richmond", "years": "2018-present", "is_primary": False},
            
            "anna paulson": {"name": "Anna Paulson", "role": "President", "voting_status": "Rotating Voter", "bank": "Philadelphia", "years": "2024-present", "is_primary": True},
            "paulson": {"name": "Anna Paulson", "role": "President", "voting_status": "Rotating Voter", "bank": "Philadelphia", "years": "2024-present", "is_primary": False},
            
            "susan collins": {"name": "Susan Collins", "role": "President", "voting_status": "Rotating Voter", "bank": "Boston", "years": "2021-present", "is_primary": True},
            "susan m. collins": {"name": "Susan Collins", "role": "President", "voting_status": "Rotating Voter", "bank": "Boston", "years": "2021-present", "is_primary": False},
            "collins": {"name": "Susan Collins", "role": "President", "voting_status": "Rotating Voter", "bank": "Boston", "years": "2021-present", "is_primary": False},
            
            "alberto musalem": {"name": "Alberto Musalem", "role": "President", "voting_status": "Rotating Voter", "bank": "St. Louis", "years": "2023-present", "is_primary": True},
            "alberto g. musalem": {"name": "Alberto Musalem", "role": "President", "voting_status": "Rotating Voter", "bank": "St. Louis", "years": "2023-present", "is_primary": False},
            "musalem": {"name": "Alberto Musalem", "role": "President", "voting_status": "Rotating Voter", "bank": "St. Louis", "years": "2023-present", "is_primary": False},
            
            "lorie logan": {"name": "Lorie Logan", "role": "President", "voting_status": "Rotating Voter", "bank": "Dallas", "years": "2022-present", "is_primary": True},
            "lorie k. logan": {"name": "Lorie Logan", "role": "President", "voting_status": "Rotating Voter", "bank": "Dallas", "years": "2022-present", "is_primary": False},
            "logan": {"name": "Lorie Logan", "role": "President", "voting_status": "Rotating Voter", "bank": "Dallas", "years": "2022-present", "is_primary": False},
            
            # Past Notable Fed Officials
            "william dudley": {"name": "William Dudley", "role": "President", "voting_status": "Always Voting", "bank": "New York", "years": "2009-2018", "is_primary": True},
            "william c. dudley": {"name": "William Dudley", "role": "President", "voting_status": "Always Voting", "bank": "New York", "years": "2009-2018", "is_primary": False},
            "dudley": {"name": "William Dudley", "role": "President", "voting_status": "Always Voting", "bank": "New York", "years": "2009-2018", "is_primary": False},
            
            "charles evans": {"name": "Charles Evans", "role": "President", "voting_status": "Rotating Voter", "bank": "Chicago", "years": "2007-2023", "is_primary": True},
            "charles l. evans": {"name": "Charles Evans", "role": "President", "voting_status": "Rotating Voter", "bank": "Chicago", "years": "2007-2023", "is_primary": False},
            "evans": {"name": "Charles Evans", "role": "President", "voting_status": "Rotating Voter", "bank": "Chicago", "years": "2007-2023", "is_primary": False},
            
            "jim bullard": {"name": "James Bullard", "role": "President", "voting_status": "Rotating Voter", "bank": "St. Louis", "years": "2008-2023", "is_primary": True},
            "james bullard": {"name": "James Bullard", "role": "President", "voting_status": "Rotating Voter", "bank": "St. Louis", "years": "2008-2023", "is_primary": False},
            "james b. bullard": {"name": "James Bullard", "role": "President", "voting_status": "Rotating Voter", "bank": "St. Louis", "years": "2008-2023", "is_primary": False},
            "bullard": {"name": "James Bullard", "role": "President", "voting_status": "Rotating Voter", "bank": "St. Louis", "years": "2008-2023", "is_primary": False},
            
            "esther george": {"name": "Esther George", "role": "President", "voting_status": "Rotating Voter", "bank": "Kansas City", "years": "2011-2023", "is_primary": True},
            "esther l. george": {"name": "Esther George", "role": "President", "voting_status": "Rotating Voter", "bank": "Kansas City", "years": "2011-2023", "is_primary": False},
            "george": {"name": "Esther George", "role": "President", "voting_status": "Rotating Voter", "bank": "Kansas City", "years": "2011-2023", "is_primary": False},
            
            "loretta mester": {"name": "Loretta Mester", "role": "President", "voting_status": "Rotating Voter", "bank": "Cleveland", "years": "2014-2024", "is_primary": True},
            "loretta j. mester": {"name": "Loretta Mester", "role": "President", "voting_status": "Rotating Voter", "bank": "Cleveland", "years": "2014-2024", "is_primary": False},
            "mester": {"name": "Loretta Mester", "role": "President", "voting_status": "Rotating Voter", "bank": "Cleveland", "years": "2014-2024", "is_primary": False},
            
            "robert kaplan": {"name": "Robert Kaplan", "role": "President", "voting_status": "Rotating Voter", "bank": "Dallas", "years": "2015-2021", "is_primary": True},
            "robert s. kaplan": {"name": "Robert Kaplan", "role": "President", "voting_status": "Rotating Voter", "bank": "Dallas", "years": "2015-2021", "is_primary": False},
            "kaplan": {"name": "Robert Kaplan", "role": "President", "voting_status": "Rotating Voter", "bank": "Dallas", "years": "2015-2021", "is_primary": False},
            
            "eric rosengren": {"name": "Eric Rosengren", "role": "President", "voting_status": "Rotating Voter", "bank": "Boston", "years": "2007-2021", "is_primary": True},
            "eric s. rosengren": {"name": "Eric Rosengren", "role": "President", "voting_status": "Rotating Voter", "bank": "Boston", "years": "2007-2021", "is_primary": False},
            "rosengren": {"name": "Eric Rosengren", "role": "President", "voting_status": "Rotating Voter", "bank": "Boston", "years": "2007-2021", "is_primary": False},
            
            "patrick harker": {"name": "Patrick Harker", "role": "President", "voting_status": "Rotating Voter", "bank": "Philadelphia", "years": "2015-2024", "is_primary": True},
            "patrick t. harker": {"name": "Patrick Harker", "role": "President", "voting_status": "Rotating Voter", "bank": "Philadelphia", "years": "2015-2024", "is_primary": False},
            "harker": {"name": "Patrick Harker", "role": "President", "voting_status": "Rotating Voter", "bank": "Philadelphia", "years": "2015-2024", "is_primary": False},
            
            # Historical Board Members
            "daniel tarullo": {"name": "Daniel Tarullo", "role": "Governor", "voting_status": "Voting Member", "years": "2009-2017", "is_primary": True},
            "daniel k. tarullo": {"name": "Daniel Tarullo", "role": "Governor", "voting_status": "Voting Member", "years": "2009-2017", "is_primary": False},
            "tarullo": {"name": "Daniel Tarullo", "role": "Governor", "voting_status": "Voting Member", "years": "2009-2017", "is_primary": False},
            
            "jeremy stein": {"name": "Jeremy Stein", "role": "Governor", "voting_status": "Voting Member", "years": "2012-2014", "is_primary": True},
            "jeremy c. stein": {"name": "Jeremy Stein", "role": "Governor", "voting_status": "Voting Member", "years": "2012-2014", "is_primary": False},
            "stein": {"name": "Jeremy Stein", "role": "Governor", "voting_status": "Voting Member", "years": "2012-2014", "is_primary": False},
        }

    def _initialize_url_patterns(self):
        """Initialize URL-based name patterns for fallback extraction."""
        self.url_name_patterns = {}
        
        # Build reverse mapping from speaker database for URL extraction
        for speaker_data in self.speaker_roles.values():
            if not speaker_data.get('is_primary', True):
                continue
                
            name = speaker_data['name'].lower()
            
            # Extract last name for URL matching
            if ' ' in name:
                last_name = name.split()[-1]
                if last_name not in self.url_name_patterns:
                    self.url_name_patterns[last_name] = []
                self.url_name_patterns[last_name].append(speaker_data['name'])
            
            # Add full name variants for URL matching
            url_friendly = name.replace(' ', '').replace('.', '')
            if url_friendly not in self.url_name_patterns:
                self.url_name_patterns[url_friendly] = []
            self.url_name_patterns[url_friendly].append(speaker_data['name'])
        
        # Add common manual patterns for better URL extraction
        manual_patterns = {
            'powell': ['Jerome Powell'],
            'jefferson': ['Philip Jefferson'],
            'barr': ['Michael Barr'],
            'cook': ['Lisa Cook'],
            'kugler': ['Adriana Kugler'],
            'waller': ['Christopher Waller'],
            'bowman': ['Michelle Bowman'],
            'yellen': ['Janet Yellen'],
            'bernanke': ['Ben Bernanke'],
            'greenspan': ['Alan Greenspan'],
            'volcker': ['Paul Volcker'],
            'brainard': ['Lael Brainard'],
            'clarida': ['Richard Clarida'],
            'quarles': ['Randal Quarles'],
            'fischer': ['Stanley Fischer'],
            'williams': ['John Williams'],
            'goolsbee': ['Austan Goolsbee'],
            'kashkari': ['Neel Kashkari'],
            'bostic': ['Raphael Bostic'],
            'daly': ['Mary Daly'],
            'hammack': ['Beth Hammack'],
            'barkin': ['Thomas Barkin'],
            'paulson': ['Anna Paulson'],
            'collins': ['Susan Collins'],
            'musalem': ['Alberto Musalem'],
            'logan': ['Lorie Logan'],
            'dudley': ['William Dudley'],
            'evans': ['Charles Evans'],
            'bullard': ['James Bullard'],
            'george': ['Esther George'],
            'mester': ['Loretta Mester'],
            'kaplan': ['Robert Kaplan'],
            'rosengren': ['Eric Rosengren'],
            'harker': ['Patrick Harker'],
            'tarullo': ['Daniel Tarullo'],
            'stein': ['Jeremy Stein'],
        }
        
        # Update with manual patterns
        for pattern, names in manual_patterns.items():
            if pattern not in self.url_name_patterns:
                self.url_name_patterns[pattern] = []
            self.url_name_patterns[pattern].extend(names)
            # Remove duplicates
            self.url_name_patterns[pattern] = list(set(self.url_name_patterns[pattern]))


# Plugin factory function for easy instantiation
def create_federal_reserve_plugin(config: Optional[Dict] = None) -> FederalReservePlugin:
    """
    Factory function to create a Federal Reserve plugin instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FederalReservePlugin instance
        
    Example:
        >>> plugin = create_federal_reserve_plugin()
        >>> assert plugin.get_institution_code() == "FED"
        >>> 
        >>> date_range = DateRange.year(2024)
        >>> speeches = plugin.discover_speeches(date_range)
        >>> print(f"Found {len(speeches)} speeches for 2024")
    """
    return FederalReservePlugin(config)


# Plugin registration for the platform
PLUGIN_CLASS = FederalReservePlugin
PLUGIN_FACTORY = create_federal_reserve_plugin

# Plugin metadata for the platform
PLUGIN_METADATA = {
    'name': 'Federal Reserve System',
    'code': 'FED',
    'country': 'United States',
    'description': 'Plugin for scraping speeches from the Federal Reserve System',
    'supported_languages': ['en'],
    'date_range': {
        'start': '1996-01-01',  # Approximate start of online speech archives
        'end': None  # Current
    },
    'capabilities': [
        'speech_discovery',
        'content_extraction',
        'speaker_recognition',
        'date_extraction',
        'validation'
    ],
    'version': '2.0.0',
    'author': 'Central Bank Speech Collector',
    'requires': [
        'requests',
        'beautifulsoup4',
        'pdfplumber',
        'lxml'
    ],
    'optional_requires': [
        'selenium'  # For dynamic content scraping
    ],
    'config_schema': {
        'base_url': {
            'type': 'string',
            'default': 'https://www.federalreserve.gov',
            'description': 'Base URL for the Federal Reserve website'
        },
        'request_delay': {
            'type': 'float',
            'default': 1.0,
            'description': 'Delay between requests in seconds'
        },
        'max_retries': {
            'type': 'integer',
            'default': 3,
            'description': 'Maximum number of retries for failed requests'
        },
        'timeout': {
            'type': 'integer',
            'default': 30,
            'description': 'Request timeout in seconds'
        },
        'use_selenium': {
            'type': 'boolean',
            'default': False,
            'description': 'Use Selenium for dynamic content scraping'
        },
        'validate_ssl': {
            'type': 'boolean',
            'default': True,
            'description': 'Validate SSL certificates'
        }
    },
    'statistics': {
        'speakers_covered': 50,  # Approximate number of speakers in database
        'historical_coverage': '1996-present',
        'average_speeches_per_year': 150,
        'success_rate': 0.85  # Expected success rate for speech extraction
    }
}


if __name__ == "__main__":
    """
    Example usage and testing of the Federal Reserve plugin.
    
    This demonstrates how the plugin integrates with the platform architecture
    and provides examples of all major functionality.
    """
    import logging
    from datetime import date
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def test_plugin_basic_functionality():
        """Test basic plugin functionality."""
        logger.info("Testing Federal Reserve plugin basic functionality...")
        
        # Create plugin instance
        plugin = create_federal_reserve_plugin()
        
        # Test institution information
        assert plugin.get_institution_code() == "FED"
        institution = plugin.get_institution()
        assert institution.name == "Federal Reserve System"
        assert institution.country == "United States"
        
        # Test speaker database
        speaker_db = plugin.get_speaker_database()
        assert len(speaker_db.speakers) > 0
        
        # Find current Fed Chair
        current_chairs = [s for s in speaker_db.speakers if s.role == "Chair" and s.is_current]
        if current_chairs:
            chair = current_chairs[0]
            logger.info(f"Current Fed Chair: {chair.name} ({chair.role})")
        
        logger.info("✓ Basic functionality tests passed")
    
    def test_speech_discovery():
        """Test speech discovery functionality."""
        logger.info("Testing speech discovery...")
        
        plugin = create_federal_reserve_plugin()
        
        # Test discovering speeches for a recent month
        date_range = DateRange(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        try:
            speeches = plugin.discover_speeches(date_range)
            logger.info(f"Found {len(speeches)} speeches for January 2024")
            
            if speeches:
                sample_speech = speeches[0]
                logger.info(f"Sample speech: {sample_speech.title}")
                logger.info(f"Speaker: {sample_speech.speaker_name}")
                logger.info(f"Date: {sample_speech.date}")
                logger.info(f"URL: {sample_speech.url}")
        
        except Exception as e:
            logger.warning(f"Speech discovery test failed (expected in test environment): {e}")
        
        logger.info("✓ Speech discovery test completed")
    
    def test_validation():
        """Test speech validation functionality."""
        logger.info("Testing speech validation...")
        
        plugin = create_federal_reserve_plugin()
        
        # Create sample metadata and content for validation
        sample_metadata = SpeechMetadata(
            url="https://www.federalreserve.gov/newsevents/speech/powell20240115a.htm",
            title="Economic Outlook and Monetary Policy",
            speaker_name="Jerome Powell",
            date=date(2024, 1, 15),
            institution_code="FED",
            speech_type=SpeechType.FORMAL_SPEECH,
            language="en"
        )
        
        sample_content = SpeechContent(
            raw_text="Thank you for the opportunity to speak today about the economic outlook and monetary policy...",
            cleaned_text="Thank you for the opportunity to speak today about the economic outlook and monetary policy. The Federal Reserve continues to monitor economic conditions and will adjust monetary policy as appropriate to achieve our dual mandate of maximum employment and price stability.",
            extraction_method="test",
            confidence_score=0.9,
            word_count=50,
            extraction_timestamp=datetime.now()
        )
        
        validation_result = plugin.validate_speech_authenticity(sample_metadata, sample_content)
        
        logger.info(f"Validation status: {validation_result.status}")
        logger.info(f"Validation confidence: {validation_result.confidence:.2f}")
        if validation_result.issues:
            logger.info(f"Validation issues: {validation_result.issues}")
        
        assert validation_result.status in [ValidationStatus.VALID, ValidationStatus.QUESTIONABLE]
        logger.info("✓ Validation test passed")
    
    def test_speaker_recognition():
        """Test speaker recognition functionality."""
        logger.info("Testing speaker recognition...")
        
        plugin = create_federal_reserve_plugin()
        
        # Test various speaker name formats
        test_cases = [
            ("Jerome Powell", "Chair"),
            ("powell", "Chair"),
            ("Jerome H. Powell", "Chair"),
            ("Philip Jefferson", "Vice Chair"),
            ("Ben Bernanke", "Chair"),
            ("janet yellen", "Chair"),
            ("Unknown Speaker", "Unknown")
        ]
        
        for speaker_name, expected_role in test_cases:
            speaker_info = plugin._get_speaker_info(speaker_name)
            logger.info(f"'{speaker_name}' -> Role: {speaker_info.get('role', 'Unknown')} (Expected: {expected_role})")
            
            # For known speakers, check if recognition worked
            if expected_role != "Unknown":
                assert speaker_info.get('source') != 'unknown', f"Failed to recognize {speaker_name}"
        
        logger.info("✓ Speaker recognition test completed")
    
    def test_date_extraction():
        """Test date extraction from URLs."""
        logger.info("Testing date extraction from URLs...")
        
        plugin = create_federal_reserve_plugin()
        
        # Test various URL formats
        test_urls = [
            ("https://www.federalreserve.gov/newsevents/speech/powell20240115a.htm", date(2024, 1, 15)),
            ("https://www.federalreserve.gov/newsevents/speech/yellen20141201.htm", date(2014, 12, 1)),
            ("https://www.federalreserve.gov/newsevents/speech/bernanke20120315a.htm", date(2012, 3, 15)),
            ("https://www.federalreserve.gov/newsevents/speech/invalid_url.htm", None)
        ]
        
        for test_url, expected_date in test_urls:
            extracted_date = plugin._extract_date_from_url(test_url)
            logger.info(f"URL: {test_url}")
            logger.info(f"Extracted: {extracted_date}, Expected: {expected_date}")
            
            if expected_date:
                assert extracted_date == expected_date, f"Date extraction failed for {test_url}"
            else:
                assert extracted_date is None, f"Should not extract date from invalid URL"
        
        logger.info("✓ Date extraction test passed")
    
    # Run all tests
    try:
        test_plugin_basic_functionality()
        test_speech_discovery()
        test_validation()
        test_speaker_recognition()
        test_date_extraction()
        
        logger.info("🎉 All Federal Reserve plugin tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Plugin test failed: {e}")
        raise#!/usr/bin/env python3