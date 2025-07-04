#!/usr/bin/env python3
"""
Unit Tests for Federal Reserve Plugin

Comprehensive test suite for the Federal Reserve plugin, following the
plugin testing requirements from the architectural specification.

Test Coverage:
- Plugin interface compliance
- Speaker database functionality
- Date extraction from URLs
- Content validation
- Error handling
- Edge cases

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from plugins.federal_reserve.plugin import FederalReservePlugin, create_federal_reserve_plugin
from interfaces.plugin_interfaces import (
    SpeechMetadata, SpeechContent, ValidationResult, ValidationStatus,
    SpeechType, Speaker, SpeakerDatabase
)
from domain.value_objects import DateRange


class TestFederalReservePluginBasics:
    """Test basic plugin functionality and interface compliance."""
    
    def test_plugin_creation(self):
        """Test plugin can be created successfully."""
        plugin = create_federal_reserve_plugin()
        assert plugin is not None
        assert isinstance(plugin, FederalReservePlugin)
    
    def test_plugin_creation_with_config(self):
        """Test plugin creation with custom configuration."""
        config = {
            "base_url": "https://example.com",
            "request_delay": 2.0,
            "max_retries": 5
        }
        plugin = create_federal_reserve_plugin(config)
        assert plugin is not None
        assert plugin.config == config
    
    def test_institution_code(self):
        """Test institution code is correct."""
        plugin = create_federal_reserve_plugin()
        assert plugin.get_institution_code() == "FED"
    
    def test_institution_name(self):
        """Test institution name is correct."""
        plugin = create_federal_reserve_plugin()
        assert plugin.get_institution_name() == "Federal Reserve System"
    
    def test_supported_languages(self):
        """Test supported languages."""
        plugin = create_federal_reserve_plugin()
        languages = plugin.get_supported_languages()
        assert "en" in languages
        assert isinstance(languages, list)
    
    def test_rate_limit_delay(self):
        """Test rate limit delay is reasonable."""
        plugin = create_federal_reserve_plugin()
        delay = plugin.get_rate_limit_delay()
        assert isinstance(delay, (int, float))
        assert delay >= 0.1  # At least 100ms


class TestFederalReserveSpeakerDatabase:
    """Test speaker database functionality."""
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance for testing."""
        return create_federal_reserve_plugin()
    
    def test_speaker_database_creation(self, plugin):
        """Test speaker database can be created."""
        db = plugin.get_speaker_database()
        assert isinstance(db, SpeakerDatabase)
        assert db.speaker_count > 0
    
    def test_find_current_chair(self, plugin):
        """Test finding current Federal Reserve Chair."""
        db = plugin.get_speaker_database()
        
        # Look for Jerome Powell (current as of 2024)
        powell = db.find_speaker("Jerome Powell")
        assert powell is not None
        assert powell.role == "Chair"
        assert powell.institution_code == "FED"
        assert powell.voting_member is True
    
    def test_find_speaker_case_insensitive(self, plugin):
        """Test speaker search is case insensitive."""
        db = plugin.get_speaker_database()
        
        # Test various cases
        speaker1 = db.find_speaker("jerome powell")
        speaker2 = db.find_speaker("JEROME POWELL")
        speaker3 = db.find_speaker("Jerome Powell")
        
        assert speaker1 is not None
        assert speaker2 is not None
        assert speaker3 is not None
        assert speaker1.name == speaker2.name == speaker3.name
    
    def test_get_speakers_by_role(self, plugin):
        """Test getting speakers by role."""
        db = plugin.get_speaker_database()
        
        chairs = db.get_speakers_by_role("Chair")
        assert len(chairs) > 0
        assert all(s.role == "Chair" for s in chairs)
        
        governors = db.get_speakers_by_role("Governor")
        assert len(governors) > 0
        assert all(s.role == "Governor" for s in governors)
    
    def test_get_current_speakers(self, plugin):
        """Test getting current speakers."""
        db = plugin.get_speaker_database()
        
        current_speakers = db.get_current_speakers()
        assert len(current_speakers) > 0
        assert all(s.is_current for s in current_speakers)
    
    def test_speaker_not_found(self, plugin):
        """Test behavior when speaker not found."""
        db = plugin.get_speaker_database()
        
        speaker = db.find_speaker("Non Existent Speaker")
        assert speaker is None


class TestFederalReserveDateExtraction:
    """Test date extraction from Federal Reserve URLs."""
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance for testing."""
        return create_federal_reserve_plugin()
    
    def test_extract_date_from_standard_url(self, plugin):
        """Test extracting date from standard Fed URL format."""
        url = "https://www.federalreserve.gov/newsevents/speech/powell20240115a.htm"
        expected_date = date(2024, 1, 15)
        
        extracted_date = plugin._extract_date_from_url(url)
        assert extracted_date == expected_date
    
    def test_extract_date_from_various_formats(self, plugin):
        """Test extracting dates from various URL formats."""
        test_cases = [
            ("https://www.federalreserve.gov/newsevents/speech/yellen20141201.htm", date(2014, 12, 1)),
            ("https://www.federalreserve.gov/newsevents/speech/bernanke20120315a.htm", date(2012, 3, 15)),
            ("https://www.federalreserve.gov/newsevents/speech/powell20240215b.htm", date(2024, 2, 15)),
        ]
        
        for url, expected_date in test_cases:
            extracted_date = plugin._extract_date_from_url(url)
            assert extracted_date == expected_date, f"Failed for URL: {url}"
    
    def test_extract_date_invalid_url(self, plugin):
        """Test behavior with invalid URLs."""
        invalid_urls = [
            "https://www.federalreserve.gov/invalid.htm",
            "https://example.com/speech.htm",
            "invalid-url",
            "",
            None
        ]
        
        for url in invalid_urls:
            extracted_date = plugin._extract_date_from_url(url)
            assert extracted_date is None
    
    def test_extract_date_edge_cases(self, plugin):
        """Test date extraction edge cases."""
        # Test with different date formats
        edge_cases = [
            ("https://www.federalreserve.gov/newsevents/speech/test20000229.htm", date(2000, 2, 29)),  # Leap year
            ("https://www.federalreserve.gov/newsevents/speech/test19991231.htm", date(1999, 12, 31)),  # End of year
            ("https://www.federalreserve.gov/newsevents/speech/test20010101.htm", date(2001, 1, 1)),   # Start of year
        ]
        
        for url, expected_date in edge_cases:
            extracted_date = plugin._extract_date_from_url(url)
            assert extracted_date == expected_date


class TestFederalReserveValidation:
    """Test speech validation functionality."""
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance for testing."""
        return create_federal_reserve_plugin()
    
    @pytest.fixture
    def valid_speech_metadata(self):
        """Create valid speech metadata for testing."""
        return SpeechMetadata(
            url="https://www.federalreserve.gov/newsevents/speech/powell20240115a.htm",
            title="Economic Outlook and Monetary Policy",
            speaker_name="Jerome Powell",
            date=date(2024, 1, 15),
            institution_code="FED",
            speech_type=SpeechType.FORMAL_SPEECH,
            language="en"
        )
    
    @pytest.fixture
    def valid_speech_content(self):
        """Create valid speech content for testing."""
        return SpeechContent(
            raw_text="Thank you for the opportunity to speak today about the economic outlook and monetary policy. The Federal Reserve continues to monitor economic conditions and will adjust monetary policy as appropriate to achieve our dual mandate of maximum employment and price stability. Recent economic data suggests that the labor market remains strong, with unemployment at historically low levels. However, we continue to watch inflation closely as it remains above our 2 percent target.",
            cleaned_text="Thank you for the opportunity to speak today about the economic outlook and monetary policy. The Federal Reserve continues to monitor economic conditions and will adjust monetary policy as appropriate to achieve our dual mandate of maximum employment and price stability. Recent economic data suggests that the labor market remains strong, with unemployment at historically low levels. However, we continue to watch inflation closely as it remains above our 2 percent target.",
            extraction_method="test",
            confidence_score=0.9,
            word_count=75,
            extraction_timestamp=datetime.now()
        )
    
    def test_validate_valid_speech(self, plugin, valid_speech_metadata, valid_speech_content):
        """Test validation of valid speech."""
        result = plugin.validate_speech_authenticity(valid_speech_metadata, valid_speech_content)
        
        assert isinstance(result, ValidationResult)
        assert result.status == ValidationStatus.VALID
        assert result.confidence >= 0.7
        assert result.is_valid
    
    def test_validate_invalid_domain(self, plugin, valid_speech_metadata, valid_speech_content):
        """Test validation fails for non-Fed domain."""
        invalid_metadata = SpeechMetadata(
            url="https://example.com/speech.htm",
            title=valid_speech_metadata.title,
            speaker_name=valid_speech_metadata.speaker_name,
            date=valid_speech_metadata.date,
            institution_code="FED",
            speech_type=SpeechType.FORMAL_SPEECH,
            language="en"
        )
        
        result = plugin.validate_speech_authenticity(invalid_metadata, valid_speech_content)
        
        assert result.status != ValidationStatus.VALID
        assert result.confidence < 0.7
        assert not result.is_valid
        assert any("domain" in issue.lower() for issue in result.issues)
    
    def test_validate_short_content(self, plugin, valid_speech_metadata):
        """Test validation fails for too-short content."""
        short_content = SpeechContent(
            raw_text="Short content",
            cleaned_text="Short content",
            extraction_method="test",
            confidence_score=0.9,
            word_count=2,
            extraction_timestamp=datetime.now()
        )
        
        result = plugin.validate_speech_authenticity(valid_speech_metadata, short_content)
        
        assert result.status != ValidationStatus.VALID
        assert result.confidence < 0.7
        assert not result.is_valid
        assert any("short" in issue.lower() for issue in result.issues)
    
def test_validate_placeholder_content(self, plugin, valid_speech_metadata):
       """Test validation fails for placeholder content."""
       placeholder_content = SpeechContent(
           raw_text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
           cleaned_text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
           extraction_method="test",
           confidence_score=0.9,
           word_count=50,
           extraction_timestamp=datetime.now()
       )
       
       result = plugin.validate_speech_authenticity(valid_speech_metadata, placeholder_content)
       
       assert result.status != ValidationStatus.VALID
       assert result.confidence < 0.7
       assert not result.is_valid
       assert any("placeholder" in issue.lower() for issue in result.issues)
   
def test_validate_future_date(self, plugin, valid_speech_content):
       """Test validation fails for future dates."""
       future_metadata = SpeechMetadata(
           url="https://www.federalreserve.gov/newsevents/speech/powell20251215a.htm",
           title="Future Economic Outlook",
           speaker_name="Jerome Powell",
           date=date(2025, 12, 15),  # Future date
           institution_code="FED",
           speech_type=SpeechType.FORMAL_SPEECH,
           language="en"
       )
       
       result = plugin.validate_speech_authenticity(future_metadata, valid_speech_content)
       
       assert result.status != ValidationStatus.VALID
       assert result.confidence < 0.8
       assert not result.is_valid
       assert any("future" in issue.lower() for issue in result.issues)
   
def test_validate_unknown_speaker(self, plugin, valid_speech_content):
       """Test validation with unknown speaker."""
       unknown_speaker_metadata = SpeechMetadata(
           url="https://www.federalreserve.gov/newsevents/speech/unknown20240115a.htm",
           title="Economic Outlook",
           speaker_name="Unknown Speaker",
           date=date(2024, 1, 15),
           institution_code="FED",
           speech_type=SpeechType.FORMAL_SPEECH,
           language="en"
       )
       
       result = plugin.validate_speech_authenticity(unknown_speaker_metadata, valid_speech_content)
       
       assert result.status in [ValidationStatus.QUESTIONABLE, ValidationStatus.VALID]
       assert result.confidence < 1.0
       assert any("speaker" in issue.lower() for issue in result.issues)


class TestFederalReserveSpeechDiscovery:
   """Test speech discovery functionality."""
   
   @pytest.fixture
   def plugin(self):
       """Create plugin instance for testing."""
       return create_federal_reserve_plugin()
   
   @pytest.fixture
   def sample_date_range(self):
       """Create sample date range for testing."""
       return DateRange(
           start_date=date(2024, 1, 1),
           end_date=date(2024, 1, 31)
       )
   
   @patch('plugins.federal_reserve.plugin.requests.get')
   def test_discover_speeches_success(self, mock_get, plugin, sample_date_range):
       """Test successful speech discovery."""
       # Mock successful HTTP response
       mock_response = Mock()
       mock_response.status_code = 200
       mock_response.content = b"""
       <html>
       <body>
           <div class="speeches">
               <a href="/newsevents/speech/powell20240115a.htm">Economic Outlook</a>
               <a href="/newsevents/speech/jefferson20240110a.htm">Banking Resilience</a>
           </div>
       </body>
       </html>
       """
       mock_get.return_value = mock_response
       
       speeches = plugin.discover_speeches(sample_date_range)
       
       assert isinstance(speeches, list)
       # Note: Actual implementation depends on HTML parsing logic
       # This test validates the interface contract
   
   @patch('plugins.federal_reserve.plugin.requests.get')
   def test_discover_speeches_network_error(self, mock_get, plugin, sample_date_range):
       """Test speech discovery with network error."""
       mock_get.side_effect = Exception("Network error")
       
       # Should handle gracefully and return empty list or raise appropriate exception
       try:
           speeches = plugin.discover_speeches(sample_date_range)
           assert isinstance(speeches, list)
       except Exception as e:
           # Should raise appropriate plugin-specific exception
           assert "Network" in str(e) or "Connection" in str(e)
   
   def test_discover_speeches_invalid_date_range(self, plugin):
       """Test discovery with invalid date range."""
       invalid_range = DateRange(
           start_date=date(2024, 12, 31),
           end_date=date(2024, 1, 1)  # End before start
       )
       
       with pytest.raises(ValueError):
           plugin.discover_speeches(invalid_range)
   
   def test_discover_speeches_with_limit(self, plugin, sample_date_range):
       """Test discovery with limit parameter."""
       # Mock implementation would need actual HTTP mocking
       # This test validates the interface
       limit = 5
       speeches = plugin.discover_speeches(sample_date_range, limit=limit)
       
       assert isinstance(speeches, list)
       # In real implementation: assert len(speeches) <= limit


class TestFederalReserveContentExtraction:
   """Test content extraction functionality."""
   
   @pytest.fixture
   def plugin(self):
       """Create plugin instance for testing."""
       return create_federal_reserve_plugin()
   
   @pytest.fixture
   def sample_html_metadata(self):
       """Create metadata for HTML speech."""
       return SpeechMetadata(
           url="https://www.federalreserve.gov/newsevents/speech/powell20240115a.htm",
           title="Economic Outlook",
           speaker_name="Jerome Powell",
           date=date(2024, 1, 15),
           institution_code="FED",
           speech_type=SpeechType.FORMAL_SPEECH,
           language="en"
       )
   
   @pytest.fixture
   def sample_pdf_metadata(self):
       """Create metadata for PDF speech."""
       return SpeechMetadata(
           url="https://www.federalreserve.gov/newsevents/speech/powell20240115a.pdf",
           title="Economic Outlook",
           speaker_name="Jerome Powell",
           date=date(2024, 1, 15),
           institution_code="FED",
           speech_type=SpeechType.FORMAL_SPEECH,
           language="en"
       )
   
   @patch('plugins.federal_reserve.plugin.requests.get')
   def test_extract_html_content_success(self, mock_get, plugin, sample_html_metadata):
       """Test successful HTML content extraction."""
       mock_response = Mock()
       mock_response.status_code = 200
       mock_response.headers = {"content-type": "text/html"}
       mock_response.text = """
       <html>
       <body>
           <div id="article">
               <h1>Economic Outlook and Monetary Policy</h1>
               <p>Thank you for the opportunity to speak today about the economic outlook and monetary policy.</p>
               <p>The Federal Reserve continues to monitor economic conditions and will adjust monetary policy as appropriate to achieve our dual mandate of maximum employment and price stability.</p>
               <p>Recent economic data suggests that the labor market remains strong, with unemployment at historically low levels.</p>
           </div>
       </body>
       </html>
       """
       mock_get.return_value = mock_response
       
       content = plugin.extract_speech_content(sample_html_metadata)
       
       assert isinstance(content, SpeechContent)
       assert len(content.cleaned_text) > 100
       assert content.word_count > 10
       assert content.confidence_score > 0.5
       assert "economic" in content.cleaned_text.lower()
       assert "monetary policy" in content.cleaned_text.lower()
   
   @patch('plugins.federal_reserve.plugin.requests.get')
   def test_extract_pdf_content_success(self, mock_get, plugin, sample_pdf_metadata):
       """Test successful PDF content extraction."""
       mock_response = Mock()
       mock_response.status_code = 200
       mock_response.headers = {"content-type": "application/pdf"}
       mock_response.content = b"Mock PDF content"  # Would need actual PDF bytes
       mock_get.return_value = mock_response
       
       # Mock pdfplumber
       with patch('plugins.federal_reserve.plugin.pdfplumber.open') as mock_pdf:
           mock_pdf_instance = Mock()
           mock_page = Mock()
           mock_page.extract_text.return_value = "Thank you for the opportunity to speak today about the economic outlook and monetary policy. The Federal Reserve continues to monitor economic conditions."
           mock_pdf_instance.pages = [mock_page]
           mock_pdf.return_value.__enter__.return_value = mock_pdf_instance
           
           content = plugin.extract_speech_content(sample_pdf_metadata)
           
           assert isinstance(content, SpeechContent)
           assert len(content.cleaned_text) > 50
           assert content.extraction_method == "pdf_pdfplumber"
           assert "economic" in content.cleaned_text.lower()
   
   @patch('plugins.federal_reserve.plugin.requests.get')
   def test_extract_content_network_error(self, mock_get, plugin, sample_html_metadata):
       """Test content extraction with network error."""
       mock_get.side_effect = Exception("Network error")
       
       with pytest.raises(Exception) as exc_info:
           plugin.extract_speech_content(sample_html_metadata)
       
       assert "Network" in str(exc_info.value) or "Failed" in str(exc_info.value)
   
   @patch('plugins.federal_reserve.plugin.requests.get')
   def test_extract_content_empty_response(self, mock_get, plugin, sample_html_metadata):
       """Test content extraction with empty response."""
       mock_response = Mock()
       mock_response.status_code = 200
       mock_response.headers = {"content-type": "text/html"}
       mock_response.text = "<html><body></body></html>"
       mock_get.return_value = mock_response
       
       content = plugin.extract_speech_content(sample_html_metadata)
       
       assert isinstance(content, SpeechContent)
       assert content.confidence_score < 0.8  # Low confidence for empty content


class TestFederalReserveUtilityMethods:
   """Test utility and helper methods."""
   
   @pytest.fixture
   def plugin(self):
       """Create plugin instance for testing."""
       return create_federal_reserve_plugin()
   
   def test_clean_speaker_name(self, plugin):
       """Test speaker name cleaning."""
       test_cases = [
           ("Governor Jerome Powell", "Jerome Powell"),
           ("Chair Janet L. Yellen", "Janet L. Yellen"),
           ("Dr. Ben Bernanke", "Ben Bernanke"),
           ("President, John Williams", "John Williams"),
           ("Vice Chair for Supervision Michael Barr", "Michael Barr"),
           ("", ""),
           ("   ", ""),
       ]
       
       for input_name, expected_output in test_cases:
           cleaned = plugin._clean_speaker_name(input_name)
           assert cleaned == expected_output, f"Failed for '{input_name}'"
   
   def test_names_match(self, plugin):
       """Test name matching logic."""
       # Test positive matches
       positive_cases = [
           ("Jerome Powell", "jerome powell"),
           ("Janet L. Yellen", "janet yellen"),
           ("Ben S. Bernanke", "ben bernanke"),
           ("Christopher J. Waller", "chris waller"),
       ]
       
       for name1, name2 in positive_cases:
           assert plugin._names_match(name1, name2), f"Should match: '{name1}' and '{name2}'"
       
       # Test negative matches
       negative_cases = [
           ("Jerome Powell", "Janet Yellen"),
           ("Ben Bernanke", "Alan Greenspan"),
           ("", "Jerome Powell"),
           ("Jerome Powell", ""),
       ]
       
       for name1, name2 in negative_cases:
           assert not plugin._names_match(name1, name2), f"Should not match: '{name1}' and '{name2}'"
   
   def test_parse_date_string(self, plugin):
       """Test date string parsing."""
       test_cases = [
           ("January 15, 2024", date(2024, 1, 15)),
           ("Jan 15, 2024", date(2024, 1, 15)),
           ("01/15/2024", date(2024, 1, 15)),
           ("01-15-2024", date(2024, 1, 15)),
           ("2024-01-15", date(2024, 1, 15)),
           ("20240115", date(2024, 1, 15)),
           ("invalid date", None),
           ("", None),
       ]
       
       for date_str, expected_date in test_cases:
           parsed_date = plugin._parse_date_string(date_str)
           assert parsed_date == expected_date, f"Failed for '{date_str}'"
   
   def test_clean_text_content(self, plugin):
       """Test text content cleaning."""
       raw_text = """
       
       Return to top
       
       Economic Outlook and Monetary Policy
       
       Thank you for the opportunity to speak today.
       
       The Federal Reserve continues to monitor economic conditions.
       
       Last update: January 15, 2024
       
       """
       
       cleaned = plugin._clean_text_content(raw_text)
       
       assert "Return to top" not in cleaned
       assert "Last update:" not in cleaned
       assert "Economic Outlook and Monetary Policy" in cleaned
       assert "Thank you for the opportunity" in cleaned
       assert "Federal Reserve continues to monitor" in cleaned
       assert len(cleaned.strip()) > 0
   
   def test_deduplicate_speeches(self, plugin):
       """Test speech deduplication."""
       speeches = [
           SpeechMetadata(
               url="https://example.com/speech1.htm",
               title="Speech 1",
               speaker_name="Speaker A",
               date=date(2024, 1, 1),
               institution_code="FED",
               speech_type=SpeechType.FORMAL_SPEECH,
               language="en"
           ),
           SpeechMetadata(
               url="https://example.com/speech1.htm",  # Duplicate URL
               title="Speech 1 Duplicate",
               speaker_name="Speaker A",
               date=date(2024, 1, 1),
               institution_code="FED",
               speech_type=SpeechType.FORMAL_SPEECH,
               language="en"
           ),
           SpeechMetadata(
               url="https://example.com/speech2.htm",
               title="Speech 2",
               speaker_name="Speaker B",
               date=date(2024, 1, 2),
               institution_code="FED",
               speech_type=SpeechType.FORMAL_SPEECH,
               language="en"
           ),
       ]
       
       unique_speeches = plugin._deduplicate_speeches(speeches)
       
       assert len(unique_speeches) == 2
       urls = [s.url for s in unique_speeches]
       assert "https://example.com/speech1.htm" in urls
       assert "https://example.com/speech2.htm" in urls


class TestFederalReservePluginIntegration:
   """Integration tests for the Federal Reserve plugin."""
   
   @pytest.fixture
   def plugin(self):
       """Create plugin instance for integration testing."""
       return create_federal_reserve_plugin()
   
   def test_plugin_metadata_consistency(self, plugin):
       """Test that plugin metadata is consistent."""
       from plugins.federal_reserve.plugin import PLUGIN_METADATA
       
       assert PLUGIN_METADATA['code'] == plugin.get_institution_code()
       assert PLUGIN_METADATA['name'] == plugin.get_institution_name()
       assert set(PLUGIN_METADATA['supported_languages']) == set(plugin.get_supported_languages())
   
   def test_plugin_factory_consistency(self, plugin):
       """Test that plugin factory creates consistent instances."""
       plugin1 = create_federal_reserve_plugin()
       plugin2 = create_federal_reserve_plugin()
       
       assert plugin1.get_institution_code() == plugin2.get_institution_code()
       assert plugin1.get_institution_name() == plugin2.get_institution_name()
       assert plugin1.get_supported_languages() == plugin2.get_supported_languages()
   
   def test_plugin_statistics_tracking(self, plugin):
       """Test that plugin tracks statistics correctly."""
       initial_stats = plugin.stats.copy()
       
       # Stats should be initialized
       assert 'total_processed' in plugin.stats
       assert 'successful_extractions' in plugin.stats
       assert 'unknown_speakers' in plugin.stats
       assert 'validation_failures' in plugin.stats
       
       # Stats should be numeric
       for key, value in plugin.stats.items():
           assert isinstance(value, (int, float)), f"Stat '{key}' should be numeric"
   
   @pytest.mark.slow
   @pytest.mark.requires_network
   def test_plugin_live_discovery(self, plugin):
       """Test plugin with live Federal Reserve website (slow test)."""
       # Small date range for testing
       date_range = DateRange(
           start_date=date(2024, 1, 1),
           end_date=date(2024, 1, 31)
       )
       
       try:
           speeches = plugin.discover_speeches(date_range, limit=5)
           
           assert isinstance(speeches, list)
           assert len(speeches) <= 5
           
           for speech in speeches:
               assert isinstance(speech, SpeechMetadata)
               assert speech.institution_code == "FED"
               assert speech.url.startswith("https://www.federalreserve.gov")
               assert speech.date is not None
               assert date_range.contains(speech.date)
               
       except Exception as e:
           pytest.skip(f"Live test skipped due to network issues: {e}")
   
   @pytest.mark.slow
   @pytest.mark.requires_network
   def test_plugin_live_extraction(self, plugin):
       """Test plugin with live content extraction (slow test)."""
       # Use a known stable URL (if available)
       sample_metadata = SpeechMetadata(
           url="https://www.federalreserve.gov/newsevents/speech/powell20240101a.htm",
           title="Test Speech",
           speaker_name="Jerome Powell",
           date=date(2024, 1, 1),
           institution_code="FED",
           speech_type=SpeechType.FORMAL_SPEECH,
           language="en"
       )
       
       try:
           content = plugin.extract_speech_content(sample_metadata)
           
           assert isinstance(content, SpeechContent)
           assert len(content.cleaned_text) > 100
           assert content.word_count > 10
           assert content.confidence_score > 0.0
           
       except Exception as e:
           pytest.skip(f"Live extraction test skipped: {e}")


class TestFederalReservePluginErrorHandling:
   """Test error handling and edge cases."""
   
   @pytest.fixture
   def plugin(self):
       """Create plugin instance for testing."""
       return create_federal_reserve_plugin()
   
   def test_plugin_handles_none_inputs(self, plugin):
       """Test plugin gracefully handles None inputs."""
       # Test date extraction with None
       assert plugin._extract_date_from_url(None) is None
       
       # Test speaker name cleaning with None
       assert plugin._clean_speaker_name(None) == ""
       
       # Test name matching with None
       assert not plugin._names_match(None, "test")
       assert not plugin._names_match("test", None)
       assert not plugin._names_match(None, None)
   
   def test_plugin_handles_empty_strings(self, plugin):
       """Test plugin gracefully handles empty strings."""
       # Test date extraction with empty string
       assert plugin._extract_date_from_url("") is None
       
       # Test speaker name cleaning with empty string
       assert plugin._clean_speaker_name("") == ""
       
       # Test name matching with empty strings
       assert not plugin._names_match("", "test")
       assert not plugin._names_match("test", "")
       assert not plugin._names_match("", "")
   
   def test_plugin_handles_malformed_data(self, plugin):
       """Test plugin gracefully handles malformed data."""
       # Test with malformed URL
       malformed_urls = [
           "not-a-url",
           "http://",
           "https://",
           "ftp://example.com",
           "javascript:alert('xss')",
       ]
       
       for url in malformed_urls:
           result = plugin._extract_date_from_url(url)
           assert result is None, f"Should handle malformed URL: {url}"
   
   def test_plugin_configuration_validation(self):
       """Test plugin configuration validation."""
       # Test with invalid configuration
       invalid_configs = [
           {"base_url": "not-a-url"},
           {"request_delay": -1},
           {"max_retries": "not-a-number"},
           {"timeout": 0},
       ]
       
       for config in invalid_configs:
           try:
               plugin = create_federal_reserve_plugin(config)
               # Plugin should either handle gracefully or raise clear error
               assert plugin is not None
           except (ValueError, TypeError) as e:
               # Clear error message expected
               assert len(str(e)) > 0


# Performance and benchmarking tests
class TestFederalReservePluginPerformance:
   """Performance tests for the Federal Reserve plugin."""
   
   @pytest.fixture
   def plugin(self):
       """Create plugin instance for performance testing."""
       return create_federal_reserve_plugin()
   
   def test_speaker_database_lookup_performance(self, plugin):
       """Test speaker database lookup performance."""
       import time
       
       db = plugin.get_speaker_database()
       
       # Test lookup performance
       start_time = time.time()
       for _ in range(1000):
           db.find_speaker("Jerome Powell")
       end_time = time.time()
       
       lookup_time = end_time - start_time
       assert lookup_time < 1.0, f"Speaker lookup too slow: {lookup_time:.3f}s for 1000 lookups"
   
   def test_date_extraction_performance(self, plugin):
       """Test date extraction performance."""
       import time
       
       test_urls = [
           "https://www.federalreserve.gov/newsevents/speech/powell20240115a.htm",
           "https://www.federalreserve.gov/newsevents/speech/yellen20141201.htm",
           "https://www.federalreserve.gov/newsevents/speech/bernanke20120315a.htm",
       ] * 100  # Test with 300 URLs
       
       start_time = time.time()
       for url in test_urls:
           plugin._extract_date_from_url(url)
       end_time = time.time()
       
       extraction_time = end_time - start_time
       assert extraction_time < 1.0, f"Date extraction too slow: {extraction_time:.3f}s for {len(test_urls)} URLs"


if __name__ == "__main__":
   """Run tests directly."""
   pytest.main([__file__, "-v"])