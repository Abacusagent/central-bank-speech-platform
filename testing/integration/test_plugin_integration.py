#!/usr/bin/env python3
"""
Integration Tests for Plugin System

Tests the integration between plugins and the core platform,
ensuring proper end-to-end functionality.

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import pytest
import asyncio
from datetime import date, datetime
from unittest.mock import Mock, patch, AsyncMock

from plugins.federal_reserve.plugin import FederalReservePlugin, create_federal_reserve_plugin
from application.services.speech_collection_service import SpeechCollectionService
from application.orchestrators.speech_collection import SpeechCollectionOrchestrator
from domain.value_objects import DateRange
from interfaces.plugin_interfaces import SpeechMetadata, SpeechContent, ValidationStatus


class TestPluginIntegration:
    """Test plugin integration with the core platform."""
    
    @pytest.fixture
    def fed_plugin(self):
        """Create Federal Reserve plugin for integration testing."""
        return create_federal_reserve_plugin()
    
    @pytest.fixture
    def mock_uow(self):
        """Create mock unit of work for testing."""
        uow = Mock()
        uow.speeches = Mock()
        uow.speakers = Mock()
        uow.institutions = Mock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        return uow
    
    @pytest.fixture
    def speech_service(self, mock_uow):
        """Create speech collection service for testing."""
        return SpeechCollectionService(uow=mock_uow)
    
    @pytest.fixture
    def orchestrator(self, mock_uow):
        """Create speech collection orchestrator for testing."""
        return SpeechCollectionOrchestrator(uow=mock_uow)
    
    @pytest.mark.asyncio
    async def test_plugin_registration(self, orchestrator, fed_plugin):
        """Test plugin registration with orchestrator."""
        # Register plugin
        await orchestrator.register_plugin(fed_plugin)
        
        # Verify registration
        assert fed_plugin.get_institution_code() in orchestrator.registered_plugins
        assert orchestrator.registered_plugins[fed_plugin.get_institution_code()] == fed_plugin
    
    @pytest.mark.asyncio
    async def test_end_to_end_collection(self, orchestrator, fed_plugin, mock_uow):
        """Test end-to-end speech collection process."""
        # Register plugin
        await orchestrator.register_plugin(fed_plugin)
        
        # Mock plugin methods
        sample_metadata = SpeechMetadata(
            url="https://www.federalreserve.gov/newsevents/speech/powell20240115a.htm",
            title="Economic Outlook",
            speaker_name="Jerome Powell",
            date=date(2024, 1, 15),
            institution_code="FED",
            speech_type="FORMAL_SPEECH",
            language="en"
        )
        
        sample_content = SpeechContent(
            raw_text="Test speech content about monetary policy...",
            cleaned_text="Test speech content about monetary policy...",
            extraction_method="test",
            confidence_score=0.9,
            word_count=50,
            extraction_timestamp=datetime.now()
        )
        
        with patch.object(fed_plugin, 'discover_speeches', return_value=[sample_metadata]):
            with patch.object(fed_plugin, 'extract_speech_content', return_value=sample_content):
                with patch.object(fed_plugin, 'validate_speech_authenticity') as mock_validate:
                    mock_validate.return_value = Mock(status=ValidationStatus.VALID, confidence=0.9)
                    
                    # Run collection
                    date_range = DateRange(date(2024, 1, 1), date(2024, 1, 31))
                    result = await orchestrator.collect_speeches(
                        institution_codes=["FED"],
                        date_range=date_range
                    )
                    
                    # Verify results
                    assert result is not None
                    assert len(result.successful_collections) > 0
                    
                    # Verify database interactions
                    mock_uow.speeches.save.assert_called()
                    mock_uow.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_plugin_error_isolation(self, orchestrator, fed_plugin, mock_uow):
        """Test that plugin errors don't crash the system."""
        # Register plugin
        await orchestrator.register_plugin(fed_plugin)
        
        # Mock plugin to raise error
        with patch.object(fed_plugin, 'discover_speeches', side_effect=Exception("Plugin error")):
            date_range = DateRange(date(2024, 1, 1), date(2024, 1, 31))
            
            # Should handle error gracefully
            result = await orchestrator.collect_speeches(
                institution_codes=["FED"],
                date_range=date_range
            )
            
            # System should still return result (with errors recorded)
            assert result is not None
            assert len(result.failed_collections) > 0
            
            # Should not crash the orchestrator
            assert orchestrator.is_healthy()
    
    def test_plugin_interface_compliance(self, fed_plugin):
        """Test that plugin fully implements the required interface."""
        from interfaces.plugin_interfaces import CentralBankScraperPlugin
        
        # Check that plugin is instance of interface
        assert isinstance(fed_plugin, CentralBankScraperPlugin)
        
        # Check all required methods are implemented
        required_methods = [
            'get_institution_code',
            'get_institution_name', 
            'get_supported_languages',
            'discover_speeches',
            'extract_speech_content',
            'get_speaker_database',
            'validate_speech_authenticity'
        ]
        
        for method_name in required_methods:
            assert hasattr(fed_plugin, method_name), f"Missing method: {method_name}"
            method = getattr(fed_plugin, method_name)
            assert callable(method), f"Method {method_name} is not callable"


if __name__ == "__main__":
    """Run integration tests."""
    pytest.main([__file__, "-v", "-m", "integration"])