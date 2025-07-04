#!/usr/bin/env python3
"""
Central Bank Speech Platform - Test Configuration and Fixtures

This module provides comprehensive test fixtures and configuration for the
speech analysis platform, following pytest best practices and DDD principles.

Key Features:
- Database fixtures with transaction rollback
- Plugin test fixtures with mock data
- NLP processor test fixtures
- HTTP client mocks
- Domain entity factories

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import asyncio
import pytest
from datetime import date, datetime, timedelta
from typing import AsyncGenerator, Generator, List, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock
from uuid import uuid4
import tempfile
import os
from pathlib import Path

# Core imports
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
import factory

# Domain imports
from domain.entities import (
    CentralBankSpeech, CentralBankSpeaker, Institution, 
    SentimentAnalysis, SpeechStatus, PolicyStance, InstitutionType
)
from domain.value_objects import DateRange, Url, ContentHash, ConfidenceLevel
from domain.repositories import UnitOfWork

# Application imports
from application.services.speech_collection_service import SpeechCollectionService
from application.services.analysis_service import AnalysisService
from application.orchestrators.speech_collection import SpeechCollectionOrchestrator

# Infrastructure imports
from infrastructure.persistence.models import Base
from infrastructure.persistence.uow import SqlAlchemyUnitOfWork
from infrastructure.web.http_client import HttpClient

# Interface imports
from interfaces.plugin_interfaces import (
    SpeechMetadata, SpeechContent, ValidationResult, ValidationStatus,
    SpeechType, Speaker, SpeakerDatabase
)

# Plugin imports
from plugins.federal_reserve.plugin import FederalReservePlugin

# Config imports
from config.settings import Settings


# Test Database Configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_database_url() -> str:
    """Create test database URL."""
    return "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
async def test_engine(test_database_url: str):
    """Create test database engine."""
    engine = create_async_engine(
        test_database_url,
        echo=False,
        future=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session with transaction rollback."""
    async_session = sessionmaker(
        test_engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        async with session.begin():
            yield session
            await session.rollback()


@pytest.fixture
async def test_uow(test_session: AsyncSession) -> UnitOfWork:
    """Create test unit of work."""
    return SqlAlchemyUnitOfWork(test_session)


# Domain Entity Factories
class InstitutionFactory(factory.Factory):
    """Factory for creating Institution entities."""
    
    class Meta:
        model = Institution
    
    code = factory.Sequence(lambda n: f"TEST{n:03d}")
    name = factory.Faker("company")
    country = factory.Faker("country")
    institution_type = InstitutionType.CENTRAL_BANK
    established_date = factory.Faker("date_between", start_date="-100y", end_date="today")
    website_url = factory.Faker("url")
    description = factory.Faker("text", max_nb_chars=200)


class CentralBankSpeakerFactory(factory.Factory):
    """Factory for creating CentralBankSpeaker entities."""
    
    class Meta:
        model = CentralBankSpeaker
    
    id = factory.LazyFunction(uuid4)
    name = factory.Faker("name")
    role = factory.Faker("job")
    institution = factory.SubFactory(InstitutionFactory)
    start_date = factory.Faker("date_between", start_date="-10y", end_date="today")
    end_date = None
    voting_member = factory.Faker("boolean")
    biographical_notes = factory.Faker("text", max_nb_chars=500)


class SpeechContentFactory(factory.Factory):
    """Factory for creating SpeechContent objects."""
    
    class Meta:
        model = SpeechContent
    
    raw_text = factory.Faker("text", max_nb_chars=2000)
    cleaned_text = factory.Faker("text", max_nb_chars=1500)
    extraction_method = "test_extraction"
    confidence_score = factory.Faker("pyfloat", min_value=0.7, max_value=1.0)
    word_count = factory.LazyAttribute(lambda obj: len(obj.cleaned_text.split()))
    extraction_timestamp = factory.LazyFunction(datetime.now)


class SpeechMetadataFactory(factory.Factory):
    """Factory for creating SpeechMetadata objects."""
    
    class Meta:
        model = SpeechMetadata
    
    url = factory.Faker("url")
    title = factory.Faker("sentence", nb_words=8)
    speaker_name = factory.Faker("name")
    date = factory.Faker("date_between", start_date="-5y", end_date="today")
    institution_code = "TEST"
    speech_type = SpeechType.FORMAL_SPEECH
    language = "en"


class ValidationResultFactory(factory.Factory):
    """Factory for creating ValidationResult objects."""
    
    class Meta:
        model = ValidationResult
    
    status = ValidationStatus.VALID
    confidence = factory.Faker("pyfloat", min_value=0.8, max_value=1.0)
    issues = factory.List([])
    metadata = factory.Dict({})


# Plugin Test Fixtures
@pytest.fixture
def mock_federal_reserve_plugin():
    """Create a mock Federal Reserve plugin for testing."""
    plugin = Mock(spec=FederalReservePlugin)
    plugin.get_institution_code.return_value = "FED"
    plugin.get_institution_name.return_value = "Federal Reserve System"
    plugin.get_supported_languages.return_value = ["en"]
    plugin.get_rate_limit_delay.return_value = 0.1  # Fast for testing
    
    # Mock speaker database
    mock_speakers = [
        Speaker(
            name="Jerome Powell",
            role="Chair",
            institution_code="FED",
            start_date=date(2018, 2, 5),
            voting_member=True
        ),
        Speaker(
            name="Philip Jefferson",
            role="Vice Chair",
            institution_code="FED",
            start_date=date(2022, 9, 23),
            voting_member=True
        )
    ]
    plugin.get_speaker_database.return_value = SpeakerDatabase(speakers=mock_speakers)
    
    return plugin


@pytest.fixture
def sample_speech_metadata():
    """Create sample speech metadata for testing."""
    return SpeechMetadataFactory()


@pytest.fixture
def sample_speech_content():
    """Create sample speech content for testing."""
    return SpeechContentFactory()


@pytest.fixture
def sample_validation_result():
    """Create sample validation result for testing."""
    return ValidationResultFactory()


@pytest.fixture
def sample_date_range():
    """Create sample date range for testing."""
    return DateRange(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31)
    )


# HTTP Client Mocks
@pytest.fixture
def mock_http_client():
    """Create mock HTTP client for testing."""
    client = Mock(spec=HttpClient)
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.session = AsyncMock()
    return client


@pytest.fixture
def mock_httpx_response():
    """Create mock HTTPX response for testing."""
    response = Mock()
    response.status_code = 200
    response.text = "<html><body>Test speech content</body></html>"
    response.content = b"Test PDF content"
    response.headers = {"content-type": "text/html"}
    response.raise_for_status = Mock()
    return response


# Service Fixtures
@pytest.fixture
def speech_collection_service(test_uow, mock_http_client):
    """Create speech collection service for testing."""
    return SpeechCollectionService(
        uow=test_uow,
        http_client=mock_http_client
    )


@pytest.fixture
def analysis_service(test_uow):
    """Create analysis service for testing."""
    return AnalysisService(uow=test_uow)


@pytest.fixture
def speech_collection_orchestrator(test_uow, mock_http_client):
    """Create speech collection orchestrator for testing."""
    return SpeechCollectionOrchestrator(
        uow=test_uow,
        http_client=mock_http_client
    )


# Configuration Fixtures
@pytest.fixture
def test_settings():
    """Create test settings configuration."""
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        log_level="DEBUG",
        metrics_enabled=False,
        redis_url=None,
        environment="test"
    )


@pytest.fixture
def temporary_directory():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# Data Fixtures
@pytest.fixture
def sample_fed_speech_html():
    """Sample Federal Reserve speech HTML content."""
    return """
    <html>
    <head><title>Monetary Policy Speech</title></head>
    <body>
        <div id="article">
            <h1>Economic Outlook and Monetary Policy</h1>
            <p>Thank you for the opportunity to speak today about the economic outlook and monetary policy.</p>
            <p>The Federal Reserve continues to monitor economic conditions and will adjust monetary policy as appropriate to achieve our dual mandate of maximum employment and price stability.</p>
            <p>Recent economic data suggests that the labor market remains strong, with unemployment at historically low levels. However, we continue to watch inflation closely as it remains above our 2 percent target.</p>
            <p>In conclusion, the Federal Reserve remains committed to using all available tools to support the economy and achieve our statutory objectives.</p>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_fed_speeches_list_html():
    """Sample Federal Reserve speeches list HTML content."""
    return """
    <html>
    <body>
        <div class="speeches-list">
            <div class="speech-item">
                <a href="/newsevents/speech/powell20240115a.htm">Economic Outlook and Monetary Policy</a>
                <p>Chair Jerome Powell, January 15, 2024</p>
            </div>
            <div class="speech-item">
                <a href="/newsevents/speech/jefferson20240110a.htm">Banking System Resilience</a>
                <p>Vice Chair Philip Jefferson, January 10, 2024</p>
            </div>
        </div>
    </body>
    </html>
    """


# Plugin Integration Test Fixtures
@pytest.fixture
def federal_reserve_plugin_integration():
    """Create Federal Reserve plugin for integration testing."""
    # Only create if network access is available
    return FederalReservePlugin()


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may require external services"
    )
    config.addinivalue_line(
        "markers", "plugin: Plugin-specific tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that take more than 5 seconds"
    )
    config.addinivalue_line(
        "markers", "requires_network: Tests requiring network access"
    )


# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatic cleanup of test data after each test."""
    yield
    # Any cleanup logic here


# Environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests."""
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup environment
    if "TESTING" in os.environ:
        del os.environ["TESTING"]