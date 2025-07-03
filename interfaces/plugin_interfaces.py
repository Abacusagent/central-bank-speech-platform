"""
Central Bank Scraper Plugin Interfaces

This module defines the core contracts that all central bank scraper plugins must implement.
These interfaces are the foundation of the entire platform - they ensure consistency,
enable plugin isolation, and provide the abstraction layer for infinite scalability.

All central bank plugins MUST implement CentralBankScraperPlugin without exception.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pathlib import Path


class ValidationStatus(Enum):
    """Speech validation result status."""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class SpeechType(Enum):
    """Classification of speech types."""
    FORMAL_SPEECH = "formal_speech"
    TESTIMONY = "testimony"
    INTERVIEW = "interview"
    REMARKS = "remarks"
    PANEL_DISCUSSION = "panel_discussion"
    LECTURE = "lecture"
    OTHER = "other"


@dataclass(frozen=True)
class DateRange:
    """Immutable date range for speech discovery queries."""
    start_date: date
    end_date: date
    
    def __post_init__(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")
    
    @property
    def days(self) -> int:
        """Number of days in the range."""
        return (self.end_date - self.start_date).days + 1


@dataclass(frozen=True)
class SpeechMetadata:
    """
    Essential metadata for a central bank speech.
    
    This is the minimal information required to identify and locate a speech
    before content extraction. Used in the discovery phase.
    """
    url: str
    title: str
    speaker_name: str
    date: date
    institution_code: str
    speech_type: SpeechType = SpeechType.FORMAL_SPEECH
    location: Optional[str] = None
    language: str = "en"
    
    def __post_init__(self):
        if not self.url.startswith(('http://', 'https://')):
            raise ValueError("URL must be a valid HTTP(S) URL")
        if len(self.title.strip()) < 5:
            raise ValueError("Title must be at least 5 characters")
        if len(self.speaker_name.strip()) < 2:
            raise ValueError("Speaker name must be at least 2 characters")


@dataclass(frozen=True)
class SpeechContent:
    """
    Extracted and validated speech content.
    
    Contains the actual speech text plus technical metadata about the extraction.
    """
    raw_text: str
    cleaned_text: str
    extraction_method: str
    confidence_score: float
    word_count: int
    extraction_timestamp: datetime
    
    def __post_init__(self):
        if self.confidence_score < 0.0 or self.confidence_score > 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        if self.word_count < 0:
            raise ValueError("Word count cannot be negative")
        if len(self.cleaned_text.strip()) < 100:
            raise ValueError("Cleaned text must be at least 100 characters")


@dataclass(frozen=True)
class Speaker:
    """
    Information about a central bank speaker.
    
    Represents someone who gives speeches at a central bank, with their
    role and institutional context.
    """
    name: str
    role: str
    institution_code: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    voting_member: bool = False
    biographical_notes: Optional[str] = None
    
    @property
    def is_current(self) -> bool:
        """Check if speaker is currently in their role."""
        if self.end_date is None:
            return True
        return self.end_date >= date.today()
    
    @property
    def tenure_years(self) -> Optional[float]:
        """Calculate years in role, if dates available."""
        if self.start_date is None:
            return None
        end = self.end_date or date.today()
        return (end - self.start_date).days / 365.25


@dataclass
class ValidationResult:
    """
    Result of speech validation process.
    
    Indicates whether extracted content represents a genuine speech
    and provides diagnostic information for quality assurance.
    """
    status: ValidationStatus
    confidence: float
    issues: List[str]
    metadata: Dict[str, Any]
    
    @property
    def is_valid(self) -> bool:
        """True if speech passed validation."""
        return self.status == ValidationStatus.VALID
    
    @property
    def has_issues(self) -> bool:
        """True if validation found any issues."""
        return len(self.issues) > 0


class SpeakerDatabase:
    """
    Interface for speaker information database.
    
    Each plugin provides a database of speakers for their institution,
    enabling accurate speaker recognition and role assignment.
    """
    
    def __init__(self, speakers: List[Speaker]):
        self._speakers = {speaker.name.lower(): speaker for speaker in speakers}
    
    def find_speaker(self, name: str) -> Optional[Speaker]:
        """
        Find speaker by name (case-insensitive).
        
        Args:
            name: Speaker name to search for
            
        Returns:
            Speaker object if found, None otherwise
            
        Example:
            >>> db = SpeakerDatabase([Speaker("Jerome Powell", "Chair", "FED")])
            >>> speaker = db.find_speaker("jerome powell")
            >>> assert speaker.role == "Chair"
        """
        return self._speakers.get(name.lower())
    
    def get_speakers_by_role(self, role: str) -> List[Speaker]:
        """Get all speakers with a specific role."""
        return [s for s in self._speakers.values() if s.role.lower() == role.lower()]
    
    def get_current_speakers(self) -> List[Speaker]:
        """Get all speakers currently in their roles."""
        return [s for s in self._speakers.values() if s.is_current]
    
    @property
    def speaker_count(self) -> int:
        """Total number of speakers in database."""
        return len(self._speakers)


class CentralBankScraperPlugin(ABC):
    """
    Abstract base class for all central bank scraper plugins.
    
    This is THE core contract of the entire platform. Every central bank
    implementation must inherit from this class and implement all abstract methods.
    
    The plugin system enables:
    - Infinite scalability (easy to add new central banks)
    - Plugin isolation (failures don't cascade)
    - Consistent interfaces (all plugins work the same way)
    - Testability (each plugin can be tested independently)
    
    Design Principles:
    - Fail fast with clear error messages
    - Log extensively for debugging
    - Handle rate limiting gracefully
    - Validate all inputs and outputs
    - Use typing everywhere
    """
    
    @abstractmethod
    def get_institution_code(self) -> str:
        """
        Get the unique institution identifier code.
        
        Returns:
            Short, uppercase institution code (e.g., "FED", "BOE", "ECB")
            
        Example:
            >>> plugin = FederalReservePlugin()
            >>> assert plugin.get_institution_code() == "FED"
        """
        pass
    
    @abstractmethod
    def get_institution_name(self) -> str:
        """
        Get the full institution name.
        
        Returns:
            Complete official name of the central bank
            
        Example:
            >>> plugin = FederalReservePlugin()
            >>> assert plugin.get_institution_name() == "Federal Reserve System"
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of languages this plugin can process.
        
        Returns:
            List of ISO 639-1 language codes
            
        Example:
            >>> plugin = ECBPlugin()
            >>> langs = plugin.get_supported_languages()
            >>> assert "en" in langs and "de" in langs
        """
        pass
    
    @abstractmethod
    def discover_speeches(self, date_range: DateRange, limit: Optional[int] = None) -> List[SpeechMetadata]:
        """
        Discover available speeches within a date range.
        
        This is the first phase of speech collection - finding what speeches exist
        without extracting their content. Must be fast and reliable.
        
        Args:
            date_range: Date range to search within
            limit: Maximum number of speeches to return (None for no limit)
            
        Returns:
            List of speech metadata objects, sorted by date (newest first)
            
        Raises:
            ConnectionError: If unable to connect to institution website
            ValueError: If date_range is invalid
            
        Example:
            >>> plugin = FederalReservePlugin()
            >>> date_range = DateRange(date(2024, 1, 1), date(2024, 12, 31))
            >>> speeches = plugin.discover_speeches(date_range, limit=10)
            >>> assert len(speeches) <= 10
            >>> assert all(s.institution_code == "FED" for s in speeches)
        """
        pass
    
    @abstractmethod
    def extract_speech_content(self, speech_metadata: SpeechMetadata) -> SpeechContent:
        """
        Extract full text content from a speech.
        
        This is the second phase - getting the actual speech text from the URL
        identified in the discovery phase. Must handle PDFs, HTML, and other formats.
        
        Args:
            speech_metadata: Metadata from discovery phase
            
        Returns:
            Extracted and cleaned speech content
            
        Raises:
            ContentExtractionError: If unable to extract readable content
            ValidationError: If extracted content doesn't appear to be a speech
            
        Example:
            >>> plugin = FederalReservePlugin()
            >>> metadata = SpeechMetadata(url="...", title="...", ...)
            >>> content = plugin.extract_speech_content(metadata)
            >>> assert content.word_count > 100
            >>> assert content.confidence_score > 0.8
        """
        pass
    
    @abstractmethod
    def get_speaker_database(self) -> SpeakerDatabase:
        """
        Get the speaker database for this institution.
        
        Returns comprehensive information about current and historical speakers,
        enabling accurate speaker recognition and role assignment.
        
        Returns:
            Database containing all known speakers for this institution
            
        Example:
            >>> plugin = FederalReservePlugin()
            >>> db = plugin.get_speaker_database()
            >>> powell = db.find_speaker("Jerome Powell")
            >>> assert powell.role == "Chair"
            >>> assert powell.voting_member == True
        """
        pass
    
    @abstractmethod
    def validate_speech_authenticity(self, speech_metadata: SpeechMetadata, content: SpeechContent) -> ValidationResult:
        """
        Validate that extracted content represents a genuine speech.
        
        Performs institution-specific validation to ensure we're collecting
        real speeches, not boilerplate, navigation text, or other noise.
        
        Args:
            speech_metadata: Original speech metadata
            content: Extracted speech content
            
        Returns:
            Validation result with status and diagnostic information
            
        Example:
            >>> plugin = FederalReservePlugin()
            >>> result = plugin.validate_speech_authenticity(metadata, content)
            >>> if result.is_valid:
            ...     print("Speech is authentic")
            >>> else:
            ...     print(f"Issues: {result.issues}")
        """
        pass
    
    def get_rate_limit_delay(self) -> float:
        """
        Get recommended delay between requests (in seconds).
        
        Returns:
            Delay in seconds to be respectful to institution servers
            
        Default implementation returns 1.0 seconds.
        Override for institution-specific requirements.
        """
        return 1.0
    
    def supports_bulk_download(self) -> bool:
        """
        Check if this plugin supports bulk downloading.
        
        Returns:
            True if plugin can download multiple speeches efficiently
            
        Default implementation returns False.
        Override for institutions that provide bulk data access.
        """
        return False
    
    def get_plugin_version(self) -> str:
        """
        Get the version of this plugin.
        
        Returns:
            Semantic version string
            
        Default implementation returns "1.0.0".
        Override to track plugin versions.
        """
        return "1.0.0"


# Custom Exceptions for Plugin System

class PluginError(Exception):
    """Base exception for all plugin-related errors."""
    pass


class ContentExtractionError(PluginError):
    """Raised when speech content cannot be extracted."""
    pass


class ValidationError(PluginError):
    """Raised when speech content fails validation."""
    pass


class SpeakerRecognitionError(PluginError):
    """Raised when speaker cannot be identified."""
    pass


class RateLimitError(PluginError):
    """Raised when rate limits are exceeded."""
    pass


# Type aliases for clarity
PluginRegistry = Dict[str, CentralBankScraperPlugin]
SpeechCollection = List[SpeechMetadata]