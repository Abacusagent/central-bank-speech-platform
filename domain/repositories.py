"""
Repository Interfaces for Central Bank Speech Analysis Platform

This module defines abstract repository interfaces that specify how domain entities
can be persisted and retrieved, without coupling to specific storage technologies.
Following the Repository pattern and Dependency Inversion Principle.

Key Principles:
- Abstract interfaces define contracts, not implementations
- Domain layer depends on abstractions, not concrete implementations
- Repository methods use domain objects and value objects
- Query methods return domain entities, not raw data
- Implementations live in infrastructure layer

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import List, Optional, Dict, Any, Set, Union, AsyncIterator, Tuple
from uuid import UUID

from domain.entities import (
    CentralBankSpeech, CentralBankSpeaker, Institution, 
    SentimentAnalysis, SpeechStatus, PolicyStance, InstitutionType
)
from domain.value_objects import DateRange, Url, ContentHash, ConfidenceLevel
from interfaces.plugin_interfaces import SpeechMetadata, SpeechContent, ValidationResult


class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass


class EntityNotFoundError(RepositoryError):
    """Raised when a requested entity is not found."""
    pass


class DuplicateEntityError(RepositoryError):
    """Raised when attempting to create an entity that already exists."""
    pass


class ConcurrencyError(RepositoryError):
    """Raised when concurrent modifications conflict."""
    pass


class QuerySpecification(ABC):
    """
    Abstract base class for query specifications.
    
    Specifications encapsulate query logic and can be combined
    for complex queries without coupling to specific storage.
    """
    
    @abstractmethod
    def is_satisfied_by(self, entity: Any) -> bool:
        """Check if entity satisfies this specification."""
        pass
    
    def and_spec(self, other: 'QuerySpecification') -> 'QuerySpecification':
        """Combine with another specification using AND logic."""
        return AndSpecification(self, other)
    
    def or_spec(self, other: 'QuerySpecification') -> 'QuerySpecification':
        """Combine with another specification using OR logic."""
        return OrSpecification(self, other)
    
    def not_spec(self) -> 'QuerySpecification':
        """Negate this specification."""
        return NotSpecification(self)


class AndSpecification(QuerySpecification):
    """Combines two specifications with AND logic."""
    
    def __init__(self, left: QuerySpecification, right: QuerySpecification):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, entity: Any) -> bool:
        return self.left.is_satisfied_by(entity) and self.right.is_satisfied_by(entity)


class OrSpecification(QuerySpecification):
    """Combines two specifications with OR logic."""
    
    def __init__(self, left: QuerySpecification, right: QuerySpecification):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, entity: Any) -> bool:
        return self.left.is_satisfied_by(entity) or self.right.is_satisfied_by(entity)


class NotSpecification(QuerySpecification):
    """Negates a specification."""
    
    def __init__(self, specification: QuerySpecification):
        self.specification = specification
    
    def is_satisfied_by(self, entity: Any) -> bool:
        return not self.specification.is_satisfied_by(entity)


class SpeechRepository(ABC):
    """
    Abstract repository for Central Bank Speech entities.
    
    Provides methods for storing, retrieving, and querying speeches
    without coupling to specific storage technology.
    """
    
    @abstractmethod
    async def save(self, speech: CentralBankSpeech) -> None:
        """
        Save a speech to the repository.
        
        Args:
            speech: Speech entity to save
            
        Raises:
            RepositoryError: If save operation fails
            
        Example:
            >>> speech = CentralBankSpeech(...)
            >>> await repository.save(speech)
        """
        pass
    
    @abstractmethod
    async def save_batch(self, speeches: List[CentralBankSpeech]) -> None:
        """
        Save multiple speeches in a single operation for performance.
        
        Args:
            speeches: List of speech entities to save
            
        Raises:
            RepositoryError: If batch save operation fails
            
        Example:
            >>> speeches = [CentralBankSpeech(...), CentralBankSpeech(...)]
            >>> await repository.save_batch(speeches)
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, speech_id: UUID) -> Optional[CentralBankSpeech]:
        """
        Retrieve a speech by its unique ID.
        
        Args:
            speech_id: Unique identifier for the speech
            
        Returns:
            Speech entity if found, None otherwise
            
        Example:
            >>> speech_id = UUID("12345678-1234-5678-9012-123456789012")
            >>> speech = await repository.get_by_id(speech_id)
        """
        pass
    
    @abstractmethod
    async def get_by_url(self, url: str) -> Optional[CentralBankSpeech]:
        """
        Retrieve a speech by its source URL from metadata.
        
        Args:
            url: Source URL of the speech
            
        Returns:
            Speech entity if found, None otherwise
            
        Example:
            >>> url = "https://www.federalreserve.gov/newsevents/speech/..."
            >>> speech = await repository.get_by_url(url)
        """
        pass
    
    @abstractmethod
    async def get_by_content_hash(self, content_hash: ContentHash) -> Optional[CentralBankSpeech]:
        """
        Retrieve a speech by its content hash (for deduplication).
        
        Args:
            content_hash: Hash of the speech content
            
        Returns:
            Speech entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_by_speaker(self, speaker: CentralBankSpeaker, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """
        Find all speeches by a specific speaker.
        
        Args:
            speaker: Speaker to search for
            limit: Maximum number of results to return
            
        Returns:
            List of speeches by the speaker, ordered by date (newest first)
            
        Example:
            >>> speaker = CentralBankSpeaker(name="Jerome Powell", ...)
            >>> speeches = await repository.find_by_speaker(speaker, limit=10)
        """
        pass
    
    @abstractmethod
    async def find_by_speaker_name(self, speaker_name: str, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """
        Find all speeches by speaker name (more flexible than find_by_speaker).
        
        Args:
            speaker_name: Name of speaker to search for
            limit: Maximum number of results to return
            
        Returns:
            List of speeches by speakers matching the name
        """
        pass
    
    @abstractmethod
    async def find_by_institution(self, institution: Institution, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """
        Find all speeches from a specific institution.
        
        Args:
            institution: Institution to search for
            limit: Maximum number of results to return
            
        Returns:
            List of speeches from the institution, ordered by date (newest first)
        """
        pass
    
    @abstractmethod
    async def find_by_institution_code(self, institution_code: str, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """
        Find all speeches from an institution by code.
        
        Args:
            institution_code: Institution code (e.g., "FED", "BOE")
            limit: Maximum number of results to return
            
        Returns:
            List of speeches from the institution
        """
        pass
    
    @abstractmethod
    async def find_by_date_range(self, date_range: DateRange, institution: Optional[Institution] = None) -> List[CentralBankSpeech]:
        """
        Find speeches within a date range.
        
        Args:
            date_range: Date range to search within
            institution: Optional institution filter
            
        Returns:
            List of speeches within the date range, ordered by date (newest first)
            
        Example:
            >>> from domain.value_objects import DateRange
            >>> date_range = DateRange.year(2024)
            >>> speeches = await repository.find_by_date_range(date_range)
        """
        pass
    
    @abstractmethod
    async def find_by_status(self, status: SpeechStatus, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """
        Find speeches with a specific processing status.
        
        Args:
            status: Processing status to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of speeches with the specified status
        """
        pass
    
    @abstractmethod
    async def find_by_policy_stance(self, stance: PolicyStance, date_range: Optional[DateRange] = None) -> List[CentralBankSpeech]:
        """
        Find speeches with a specific policy stance.
        
        Args:
            stance: Policy stance to filter by
            date_range: Optional date range filter
            
        Returns:
            List of speeches with the specified stance
        """
        pass
    
    @abstractmethod
    async def find_by_sentiment_range(self, min_score: float, max_score: float, 
                                     institution: Optional[Institution] = None) -> List[CentralBankSpeech]:
        """
        Find speeches with sentiment scores within a range.
        
        Args:
            min_score: Minimum sentiment score (inclusive)
            max_score: Maximum sentiment score (inclusive)
            institution: Optional institution filter
            
        Returns:
            List of speeches with sentiment scores in the range
        """
        pass
    
    @abstractmethod
    async def find_by_specification(self, specification: QuerySpecification, 
                                   limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """
        Find speeches matching a complex specification.
        
        Args:
            specification: Query specification to match
            limit: Maximum number of results to return
            
        Returns:
            List of speeches matching the specification
            
        Example:
            >>> spec = SpeakerNameSpec("Powell").and_spec(DateRangeSpec(DateRange.year(2024)))
            >>> speeches = await repository.find_by_specification(spec, limit=50)
        """
        pass
    
    @abstractmethod
    async def stream_by_institution(self, institution: Institution) -> AsyncIterator[CentralBankSpeech]:
        """
        Stream speeches from an institution for memory-efficient processing.
        
        Args:
            institution: Institution to stream speeches from
            
        Yields:
            Speech entities one at a time
            
        Example:
            >>> async for speech in repository.stream_by_institution(fed):
            ...     await process_speech(speech)
        """
        pass
    
    @abstractmethod
    async def get_unprocessed_speeches(self, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """
        Get speeches that haven't been analyzed yet.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of speeches without sentiment analysis
        """
        pass
    
    @abstractmethod
    async def count_by_institution(self, institution: Institution) -> int:
        """
        Count speeches from a specific institution.
        
        Args:
            institution: Institution to count speeches for
            
        Returns:
            Number of speeches from the institution
        """
        pass
    
    @abstractmethod
    async def count_by_status(self, status: SpeechStatus) -> int:
        """
        Count speeches with a specific status.
        
        Args:
            status: Status to count
            
        Returns:
            Number of speeches with the status
        """
        pass
    
    @abstractmethod
    async def count_by_date_range(self, date_range: DateRange, institution: Optional[Institution] = None) -> int:
        """
        Count speeches within a date range.
        
        Args:
            date_range: Date range to count within
            institution: Optional institution filter
            
        Returns:
            Number of speeches in the date range
        """
        pass
    
    @abstractmethod
    async def get_all_institutions(self) -> List[Institution]:
        """
        Get all institutions that have speeches in the repository.
        
        Returns:
            List of institutions with speeches
        """
        pass
    
    @abstractmethod
    async def get_date_range_for_institution(self, institution: Institution) -> Optional[DateRange]:
        """
        Get the date range of speeches for an institution.
        
        Args:
            institution: Institution to get date range for
            
        Returns:
            Date range covering all speeches from the institution, or None if no speeches
        """
        pass
    
    @abstractmethod
    async def delete(self, speech: CentralBankSpeech) -> None:
        """
        Delete a speech from the repository.
        
        Args:
            speech: Speech to delete
            
        Raises:
            EntityNotFoundError: If speech doesn't exist
            RepositoryError: If deletion fails
        """
        pass
    
    @abstractmethod
    async def exists(self, speech_id: UUID) -> bool:
        """
        Check if a speech exists in the repository.
        
        Args:
            speech_id: ID of speech to check
            
        Returns:
            True if speech exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def exists_by_url(self, url: str) -> bool:
        """
        Check if a speech exists by URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if speech with URL exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_processing_metrics(self, date_range: Optional[DateRange] = None) -> Dict[str, Any]:
        """
        Get processing metrics for speeches.
        
        Args:
            date_range: Optional date range to filter metrics
            
        Returns:
            Dictionary containing processing metrics
        """
        pass


class SpeakerRepository(ABC):
    """
    Abstract repository for Central Bank Speaker entities.
    
    Manages speaker information and provides lookup capabilities
    for speaker recognition and role assignment.
    """
    
    @abstractmethod
    async def save(self, speaker: CentralBankSpeaker) -> None:
        """
        Save a speaker to the repository.
        
        Args:
            speaker: Speaker entity to save
            
        Raises:
            RepositoryError: If save operation fails
        """
        pass
    
    @abstractmethod
    async def save_batch(self, speakers: List[CentralBankSpeaker]) -> None:
        """
        Save multiple speakers in a single operation.
        
        Args:
            speakers: List of speaker entities to save
            
        Raises:
            RepositoryError: If batch save operation fails
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, speaker_id: UUID) -> Optional[CentralBankSpeaker]:
        """
        Retrieve a speaker by their unique ID.
        
        Args:
            speaker_id: Unique identifier for the speaker
            
        Returns:
            Speaker entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> List[CentralBankSpeaker]:
        """
        Find speakers by name (including partial matches).
        
        Args:
            name: Name to search for
            
        Returns:
            List of speakers matching the name
        """
        pass
    
    @abstractmethod
    async def find_by_name_fuzzy(self, name: str, threshold: float = 0.8) -> List[CentralBankSpeaker]:
        """
        Find speakers using fuzzy name matching.
        
        Args:
            name: Name to search for
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of speakers with similar names
        """
        pass
    
    @abstractmethod
    async def find_by_institution(self, institution: Institution) -> List[CentralBankSpeaker]:
        """
        Find all speakers from a specific institution.
        
        Args:
            institution: Institution to search for
            
        Returns:
            List of speakers from the institution
        """
        pass
    
    @abstractmethod
    async def find_current_speakers(self, institution: Optional[Institution] = None) -> List[CentralBankSpeaker]:
        """
        Find all currently active speakers.
        
        Args:
            institution: Optional institution filter
            
        Returns:
            List of currently active speakers
        """
        pass
    
    @abstractmethod
    async def find_by_role(self, role: str, institution: Optional[Institution] = None) -> List[CentralBankSpeaker]:
        """
        Find speakers by their role.
        
        Args:
            role: Role to search for (e.g., "Chair", "Governor")
            institution: Optional institution filter
            
        Returns:
            List of speakers with the specified role
        """
        pass
    
    @abstractmethod
    async def find_voting_members(self, institution: Institution) -> List[CentralBankSpeaker]:
        """
        Find all voting members at an institution.
        
        Args:
            institution: Institution to search within
            
        Returns:
            List of voting members
        """
        pass
    
    @abstractmethod
    async def search_by_alternate_names(self, name: str) -> List[CentralBankSpeaker]:
        """
        Search for speakers by alternate names/spellings.
        
        Args:
            name: Name to search for in alternate names
            
        Returns:
            List of speakers with matching alternate names
        """
        pass
    
    @abstractmethod
    async def get_speaker_statistics(self, speaker: CentralBankSpeaker) -> Dict[str, Any]:
        """
        Get statistics for a speaker (speech count, date range, etc.).
        
        Args:
            speaker: Speaker to get statistics for
            
        Returns:
            Dictionary with speaker statistics
        """
        pass
    
    @abstractmethod
    async def get_speakers_by_speech_count(self, min_speeches: int = 1) -> List[Tuple[CentralBankSpeaker, int]]:
        """
        Get speakers with their speech counts.
        
        Args:
            min_speeches: Minimum number of speeches to include speaker
            
        Returns:
            List of tuples (speaker, speech_count) ordered by speech count desc
        """
        pass
    
    @abstractmethod
    async def delete(self, speaker: CentralBankSpeaker) -> None:
        """
        Delete a speaker from the repository.
        
        Args:
            speaker: Speaker to delete
            
        Raises:
            EntityNotFoundError: If speaker doesn't exist
            RepositoryError: If deletion fails
        """
        pass
    
    @abstractmethod
    async def exists(self, speaker_id: UUID) -> bool:
        """
        Check if a speaker exists in the repository.
        
        Args:
            speaker_id: ID of speaker to check
            
        Returns:
            True if speaker exists, False otherwise
        """
        pass


class InstitutionRepository(ABC):
    """
    Abstract repository for Institution entities.
    
    Manages central bank institution information and provides
    lookup capabilities for institutional context.
    """
    
    @abstractmethod
    async def save(self, institution: Institution) -> None:
        """
        Save an institution to the repository.
        
        Args:
            institution: Institution entity to save
            
        Raises:
            RepositoryError: If save operation fails
        """
        pass
    
    @abstractmethod
    async def get_by_code(self, code: str) -> Optional[Institution]:
        """
        Retrieve an institution by its code.
        
        Args:
            code: Institution code (e.g., "FED", "BOE", "ECB")
            
        Returns:
            Institution entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Institution]:
        """
        Retrieve an institution by its name.
        
        Args:
            name: Institution name
            
        Returns:
            Institution entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Institution]:
        """
        Get all institutions in the repository.
        
        Returns:
            List of all institutions
        """
        pass
    
    @abstractmethod
    async def find_by_country(self, country: str) -> List[Institution]:
        """
        Find institutions by country.
        
        Args:
            country: Country name or code
            
        Returns:
            List of institutions in the country
        """
        pass
    
    @abstractmethod
    async def find_by_type(self, institution_type: InstitutionType) -> List[Institution]:
        """
        Find institutions by type.
        
        Args:
            institution_type: Type of institution
            
        Returns:
            List of institutions of the specified type
        """
        pass
    
    @abstractmethod
    async def get_institution_statistics(self, institution: Institution) -> Dict[str, Any]:
        """
        Get statistics for an institution.
        
        Args:
            institution: Institution to get statistics for
            
        Returns:
            Dictionary with institution statistics (speech count, speakers, etc.)
        """
        pass
    
    @abstractmethod
    async def delete(self, institution: Institution) -> None:
        """
        Delete an institution from the repository.
        
        Args:
            institution: Institution to delete
            
        Raises:
            EntityNotFoundError: If institution doesn't exist
            RepositoryError: If deletion fails
        """
        pass
    
    @abstractmethod
    async def exists(self, code: str) -> bool:
        """
        Check if an institution exists in the repository.
        
        Args:
            code: Institution code to check
            
        Returns:
            True if institution exists, False otherwise
        """
        pass


class AnalysisRepository(ABC):
    """
    Abstract repository for sentiment analysis results.
    
    Stores and retrieves NLP analysis results separately from speeches
    to support different analysis versions and approaches.
    """
    
    @abstractmethod
    async def save_analysis(self, speech_id: UUID, analysis: SentimentAnalysis) -> None:
        """
        Save sentiment analysis results for a speech.
        
        Args:
            speech_id: ID of the speech that was analyzed
            analysis: Analysis results to save
            
        Raises:
            RepositoryError: If save operation fails
        """
        pass
    
    @abstractmethod
    async def save_analyses_batch(self, analyses: List[Tuple[UUID, SentimentAnalysis]]) -> None:
        """
        Save multiple analysis results in a single operation.
        
        Args:
            analyses: List of (speech_id, analysis) tuples
            
        Raises:
            RepositoryError: If batch save operation fails
        """
        pass
    
    @abstractmethod
    async def get_latest_analysis(self, speech_id: UUID) -> Optional[SentimentAnalysis]:
        """
        Get the latest analysis for a speech.
        
        Args:
            speech_id: ID of the speech
            
        Returns:
            Latest analysis results if available, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_analysis_by_version(self, speech_id: UUID, analyzer_version: str) -> Optional[SentimentAnalysis]:
        """
        Get analysis results for a specific analyzer version.
        
        Args:
            speech_id: ID of the speech
            analyzer_version: Version of the analyzer
            
        Returns:
            Analysis results for the specified version, None if not found
        """
        pass
    
    @abstractmethod
    async def get_all_analyses(self, speech_id: UUID) -> List[SentimentAnalysis]:
        """
        Get all analysis results for a speech.
        
        Args:
            speech_id: ID of the speech
            
        Returns:
            List of all analysis results for the speech, ordered by timestamp
        """
        pass
    
    @abstractmethod
    async def delete_analysis(self, speech_id: UUID, analyzer_version: str) -> None:
        """
        Delete specific analysis results.
        
        Args:
            speech_id: ID of the speech
            analyzer_version: Version of the analyzer
            
        Raises:
            EntityNotFoundError: If analysis doesn't exist
            RepositoryError: If deletion fails
        """
        pass
    
    @abstractmethod
    async def get_analysis_summary(self, date_range: Optional[DateRange] = None) -> Dict[str, Any]:
        """
        Get summary statistics for analysis results.
        
        Args:
            date_range: Optional date range to filter results
            
        Returns:
            Dictionary containing analysis summary statistics
        """
        pass
    
    @abstractmethod
    async def find_analyses_by_stance(self, stance: PolicyStance, 
                                     date_range: Optional[DateRange] = None) -> List[Tuple[UUID, SentimentAnalysis]]:
        """
        Find analyses by policy stance.
        
        Args:
            stance: Policy stance to filter by
            date_range: Optional date range filter
            
        Returns:
            List of (speech_id, analysis) tuples
        """
        pass
    
    @abstractmethod
    async def get_confidence_distribution(self) -> Dict[ConfidenceLevel, int]:
        """
        Get distribution of confidence levels across all analyses.
        
        Returns:
            Dictionary mapping confidence levels to counts
        """
        pass


class UnitOfWork(ABC):
    """
    Abstract Unit of Work for managing transactions across repositories.
    
    Provides atomic operations across multiple repositories and ensures
    consistency in complex operations.
    """
    
    @abstractmethod
    async def __aenter__(self) -> 'UnitOfWork':
        """Start a new unit of work."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Complete or rollback the unit of work."""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit all changes in this unit of work."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback all changes in this unit of work."""
        pass
    
    @property
    @abstractmethod
    def speeches(self) -> SpeechRepository:
        """Get the speech repository for this unit of work."""
        pass
    
    @property
    @abstractmethod
    def speakers(self) -> SpeakerRepository:
        """Get the speaker repository for this unit of work."""
        pass
    
    @property
    @abstractmethod
    def institutions(self) -> InstitutionRepository:
        """Get the institution repository for this unit of work."""
        pass
    
    @property
    @abstractmethod
    def analyses(self) -> AnalysisRepository:
        """Get the analysis repository for this unit of work."""
        pass


# Common Query Specifications

class SpeakerNameSpecification(QuerySpecification):
    """Specification for filtering speeches by speaker name."""
    
    def __init__(self, speaker_name: str):
        self.speaker_name = speaker_name.lower()
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return (speech.speaker is not None and 
                self.speaker_name in speech.speaker.name.lower())


class DateRangeSpecification(QuerySpecification):
    """Specification for filtering speeches by date range."""
    
    def __init__(self, date_range: DateRange):
        self.date_range = date_range
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        speech_date = None
        if speech.metadata and speech.metadata.date:
            speech_date = speech.metadata.date
        return speech_date is not None and self.date_range.contains(speech_date)


class InstitutionSpecification(QuerySpecification):
    """Specification for filtering speeches by institution."""
    
    def __init__(self, institution: Institution):
        self.institution = institution
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return speech.institution == self.institution


class InstitutionCodeSpecification(QuerySpecification):
    """Specification for filtering speeches by institution code."""
    
    def __init__(self, institution_code: str):
        self.institution_code = institution_code.upper()
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        if speech.institution:
            return speech.institution.code.upper() == self.institution_code
        elif speech.metadata:
            return speech.metadata.institution_code.upper() == self.institution_code
        return False


class PolicyStanceSpecification(QuerySpecification):
    """Specification for filtering speeches by policy stance."""
    
    def __init__(self, stance: PolicyStance):
        self.stance = stance
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return (speech.sentiment_analysis is not None and 
                speech.sentiment_analysis.policy_stance == self.stance)


class SentimentScoreRangeSpecification(QuerySpecification):
    """Specification for filtering speeches by sentiment score range."""
    
    def __init__(self, min_score: float, max_score: float):
        self.min_score = min_score
        self.max_score = max_score
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return (speech.sentiment_analysis is not None and 
                self.min_score <= speech.sentiment_analysis.hawkish_dovish_score <= self.max_score)


class StatusSpecification(QuerySpecification):
    """Specification for filtering speeches by processing status."""
    
    def __init__(self, status: SpeechStatus):
        self.status = status
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return speech.status == self.status


class TagSpecification(QuerySpecification):
    """Specification for filtering speeches by tag."""
    
    def __init__(self, tag: str):
        self.tag = tag.lower()
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return any(self.tag in existing_tag.lower() for existing_tag in speech.tags)


class WordCountRangeSpecification(QuerySpecification):
    """Specification for filtering speeches by word count range."""
    
    def __init__(self, min_words: int, max_words: int):
        self.min_words = min_words
        self.max_words = max_words
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        word_count = 0
        if speech.content and speech.content.word_count:
            word_count = speech.content.word_count
        return self.min_words <= word_count <= self.max_words


class ConfidenceThresholdSpecification(QuerySpecification):
    """Specification for filtering speeches by analysis confidence."""
    
    def __init__(self, min_confidence: float):
        self.min_confidence = min_confidence
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return (speech.sentiment_analysis is not None and 
                speech.sentiment_analysis.confidence_score >= self.min_confidence)


class UncertaintyRangeSpecification(QuerySpecification):
    """Specification for filtering speeches by uncertainty score range."""
    
    def __init__(self, min_uncertainty: float, max_uncertainty: float):
        self.min_uncertainty = min_uncertainty
        self.max_uncertainty = max_uncertainty
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return (speech.sentiment_analysis is not None and 
                self.min_uncertainty <= speech.sentiment_analysis.uncertainty_score <= self.max_uncertainty)


class TopicSpecification(QuerySpecification):
    """Specification for filtering speeches by topic classification."""
    
    def __init__(self, topic: str):
        self.topic = topic.lower()
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        if speech.sentiment_analysis and speech.sentiment_analysis.topic_classifications:
            return any(self.topic in topic.lower() for topic in speech.sentiment_analysis.topic_classifications)
        return False


class HasAnalysisSpecification(QuerySpecification):
    """Specification for filtering speeches that have been analyzed."""
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return speech.sentiment_analysis is not None


class HasContentSpecification(QuerySpecification):
    """Specification for filtering speeches that have extracted content."""
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return speech.content is not None and speech.content.cleaned_text.strip() != ""


class ValidationStatusSpecification(QuerySpecification):
    """Specification for filtering speeches by validation status."""
    
    def __init__(self, validation_status: str):
        self.validation_status = validation_status.lower()
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        return (speech.validation_result is not None and 
                speech.validation_result.status.value.lower() == self.validation_status)


class RecentSpeechSpecification(QuerySpecification):
    """Specification for filtering recent speeches (within last N days)."""
    
    def __init__(self, days: int):
        from datetime import date, timedelta
        self.cutoff_date = date.today() - timedelta(days=days)
    
    def is_satisfied_by(self, speech: CentralBankSpeech) -> bool:
        speech_date = None
        if speech.metadata and speech.metadata.date:
            speech_date = speech.metadata.date
        return speech_date is not None and speech_date >= self.cutoff_date


# Convenience factory functions for common specifications

def create_federal_reserve_spec() -> QuerySpecification:
    """Create specification for Federal Reserve speeches."""
    return InstitutionCodeSpecification("FED")


def create_analyzed_speeches_spec() -> QuerySpecification:
    """Create specification for speeches with analysis."""
    return HasAnalysisSpecification()


def create_high_confidence_spec(min_confidence: float = 0.8) -> QuerySpecification:
    """Create specification for high-confidence analyses."""
    return ConfidenceThresholdSpecification(min_confidence)


def create_hawkish_speeches_spec() -> QuerySpecification:
    """Create specification for hawkish speeches."""
    return PolicyStanceSpecification(PolicyStance.HAWKISH)


def create_dovish_speeches_spec() -> QuerySpecification:
    """Create specification for dovish speeches."""
    return PolicyStanceSpecification(PolicyStance.DOVISH)


def create_recent_fed_speeches_spec(days: int = 30) -> QuerySpecification:
    """Create specification for recent Federal Reserve speeches."""
    return create_federal_reserve_spec().and_spec(RecentSpeechSpecification(days))


def create_analyzed_hawkish_spec() -> QuerySpecification:
    """Create specification for analyzed hawkish speeches."""
    return HasAnalysisSpecification().and_spec(PolicyStanceSpecification(PolicyStance.HAWKISH))