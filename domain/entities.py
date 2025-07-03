"""
Domain Entities for Central Bank Speech Analysis Platform

This module contains the core domain entities that represent the fundamental
business concepts of central bank speech analysis. These are rich domain objects
that encapsulate both data and behavior, following Domain-Driven Design principles.

Key Principles:
- Entities have identity (can be distinguished even if attributes are identical)
- Entities encapsulate business rules and invariants
- Entities are mutable but changes are controlled through methods
- Entities should be technology-agnostic (no database, web, or framework dependencies)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Set, Union
from uuid import UUID, uuid4

from interfaces.plugin_interfaces import (
    SpeechMetadata, SpeechContent, Speaker, ValidationResult, 
    SpeechType, ValidationStatus
)


class PolicyStance(Enum):
    """Monetary policy stance classifications."""
    HAWKISH = "hawkish"
    DOVISH = "dovish"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class SpeechStatus(Enum):
    """Processing status of a speech."""
    DISCOVERED = "discovered"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    ANALYZED = "analyzed"
    PUBLISHED = "published"
    FAILED = "failed"
    ARCHIVED = "archived"


class InstitutionType(Enum):
    """Types of central banking institutions."""
    CENTRAL_BANK = "central_bank"
    FEDERAL_RESERVE_BANK = "federal_reserve_bank"
    SUPERVISORY_AUTHORITY = "supervisory_authority"
    INTERNATIONAL_ORGANIZATION = "international_organization"


@dataclass
class Institution:
    """
    Represents a central banking institution.
    
    This is an entity with identity - two institutions are the same if they have
    the same code, even if other attributes differ.
    """
    code: str  # Unique identifier (e.g., "FED", "BOE", "ECB")
    name: str
    country: str
    institution_type: InstitutionType
    established_date: Optional[date] = None
    website_url: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate institution invariants."""
        if len(self.code) < 2:
            raise ValueError("Institution code must be at least 2 characters")
        if len(self.name.strip()) < 3:
            raise ValueError("Institution name must be at least 3 characters")
        if self.website_url and not self.website_url.startswith(('http://', 'https://')):
            raise ValueError("Website URL must be a valid HTTP(S) URL")
    
    def __eq__(self, other) -> bool:
        """Institutions are equal if they have the same code."""
        if not isinstance(other, Institution):
            return False
        return self.code == other.code
    
    def __hash__(self) -> int:
        """Hash based on code for use in sets and dictionaries."""
        return hash(self.code)
    
    def __repr__(self) -> str:
        return f"Institution(code='{self.code}', name='{self.name}')"


@dataclass
class CentralBankSpeaker:
    """
    Represents a person who gives speeches at a central bank.
    
    This is an entity with identity based on name and institution.
    Extends the basic Speaker from interfaces with domain-specific behavior.
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    role: str = ""
    institution: Optional[Institution] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    voting_member: bool = False
    biographical_notes: Optional[str] = None
    alternate_names: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Validate speaker invariants."""
        if len(self.name.strip()) < 2:
            raise ValueError("Speaker name must be at least 2 characters")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")
    
    def add_alternate_name(self, name: str) -> None:
        """
        Add an alternate name/spelling for this speaker.
        
        Args:
            name: Alternate name to add
            
        Example:
            >>> speaker = CentralBankSpeaker(name="Jerome Powell")
            >>> speaker.add_alternate_name("Jerome H. Powell")
            >>> assert "Jerome H. Powell" in speaker.alternate_names
        """
        if name.strip() and name != self.name:
            self.alternate_names.add(name.strip())
    
    def matches_name(self, name: str) -> bool:
        """
        Check if a name matches this speaker (including alternates).
        
        Args:
            name: Name to check against
            
        Returns:
            True if name matches this speaker
            
        Example:
            >>> speaker = CentralBankSpeaker(name="Jerome Powell")
            >>> speaker.add_alternate_name("J. Powell")
            >>> assert speaker.matches_name("J. Powell")
        """
        name_lower = name.strip().lower()
        return (name_lower == self.name.lower() or 
                any(alt.lower() == name_lower for alt in self.alternate_names))
    
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
    
    @property
    def display_name(self) -> str:
        """Get the display name with role information."""
        if self.role:
            return f"{self.name} ({self.role})"
        return self.name
    
    def __eq__(self, other) -> bool:
        """Speakers are equal if they have the same ID."""
        if not isinstance(other, CentralBankSpeaker):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID for use in sets and dictionaries."""
        return hash(self.id)
    
    def __repr__(self) -> str:
        return f"CentralBankSpeaker(name='{self.name}', role='{self.role}')"


@dataclass
class SentimentAnalysis:
    """
    Results of sentiment analysis on a speech.
    
    Contains various sentiment scores and classifications from different
    analytical approaches.
    """
    hawkish_dovish_score: float  # -1.0 (very dovish) to +1.0 (very hawkish)
    policy_stance: PolicyStance
    uncertainty_score: float  # 0.0 (certain) to 1.0 (very uncertain)
    confidence_score: float  # 0.0 (low confidence) to 1.0 (high confidence)
    analysis_timestamp: datetime
    analyzer_version: str
    raw_scores: Dict[str, float] = field(default_factory=dict)
    topic_classifications: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate sentiment analysis scores."""
        if not -1.0 <= self.hawkish_dovish_score <= 1.0:
            raise ValueError("Hawkish-dovish score must be between -1.0 and 1.0")
        if not 0.0 <= self.uncertainty_score <= 1.0:
            raise ValueError("Uncertainty score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
    
    @property
    def is_hawkish(self) -> bool:
        """True if speech is classified as hawkish."""
        return self.policy_stance == PolicyStance.HAWKISH
    
    @property
    def is_dovish(self) -> bool:
        """True if speech is classified as dovish."""
        return self.policy_stance == PolicyStance.DOVISH
    
    @property
    def is_neutral(self) -> bool:
        """True if speech is classified as neutral."""
        return self.policy_stance == PolicyStance.NEUTRAL
    
    @property
    def stance_strength(self) -> float:
        """
        Calculate the strength of the policy stance.
        
        Returns:
            Absolute value of hawkish-dovish score (0.0 to 1.0)
        """
        return abs(self.hawkish_dovish_score)
    
    def add_topic_classification(self, topic: str) -> None:
        """Add a topic classification to this analysis."""
        if topic.strip() and topic not in self.topic_classifications:
            self.topic_classifications.append(topic.strip())


@dataclass
class CentralBankSpeech:
    """
    The core domain entity representing a central bank speech.
    
    This is the heart of the domain model - a complete speech with all its
    associated metadata, content, and analysis results. This entity encapsulates
    the full lifecycle of a speech from discovery to analysis.
    
    Identity is based on the unique ID, allowing the same speech to be
    updated as it moves through processing stages.
    """
    id: UUID = field(default_factory=uuid4)
    metadata: Optional[SpeechMetadata] = None
    content: Optional[SpeechContent] = None
    speaker: Optional[CentralBankSpeaker] = None
    institution: Optional[Institution] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    validation_result: Optional[ValidationResult] = None
    status: SpeechStatus = SpeechStatus.DISCOVERED
    tags: Set[str] = field(default_factory=set)
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize speech with basic validation."""
        if self.metadata and self.institution:
            if self.metadata.institution_code != self.institution.code:
                raise ValueError("Metadata institution code must match institution code")
    
    def update_metadata(self, metadata: SpeechMetadata) -> None:
        """
        Update speech metadata and record the change.
        
        Args:
            metadata: New metadata to set
            
        Example:
            >>> speech = CentralBankSpeech()
            >>> metadata = SpeechMetadata(url="...", title="...", ...)
            >>> speech.update_metadata(metadata)
            >>> assert speech.status == SpeechStatus.DISCOVERED
        """
        self.metadata = metadata
        self.status = SpeechStatus.DISCOVERED
        self.updated_at = datetime.now()
        self._record_processing_event("metadata_updated")
    
    def set_content(self, content: SpeechContent) -> None:
        """
        Set the extracted content and update status.
        
        Args:
            content: Extracted speech content
            
        Example:
            >>> speech = CentralBankSpeech()
            >>> content = SpeechContent(raw_text="...", cleaned_text="...", ...)
            >>> speech.set_content(content)
            >>> assert speech.status == SpeechStatus.EXTRACTED
        """
        self.content = content
        self.status = SpeechStatus.EXTRACTED
        self.updated_at = datetime.now()
        self._record_processing_event("content_extracted")
    
    def set_validation_result(self, result: ValidationResult) -> None:
        """
        Set validation result and update status accordingly.
        
        Args:
            result: Validation result from plugin
            
        Example:
            >>> speech = CentralBankSpeech()
            >>> result = ValidationResult(ValidationStatus.VALID, 0.95, [], {})
            >>> speech.set_validation_result(result)
            >>> assert speech.status == SpeechStatus.VALIDATED
        """
        self.validation_result = result
        if result.is_valid:
            self.status = SpeechStatus.VALIDATED
        else:
            self.status = SpeechStatus.FAILED
        self.updated_at = datetime.now()
        self._record_processing_event("validation_completed", {
            "status": result.status.value,
            "confidence": result.confidence,
            "issues_count": len(result.issues)
        })
    
    def set_sentiment_analysis(self, analysis: SentimentAnalysis) -> None:
        """
        Set sentiment analysis results and update status.
        
        Args:
            analysis: Completed sentiment analysis
            
        Example:
            >>> speech = CentralBankSpeech()
            >>> analysis = SentimentAnalysis(0.3, PolicyStance.HAWKISH, ...)
            >>> speech.set_sentiment_analysis(analysis)
            >>> assert speech.status == SpeechStatus.ANALYZED
        """
        self.sentiment_analysis = analysis
        self.status = SpeechStatus.ANALYZED
        self.updated_at = datetime.now()
        self._record_processing_event("sentiment_analysis_completed", {
            "hawkish_dovish_score": analysis.hawkish_dovish_score,
            "policy_stance": analysis.policy_stance.value,
            "confidence": analysis.confidence_score
        })
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to this speech.
        
        Args:
            tag: Tag to add
            
        Example:
            >>> speech = CentralBankSpeech()
            >>> speech.add_tag("monetary_policy")
            >>> assert "monetary_policy" in speech.tags
        """
        if tag.strip():
            self.tags.add(tag.strip().lower())
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this speech."""
        self.tags.discard(tag.strip().lower())
        self.updated_at = datetime.now()
    
    def has_tag(self, tag: str) -> bool:
        """Check if speech has a specific tag."""
        return tag.strip().lower() in self.tags
    
    def _record_processing_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Record a processing event in the history."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details or {}
        }
        self.processing_history.append(event)
    
    @property
    def is_complete(self) -> bool:
        """Check if speech has been fully processed."""
        return (self.metadata is not None and 
                self.content is not None and 
                self.validation_result is not None and 
                self.validation_result.is_valid)
    
    @property
    def is_analyzed(self) -> bool:
        """Check if speech has been analyzed."""
        return self.sentiment_analysis is not None
    
    @property
    def word_count(self) -> Optional[int]:
        """Get word count if content is available."""
        return self.content.word_count if self.content else None
    
    @property
    def speech_date(self) -> Optional[date]:
        """Get speech date if metadata is available."""
        return self.metadata.date if self.metadata else None
    
    @property
    def title(self) -> Optional[str]:
        """Get speech title if metadata is available."""
        return self.metadata.title if self.metadata else None
    
    @property
    def url(self) -> Optional[str]:
        """Get speech URL if metadata is available."""
        return self.metadata.url if self.metadata else None
    
    @property
    def hawkish_dovish_score(self) -> Optional[float]:
        """Get hawkish-dovish score if analysis is available."""
        return self.sentiment_analysis.hawkish_dovish_score if self.sentiment_analysis else None
    
    @property
    def policy_stance(self) -> Optional[PolicyStance]:
        """Get policy stance if analysis is available."""
        return self.sentiment_analysis.policy_stance if self.sentiment_analysis else None
    
    def get_processing_duration(self) -> float:
        """
        Calculate total processing duration in seconds.
        
        Returns:
            Seconds from creation to last update
        """
        return (self.updated_at - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert speech to dictionary representation.
        
        Returns:
            Dictionary with all speech data
            
        Example:
            >>> speech = CentralBankSpeech()
            >>> data = speech.to_dict()
            >>> assert "id" in data
            >>> assert "status" in data
        """
        return {
            "id": str(self.id),
            "status": self.status.value,
            "title": self.title,
            "speaker_name": self.speaker.name if self.speaker else None,
            "institution_code": self.institution.code if self.institution else None,
            "speech_date": self.speech_date.isoformat() if self.speech_date else None,
            "url": self.url,
            "word_count": self.word_count,
            "hawkish_dovish_score": self.hawkish_dovish_score,
            "policy_stance": self.policy_stance.value if self.policy_stance else None,
            "tags": list(self.tags),
            "is_complete": self.is_complete,
            "is_analyzed": self.is_analyzed,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "processing_duration_seconds": self.get_processing_duration()
        }
    
    def __eq__(self, other) -> bool:
        """Speeches are equal if they have the same ID."""
        if not isinstance(other, CentralBankSpeech):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID for use in sets and dictionaries."""
        return hash(self.id)
    
    def __repr__(self) -> str:
        return f"CentralBankSpeech(id={self.id}, status={self.status.value}, title='{self.title}')"


@dataclass
class SpeechCollection:
    """
    A collection of related speeches with metadata.
    
    This entity represents a logical grouping of speeches, such as all speeches
    from a particular speaker, time period, or topic.
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: Optional[str] = None
    speeches: List[CentralBankSpeech] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_speech(self, speech: CentralBankSpeech) -> None:
        """
        Add a speech to this collection.
        
        Args:
            speech: Speech to add
            
        Example:
            >>> collection = SpeechCollection(name="Powell Speeches 2024")
            >>> speech = CentralBankSpeech()
            >>> collection.add_speech(speech)
            >>> assert len(collection.speeches) == 1
        """
        if speech not in self.speeches:
            self.speeches.append(speech)
            self.updated_at = datetime.now()
    
    def remove_speech(self, speech: CentralBankSpeech) -> None:
        """Remove a speech from this collection."""
        if speech in self.speeches:
            self.speeches.remove(speech)
            self.updated_at = datetime.now()
    
    def get_speeches_by_speaker(self, speaker_name: str) -> List[CentralBankSpeech]:
        """Get all speeches by a specific speaker."""
        return [s for s in self.speeches 
                if s.speaker and s.speaker.matches_name(speaker_name)]
    
    def get_speeches_by_date_range(self, start_date: date, end_date: date) -> List[CentralBankSpeech]:
        """Get all speeches within a date range."""
        return [s for s in self.speeches 
                if s.speech_date and start_date <= s.speech_date <= end_date]
    
    def get_speeches_by_stance(self, stance: PolicyStance) -> List[CentralBankSpeech]:
        """Get all speeches with a specific policy stance."""
        return [s for s in self.speeches 
                if s.policy_stance == stance]
    
    @property
    def speech_count(self) -> int:
        """Total number of speeches in collection."""
        return len(self.speeches)
    
    @property
    def completed_speech_count(self) -> int:
        """Number of fully processed speeches."""
        return len([s for s in self.speeches if s.is_complete])
    
    @property
    def analyzed_speech_count(self) -> int:
        """Number of analyzed speeches."""
        return len([s for s in self.speeches if s.is_analyzed])
    
    @property
    def average_hawkish_dovish_score(self) -> Optional[float]:
        """Calculate average hawkish-dovish score across all analyzed speeches."""
        scores = [s.hawkish_dovish_score for s in self.speeches 
                 if s.hawkish_dovish_score is not None]
        return sum(scores) / len(scores) if scores else None
    
    def __eq__(self, other) -> bool:
        """Collections are equal if they have the same ID."""
        if not isinstance(other, SpeechCollection):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID for use in sets and dictionaries."""
        return hash(self.id)
    
    def __repr__(self) -> str:
        return f"SpeechCollection(name='{self.name}', speech_count={self.speech_count})"