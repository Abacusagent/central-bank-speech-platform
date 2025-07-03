"""
Value Objects for Central Bank Speech Analysis Platform

This module contains immutable value objects that represent concepts without identity.
Value objects are compared by their attributes rather than identity, and once created,
they cannot be modified. They encapsulate domain concepts and provide type safety.

Key Principles:
- Value objects are immutable (frozen dataclasses)
- Equality is based on all attributes, not identity
- Value objects can be freely shared and copied
- They encapsulate validation logic and business rules
- They provide rich behavior through methods and properties
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any, Set, Tuple, Union, FrozenSet
from uuid import UUID
import re
from urllib.parse import urlparse


class ConfidenceLevel(Enum):
    """Confidence levels for various analyses."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ContentFormat(Enum):
    """Supported content formats for speech extraction."""
    HTML = "html"
    PDF = "pdf"
    PLAIN_TEXT = "plain_text"
    WORD_DOCUMENT = "word_document"
    UNKNOWN = "unknown"


class LanguageCode(Enum):
    """ISO 639-1 language codes for supported languages."""
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    JAPANESE = "ja"
    CHINESE = "zh"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"


@dataclass(frozen=True)
class SpeechId:
    """
    Unique identifier for a speech.
    
    Combines institution code and a unique identifier to ensure global uniqueness
    across all institutions in the platform.
    """
    institution_code: str
    unique_id: str
    
    def __post_init__(self):
        """Validate speech ID format."""
        if not self.institution_code or len(self.institution_code) < 2:
            raise ValueError("Institution code must be at least 2 characters")
        if not self.unique_id or len(self.unique_id) < 8:
            raise ValueError("Unique ID must be at least 8 characters")
        if not re.match(r'^[A-Z0-9_]+$', self.institution_code):
            raise ValueError("Institution code must be uppercase alphanumeric with underscores")
    
    @classmethod
    def from_string(cls, speech_id_str: str) -> 'SpeechId':
        """
        Create SpeechId from string representation.
        
        Args:
            speech_id_str: String in format "INSTITUTION_CODE:unique_id"
            
        Returns:
            SpeechId instance
            
        Example:
            >>> speech_id = SpeechId.from_string("FED:20241201_powell_speech_001")
            >>> assert speech_id.institution_code == "FED"
        """
        if ':' not in speech_id_str:
            raise ValueError("Speech ID string must contain ':' separator")
        
        institution_code, unique_id = speech_id_str.split(':', 1)
        return cls(institution_code=institution_code, unique_id=unique_id)
    
    def to_string(self) -> str:
        """
        Convert to string representation.
        
        Returns:
            String in format "INSTITUTION_CODE:unique_id"
            
        Example:
            >>> speech_id = SpeechId("FED", "20241201_powell_speech_001")
            >>> assert speech_id.to_string() == "FED:20241201_powell_speech_001"
        """
        return f"{self.institution_code}:{self.unique_id}"
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass(frozen=True)
class Url:
    """
    Value object representing a validated URL.
    
    Ensures URLs are properly formatted and provides utility methods
    for URL manipulation and validation.
    """
    value: str
    
    def __post_init__(self):
        """Validate URL format."""
        if not self.value:
            raise ValueError("URL cannot be empty")
        
        if not self.value.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        
        try:
            parsed = urlparse(self.value)
            if not parsed.netloc:
                raise ValueError("URL must have a valid domain")
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")
    
    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        return urlparse(self.value).netloc
    
    @property
    def path(self) -> str:
        """Extract path from URL."""
        return urlparse(self.value).path
    
    @property
    def is_secure(self) -> bool:
        """Check if URL uses HTTPS."""
        return self.value.startswith('https://')
    
    @property
    def is_pdf(self) -> bool:
        """Check if URL appears to point to a PDF."""
        return self.value.lower().endswith('.pdf')
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DateRange:
    """
    Immutable date range with validation and utility methods.
    
    Represents a span of time with inclusive start and end dates.
    """
    start_date: date
    end_date: date
    
    def __post_init__(self):
        """Validate date range."""
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before or equal to end date")
    
    @classmethod
    def single_day(cls, target_date: date) -> 'DateRange':
        """
        Create a single-day date range.
        
        Args:
            target_date: The date for the range
            
        Returns:
            DateRange spanning just the target date
        """
        return cls(start_date=target_date, end_date=target_date)
    
    @classmethod
    def year(cls, year: int) -> 'DateRange':
        """
        Create a date range spanning an entire year.
        
        Args:
            year: The year to span
            
        Returns:
            DateRange from January 1 to December 31 of the year
        """
        return cls(
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31)
        )
    
    @classmethod
    def month(cls, year: int, month: int) -> 'DateRange':
        """
        Create a date range spanning an entire month.
        
        Args:
            year: The year
            month: The month (1-12)
            
        Returns:
            DateRange spanning the entire month
        """
        from calendar import monthrange
        
        start_date = date(year, month, 1)
        _, last_day = monthrange(year, month)
        end_date = date(year, month, last_day)
        
        return cls(start_date=start_date, end_date=end_date)
    
    @property
    def days(self) -> int:
        """Number of days in the range (inclusive)."""
        return (self.end_date - self.start_date).days + 1
    
    @property
    def weeks(self) -> float:
        """Number of weeks in the range."""
        return self.days / 7.0
    
    @property
    def months(self) -> float:
        """Approximate number of months in the range."""
        return self.days / 30.44  # Average days per month
    
    @property
    def years(self) -> float:
        """Number of years in the range."""
        return self.days / 365.25  # Account for leap years
    
    def contains(self, target_date: date) -> bool:
        """
        Check if a date falls within this range.
        
        Args:
            target_date: Date to check
            
        Returns:
            True if date is within range (inclusive)
        """
        return self.start_date <= target_date <= self.end_date
    
    def overlaps(self, other: 'DateRange') -> bool:
        """
        Check if this range overlaps with another.
        
        Args:
            other: Another date range
            
        Returns:
            True if ranges overlap
        """
        return (self.start_date <= other.end_date and 
                self.end_date >= other.start_date)
    
    def intersection(self, other: 'DateRange') -> Optional['DateRange']:
        """
        Get the intersection of two date ranges.
        
        Args:
            other: Another date range
            
        Returns:
            DateRange representing intersection, or None if no overlap
        """
        if not self.overlaps(other):
            return None
        
        start = max(self.start_date, other.start_date)
        end = min(self.end_date, other.end_date)
        
        return DateRange(start_date=start, end_date=end)
    
    def __str__(self) -> str:
        return f"{self.start_date} to {self.end_date}"


@dataclass(frozen=True)
class MonetaryAmount:
    """
    Value object representing a monetary amount with currency.
    
    Uses Decimal for precise financial calculations.
    """
    amount: Decimal
    currency: str
    
    def __post_init__(self):
        """Validate monetary amount."""
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, 'amount', Decimal(str(self.amount)))
        
        if len(self.currency) != 3:
            raise ValueError("Currency must be 3-character ISO code")
        
        if not self.currency.isupper():
            raise ValueError("Currency must be uppercase")
    
    @classmethod
    def from_float(cls, amount: float, currency: str) -> 'MonetaryAmount':
        """
        Create MonetaryAmount from float (not recommended for precision).
        
        Args:
            amount: Amount as float
            currency: 3-character currency code
            
        Returns:
            MonetaryAmount instance
        """
        return cls(amount=Decimal(str(amount)), currency=currency)
    
    @classmethod
    def usd(cls, amount: Union[str, int, float, Decimal]) -> 'MonetaryAmount':
        """Create USD amount."""
        return cls(amount=Decimal(str(amount)), currency="USD")
    
    @classmethod
    def eur(cls, amount: Union[str, int, float, Decimal]) -> 'MonetaryAmount':
        """Create EUR amount."""
        return cls(amount=Decimal(str(amount)), currency="EUR")
    
    @classmethod
    def gbp(cls, amount: Union[str, int, float, Decimal]) -> 'MonetaryAmount':
        """Create GBP amount."""
        return cls(amount=Decimal(str(amount)), currency="GBP")
    
    def __add__(self, other: 'MonetaryAmount') -> 'MonetaryAmount':
        """Add two monetary amounts (must be same currency)."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return MonetaryAmount(amount=self.amount + other.amount, currency=self.currency)
    
    def __sub__(self, other: 'MonetaryAmount') -> 'MonetaryAmount':
        """Subtract two monetary amounts (must be same currency)."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {self.currency} and {other.currency}")
        return MonetaryAmount(amount=self.amount - other.amount, currency=self.currency)
    
    def __mul__(self, factor: Union[int, float, Decimal]) -> 'MonetaryAmount':
        """Multiply monetary amount by a factor."""
        return MonetaryAmount(amount=self.amount * Decimal(str(factor)), currency=self.currency)
    
    def __truediv__(self, divisor: Union[int, float, Decimal]) -> 'MonetaryAmount':
        """Divide monetary amount by a divisor."""
        return MonetaryAmount(amount=self.amount / Decimal(str(divisor)), currency=self.currency)
    
    def __str__(self) -> str:
        return f"{self.amount} {self.currency}"


@dataclass(frozen=True)
class SentimentScore:
    """
    Value object representing a sentiment score with confidence.
    
    Encapsulates both the score value and the confidence in that score.
    """
    value: float
    confidence: float
    scale_min: float = -1.0
    scale_max: float = 1.0
    
    def __post_init__(self):
        """Validate sentiment score."""
        if not self.scale_min <= self.value <= self.scale_max:
            raise ValueError(f"Score {self.value} must be between {self.scale_min} and {self.scale_max}")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence {self.confidence} must be between 0.0 and 1.0")
        
        if self.scale_min >= self.scale_max:
            raise ValueError("Scale minimum must be less than scale maximum")
    
    @classmethod
    def hawkish_dovish(cls, value: float, confidence: float) -> 'SentimentScore':
        """
        Create hawkish-dovish sentiment score.
        
        Args:
            value: Score from -1.0 (very dovish) to 1.0 (very hawkish)
            confidence: Confidence in score from 0.0 to 1.0
            
        Returns:
            SentimentScore instance
        """
        return cls(value=value, confidence=confidence, scale_min=-1.0, scale_max=1.0)
    
    @classmethod
    def uncertainty(cls, value: float, confidence: float) -> 'SentimentScore':
        """
        Create uncertainty score.
        
        Args:
            value: Score from 0.0 (certain) to 1.0 (very uncertain)
            confidence: Confidence in score from 0.0 to 1.0
            
        Returns:
            SentimentScore instance
        """
        return cls(value=value, confidence=confidence, scale_min=0.0, scale_max=1.0)
    
    @property
    def is_positive(self) -> bool:
        """Check if score is positive (above scale midpoint)."""
        midpoint = (self.scale_min + self.scale_max) / 2
        return self.value > midpoint
    
    @property
    def is_negative(self) -> bool:
        """Check if score is negative (below scale midpoint)."""
        midpoint = (self.scale_min + self.scale_max) / 2
        return self.value < midpoint
    
    @property
    def is_neutral(self) -> bool:
        """Check if score is neutral (at scale midpoint)."""
        midpoint = (self.scale_min + self.scale_max) / 2
        return abs(self.value - midpoint) < 0.01  # Allow small floating point errors
    
    @property
    def normalized_value(self) -> float:
        """
        Get score normalized to 0.0-1.0 range.
        
        Returns:
            Score normalized to 0.0-1.0 where 0.0 is scale_min and 1.0 is scale_max
        """
        return (self.value - self.scale_min) / (self.scale_max - self.scale_min)
    
    @property
    def strength(self) -> float:
        """
        Get the strength of the sentiment (distance from neutral).
        
        Returns:
            Value from 0.0 (neutral) to 1.0 (maximum strength)
        """
        midpoint = (self.scale_min + self.scale_max) / 2
        max_distance = max(abs(self.scale_max - midpoint), abs(self.scale_min - midpoint))
        return abs(self.value - midpoint) / max_distance
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def __str__(self) -> str:
        return f"{self.value:.3f} (confidence: {self.confidence:.3f})"


@dataclass(frozen=True)
class TextStatistics:
    """
    Value object containing statistics about text content.
    
    Provides various metrics about text complexity and characteristics.
    """
    character_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    average_words_per_sentence: float
    average_sentences_per_paragraph: float
    readability_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate text statistics."""
        if self.character_count < 0:
            raise ValueError("Character count cannot be negative")
        if self.word_count < 0:
            raise ValueError("Word count cannot be negative")
        if self.sentence_count < 0:
            raise ValueError("Sentence count cannot be negative")
        if self.paragraph_count < 0:
            raise ValueError("Paragraph count cannot be negative")
        if self.readability_score is not None and not 0.0 <= self.readability_score <= 100.0:
            raise ValueError("Readability score must be between 0.0 and 100.0")
    
    @classmethod
    def from_text(cls, text: str) -> 'TextStatistics':
        """
        Calculate statistics from text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            TextStatistics instance
            
        Example:
            >>> stats = TextStatistics.from_text("Hello world. This is a test.")
            >>> assert stats.word_count == 7
            >>> assert stats.sentence_count == 2
        """
        import re
        
        # Basic counts
        character_count = len(text)
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        # Sentence counting (basic approach)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Paragraph counting
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Averages
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        avg_sentences_per_paragraph = sentence_count / max(paragraph_count, 1)
        
        return cls(
            character_count=character_count,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            average_words_per_sentence=avg_words_per_sentence,
            average_sentences_per_paragraph=avg_sentences_per_paragraph
        )
    
    @property
    def is_substantial(self) -> bool:
        """Check if text is substantial (more than 500 words)."""
        return self.word_count > 500
    
    @property
    def is_short(self) -> bool:
        """Check if text is short (less than 100 words)."""
        return self.word_count < 100
    
    @property
    def complexity_level(self) -> str:
        """
        Estimate complexity level based on sentence length.
        
        Returns:
            String indicating complexity: "simple", "moderate", "complex"
        """
        if self.average_words_per_sentence < 15:
            return "simple"
        elif self.average_words_per_sentence < 25:
            return "moderate"
        else:
            return "complex"


@dataclass(frozen=True)
class ProcessingMetrics:
    """
    Value object containing metrics about processing performance.
    
    Tracks timing, success rates, and other operational metrics.
    """
    start_time: datetime
    end_time: datetime
    success: bool
    error_count: int = 0
    warning_count: int = 0
    items_processed: int = 0
    bytes_processed: int = 0
    
    def __post_init__(self):
        """Validate processing metrics."""
        if self.start_time > self.end_time:
            raise ValueError("Start time must be before end time")
        if self.error_count < 0:
            raise ValueError("Error count cannot be negative")
        if self.warning_count < 0:
            raise ValueError("Warning count cannot be negative")
        if self.items_processed < 0:
            raise ValueError("Items processed cannot be negative")
        if self.bytes_processed < 0:
            raise ValueError("Bytes processed cannot be negative")
    
    @classmethod
    def start_processing(cls) -> 'ProcessingMetrics':
        """
        Create metrics for the start of processing.
        
        Returns:
            ProcessingMetrics with start time set to now
        """
        now = datetime.now(timezone.utc)
        return cls(
            start_time=now,
            end_time=now,  # Will be updated when processing completes
            success=False,
            error_count=0,
            warning_count=0,
            items_processed=0,
            bytes_processed=0
        )
    
    def complete_successfully(self, items_processed: int = 0, bytes_processed: int = 0) -> 'ProcessingMetrics':
        """
        Mark processing as complete and successful.
        
        Args:
            items_processed: Number of items processed
            bytes_processed: Number of bytes processed
            
        Returns:
            New ProcessingMetrics instance with completion data
        """
        return ProcessingMetrics(
            start_time=self.start_time,
            end_time=datetime.now(timezone.utc),
            success=True,
            error_count=self.error_count,
            warning_count=self.warning_count,
            items_processed=items_processed,
            bytes_processed=bytes_processed
        )
    
    def complete_with_failure(self, error_count: int = 1) -> 'ProcessingMetrics':
        """
        Mark processing as complete but failed.
        
        Args:
            error_count: Number of errors encountered
            
        Returns:
            New ProcessingMetrics instance with failure data
        """
        return ProcessingMetrics(
            start_time=self.start_time,
            end_time=datetime.now(timezone.utc),
            success=False,
            error_count=error_count,
            warning_count=self.warning_count,
            items_processed=self.items_processed,
            bytes_processed=self.bytes_processed
        )
    
    @property
    def duration(self) -> float:
        """Processing duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def items_per_second(self) -> float:
        """Items processed per second."""
        return self.items_processed / max(self.duration, 0.001)
    
    @property
    def bytes_per_second(self) -> float:
        """Bytes processed per second."""
        return self.bytes_processed / max(self.duration, 0.001)
    
    @property
    def has_errors(self) -> bool:
        """Check if processing had errors."""
        return self.error_count > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if processing had warnings."""
        return self.warning_count > 0
    
    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"Processing {status}: {self.duration:.2f}s, {self.items_processed} items"


@dataclass(frozen=True)
class ContentHash:
    """
    Value object representing a content hash for deduplication.
    
    Provides multiple hash algorithms for different use cases.
    """
    sha256: str
    md5: str
    content_length: int
    
    def __post_init__(self):
        """Validate content hash."""
        if len(self.sha256) != 64:
            raise ValueError("SHA256 hash must be 64 characters")
        if len(self.md5) != 32:
            raise ValueError("MD5 hash must be 32 characters")
        if self.content_length < 0:
            raise ValueError("Content length cannot be negative")
        if not re.match(r'^[a-f0-9]+$', self.sha256.lower()):
            raise ValueError("SHA256 hash must be hexadecimal")
        if not re.match(r'^[a-f0-9]+$', self.md5.lower()):
            raise ValueError("MD5 hash must be hexadecimal")
    
    @classmethod
    def from_content(cls, content: str) -> 'ContentHash':
        """
        Generate hash from content.
        
        Args:
            content: Content to hash
            
        Returns:
            ContentHash instance
        """
        import hashlib
        
        content_bytes = content.encode('utf-8')
        sha256_hash = hashlib.sha256(content_bytes).hexdigest()
        md5_hash = hashlib.md5(content_bytes).hexdigest()
        
        return cls(
            sha256=sha256_hash,
            md5=md5_hash,
            content_length=len(content)
        )
    
    @property
    def short_hash(self) -> str:
        """Get short hash (first 8 characters of SHA256)."""
        return self.sha256[:8]
    
    def __str__(self) -> str:
        return f"ContentHash({self.short_hash})"


@dataclass(frozen=True)
class Version:
    """
    Value object representing a semantic version.
    
    Follows semantic versioning specification (semver.org).
    """
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build_metadata: Optional[str] = None
    
    def __post_init__(self):
        """Validate version numbers."""
        if self.major < 0:
            raise ValueError("Major version cannot be negative")
        if self.minor < 0:
            raise ValueError("Minor version cannot be negative")
        if self.patch < 0:
            raise ValueError("Patch version cannot be negative")
    
    @classmethod
    def from_string(cls, version_str: str) -> 'Version':
        """
        Parse version from string.
        
        Args:
            version_str: Version string like "1.2.3-alpha.1+build.123"
            
        Returns:
            Version instance
        """
        import re
        
        # Regex for semantic version parsing
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'
        match = re.match(pattern, version_str)
        
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        
        major, minor, patch, pre_release, build_metadata = match.groups()
        
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            pre_release=pre_release,
            build_metadata=build_metadata
        )
    
    def to_string(self) -> str:
        """
        Convert to string representation.
        
        Returns:
            Semantic version string
        """
        version_str = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.pre_release:
            version_str += f"-{self.pre_release}"
        
        if self.build_metadata:
            version_str += f"+{self.build_metadata}"
        
        return version_str
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __lt__(self, other: 'Version') -> bool:
        """Compare versions for sorting."""
        if not isinstance(other, Version):
            return NotImplemented
        
        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        # Handle pre-release versions
        if self.pre_release is None and other.pre_release is not None:
            return False  # Release version is greater than pre-release
        if self.pre_release is not None and other.pre_release is None:
            return True  # Pre-release version is less than release
        if self.pre_release is not None and other.pre_release is not None:
            return self.pre_release < other.pre_release
        
        return False  # Versions are equal
    
    def __le__(self, other: 'Version') -> bool:
        return self < other or self == other
    
    def __gt__(self, other: 'Version') -> bool:
        return not self <= other
    
    def __ge__(self, other: 'Version') -> bool:
        return not self < other
    
    def is_compatible_with(self, other: 'Version') -> bool:
        """
        Check if this version is compatible with another (same major version).
        
        Args:
            other: Version to check compatibility with
            
        Returns:
            True if versions are compatible
        """
        return self.major == other.major


# Type aliases for commonly used value object combinations
SpeechIdentifier = Tuple[SpeechId, ContentHash]
DateScore = Tuple[date, SentimentScore]
UrlContentPair = Tuple[Url, ContentHash]
ProcessingResult = Tuple[ProcessingMetrics, Optional[str]]  # Metrics and optional error message