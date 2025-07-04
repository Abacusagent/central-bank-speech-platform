"""
Validation Framework for Central Bank Speech Analysis Platform

Comprehensive validation for speech content, metadata, and analysis results.
Ensures data quality and consistency across the entire pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import date, datetime
import re
from dataclasses import dataclass
from enum import Enum

from domain.entities import CentralBankSpeech
from domain.value_objects import ValidationResult, ValidationStatus
from interfaces.plugin_interfaces import SpeechMetadata, SpeechContent


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class BaseValidator(ABC):
    """Base class for all validators."""
    
    @abstractmethod
    def validate(self, item: Any) -> List[ValidationIssue]:
        """Validate an item and return issues."""
        pass
    
    def is_valid(self, item: Any) -> bool:
        """Check if item is valid (no errors or critical issues)."""
        issues = self.validate(item)
        return not any(
            issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for issue in issues
        )


class SpeechContentValidator(BaseValidator):
    """Validates speech content for quality and consistency."""
    
    MIN_CONTENT_LENGTH = 100
    MAX_CONTENT_LENGTH = 1_000_000
    MIN_WORD_COUNT = 20
    
    def validate(self, content: SpeechContent) -> List[ValidationIssue]:
        """Validate speech content."""
        issues = []
        
        # Check content length
        if len(content.raw_text) < self.MIN_CONTENT_LENGTH:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Content too short: {len(content.raw_text)} chars (min: {self.MIN_CONTENT_LENGTH})",
                field="raw_text"
            ))
        
        if len(content.raw_text) > self.MAX_CONTENT_LENGTH:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Content unusually long: {len(content.raw_text)} chars",
                field="raw_text"
            ))
        
        # Check word count
        word_count = len(content.raw_text.split())
        if word_count < self.MIN_WORD_COUNT:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Too few words: {word_count} (min: {self.MIN_WORD_COUNT})",
                field="raw_text"
            ))
        
        # Check for common extraction issues
        if content.raw_text.count('\n') > len(content.raw_text) * 0.1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Unusually high line break count, possible extraction issue",
                field="raw_text"
            ))
        
        # Check for repetitive content
        if self._has_repetitive_content(content.raw_text):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Repetitive content detected, possible extraction issue",
                field="raw_text"
            ))
        
        # Check language consistency
        if content.language and not self._is_language_consistent(content.raw_text, content.language):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Content language may not match declared language: {content.language}",
                field="language"
            ))
        
        return issues
    
    def _has_repetitive_content(self, text: str) -> bool:
        """Check if text has repetitive patterns."""
        # Simple heuristic: check for repeated phrases
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for repeated 3-word sequences
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        unique_trigrams = set(trigrams)
        
        return len(unique_trigrams) < len(trigrams) * 0.7
    
    def _is_language_consistent(self, text: str, declared_language: str) -> bool:
        """Basic language consistency check."""
        # This is a simplified check - in production, use proper language detection
        if declared_language.lower() == 'en':
            # Check for common English patterns
            english_patterns = [
                r'\bthe\b', r'\band\b', r'\bof\b', r'\bto\b', r'\bin\b',
                r'\ba\b', r'\bis\b', r'\bthat\b', r'\bfor\b', r'\bwith\b'
            ]
            
            matches = sum(1 for pattern in english_patterns if re.search(pattern, text.lower()))
            return matches >= 3
        
        return True  # Default to consistent for other languages


class SpeechMetadataValidator(BaseValidator):
    """Validates speech metadata for completeness and accuracy."""
    
    def validate(self, metadata: SpeechMetadata) -> List[ValidationIssue]:
        """Validate speech metadata."""
        issues = []
        
        # Check required fields
        if not metadata.title or not metadata.title.strip():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Title is required",
                field="title"
            ))
        
        if not metadata.speaker_name or not metadata.speaker_name.strip():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Speaker name is required",
                field="speaker_name"
            ))
        
        if not metadata.date:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Date is required",
                field="date"
            ))
        
        # Check date reasonableness
        if metadata.date:
            if metadata.date < date(1900, 1, 1):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Date too early: {metadata.date}",
                    field="date"
                ))
            
            if metadata.date > date.today():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Future date: {metadata.date}",
                    field="date"
                ))
        
        # Check URL validity
        if not metadata.url or not str(metadata.url).strip():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="URL is required",
                field="url"
            ))
        else:
            url_str = str(metadata.url)
            if not (url_str.startswith('http://') or url_str.startswith('https://')):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid URL format: {url_str}",
                    field="url"
                ))
        
        # Check institution code
        valid_institutions = ['FED', 'ECB', 'BOE', 'BOJ', 'BIS']
        if metadata.institution_code not in valid_institutions:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Unknown institution code: {metadata.institution_code}",
                field="institution_code"
            ))
        
        return issues


class ComprehensiveSpeechValidator(BaseValidator):
    """Comprehensive validator that combines multiple validation checks."""
    
    def __init__(self):
        self.metadata_validator = SpeechMetadataValidator()
        self.content_validator = SpeechContentValidator()
    
    def validate(self, speech: CentralBankSpeech) -> List[ValidationIssue]:
        """Validate a complete speech entity."""
        issues = []
        
        # Validate metadata
        if hasattr(speech, 'metadata') and speech.metadata:
            issues.extend(self.metadata_validator.validate(speech.metadata))
        
        # Validate content
        if hasattr(speech, 'content') and speech.content:
            issues.extend(self.content_validator.validate(speech.content))
        
        # Cross-validation checks
        if speech.title and speech.content:
            # Check if title matches content
            if not self._title_matches_content(speech.title, speech.content.raw_text):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Title may not match content",
                    details={"title": speech.title[:100]}
                ))
        
        return issues
    
    def _title_matches_content(self, title: str, content: str) -> bool:
        """Check if title is reasonably related to content."""
        # Simple heuristic: check if key words from title appear in content
        title_words = set(word.lower() for word in title.split() if len(word) > 3)
        content_words = set(word.lower() for word in content.split())
        
        if not title_words:
            return True
        
        overlap = len(title_words.intersection(content_words))
        return overlap >= min(2, len(title_words) * 0.5)


# Factory function for creating validators
def create_validator(validator_type: str) -> BaseValidator:
    """Create a validator instance of the specified type."""
    validators = {
        'content': SpeechContentValidator,
        'metadata': SpeechMetadataValidator,
        'comprehensive': ComprehensiveSpeechValidator
    }
    
    if validator_type not in validators:
        raise ValueError(f"Unknown validator type: {validator_type}")
    
    return validators[validator_type]()