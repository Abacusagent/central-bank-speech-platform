# validation/validator.py

"""
Validation Framework for Central Bank Speech Analysis Platform

Chains together modular validators for speech content and metadata,
ensuring data quality and authenticity.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

from typing import List, Dict, Any
from datetime import date
import logging

from domain.entities import CentralBankSpeech
from interfaces.plugin_interfaces import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

# -------------------
# Individual Validators
# -------------------

class ContentLengthValidator:
    """Ensures speech content is of minimum viable length."""

    def __call__(self, speech: CentralBankSpeech) -> ValidationResult:
        text = (speech.content.cleaned_text or speech.content.raw_text or "").strip()
        if len(text) < 200:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                confidence=0.1,
                issues=["Content is too short to be a genuine speech."],
                metadata={"validator": "ContentLengthValidator"}
            )
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            issues=[],
            metadata={"validator": "ContentLengthValidator"}
        )

class LanguageDetectionValidator:
    """Ensures the language matches expected."""

    def __call__(self, speech: CentralBankSpeech) -> ValidationResult:
        # Very simple check: compare language field to expected
        expected = getattr(speech, 'expected_language', 'en')
        if speech.content and getattr(speech.content, 'language', None) not in (None, expected):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                confidence=0.3,
                issues=[f"Language mismatch: expected {expected}, found {speech.content.language}"],
                metadata={"validator": "LanguageDetectionValidator"}
            )
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            issues=[],
            metadata={"validator": "LanguageDetectionValidator"}
        )

class SpeechStructureValidator:
    """Checks for plausible speech structure."""

    def __call__(self, speech: CentralBankSpeech) -> ValidationResult:
        text = (speech.content.cleaned_text or speech.content.raw_text or "")
        num_paragraphs = text.count('\n\n')
        if num_paragraphs < 2:
            return ValidationResult(
                status=ValidationStatus.QUESTIONABLE,
                confidence=0.4,
                issues=["Speech does not appear to have multiple paragraphs."],
                metadata={"validator": "SpeechStructureValidator"}
            )
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            issues=[],
            metadata={"validator": "SpeechStructureValidator"}
        )

class BoilerplateDetectionValidator:
    """Detects if content is mostly boilerplate or placeholder text."""

    BOILERPLATE_PATTERNS = [
        'lorem ipsum', 'placeholder', 'test content', 'coming soon',
        'under construction', 'page not found'
    ]

    def __call__(self, speech: CentralBankSpeech) -> ValidationResult:
        text = (speech.content.cleaned_text or speech.content.raw_text or "").lower()
        for pat in self.BOILERPLATE_PATTERNS:
            if pat in text:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    confidence=0.0,
                    issues=[f"Boilerplate detected: {pat!r}"],
                    metadata={"validator": "BoilerplateDetectionValidator"}
                )
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            issues=[],
            metadata={"validator": "BoilerplateDetectionValidator"}
        )

class DuplicateDetectionValidator:
    """Checks for duplicate speeches by URL or content hash."""

    def __init__(self, known_hashes: set, known_urls: set):
        self.known_hashes = known_hashes
        self.known_urls = known_urls

    def __call__(self, speech: CentralBankSpeech) -> ValidationResult:
        if speech.metadata.url in self.known_urls:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                confidence=0.0,
                issues=[f"Duplicate URL: {speech.metadata.url}"],
                metadata={"validator": "DuplicateDetectionValidator"}
            )
        if speech.content and getattr(speech.content, "content_hash_sha256", None) in self.known_hashes:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                confidence=0.0,
                issues=["Duplicate speech content hash detected."],
                metadata={"validator": "DuplicateDetectionValidator"}
            )
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            issues=[],
            metadata={"validator": "DuplicateDetectionValidator"}
        )

class SpeakerConsistencyValidator:
    """Ensures speaker entity exists and is valid."""

    def __call__(self, speech: CentralBankSpeech) -> ValidationResult:
        if not getattr(speech, "speaker", None):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                confidence=0.0,
                issues=["No speaker assigned or recognized."],
                metadata={"validator": "SpeakerConsistencyValidator"}
            )
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            issues=[],
            metadata={"validator": "SpeakerConsistencyValidator"}
        )

class DateRangeValidator:
    """Ensures speech date is reasonable (not before central bank existed, not in the future)."""

    def __call__(self, speech: CentralBankSpeech) -> ValidationResult:
        speech_date = getattr(speech.metadata, 'date', None)
        institution = getattr(speech, 'institution', None)
        today = date.today()
        earliest = getattr(institution, 'established_date', date(1900, 1, 1)) if institution else date(1900, 1, 1)
        if speech_date is None:
            return ValidationResult(
                status=ValidationStatus.QUESTIONABLE,
                confidence=0.2,
                issues=["Speech date is missing."],
                metadata={"validator": "DateRangeValidator"}
            )
        if speech_date < earliest:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                confidence=0.0,
                issues=["Speech date predates institution's founding."],
                metadata={"validator": "DateRangeValidator"}
            )
        if speech_date > today:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                confidence=0.0,
                issues=["Speech date is in the future."],
                metadata={"validator": "DateRangeValidator"}
            )
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            issues=[],
            metadata={"validator": "DateRangeValidator"}
        )

class InstitutionConsistencyValidator:
    """Ensures speech institution matches known code and type."""

    def __call__(self, speech: CentralBankSpeech) -> ValidationResult:
        institution = getattr(speech, 'institution', None)
        meta_code = getattr(speech.metadata, 'institution_code', None)
        if not institution or institution.code != meta_code:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                confidence=0.0,
                issues=["Institution code mismatch or not found."],
                metadata={"validator": "InstitutionConsistencyValidator"}
            )
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            issues=[],
            metadata={"validator": "InstitutionConsistencyValidator"}
        )

class RoleValidationValidator:
    """Ensures speaker role is plausible (e.g., not empty or 'unknown')."""

    def __call__(self, speech: CentralBankSpeech) -> ValidationResult:
        speaker = getattr(speech, "speaker", None)
        if not speaker or not getattr(speaker, "role", None) or speaker.role.lower() == "unknown":
            return ValidationResult(
                status=ValidationStatus.QUESTIONABLE,
                confidence=0.3,
                issues=["Speaker role is missing or unknown."],
                metadata={"validator": "RoleValidationValidator"}
            )
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            issues=[],
            metadata={"validator": "RoleValidationValidator"}
        )

# -------------------------
# Speech Validation Framework
# -------------------------

class SpeechValidationFramework:
    """
    Chains together all content and metadata validators and aggregates results.
    """

    def __init__(self, known_hashes: set = None, known_urls: set = None):
        # Content validators
        self.content_validators = [
            ContentLengthValidator(),
            LanguageDetectionValidator(),
            SpeechStructureValidator(),
            BoilerplateDetectionValidator(),
            DuplicateDetectionValidator(known_hashes or set(), known_urls or set())
        ]
        # Metadata validators
        self.metadata_validators = [
            SpeakerConsistencyValidator(),
            DateRangeValidator(),
            InstitutionConsistencyValidator(),
            RoleValidationValidator()
        ]

    def validate(self, speech: CentralBankSpeech) -> ValidationResult:
        """
        Runs all validators, aggregates results, and returns a single ValidationResult.
        """
        issues = []
        confidence = 1.0
        status = ValidationStatus.VALID
        validator_results = []

        # Content validators
        for validator in self.content_validators:
            result = validator(speech)
            validator_results.append(result)
            if result.status != ValidationStatus.VALID:
                issues.extend(result.issues)
                confidence *= result.confidence
                if result.status == ValidationStatus.INVALID:
                    status = ValidationStatus.INVALID
                elif result.status == ValidationStatus.QUESTIONABLE and status != ValidationStatus.INVALID:
                    status = ValidationStatus.QUESTIONABLE

        # Metadata validators
        for validator in self.metadata_validators:
            result = validator(speech)
            validator_results.append(result)
            if result.status != ValidationStatus.VALID:
                issues.extend(result.issues)
                confidence *= result.confidence
                if result.status == ValidationStatus.INVALID:
                    status = ValidationStatus.INVALID
                elif result.status == ValidationStatus.QUESTIONABLE and status != ValidationStatus.INVALID:
                    status = ValidationStatus.QUESTIONABLE

        return ValidationResult(
            status=status,
            confidence=max(0.0, min(1.0, confidence)),
            issues=issues,
            metadata={
                "validators_run": [type(v).__name__ for v in self.content_validators + self.metadata_validators],
                "results": [r.__dict__ for r in validator_results]
            }
        )
