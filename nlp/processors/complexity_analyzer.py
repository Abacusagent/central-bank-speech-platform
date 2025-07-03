# nlp/processors/complexity_analyzer.py

"""
Complexity Analyzer for Central Bank Speech Analysis Platform

Measures the communication complexity of central bank speeches using
readability, sentence structure, and vocabulary sophistication.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from nlp.pipeline import NLPProcessor, ProcessorResult
from domain.entities import CentralBankSpeech

logger = logging.getLogger(__name__)

class ComplexityAnalyzer(NLPProcessor):
    """
    Analyzer for measuring speech complexity using readability, sentence length,
    and vocabulary diversity.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """
        Analyze complexity of the speech and return a ProcessorResult.
        """
        start_time = datetime.now()
        try:
            text = self._extract_text_for_analysis(speech)
            if not text:
                return ProcessorResult(
                    processor_name=self.name,
                    success=False,
                    confidence=0.0,
                    processing_time=0.0,
                    results={},
                    error_message="No text content available"
                )

            # Calculate metrics
            metrics = {
                'readability_score': self._calculate_readability(text),
                'avg_sentence_length': self._calculate_avg_sentence_length(text),
                'vocabulary_diversity': self._calculate_vocabulary_diversity(text),
                'complexity_score': 0.0
            }

            # Combine into overall complexity score (weighted)
            metrics['complexity_score'] = (
                metrics['readability_score'] * 0.4 +
                min(1.0, metrics['avg_sentence_length'] / 30) * 0.3 +
                metrics['vocabulary_diversity'] * 0.3
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessorResult(
                processor_name=self.name,
                success=True,
                confidence=0.8,
                processing_time=processing_time,
                results=metrics
            )
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Complexity analysis failed: {e}")
            return ProcessorResult(
                processor_name=self.name,
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                results={},
                error_message=str(e)
            )

    def _calculate_readability(self, text: str) -> float:
        """
        Calculate normalized Flesch Reading Ease score (0 = simple, 1 = very complex).
        """
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text)
        syllables = sum(self._count_syllables(word) for word in words)
        if not sentences or not words:
            return 0.5  # neutral if not enough data

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        # Simplified Flesch Reading Ease formula (normalized to 0-1, higher = more complex)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        normalized_score = max(0, min(1, (100 - flesch_score) / 100))
        return normalized_score

    def _count_syllables(self, word: str) -> int:
        """
        Counts syllables in a word (simplified).
        """
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word and word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        return max(1, count)

    def _calculate_avg_sentence_length(self, text: str) -> float:
        """
        Average sentence length (in words).
        """
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        total_words = sum(len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences)
        return total_words / len(sentences)

    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """
        Vocabulary diversity: type-token ratio (unique words / total words).
        """
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        unique_words = set(words)
        return len(unique_words) / len(words)

    def get_confidence_score(self) -> float:
        """Returns static confidence in complexity analysis."""
        return 0.8
