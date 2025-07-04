# nlp/processors/uncertainty_quantifier.py

"""
Uncertainty Quantifier for Central Bank Speech Analysis Platform

Measures the degree of uncertainty in monetary policy communications
using curated linguistic indicators and heuristic scoring.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from infrastructure.nlp.pipeline import NLPProcessor, ProcessorResult
from domain.entities import CentralBankSpeech

logger = logging.getLogger(__name__)

class UncertaintyQuantifier(NLPProcessor):
    """
    Quantifies linguistic uncertainty in a speech.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.uncertainty_words = {
            'high': ['uncertain', 'unclear', 'ambiguous', 'unknown', 'unpredictable'],
            'medium': ['may', 'might', 'could', 'possibly', 'perhaps', 'likely'],
            'low': ['depends', 'conditional', 'subject to', 'monitor', 'assess']
        }

    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """
        Quantify uncertainty in a speech and return a ProcessorResult.
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

            # Main scoring
            uncertainty_score = self._calculate_uncertainty_score(text)
            indicators = self._find_uncertainty_indicators(text)
            if uncertainty_score >= 0.7:
                uncertainty_level = "High"
            elif uncertainty_score >= 0.4:
                uncertainty_level = "Medium"
            else:
                uncertainty_level = "Low"

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessorResult(
                processor_name=self.name,
                success=True,
                confidence=0.75,
                processing_time=processing_time,
                results={
                    'uncertainty_score': uncertainty_score,
                    'uncertainty_level': uncertainty_level,
                    'uncertainty_indicators': indicators
                }
            )
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Uncertainty quantification failed: {e}")
            return ProcessorResult(
                processor_name=self.name,
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                results={},
                error_message=str(e)
            )

    def _calculate_uncertainty_score(self, text: str) -> float:
        """
        Calculates uncertainty score using word frequency and weighted sum.
        Returns a normalized value between 0 and 1.
        """
        import re

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        if not words:
            return 0.0

        uncertainty_count = 0.0
        for category, word_list in self.uncertainty_words.items():
            weight = {'high': 1.0, 'medium': 0.6, 'low': 0.3}[category]
            for word in word_list:
                count = text_lower.count(word)
                uncertainty_count += count * weight

        normalized_score = uncertainty_count / len(words) * 100
        return min(1.0, normalized_score)

    def _find_uncertainty_indicators(self, text: str) -> List[str]:
        """
        Finds and returns a list of unique uncertainty-related terms present in the text.
        """
        text_lower = text.lower()
        found = set()
        for category, word_list in self.uncertainty_words.items():
            for word in word_list:
                if word in text_lower:
                    found.add(word)
        return list(found)

    def get_confidence_score(self) -> float:
        """Returns static confidence in this processor's output."""
        return 0.75
