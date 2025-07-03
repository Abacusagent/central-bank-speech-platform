# nlp/processors/stance_detector.py

"""
Stance Detector for Central Bank Speech Analysis Platform

Detects nuanced policy stances in central bank communications,
including commitment, flexibility, forward guidance, and data dependency,
using curated phrase indicators.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from nlp.pipeline import NLPProcessor, ProcessorResult
from domain.entities import CentralBankSpeech

logger = logging.getLogger(__name__)

class StanceDetector(NLPProcessor):
    """
    Stance detector for identifying subtle policy positions and communication strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.stance_indicators = {
            'commitment': ['will', 'committed to', 'determined to', 'pledge'],
            'flexibility': ['flexible', 'adapt', 'adjust', 'respond to'],
            'forward_guidance': ['expect', 'anticipate', 'project', 'outlook'],
            'data_dependency': ['data dependent', 'economic conditions', 'indicators']
        }

    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """
        Detect policy stance dimensions in a speech and return ProcessorResult.
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

            # Calculate scores for each stance type
            stance_scores = {}
            for stance_type, indicators in self.stance_indicators.items():
                stance_scores[stance_type] = self._calculate_stance_score(text, indicators)

            # Determine dominant stance (the highest score)
            dominant_stance = max(stance_scores, key=stance_scores.get)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessorResult(
                processor_name=self.name,
                success=True,
                confidence=0.7,
                processing_time=processing_time,
                results={
                    'stance_scores': stance_scores,
                    'dominant_stance': dominant_stance,
                    'stance_strength': stance_scores[dominant_stance]
                }
            )
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Stance detection failed: {e}")
            return ProcessorResult(
                processor_name=self.name,
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                results={},
                error_message=str(e)
            )

    def _calculate_stance_score(self, text: str, indicators: List[str]) -> float:
        """
        Counts frequency of each indicator phrase and normalizes by text length.
        Returns a value between 0 and 1.
        """
        text_lower = text.lower()
        total_score = 0
        for indicator in indicators:
            total_score += text_lower.count(indicator)
        words = len(text.split())
        if words == 0:
            return 0.0
        return min(1.0, total_score / words * 100)

    def get_confidence_score(self) -> float:
        """Returns static confidence in stance detection."""
        return 0.7
