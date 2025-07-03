# application/services/analysis_service.py

"""
Speech Analysis Service - Application Layer

Handles NLP analysis for one or more speeches using the modular NLP pipeline.
Manages orchestration, aggregation of results, error handling, and domain updates.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from domain.entities import CentralBankSpeech, SentimentAnalysis
from domain.repositories import UnitOfWork, RepositoryError, EntityNotFoundError
from nlp.pipeline import NLPProcessingPipeline, NLPAnalysis

logger = logging.getLogger(__name__)

class SpeechAnalysisService:
    """
    Service for running the NLP pipeline on speeches, storing results, and aggregating metrics.
    """

    def __init__(self, unit_of_work: UnitOfWork, nlp_pipeline: NLPProcessingPipeline):
        self.uow = unit_of_work
        self.nlp_pipeline = nlp_pipeline

    async def analyze_speech(
        self,
        speech: CentralBankSpeech,
        persist: bool = True
    ) -> Dict[str, Any]:
        """
        Run the NLP pipeline on a single speech. Optionally persists the analysis result.
        Returns the NLP analysis results and success/error details.
        """
        try:
            await self._ensure_pipeline_initialized()
            logger.info(f"Analyzing speech {getattr(speech, 'id', None)} ({getattr(speech, 'title', '')})")
            start_time = datetime.now()

            nlp_analysis: NLPAnalysis = await self.nlp_pipeline.process_speech(speech)

            sentiment_analysis = nlp_analysis.to_sentiment_analysis()
            speech.set_sentiment_analysis(sentiment_analysis)

            if persist:
                async with self.uow:
                    await self.uow.speeches.save(speech)
                    await self.uow.commit()

            end_time = datetime.now()
            logger.info(f"NLP analysis complete for speech {getattr(speech, 'id', None)}")
            return {
                "speech_id": getattr(speech, 'id', None),
                "analysis": nlp_analysis,
                "processing_time_seconds": (end_time - start_time).total_seconds(),
                "success": True,
                "errors": []
            }

        except Exception as e:
            logger.error(f"Speech NLP analysis failed: {e}")
            return {
                "speech_id": getattr(speech, 'id', None),
                "analysis": None,
                "processing_time_seconds": 0,
                "success": False,
                "errors": [str(e)]
            }

    async def analyze_speeches(
        self,
        speeches: List[CentralBankSpeech],
        persist: bool = True,
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Run NLP analysis on a list of speeches concurrently.
        Returns aggregate metrics and a list of per-speech results.
        """
        if not speeches:
            return {"results": [], "aggregate": {}, "errors": ["No speeches provided"]}

        await self._ensure_pipeline_initialized()
        logger.info(f"Starting NLP analysis for {len(speeches)} speeches (max_concurrent={max_concurrent})")
        start_time = datetime.now()

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def analyze_one(speech):
            async with semaphore:
                return await self.analyze_speech(speech, persist=persist)

        tasks = [analyze_one(s) for s in speeches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=False)

        num_success = sum(1 for r in batch_results if r["success"])
        num_failed = len(batch_results) - num_success

        end_time = datetime.now()
        aggregate = {
            "total_speeches": len(speeches),
            "successfully_analyzed": num_success,
            "failed": num_failed,
            "success_rate": (num_success / max(1, len(speeches))) * 100,
            "total_duration_seconds": (end_time - start_time).total_seconds()
        }

        logger.info(f"Batch NLP analysis complete: {aggregate}")
        return {"results": batch_results, "aggregate": aggregate}

    async def _ensure_pipeline_initialized(self):
        """
        Ensures the NLP pipeline is initialized before processing.
        """
        if not getattr(self.nlp_pipeline, "is_initialized", False):
            logger.info("Initializing NLP pipeline...")
            await self.nlp_pipeline.initialize()

