# application/services/collection_service.py

"""
Speech Collection Service - Single-Institution Application Layer

Handles the full workflow for collecting speeches from one central bank via its plugin.
Coordinates discovery, extraction, validation, speaker assignment, and persistence.

Key Responsibilities:
- Plugin isolation and error resilience
- Transactional safety using UnitOfWork
- Full metrics and error aggregation
- Clear separation from orchestration (multi-institution handled at orchestrator level)

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from domain.value_objects import DateRange
from domain.entities import CentralBankSpeech, Institution
from domain.repositories import UnitOfWork, RepositoryError, EntityNotFoundError
from interfaces.plugin_interfaces import (
    CentralBankScraperPlugin, SpeechMetadata, SpeechContent, ValidationResult,
    PluginError, ContentExtractionError, ValidationError, RateLimitError
)

logger = logging.getLogger(__name__)

class SpeechCollectionService:
    """
    Service to collect and persist speeches from a single central bank (plugin).
    Handles discovery, extraction, validation, speaker assignment, and persistence.
    """

    def __init__(self, unit_of_work: UnitOfWork):
        self.uow = unit_of_work

    async def collect(
        self,
        plugin: CentralBankScraperPlugin,
        date_range: DateRange,
        limit: Optional[int] = None,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Collects speeches from a single plugin within the specified date range.
        Returns comprehensive collection metrics and error details.
        """
        logger.info(f"Starting speech collection for {plugin.get_institution_code()} [{date_range.start_date} - {date_range.end_date}]")
        start_time = datetime.now()
        discovered: List[SpeechMetadata] = []
        filtered: List[SpeechMetadata] = []
        collected = 0
        failed = 0
        skipped = 0
        errors: List[str] = []
        warnings: List[str] = []

        try:
            async with self.uow:
                # Ensure institution exists
                institution = await self._get_or_create_institution(plugin)

                # Step 1: Discover
                discovered = await asyncio.get_event_loop().run_in_executor(
                    None, plugin.discover_speeches, date_range
                )
                logger.info(f"Discovered {len(discovered)} speeches for {plugin.get_institution_code()}")

                # Step 2: Filter out existing
                if skip_existing:
                    for meta in discovered:
                        exists = await self.uow.speeches.get_by_url(meta.url)
                        if not exists:
                            filtered.append(meta)
                        else:
                            skipped += 1
                else:
                    filtered = discovered

                # Step 3: Optionally limit batch
                if limit:
                    filtered = filtered[:limit]

                # Step 4: Main processing loop
                for i, meta in enumerate(filtered):
                    try:
                        logger.debug(f"[{i+1}/{len(filtered)}] Processing {meta.title} ({meta.url})")
                        speech = CentralBankSpeech()
                        speech.update_metadata(meta)
                        speech.institution = institution

                        # Extraction
                        content = await asyncio.get_event_loop().run_in_executor(
                            None, plugin.extract_speech_content, meta
                        )
                        speech.set_content(content)

                        # Validation
                        validation_result = await asyncio.get_event_loop().run_in_executor(
                            None, plugin.validate_speech_authenticity, meta, content
                        )
                        speech.set_validation_result(validation_result)

                        if not validation_result.is_valid:
                            failed += 1
                            errors.append(f"Validation failed: {meta.url} | {validation_result.issues}")
                            continue

                        # Speaker assignment
                        await self._assign_speaker(plugin, speech, meta)

                        # Persist speech
                        await self.uow.speeches.save(speech)
                        collected += 1

                        # Rate limiting
                        if hasattr(plugin, "get_rate_limit_delay"):
                            await asyncio.sleep(plugin.get_rate_limit_delay())

                    except ContentExtractionError as e:
                        failed += 1
                        errors.append(f"Extraction error: {meta.url} | {e}")
                    except ValidationError as e:
                        failed += 1
                        errors.append(f"Validation error: {meta.url} | {e}")
                    except Exception as e:
                        failed += 1
                        errors.append(f"Unknown error for {meta.url}: {e}")

                await self.uow.commit()
                end_time = datetime.now()
                return {
                    "institution_code": plugin.get_institution_code(),
                    "institution_name": plugin.get_institution_name(),
                    "date_range": {
                        "start_date": date_range.start_date.isoformat(),
                        "end_date": date_range.end_date.isoformat(),
                    },
                    "speeches_discovered": len(discovered),
                    "speeches_collected": collected,
                    "speeches_failed": failed,
                    "speeches_skipped": skipped,
                    "processing_duration_seconds": (end_time - start_time).total_seconds(),
                    "success_rate": (collected / max(1, len(filtered))) * 100 if filtered else 0,
                    "errors": errors,
                    "warnings": warnings,
                }

        except Exception as e:
            await self.uow.rollback()
            logger.error(f"Collection failed for {plugin.get_institution_code()}: {e}")
            raise

    async def _get_or_create_institution(self, plugin: CentralBankScraperPlugin) -> Institution:
        """
        Retrieve or create the Institution entity for the plugin.
        """
        code = plugin.get_institution_code()
        institution = await self.uow.institutions.get_by_code(code)
        if not institution:
            institution = plugin.get_institution() if hasattr(plugin, "get_institution") else Institution(
                code=code,
                name=plugin.get_institution_name(),
                country="Unknown",
                institution_type="central_bank"
            )
            await self.uow.institutions.save(institution)
            logger.info(f"Created institution entity for {code}")
        return institution

    async def _assign_speaker(self, plugin: CentralBankScraperPlugin, speech: CentralBankSpeech, meta: SpeechMetadata) -> None:
        """
        Assign the best-matching speaker to the speech, using plugin's database and platform repository.
        """
        speaker_db = plugin.get_speaker_database()
        plugin_speaker = speaker_db.find_speaker(meta.speaker_name)
        if not plugin_speaker:
            logger.warning(f"Unknown speaker: {meta.speaker_name} (plugin: {plugin.get_institution_code()})")
            return
        # Try to find existing speaker in DB
        candidates = await self.uow.speakers.find_by_name(plugin_speaker.name)
        if candidates:
            speech.speaker = candidates[0]
        else:
            await self.uow.speakers.save(plugin_speaker)
            speech.speaker = plugin_speaker

