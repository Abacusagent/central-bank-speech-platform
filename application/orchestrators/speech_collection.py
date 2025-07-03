# application/orchestrators/speech_collection.py

"""
Speech Collection Orchestrator - Multi-Plugin Workflow
Orchestrates the complete, cross-institution speech discovery and collection process
for the Central Bank Speech Analysis Platform.

Core Responsibilities:
- Registers and validates plugins
- Iterates over each institution/plugin
- Coordinates discovery, extraction, validation, and speaker assignment
- Aggregates results and error metrics for monitoring
- Ensures plugin isolation and transactional safety

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from typing import Tuple
from datetime import datetime
from domain.value_objects import DateRange
from domain.entities import CentralBankSpeech, Institution, SpeechStatus
from domain.repositories import UnitOfWork, RepositoryError, EntityNotFoundError
from interfaces.plugin_interfaces import CentralBankScraperPlugin, SpeechMetadata, SpeechContent, ValidationResult, PluginError, ContentExtractionError, ValidationError, RateLimitError

logger = logging.getLogger(__name__)

class SpeechCollectionOrchestrator:
    """
    Orchestrates the end-to-end collection of speeches across all registered plugins.
    """

    def __init__(self, unit_of_work: UnitOfWork):
        """
        Initialize the orchestrator with a UnitOfWork (repo context).
        """
        self.uow = unit_of_work
        self.plugins: Dict[str, CentralBankScraperPlugin] = {}
        self.collection_metrics: Dict[str, Any] = {}

    def register_plugin(self, plugin: CentralBankScraperPlugin) -> None:
        """
        Register a plugin for a central bank. Prevents duplicate registration.
        """
        code = plugin.get_institution_code()
        if code in self.plugins:
            raise ValueError(f"Plugin for {code} already registered")
        self.plugins[code] = plugin
        logger.info(f"Registered plugin: {code}")

    def get_registered_plugins(self) -> Dict[str, str]:
        """
        Returns {institution_code: institution_name} for all plugins.
        """
        return {code: plugin.get_institution_name() for code, plugin in self.plugins.items()}

    async def collect_all(self, date_range: DateRange, limit_per_institution: Optional[int] = None, skip_existing: bool = True, max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Collects speeches from all registered plugins concurrently.
        Returns aggregate and per-institution collection results.
        """
        if not self.plugins:
            raise RuntimeError("No plugins registered")

        logger.info(f"Starting cross-institution speech collection ({len(self.plugins)} plugins)")

        start_time = datetime.now()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def collect_one(code: str) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                try:
                    result = await self.collect_by_institution(code, date_range, limit=limit_per_institution, skip_existing=skip_existing)
                    return code, result
                except Exception as e:
                    logger.error(f"Collection failed for {code}: {e}")
                    return code, {"error": str(e)}

        tasks = [collect_one(code) for code in self.plugins]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        metrics = {
            "institutions_processed": len(self.plugins),
            "aggregate": {},
            "per_institution": {},
            "errors": [],
            "collection_started_at": start_time.isoformat(),
        }
        total_discovered = 0
        total_collected = 0
        total_failed = 0

        for code, result in results:
            metrics["per_institution"][code] = result
            total_discovered += result.get("speeches_discovered", 0)
            total_collected += result.get("speeches_collected", 0)
            total_failed += result.get("speeches_failed", 0)
            if "error" in result:
                metrics["errors"].append({code: result["error"]})

        end_time = datetime.now()
        metrics["aggregate"] = {
            "total_speeches_discovered": total_discovered,
            "total_speeches_collected": total_collected,
            "total_speeches_failed": total_failed,
            "success_rate": (total_collected / max(1, total_discovered)) * 100 if total_discovered else 0,
            "total_duration_seconds": (end_time - start_time).total_seconds()
        }
        metrics["collection_finished_at"] = end_time.isoformat()
        logger.info(f"Speech collection completed: {metrics['aggregate']}")
        return metrics

    async def collect_by_institution(self, institution_code: str, date_range: DateRange, limit: Optional[int] = None, skip_existing: bool = True) -> Dict[str, Any]:
        """
        Collects speeches for a single institution/plugin. Handles full speech lifecycle.
        Returns detailed collection results and error/warning lists.
        """
        if institution_code not in self.plugins:
            raise ValueError(f"No plugin registered for {institution_code}")

        plugin = self.plugins[institution_code]
        logger.info(f"Collecting speeches for {institution_code} from {date_range.start_date} to {date_range.end_date}")
        start_time = datetime.now()

        # All collection results
        discovered: List[SpeechMetadata] = []
        filtered: List[SpeechMetadata] = []
        collected: int = 0
        failed: int = 0
        skipped: int = 0
        errors: List[str] = []
        warnings: List[str] = []

        try:
            async with self.uow:
                # Ensure institution entity exists
                institution = await self._get_or_create_institution(plugin)

                # Discovery
                discovered = await asyncio.get_event_loop().run_in_executor(
                    None, plugin.discover_speeches, date_range
                )
                logger.info(f"{len(discovered)} speeches discovered for {institution_code}")

                # Filter out existing speeches (by URL)
                if skip_existing:
                    filtered = []
                    for meta in discovered:
                        exists = await self.uow.speeches.get_by_url(meta.url)
                        if not exists:
                            filtered.append(meta)
                        else:
                            skipped += 1
                    logger.info(f"{len(filtered)} speeches to process after filtering ({skipped} skipped)")
                else:
                    filtered = discovered

                # Optionally apply limit
                if limit:
                    filtered = filtered[:limit]

                # Main speech processing loop
                for i, meta in enumerate(filtered):
                    try:
                        logger.debug(f"[{i+1}/{len(filtered)}] Processing {meta.title} ({meta.url})")
                        # Create speech entity
                        speech = CentralBankSpeech()
                        speech.update_metadata(meta)
                        speech.institution = institution

                        # Extract content
                        content = await asyncio.get_event_loop().run_in_executor(
                            None, plugin.extract_speech_content, meta
                        )
                        speech.set_content(content)

                        # Validate
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

                        # Respect rate limits (if plugin provides)
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

                # Compile result
                return {
                    "institution_code": institution_code,
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
            logger.error(f"Collection failed for {institution_code}: {e}")
            raise

    async def _get_or_create_institution(self, plugin: CentralBankScraperPlugin) -> Institution:
        """
        Retrieve or create the Institution entity for a plugin.
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
            # Save and assign new speaker
            await self.uow.speakers.save(plugin_speaker)
            speech.speaker = plugin_speaker

