"""
Speech Collection Service - Application Layer

This service orchestrates the complete speech collection workflow across multiple
central bank plugins. It coordinates discovery, extraction, validation, and analysis
while maintaining plugin isolation and providing comprehensive error handling.

Key Responsibilities:
- Orchestrate multi-plugin speech collection
- Manage the complete speech lifecycle (discovery -> extraction -> validation -> analysis)
- Provide transactional consistency across operations
- Handle plugin failures gracefully without affecting other plugins
- Collect and report comprehensive metrics
"""

import asyncio
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID

from domain.entities import (
    CentralBankSpeech, CentralBankSpeaker, Institution, SpeechStatus, 
    PolicyStance, SentimentAnalysis
)
from domain.value_objects import DateRange, ProcessingMetrics, ContentHash
from domain.repositories import (
    UnitOfWork, SpeechRepository, SpeakerRepository, InstitutionRepository,
    RepositoryError, EntityNotFoundError
)
from interfaces.plugin_interfaces import (
    CentralBankScraperPlugin, SpeechMetadata, SpeechContent, ValidationResult,
    PluginError, ContentExtractionError, ValidationError, RateLimitError
)

logger = logging.getLogger(__name__)


class SpeechCollectionError(Exception):
    """Base exception for speech collection operations."""
    pass


class PluginRegistrationError(SpeechCollectionError):
    """Raised when plugin registration fails."""
    pass


class CollectionTimeoutError(SpeechCollectionError):
    """Raised when collection operations timeout."""
    pass


class SpeechCollectionService:
    """
    Orchestrates the complete speech collection workflow.
    
    This service manages the entire lifecycle of speech collection from multiple
    central bank plugins, ensuring consistency, reliability, and observability.
    """
    
    def __init__(self, unit_of_work: UnitOfWork):
        """
        Initialize the speech collection service.
        
        Args:
            unit_of_work: Unit of work for transactional operations
        """
        self.uow = unit_of_work
        self.plugins: Dict[str, CentralBankScraperPlugin] = {}
        self.collection_stats: Dict[str, Any] = {}
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure structured logging for the service."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Ensure logger has appropriate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def register_plugin(self, plugin: CentralBankScraperPlugin) -> None:
        """
        Register a central bank scraper plugin.
        
        Args:
            plugin: Plugin instance to register
            
        Raises:
            PluginRegistrationError: If plugin registration fails
            
        Example:
            >>> service = SpeechCollectionService(uow)
            >>> plugin = FederalReservePlugin()
            >>> service.register_plugin(plugin)
        """
        try:
            institution_code = plugin.get_institution_code()
            
            if institution_code in self.plugins:
                raise PluginRegistrationError(
                    f"Plugin for institution {institution_code} already registered"
                )
            
            # Validate plugin implementation
            self._validate_plugin(plugin)
            
            self.plugins[institution_code] = plugin
            self.logger.info(f"Registered plugin: {institution_code}")
            
        except Exception as e:
            raise PluginRegistrationError(f"Failed to register plugin: {e}")
    
    def _validate_plugin(self, plugin: CentralBankScraperPlugin) -> None:
        """Validate that plugin implements required interface correctly."""
        required_methods = [
            'get_institution_code', 'get_institution_name', 'get_supported_languages',
            'discover_speeches', 'extract_speech_content', 'get_speaker_database',
            'validate_speech_authenticity'
        ]
        
        for method_name in required_methods:
            if not hasattr(plugin, method_name):
                raise PluginRegistrationError(f"Plugin missing required method: {method_name}")
            
            method = getattr(plugin, method_name)
            if not callable(method):
                raise PluginRegistrationError(f"Plugin attribute {method_name} is not callable")
    
    def get_registered_plugins(self) -> Dict[str, str]:
        """
        Get information about registered plugins.
        
        Returns:
            Dictionary mapping institution codes to institution names
        """
        return {
            code: plugin.get_institution_name() 
            for code, plugin in self.plugins.items()
        }
    
    async def collect_speeches_by_institution(
        self, 
        institution_code: str, 
        date_range: DateRange,
        limit: Optional[int] = None,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Collect speeches from a specific institution.
        
        Args:
            institution_code: Code of the institution to collect from
            date_range: Date range for speech collection
            limit: Maximum number of speeches to collect
            skip_existing: Whether to skip speeches that already exist
            
        Returns:
            Dictionary containing collection results and metrics
            
        Example:
            >>> service = SpeechCollectionService(uow)
            >>> date_range = DateRange.year(2024)
            >>> result = await service.collect_speeches_by_institution("FED", date_range)
            >>> print(f"Collected {result['speeches_collected']} speeches")
        """
        if institution_code not in self.plugins:
            raise SpeechCollectionError(f"No plugin registered for institution: {institution_code}")
        
        plugin = self.plugins[institution_code]
        start_time = datetime.now()
        
        self.logger.info(
            f"Starting speech collection for {institution_code} "
            f"({date_range.start_date} to {date_range.end_date})"
        )
        
        try:
            # Get or create institution entity
            async with self.uow:
                institution = await self._get_or_create_institution(plugin)
                
                # Discovery phase
                discovered_speeches = await self._discover_speeches(
                    plugin, date_range, limit
                )
                
                # Filter existing speeches if requested
                if skip_existing:
                    discovered_speeches = await self._filter_existing_speeches(
                        discovered_speeches
                    )
                
                # Process speeches through complete lifecycle
                processing_results = await self._process_speeches_batch(
                    plugin, institution, discovered_speeches
                )
                
                await self.uow.commit()
                
                # Compile results
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                result = {
                    'institution_code': institution_code,
                    'institution_name': plugin.get_institution_name(),
                    'date_range': {
                        'start_date': date_range.start_date.isoformat(),
                        'end_date': date_range.end_date.isoformat(),
                        'days': date_range.days
                    },
                    'speeches_discovered': len(discovered_speeches),
                    'speeches_collected': processing_results['successful_count'],
                    'speeches_failed': processing_results['failed_count'],
                    'speeches_skipped': processing_results['skipped_count'],
                    'processing_duration_seconds': duration,
                    'average_processing_time_per_speech': (
                        duration / max(len(discovered_speeches), 1)
                    ),
                    'success_rate': (
                        processing_results['successful_count'] / 
                        max(len(discovered_speeches), 1) * 100
                    ),
                    'errors': processing_results['errors'],
                    'warnings': processing_results['warnings']
                }
                
                self.logger.info(
                    f"Completed speech collection for {institution_code}: "
                    f"{result['speeches_collected']} speeches collected "
                    f"({result['success_rate']:.1f}% success rate)"
                )
                
                return result
                
        except Exception as e:
            await self.uow.rollback()
            self.logger.error(f"Speech collection failed for {institution_code}: {e}")
            raise SpeechCollectionError(f"Collection failed: {e}")
    
    async def collect_speeches_all_institutions(
        self, 
        date_range: DateRange,
        limit_per_institution: Optional[int] = None,
        skip_existing: bool = True,
        max_concurrent_institutions: int = 3
    ) -> Dict[str, Any]:
        """
        Collect speeches from all registered institutions concurrently.
        
        Args:
            date_range: Date range for speech collection
            limit_per_institution: Maximum speeches per institution
            skip_existing: Whether to skip existing speeches
            max_concurrent_institutions: Maximum concurrent institution processing
            
        Returns:
            Dictionary containing aggregate collection results
            
        Example:
            >>> service = SpeechCollectionService(uow)
            >>> date_range = DateRange.year(2024)
            >>> result = await service.collect_speeches_all_institutions(date_range)
            >>> print(f"Total speeches collected: {result['total_speeches_collected']}")
        """
        if not self.plugins:
            raise SpeechCollectionError("No plugins registered")
        
        start_time = datetime.now()
        self.logger.info(
            f"Starting multi-institution speech collection "
            f"({len(self.plugins)} institutions, {date_range.start_date} to {date_range.end_date})"
        )
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent_institutions)
        
        async def collect_with_semaphore(institution_code: str) -> Tuple[str, Dict[str, Any]]:
            """Collect speeches for one institution with concurrency control."""
            async with semaphore:
                try:
                    result = await self.collect_speeches_by_institution(
                        institution_code, date_range, limit_per_institution, skip_existing
                    )
                    return institution_code, result
                except Exception as e:
                    self.logger.error(f"Failed to collect from {institution_code}: {e}")
                    return institution_code, {
                        'error': str(e),
                        'speeches_collected': 0,
                        'speeches_failed': 0,
                        'speeches_discovered': 0
                    }
        
        # Process all institutions concurrently
        tasks = [
            collect_with_semaphore(institution_code) 
            for institution_code in self.plugins.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile aggregate results
        institution_results = {}
        total_discovered = 0
        total_collected = 0
        total_failed = 0
        total_errors = []
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Institution processing failed: {result}")
                total_errors.append(str(result))
                continue
                
            institution_code, institution_result = result
            institution_results[institution_code] = institution_result
            
            total_discovered += institution_result.get('speeches_discovered', 0)
            total_collected += institution_result.get('speeches_collected', 0)
            total_failed += institution_result.get('speeches_failed', 0)
            
            if 'error' in institution_result:
                total_errors.append(f"{institution_code}: {institution_result['error']}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        aggregate_result = {
            'collection_timestamp': end_time.isoformat(),
            'date_range': {
                'start_date': date_range.start_date.isoformat(),
                'end_date': date_range.end_date.isoformat(),
                'days': date_range.days
            },
            'institutions_processed': len(institution_results),
            'institutions_total': len(self.plugins),
            'total_speeches_discovered': total_discovered,
            'total_speeches_collected': total_collected,
            'total_speeches_failed': total_failed,
            'overall_success_rate': (
                total_collected / max(total_discovered, 1) * 100
            ),
            'total_processing_duration_seconds': duration,
            'institution_results': institution_results,
            'errors': total_errors
        }
        
        self.logger.info(
            f"Completed multi-institution collection: "
            f"{total_collected} speeches collected from {len(institution_results)} institutions "
            f"({aggregate_result['overall_success_rate']:.1f}% success rate)"
        )
        
        return aggregate_result
    
    async def _get_or_create_institution(self, plugin: CentralBankScraperPlugin) -> Institution:
        """Get existing institution or create new one."""
        institution_code = plugin.get_institution_code()
        
        # Try to get existing institution
        institution = await self.uow.institutions.get_by_code(institution_code)
        
        if institution is None:
            # Create new institution
            from domain.entities import InstitutionType
            
            institution = Institution(
                code=institution_code,
                name=plugin.get_institution_name(),
                country="Unknown",  # Plugin could be extended to provide this
                institution_type=InstitutionType.CENTRAL_BANK
            )
            
            await self.uow.institutions.save(institution)
            self.logger.info(f"Created new institution: {institution_code}")
        
        return institution
    
    async def _discover_speeches(
        self, 
        plugin: CentralBankScraperPlugin, 
        date_range: DateRange, 
        limit: Optional[int] = None
    ) -> List[SpeechMetadata]:
        """Discover speeches using the plugin."""
        try:
            self.logger.info(f"Discovering speeches for {plugin.get_institution_code()}")
            
            speeches = await asyncio.get_event_loop().run_in_executor(
                None, plugin.discover_speeches, date_range, limit
            )
            
            self.logger.info(f"Discovered {len(speeches)} speeches")
            return speeches
            
        except RateLimitError as e:
            self.logger.warning(f"Rate limit hit during discovery: {e}")
            # Wait and retry once
            await asyncio.sleep(plugin.get_rate_limit_delay() * 2)
            return await asyncio.get_event_loop().run_in_executor(
                None, plugin.discover_speeches, date_range, limit
            )
        except Exception as e:
            self.logger.error(f"Speech discovery failed: {e}")
            raise SpeechCollectionError(f"Discovery failed: {e}")
    
    async def _filter_existing_speeches(
        self, 
        discovered_speeches: List[SpeechMetadata]
    ) -> List[SpeechMetadata]:
        """Filter out speeches that already exist in the repository."""
        from domain.value_objects import Url
        
        filtered_speeches = []
        
        for speech_metadata in discovered_speeches:
            try:
                url = Url(speech_metadata.url)
                existing_speech = await self.uow.speeches.get_by_url(url)
                
                if existing_speech is None:
                    filtered_speeches.append(speech_metadata)
                else:
                    self.logger.debug(f"Skipping existing speech: {speech_metadata.url}")
                    
            except Exception as e:
                self.logger.warning(f"Error checking existing speech {speech_metadata.url}: {e}")
                # Include speech if we can't check (better to have duplicates than miss speeches)
                filtered_speeches.append(speech_metadata)
        
        self.logger.info(f"Filtered to {len(filtered_speeches)} new speeches")
        return filtered_speeches
    
    async def _process_speeches_batch(
        self, 
        plugin: CentralBankScraperPlugin,
        institution: Institution,
        speeches_metadata: List[SpeechMetadata]
    ) -> Dict[str, Any]:
        """Process a batch of speeches through the complete lifecycle."""
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        errors = []
        warnings = []
        
        for i, speech_metadata in enumerate(speeches_metadata):
            try:
                self.logger.info(f"Processing speech {i+1}/{len(speeches_metadata)}: {speech_metadata.title}")
                
                # Create speech entity
                speech = CentralBankSpeech()
                speech.update_metadata(speech_metadata)
                speech.institution = institution
                
                # Extract content
                content = await self._extract_speech_content(plugin, speech_metadata)
                speech.set_content(content)
                
                # Validate speech authenticity
                validation_result = await self._validate_speech(plugin, speech_metadata, content)
                speech.set_validation_result(validation_result)
                
                if not validation_result.is_valid:
                    self.logger.warning(f"Speech failed validation: {speech_metadata.url}")
                    failed_count += 1
                    errors.append(f"Validation failed for {speech_metadata.url}: {validation_result.issues}")
                    continue
                
                # Assign speaker
                await self._assign_speaker(plugin, speech, speech_metadata)
                
                # Save speech
                await self.uow.speeches.save(speech)
                successful_count += 1
                
                # Respect rate limits
                await asyncio.sleep(plugin.get_rate_limit_delay())
                
            except ContentExtractionError as e:
                self.logger.error(f"Content extraction failed for {speech_metadata.url}: {e}")
                failed_count += 1
                errors.append(f"Content extraction failed: {e}")
                
            except ValidationError as e:
                self.logger.error(f"Validation failed for {speech_metadata.url}: {e}")
                failed_count += 1
                errors.append(f"Validation failed: {e}")
                
            except Exception as e:
                self.logger.error(f"Unexpected error processing {speech_metadata.url}: {e}")
                failed_count += 1
                errors.append(f"Unexpected error: {e}")
        
        return {
            'successful_count': successful_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
            'errors': errors,
            'warnings': warnings
        }
    
    async def _extract_speech_content(
        self, 
        plugin: CentralBankScraperPlugin,
        speech_metadata: SpeechMetadata
    ) -> SpeechContent:
        """Extract content from a speech using the plugin."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, plugin.extract_speech_content, speech_metadata
            )
        except RateLimitError as e:
            self.logger.warning(f"Rate limit hit during content extraction: {e}")
            # Wait and retry once
            await asyncio.sleep(plugin.get_rate_limit_delay() * 2)
            return await asyncio.get_event_loop().run_in_executor(
                None, plugin.extract_speech_content, speech_metadata
            )
    
    async def _validate_speech(
        self, 
        plugin: CentralBankScraperPlugin,
        speech_metadata: SpeechMetadata,
        content: SpeechContent
    ) -> ValidationResult:
        """Validate speech authenticity using the plugin."""
        return await asyncio.get_event_loop().run_in_executor(
            None, plugin.validate_speech_authenticity, speech_metadata, content
        )
    
    async def _assign_speaker(
        self, 
        plugin: CentralBankScraperPlugin,
        speech: CentralBankSpeech,
        speech_metadata: SpeechMetadata
    ) -> None:
        """Assign speaker to speech using plugin speaker database."""
        try:
            speaker_db = plugin.get_speaker_database()
            plugin_speaker = speaker_db.find_speaker(speech_metadata.speaker_name)
            
            if plugin_speaker is None:
                self.logger.warning(f"Speaker not found in plugin database: {speech_metadata.speaker_name}")
                return
            
            # Try to find existing speaker in our repository
            existing_speakers = await self.uow.speakers.find_by_name(plugin_speaker.name)
            
            if existing_speakers:
                # Use existing speaker
                speech.speaker = existing_speakers[0]
            else:
                # Create new speaker entity
                speaker = CentralBankSpeaker(
                    name=plugin_speaker.name,
                    role=plugin_speaker.role,
                    institution=speech.institution,
                    start_date=plugin_speaker.start_date,
                    end_date=plugin_speaker.end_date,
                    voting_member=plugin_speaker.voting_member,
                    biographical_notes=plugin_speaker.biographical_notes
                )
                
                await self.uow.speakers.save(speaker)
                speech.speaker = speaker
                
        except Exception as e:
            self.logger.error(f"Error assigning speaker: {e}")
            # Continue without speaker assignment rather than fail the whole speech
    
    async def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the speech collection.
        
        Returns:
            Dictionary containing various statistics about collected speeches
        """
        async with self.uow:
            # Get basic counts
            institutions = await self.uow.institutions.get_all()
            total_speeches = 0
            institution_stats = {}
            
            for institution in institutions:
                count = await self.uow.speeches.count_by_institution(institution)
                total_speeches += count
                
                # Get date range for this institution
                date_range = await self.uow.speeches.get_date_range_for_institution(institution)
                
                institution_stats[institution.code] = {
                    'name': institution.name,
                    'speech_count': count,
                    'date_range': {
                        'start_date': date_range.start_date.isoformat() if date_range else None,
                        'end_date': date_range.end_date.isoformat() if date_range else None,
                        'days': date_range.days if date_range else 0
                    }
                }
            
            # Get status counts
            from domain.entities import SpeechStatus
            status_counts = {}
            for status in SpeechStatus:
                count = await self.uow.speeches.count_by_status(status)
                status_counts[status.value] = count
            
            return {
                'total_speeches': total_speeches,
                'total_institutions': len(institutions),
                'registered_plugins': len(self.plugins),
                'institution_statistics': institution_stats,
                'status_distribution': status_counts,
                'last_updated': datetime.now().isoformat()
            }
    
    async def reprocess_failed_speeches(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Reprocess speeches that previously failed validation or extraction.
        
        Args:
            limit: Maximum number of speeches to reprocess
            
        Returns:
            Dictionary containing reprocessing results
        """
        async with self.uow:
            # Get failed speeches
            failed_speeches = await self.uow.speeches.find_by_status(
                SpeechStatus.FAILED, limit=limit
            )
            
            if not failed_speeches:
                return {
                    'speeches_found': 0,
                    'speeches_reprocessed': 0,
                    'speeches_fixed': 0,
                    'speeches_still_failed': 0
                }
            
            reprocessed_count = 0
            fixed_count = 0
            still_failed_count = 0
            
            for speech in failed_speeches:
                try:
                    if not speech.institution or speech.institution.code not in self.plugins:
                        self.logger.warning(f"No plugin for speech institution: {speech.institution}")
                        continue
                    
                    plugin = self.plugins[speech.institution.code]
                    
                    # Retry content extraction if needed
                    if speech.content is None and speech.metadata:
                        try:
                            content = await self._extract_speech_content(plugin, speech.metadata)
                            speech.set_content(content)
                        except Exception as e:
                            self.logger.error(f"Reprocessing content extraction failed: {e}")
                            still_failed_count += 1
                            continue
                    
                    # Retry validation if we have content
                    if speech.content and speech.metadata:
                        try:
                            validation_result = await self._validate_speech(
                                plugin, speech.metadata, speech.content
                            )
                            speech.set_validation_result(validation_result)
                            
                            if validation_result.is_valid:
                                fixed_count += 1
                            else:
                                still_failed_count += 1
                                
                        except Exception as e:
                            self.logger.error(f"Reprocessing validation failed: {e}")
                            still_failed_count += 1
                            continue
                    
                    await self.uow.speeches.save(speech)
                    reprocessed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error reprocessing speech {speech.id}: {e}")
                    still_failed_count += 1
            
            await self.uow.commit()
            
            return {
                'speeches_found': len(failed_speeches),
                'speeches_reprocessed': reprocessed_count,
                'speeches_fixed': fixed_count,
                'speeches_still_failed': still_failed_count
            }