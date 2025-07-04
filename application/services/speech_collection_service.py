# application/services/speech_collection_service.py

"""
Speech Collection Service - Production-Ready Application Layer

This service orchestrates the complete speech collection workflow across multiple
central bank plugins with enterprise-grade reliability, observability, and performance.
Implements the Command pattern for operations and provides comprehensive error handling,
circuit breaker protection, and detailed metrics collection.

Key Features:
- Plugin lifecycle management with health monitoring
- Circuit breaker pattern for failing plugins
- Comprehensive retry logic with exponential backoff
- Batch processing for optimal database performance
- Real-time metrics and observability
- Event-driven architecture with domain events
- Production-ready error handling and logging

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Set, Tuple, AsyncIterator
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from domain.entities import (
    CentralBankSpeech, CentralBankSpeaker, Institution, SpeechStatus, 
    PolicyStance, SentimentAnalysis, InstitutionType
)
from domain.value_objects import DateRange, Url, ContentHash, ConfidenceLevel
from domain.repositories import (
    UnitOfWork, SpeechRepository, SpeakerRepository, InstitutionRepository,
    RepositoryError, EntityNotFoundError, DuplicateEntityError
)
from interfaces.plugin_interfaces import (
    CentralBankScraperPlugin, SpeechMetadata, SpeechContent, ValidationResult,
    ValidationStatus, PluginError, ContentExtractionError, ValidationError, 
    RateLimitError, SpeakerDatabase
)

logger = logging.getLogger(__name__)


# Domain Events for Speech Collection
class SpeechCollectionEventType(Enum):
    """Types of speech collection events."""
    PLUGIN_REGISTERED = "plugin_registered"
    COLLECTION_STARTED = "collection_started"
    COLLECTION_COMPLETED = "collection_completed"
    SPEECH_DISCOVERED = "speech_discovered"
    SPEECH_EXTRACTED = "speech_extracted"
    SPEECH_VALIDATED = "speech_validated"
    SPEECH_FAILED = "speech_failed"
    PLUGIN_FAILED = "plugin_failed"
    RATE_LIMIT_HIT = "rate_limit_hit"


@dataclass
class SpeechCollectionEvent:
    """Domain event for speech collection operations."""
    event_type: SpeechCollectionEventType
    timestamp: datetime
    institution_code: str
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid4()))


# Exception Hierarchy
class SpeechCollectionError(Exception):
    """Base exception for speech collection operations."""
    pass


class PluginRegistrationError(SpeechCollectionError):
    """Raised when plugin registration fails."""
    pass


class CollectionTimeoutError(SpeechCollectionError):
    """Raised when collection operations timeout."""
    pass


class PluginHealthError(SpeechCollectionError):
    """Raised when plugin health check fails."""
    pass


class BatchProcessingError(SpeechCollectionError):
    """Raised when batch processing fails."""
    pass


# Circuit Breaker for Plugin Protection
class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3


class CircuitBreaker:
    """Circuit breaker implementation for plugin protection."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker moved to HALF_OPEN state")
            else:
                raise SpeechCollectionError("Circuit breaker is OPEN - plugin unavailable")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info("Circuit breaker moved to CLOSED state")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning("Circuit breaker moved to OPEN state")


# Plugin Management
@dataclass
class PluginHealthStatus:
    """Health status of a plugin."""
    is_healthy: bool
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    consecutive_failures: int = 0


class PluginManager:
    """Manages plugin lifecycle and health monitoring."""
    
    def __init__(self):
        self.plugins: Dict[str, CentralBankScraperPlugin] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_status: Dict[str, PluginHealthStatus] = {}
        self.logger = logging.getLogger(f"{__name__}.PluginManager")
    
    def register_plugin(self, plugin: CentralBankScraperPlugin) -> None:
        """Register a plugin with health monitoring."""
        institution_code = plugin.get_institution_code()
        
        if institution_code in self.plugins:
            raise PluginRegistrationError(
                f"Plugin for institution {institution_code} already registered"
            )
        
        # Validate plugin
        self._validate_plugin(plugin)
        
        # Register plugin
        self.plugins[institution_code] = plugin
        self.circuit_breakers[institution_code] = CircuitBreaker(CircuitBreakerConfig())
        self.health_status[institution_code] = PluginHealthStatus(
            is_healthy=True,
            last_check=datetime.now(),
            response_time_ms=0.0
        )
        
        self.logger.info(f"Registered plugin: {institution_code}")
    
    def _validate_plugin(self, plugin: CentralBankScraperPlugin) -> None:
        """Validate plugin implementation."""
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
    
    async def get_healthy_plugins(self) -> Dict[str, CentralBankScraperPlugin]:
        """Get only healthy plugins."""
        healthy_plugins = {}
        
        for code, plugin in self.plugins.items():
            if await self.check_plugin_health(code):
                healthy_plugins[code] = plugin
        
        return healthy_plugins
    
    async def check_plugin_health(self, institution_code: str) -> bool:
        """Check health of a specific plugin."""
        if institution_code not in self.plugins:
            return False
        
        plugin = self.plugins[institution_code]
        start_time = datetime.now()
        
        try:
            # Simple health check - try to get institution info
            _ = plugin.get_institution_code()
            _ = plugin.get_institution_name()
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.health_status[institution_code] = PluginHealthStatus(
                is_healthy=True,
                last_check=datetime.now(),
                response_time_ms=response_time
            )
            
            return True
            
        except Exception as e:
            self.health_status[institution_code] = PluginHealthStatus(
                is_healthy=False,
                last_check=datetime.now(),
                response_time_ms=0.0,
                error_message=str(e),
                consecutive_failures=self.health_status[institution_code].consecutive_failures + 1
            )
            
            self.logger.error(f"Health check failed for {institution_code}: {e}")
            return False
    
    async def execute_with_circuit_breaker(
        self, 
        institution_code: str, 
        operation: str, 
        func, 
        *args, 
        **kwargs
    ):
        """Execute operation with circuit breaker protection."""
        if institution_code not in self.circuit_breakers:
            raise PluginRegistrationError(f"No circuit breaker for {institution_code}")
        
        circuit_breaker = self.circuit_breakers[institution_code]
        
        try:
            return await circuit_breaker.call(func, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Circuit breaker protected operation failed: {operation} for {institution_code}: {e}")
            raise


# Retry Logic with Exponential Backoff
@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RetryHandler")
    
    async def execute_with_retry(self, operation_name: str, func, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except (RateLimitError, ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt == self.config.max_attempts - 1:
                    break
                
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_attempts} for {operation_name} "
                    f"failed: {e}. Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                # Don't retry for other exceptions
                self.logger.error(f"Non-retryable error in {operation_name}: {e}")
                raise
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


# Main Service Implementation
class SpeechCollectionService:
    """
    Production-ready speech collection service with comprehensive error handling,
    circuit breaker protection, retry logic, and observability.
    """
    
    def __init__(self, unit_of_work: UnitOfWork):
        """Initialize the speech collection service."""
        self.uow = unit_of_work
        self.plugin_manager = PluginManager()
        self.retry_handler = RetryHandler(RetryConfig())
        self.events: List[SpeechCollectionEvent] = []
        self.collection_stats: Dict[str, Any] = {}
        self._setup_logging()
        self._setup_metrics()
    
    def _setup_logging(self) -> None:
        """Configure structured logging for the service."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Ensure proper logging configuration
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_metrics(self) -> None:
        """Initialize metrics collection."""
        self.metrics = {
            'total_collections': 0,
            'total_speeches_discovered': 0,
            'total_speeches_collected': 0,
            'total_speeches_failed': 0,
            'total_processing_time': 0.0,
            'plugin_health_checks': 0,
            'circuit_breaker_trips': 0,
            'retry_attempts': 0
        }
    
    def register_plugin(self, plugin: CentralBankScraperPlugin) -> None:
        """Register a central bank scraper plugin."""
        self.plugin_manager.register_plugin(plugin)
        
        # Emit registration event
        event = SpeechCollectionEvent(
            event_type=SpeechCollectionEventType.PLUGIN_REGISTERED,
            timestamp=datetime.now(),
            institution_code=plugin.get_institution_code(),
            data={
                'institution_name': plugin.get_institution_name(),
                'supported_languages': plugin.get_supported_languages()
            }
        )
        self.events.append(event)
    
    def get_registered_plugins(self) -> Dict[str, str]:
        """Get information about registered plugins."""
        return {
            code: plugin.get_institution_name() 
            for code, plugin in self.plugin_manager.plugins.items()
        }
    
    async def collect_speeches_by_institution(
        self, 
        institution_code: str, 
        date_range: DateRange,
        limit: Optional[int] = None,
        skip_existing: bool = True,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Collect speeches from a specific institution with production-grade reliability.
        
        Args:
            institution_code: Code of the institution to collect from
            date_range: Date range for speech collection
            limit: Maximum number of speeches to collect
            skip_existing: Whether to skip speeches that already exist
            batch_size: Size of batches for processing
            
        Returns:
            Dictionary containing collection results and comprehensive metrics
        """
        if institution_code not in self.plugin_manager.plugins:
            raise SpeechCollectionError(f"No plugin registered for institution: {institution_code}")
        
        plugin = self.plugin_manager.plugins[institution_code]
        start_time = datetime.now()
        correlation_id = str(uuid4())
        
        # Emit collection started event
        self._emit_event(
            SpeechCollectionEventType.COLLECTION_STARTED,
            institution_code,
            {
                'date_range': {
                    'start_date': date_range.start_date.isoformat(),
                    'end_date': date_range.end_date.isoformat()
                },
                'limit': limit,
                'correlation_id': correlation_id
            }
        )
        
        self.logger.info(
            f"Starting speech collection for {institution_code} "
            f"({date_range.start_date} to {date_range.end_date}) "
            f"[{correlation_id}]"
        )
        
        try:
            # Check plugin health first
            if not await self.plugin_manager.check_plugin_health(institution_code):
                raise PluginHealthError(f"Plugin {institution_code} is unhealthy")
            
            # Get or create institution entity
            institution = await self._get_or_create_institution(plugin)
            
            # Discovery phase with circuit breaker protection
            discovered_speeches = await self._discover_speeches_with_protection(
                plugin, institution_code, date_range, limit, correlation_id
            )
            
            # Filter existing speeches if requested
            if skip_existing:
                discovered_speeches = await self._filter_existing_speeches(
                    discovered_speeches, correlation_id
                )
            
            # Process speeches in batches
            processing_results = await self._process_speeches_in_batches(
                plugin, institution, discovered_speeches, batch_size, correlation_id
            )
            
            # Compile comprehensive results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = self._compile_collection_result(
                institution_code, plugin, date_range, discovered_speeches,
                processing_results, duration, correlation_id
            )
            
            # Emit collection completed event
            self._emit_event(
                SpeechCollectionEventType.COLLECTION_COMPLETED,
                institution_code,
                {
                    'result': result,
                    'correlation_id': correlation_id
                }
            )
            
            # Update metrics
            self.metrics['total_collections'] += 1
            self.metrics['total_speeches_discovered'] += result['speeches_discovered']
            self.metrics['total_speeches_collected'] += result['speeches_collected']
            self.metrics['total_speeches_failed'] += result['speeches_failed']
            self.metrics['total_processing_time'] += duration
            
            self.logger.info(
                f"Completed speech collection for {institution_code}: "
                f"{result['speeches_collected']} speeches collected "
                f"({result['success_rate']:.1f}% success rate) "
                f"[{correlation_id}]"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Speech collection failed for {institution_code} [{correlation_id}]: {e}")
            
            # Emit failure event
            self._emit_event(
                SpeechCollectionEventType.PLUGIN_FAILED,
                institution_code,
                {
                    'error': str(e),
                    'correlation_id': correlation_id
                }
            )
            
            raise SpeechCollectionError(f"Collection failed for {institution_code}: {e}")
    
    async def collect_speeches_all_institutions(
        self, 
        date_range: DateRange,
        limit_per_institution: Optional[int] = None,
        skip_existing: bool = True,
        max_concurrent_institutions: int = 3,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Collect speeches from all registered institutions with advanced error handling.
        
        Uses circuit breaker protection, retry logic, and comprehensive monitoring.
        """
        # Only process healthy plugins
        healthy_plugins = await self.plugin_manager.get_healthy_plugins()
        
        if not healthy_plugins:
            raise SpeechCollectionError("No healthy plugins available")
        
        start_time = datetime.now()
        correlation_id = str(uuid4())
        
        self.logger.info(
            f"Starting multi-institution speech collection "
            f"({len(healthy_plugins)} healthy institutions, "
            f"{date_range.start_date} to {date_range.end_date}) "
            f"[{correlation_id}]"
        )
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent_institutions)
        
        async def collect_with_protection(institution_code: str) -> Tuple[str, Dict[str, Any]]:
            """Collect speeches for one institution with full protection."""
            async with semaphore:
                try:
                    result = await self.collect_speeches_by_institution(
                        institution_code, date_range, limit_per_institution, 
                        skip_existing, batch_size
                    )
                    return institution_code, result
                except Exception as e:
                    self.logger.error(f"Failed to collect from {institution_code}: {e}")
                    return institution_code, {
                        'error': str(e),
                        'speeches_collected': 0,
                        'speeches_failed': 0,
                        'speeches_discovered': 0,
                        'success_rate': 0.0
                    }
        
        # Process all institutions concurrently
        tasks = [
            collect_with_protection(institution_code) 
            for institution_code in healthy_plugins.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile comprehensive aggregate results
        return self._compile_aggregate_results(
            results, healthy_plugins, date_range, start_time, correlation_id
        )
    
    async def _discover_speeches_with_protection(
        self,
        plugin: CentralBankScraperPlugin,
        institution_code: str,
        date_range: DateRange,
        limit: Optional[int],
        correlation_id: str
    ) -> List[SpeechMetadata]:
        """Discover speeches with circuit breaker and retry protection."""
        
        async def discover_operation():
            return await asyncio.get_event_loop().run_in_executor(
                None, plugin.discover_speeches, date_range, limit
            )
        
        try:
            speeches = await self.plugin_manager.execute_with_circuit_breaker(
                institution_code,
                "discover_speeches",
                lambda: self.retry_handler.execute_with_retry(
                    f"discover_speeches_{institution_code}",
                    discover_operation
                )
            )
            
            self.logger.info(f"Discovered {len(speeches)} speeches for {institution_code}")
            
            # Emit discovery events
            for speech in speeches:
                self._emit_event(
                    SpeechCollectionEventType.SPEECH_DISCOVERED,
                    institution_code,
                    {
                        'speech_url': speech.url,
                        'speech_title': speech.title,
                        'correlation_id': correlation_id
                    }
                )
            
            return speeches
            
        except Exception as e:
            self.logger.error(f"Speech discovery failed for {institution_code}: {e}")
            raise SpeechCollectionError(f"Discovery failed for {institution_code}: {e}")
    
    async def _filter_existing_speeches(
        self, 
        discovered_speeches: List[SpeechMetadata],
        correlation_id: str
    ) -> List[SpeechMetadata]:
        """Filter out speeches that already exist with optimized batch checking."""
        if not discovered_speeches:
            return []
        
        # Batch check for existing speeches
        urls = [speech.url for speech in discovered_speeches]
        
        async with self.uow:
            existing_urls = set()
            
            # Check in batches to avoid overwhelming the database
            batch_size = 50
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                
                for url in batch_urls:
                    try:
                        existing_speech = await self.uow.speeches.get_by_url(url)
                        if existing_speech:
                            existing_urls.add(url)
                    except Exception as e:
                        self.logger.warning(f"Error checking existing speech {url}: {e}")
                        # Continue processing - better to have duplicates than miss speeches
        
        # Filter out existing speeches
        filtered_speeches = [
            speech for speech in discovered_speeches 
            if speech.url not in existing_urls
        ]
        
        skipped_count = len(discovered_speeches) - len(filtered_speeches)
        self.logger.info(
            f"Filtered to {len(filtered_speeches)} new speeches "
            f"(skipped {skipped_count} existing) "
            f"[{correlation_id}]"
        )
        
        return filtered_speeches
    
    async def _process_speeches_in_batches(
        self,
        plugin: CentralBankScraperPlugin,
        institution: Institution,
        speeches_metadata: List[SpeechMetadata],
        batch_size: int,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Process speeches in batches for optimal performance."""
        if not speeches_metadata:
            return {
                'successful_count': 0,
                'failed_count': 0,
                'skipped_count': 0,
                'errors': [],
                'warnings': []
            }
        
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        errors = []
        warnings = []
        
        # Process in batches
        for i in range(0, len(speeches_metadata), batch_size):
            batch = speeches_metadata[i:i + batch_size]
            
            self.logger.info(
                f"Processing batch {i//batch_size + 1} "
                f"({len(batch)} speeches) [{correlation_id}]"
            )
            
            try:
                batch_result = await self._process_speech_batch(
                    plugin, institution, batch, correlation_id
                )
                
                successful_count += batch_result['successful_count']
                failed_count += batch_result['failed_count']
                skipped_count += batch_result['skipped_count']
                errors.extend(batch_result['errors'])
                warnings.extend(batch_result['warnings'])
                
                # Commit batch to database
                await self.uow.commit()
                
            except Exception as e:
                self.logger.error(f"Batch processing failed [{correlation_id}]: {e}")
                await self.uow.rollback()
                
                # Count all speeches in failed batch as failed
                failed_count += len(batch)
                errors.append(f"Batch processing failed: {e}")
        
        return {
            'successful_count': successful_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
            'errors': errors,
            'warnings': warnings
        }
    
    async def _process_speech_batch(
        self,
        plugin: CentralBankScraperPlugin,
        institution: Institution,
        batch: List[SpeechMetadata],
        correlation_id: str
    ) -> Dict[str, Any]:
        """Process a single batch of speeches."""
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        errors = []
        warnings = []
        
        for speech_metadata in batch:
            try:
                # Create speech entity
                speech = CentralBankSpeech()
                speech.update_metadata(speech_metadata)
                speech.institution = institution
                
                # Extract content with protection
                content = await self._extract_speech_content_with_protection(
                    plugin, speech_metadata, correlation_id
                )
                speech.set_content(content)
                
                # Validate speech
                validation_result = await self._validate_speech_with_protection(
                    plugin, speech_metadata, content, correlation_id
                )
                speech.set_validation_result(validation_result)
                
                if not validation_result.is_valid:
                    self.logger.warning(
                        f"Speech failed validation: {speech_metadata.url} "
                        f"[{correlation_id}]"
                    )
                    failed_count += 1
                    errors.append(f"Validation failed for {speech_metadata.url}: {validation_result.issues}")
                    continue
                
                # Assign speaker
                await self._assign_speaker_with_protection(
                    plugin, speech, speech_metadata, correlation_id
                )
                
                # Save speech
                await self.uow.speeches.save(speech)
                successful_count += 1
                
                # Emit success event
                self._emit_event(
                    SpeechCollectionEventType.SPEECH_EXTRACTED,
                    institution.code,
                    {
                        'speech_url': speech_metadata.url,
                        'speech_title': speech_metadata.title,
                        'correlation_id': correlation_id
                    }
                )
                
            except Exception as e:
                self.logger.error(
                    f"Error processing speech {speech_metadata.url} "
                    f"[{correlation_id}]: {e}"
                )
                failed_count += 1
                errors.append(f"Processing failed for {speech_metadata.url}: {e}")
                
                # Emit failure event
                self._emit_event(
                    SpeechCollectionEventType.SPEECH_FAILED,
                    institution.code,
                    {
                        'speech_url': speech_metadata.url,
                        'error': str(e),
                        'correlation_id': correlation_id
                    }
                )
        
        return {
            'successful_count': successful_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
            'errors': errors,
            'warnings': warnings
        }
    
async def _extract_speech_content_with_protection(
       self,
       plugin: CentralBankScraperPlugin,
       speech_metadata: SpeechMetadata,
       correlation_id: str
   ) -> SpeechContent:
       """Extract content with circuit breaker and retry protection."""
       
       async def extract_operation():
           return await asyncio.get_event_loop().run_in_executor(
               None, plugin.extract_speech_content, speech_metadata
           )
       
       try:
           return await self.plugin_manager.execute_with_circuit_breaker(
               speech_metadata.institution_code,
               "extract_speech_content",
               lambda: self.retry_handler.execute_with_retry(
                   f"extract_content_{speech_metadata.url}",
                   extract_operation
               )
           )
       except RateLimitError as e:
           self.logger.warning(f"Rate limit hit during extraction [{correlation_id}]: {e}")
           
           # Emit rate limit event
           self._emit_event(
               SpeechCollectionEventType.RATE_LIMIT_HIT,
               speech_metadata.institution_code,
               {
                   'operation': 'extract_speech_content',
                   'correlation_id': correlation_id
               }
           )
           
           # Wait for rate limit and retry
           await asyncio.sleep(plugin.get_rate_limit_delay() * 2)
           return await extract_operation()
       except Exception as e:
           self.logger.error(f"Content extraction failed [{correlation_id}]: {e}")
           raise ContentExtractionError(f"Failed to extract content: {e}")
   
async def _validate_speech_with_protection(
       self,
       plugin: CentralBankScraperPlugin,
       speech_metadata: SpeechMetadata,
       content: SpeechContent,
       correlation_id: str
   ) -> ValidationResult:
       """Validate speech with circuit breaker protection."""
       
       async def validate_operation():
           return await asyncio.get_event_loop().run_in_executor(
               None, plugin.validate_speech_authenticity, speech_metadata, content
           )
       
       try:
           result = await self.plugin_manager.execute_with_circuit_breaker(
               speech_metadata.institution_code,
               "validate_speech_authenticity",
               lambda: self.retry_handler.execute_with_retry(
                   f"validate_speech_{speech_metadata.url}",
                   validate_operation
               )
           )
           
           # Emit validation event
           self._emit_event(
               SpeechCollectionEventType.SPEECH_VALIDATED,
               speech_metadata.institution_code,
               {
                   'speech_url': speech_metadata.url,
                   'validation_status': result.status.value,
                   'correlation_id': correlation_id
               }
           )
           
           return result
           
       except Exception as e:
           self.logger.error(f"Speech validation failed [{correlation_id}]: {e}")
           raise ValidationError(f"Failed to validate speech: {e}")
   
async def _assign_speaker_with_protection(
       self,
       plugin: CentralBankScraperPlugin,
       speech: CentralBankSpeech,
       speech_metadata: SpeechMetadata,
       correlation_id: str
   ) -> None:
       """Assign speaker with comprehensive error handling."""
       try:
           # Get speaker database from plugin
           speaker_db = plugin.get_speaker_database()
           plugin_speaker = speaker_db.find_speaker(speech_metadata.speaker_name)
           
           if plugin_speaker is None:
               self.logger.warning(
                   f"Speaker not found in plugin database: {speech_metadata.speaker_name} "
                   f"[{correlation_id}]"
               )
               return
           
           # Try to find existing speaker in repository
           existing_speakers = await self.uow.speakers.find_by_name(plugin_speaker.name)
           
           if existing_speakers:
               # Use existing speaker
               speech.speaker = existing_speakers[0]
               self.logger.debug(f"Assigned existing speaker: {plugin_speaker.name} [{correlation_id}]")
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
               self.logger.info(f"Created new speaker: {plugin_speaker.name} [{correlation_id}]")
               
       except Exception as e:
           self.logger.error(f"Error assigning speaker [{correlation_id}]: {e}")
           # Continue without speaker assignment rather than fail the whole speech
   
async def _get_or_create_institution(self, plugin: CentralBankScraperPlugin) -> Institution:
        """Get existing institution or create new one."""
        institution_code = plugin.get_institution_code()
        
        async with self.uow:
            # Try to get existing institution
            institution = await self.uow.institutions.get_by_code(institution_code)
            
            if institution is None:
                # Create new institution
                institution = Institution(
                    code=institution_code,
                    name=plugin.get_institution_name(),
                    country="Unknown",  # Could be extended in plugin interface
                    institution_type=InstitutionType.CENTRAL_BANK
                )
                
                await self.uow.institutions.save(institution)
                self.logger.info(f"Created new institution: {institution_code}")
            
            return institution
    
def _compile_collection_result(
        self,
        institution_code: str,
        plugin: CentralBankScraperPlugin,
        date_range: DateRange,
        discovered_speeches: List[SpeechMetadata],
        processing_results: Dict[str, Any],
        duration: float,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Compile comprehensive collection results."""
        return {
            'institution_code': institution_code,
            'institution_name': plugin.get_institution_name(),
            'correlation_id': correlation_id,
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
            'warnings': processing_results['warnings'],
            'plugin_health': self.plugin_manager.health_status.get(institution_code, {})
        }
    
def _compile_aggregate_results(
        self,
        results: List[Any],
        healthy_plugins: Dict[str, CentralBankScraperPlugin],
        date_range: DateRange,
        start_time: datetime,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Compile aggregate results from multiple institutions."""
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
        
        return {
            'collection_timestamp': end_time.isoformat(),
            'correlation_id': correlation_id,
            'date_range': {
                'start_date': date_range.start_date.isoformat(),
                'end_date': date_range.end_date.isoformat(),
                'days': date_range.days
            },
            'institutions_processed': len(institution_results),
            'institutions_healthy': len(healthy_plugins),
            'institutions_total': len(self.plugin_manager.plugins),
            'total_speeches_discovered': total_discovered,
            'total_speeches_collected': total_collected,
            'total_speeches_failed': total_failed,
            'overall_success_rate': (
                total_collected / max(total_discovered, 1) * 100
            ),
            'total_processing_duration_seconds': duration,
            'institution_results': institution_results,
            'errors': total_errors,
            'plugin_health_summary': self._get_plugin_health_summary()
        }
    
def _emit_event(
        self,
        event_type: SpeechCollectionEventType,
        institution_code: str,
        data: Dict[str, Any]
    ) -> None:
        """Emit a domain event."""
        event = SpeechCollectionEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            institution_code=institution_code,
            data=data
        )
        self.events.append(event)
        
        # Keep only last 1000 events to prevent memory issues
        if len(self.events) > 1000:
            self.events = self.events[-1000:]
    
def _get_plugin_health_summary(self) -> Dict[str, Any]:
        """Get summary of plugin health status."""
        healthy_count = sum(1 for status in self.plugin_manager.health_status.values() if status.is_healthy)
        total_count = len(self.plugin_manager.health_status)
        
        return {
            'healthy_plugins': healthy_count,
            'total_plugins': total_count,
            'health_percentage': (healthy_count / max(total_count, 1)) * 100,
            'unhealthy_plugins': [
                code for code, status in self.plugin_manager.health_status.items()
                if not status.is_healthy
            ]
        }
    
    # Public API Methods
    
async def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the speech collection."""
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
            status_counts = {}
            for status in SpeechStatus:
                count = await self.uow.speeches.count_by_status(status)
                status_counts[status.value] = count
            
            return {
                'total_speeches': total_speeches,
                'total_institutions': len(institutions),
                'registered_plugins': len(self.plugin_manager.plugins),
                'healthy_plugins': len(await self.plugin_manager.get_healthy_plugins()),
                'institution_statistics': institution_stats,
                'status_distribution': status_counts,
                'collection_metrics': self.metrics,
                'plugin_health_summary': self._get_plugin_health_summary(),
                'last_updated': datetime.now().isoformat()
            }
    
async def get_plugin_health_status(self) -> Dict[str, Any]:
        """Get detailed health status of all plugins."""
        health_results = {}
        
        for institution_code in self.plugin_manager.plugins.keys():
            is_healthy = await self.plugin_manager.check_plugin_health(institution_code)
            health_results[institution_code] = {
                'is_healthy': is_healthy,
                'status': self.plugin_manager.health_status.get(institution_code, {}),
                'circuit_breaker_state': self.plugin_manager.circuit_breakers[institution_code].state.value
            }
        
        return health_results
    
async def reprocess_failed_speeches(
        self,
        institution_code: Optional[str] = None,
        limit: Optional[int] = None,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """Reprocess speeches that previously failed with advanced error handling."""
        async with self.uow:
            # Build query conditions
            query_conditions = [SpeechStatus.FAILED]
            
            if institution_code:
                if institution_code not in self.plugin_manager.plugins:
                    raise SpeechCollectionError(f"No plugin for institution: {institution_code}")
                
                # Get failed speeches for specific institution
                institution = await self.uow.institutions.get_by_code(institution_code)
                if not institution:
                    raise SpeechCollectionError(f"Institution not found: {institution_code}")
                
                failed_speeches = await self.uow.speeches.find_by_status(
                    SpeechStatus.FAILED, limit=limit
                )
                # Filter by institution
                failed_speeches = [
                    speech for speech in failed_speeches 
                    if speech.institution and speech.institution.code == institution_code
                ]
            else:
                # Get all failed speeches
                failed_speeches = await self.uow.speeches.find_by_status(
                    SpeechStatus.FAILED, limit=limit
                )
            
            if not failed_speeches:
                return {
                    'speeches_found': 0,
                    'speeches_reprocessed': 0,
                    'speeches_fixed': 0,
                    'speeches_still_failed': 0,
                    'errors': []
                }
            
            correlation_id = str(uuid4())
            self.logger.info(f"Starting reprocessing of {len(failed_speeches)} failed speeches [{correlation_id}]")
            
            # Process in batches
            reprocessed_count = 0
            fixed_count = 0
            still_failed_count = 0
            errors = []
            
            for i in range(0, len(failed_speeches), batch_size):
                batch = failed_speeches[i:i + batch_size]
                
                for speech in batch:
                    try:
                        if not speech.institution or speech.institution.code not in self.plugin_manager.plugins:
                            self.logger.warning(f"No plugin for speech institution: {speech.institution}")
                            still_failed_count += 1
                            continue
                        
                        plugin = self.plugin_manager.plugins[speech.institution.code]
                        
                        # Check plugin health first
                        if not await self.plugin_manager.check_plugin_health(speech.institution.code):
                            self.logger.warning(f"Plugin unhealthy, skipping: {speech.institution.code}")
                            still_failed_count += 1
                            continue
                        
                        # Retry content extraction if needed
                        if speech.content is None and speech.metadata:
                            try:
                                content = await self._extract_speech_content_with_protection(
                                    plugin, speech.metadata, correlation_id
                                )
                                speech.set_content(content)
                            except Exception as e:
                                self.logger.error(f"Reprocessing content extraction failed: {e}")
                                errors.append(f"Content extraction failed for {speech.id}: {e}")
                                still_failed_count += 1
                                continue
                        
                        # Retry validation if we have content
                        if speech.content and speech.metadata:
                            try:
                                validation_result = await self._validate_speech_with_protection(
                                    plugin, speech.metadata, speech.content, correlation_id
                                )
                                speech.set_validation_result(validation_result)
                                
                                if validation_result.is_valid:
                                    speech.status = SpeechStatus.VALIDATED
                                    fixed_count += 1
                                else:
                                    still_failed_count += 1
                                    
                            except Exception as e:
                                self.logger.error(f"Reprocessing validation failed: {e}")
                                errors.append(f"Validation failed for {speech.id}: {e}")
                                still_failed_count += 1
                                continue
                        
                        await self.uow.speeches.save(speech)
                        reprocessed_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error reprocessing speech {speech.id}: {e}")
                        errors.append(f"Reprocessing failed for {speech.id}: {e}")
                        still_failed_count += 1
                
                # Commit batch
                await self.uow.commit()
            
            self.logger.info(
                f"Completed reprocessing: {reprocessed_count} processed, "
                f"{fixed_count} fixed, {still_failed_count} still failed [{correlation_id}]"
            )
            
            return {
                'speeches_found': len(failed_speeches),
                'speeches_reprocessed': reprocessed_count,
                'speeches_fixed': fixed_count,
                'speeches_still_failed': still_failed_count,
                'errors': errors,
                'correlation_id': correlation_id
            }
    
async def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent collection events for monitoring."""
        recent_events = self.events[-limit:] if len(self.events) > limit else self.events
        
        return [
            {
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'institution_code': event.institution_code,
                'correlation_id': event.correlation_id,
                'data': event.data
            }
            for event in recent_events
        ]
    
async def reset_plugin_circuit_breaker(self, institution_code: str) -> None:
        """Manually reset circuit breaker for a plugin."""
        if institution_code not in self.plugin_manager.circuit_breakers:
            raise SpeechCollectionError(f"No circuit breaker for institution: {institution_code}")
        
        circuit_breaker = self.plugin_manager.circuit_breakers[institution_code]
        circuit_breaker.state = CircuitBreakerState.CLOSED
        circuit_breaker.failure_count = 0
        circuit_breaker.success_count = 0
        circuit_breaker.last_failure_time = None
        
        self.logger.info(f"Circuit breaker reset for {institution_code}")
    
async def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        return {
            'collection_metrics': self.metrics,
            'plugin_health': await self.get_plugin_health_status(),
            'recent_events_count': len(self.events),
            'circuit_breaker_states': {
                code: cb.state.value 
                for code, cb in self.plugin_manager.circuit_breakers.items()
            },
            'service_uptime': datetime.now().isoformat(),  # Could track actual uptime
            'memory_usage': {
                'events_count': len(self.events),
                'plugins_count': len(self.plugin_manager.plugins)
            }
        }


# Factory function for creating service with dependencies
async def create_speech_collection_service(
   unit_of_work: UnitOfWork,
   plugins: Optional[List[CentralBankScraperPlugin]] = None
) -> SpeechCollectionService:
   """
   Factory function to create a fully configured speech collection service.
   
   Args:
       unit_of_work: Unit of work for database operations
       plugins: Optional list of plugins to register
       
   Returns:
       Configured speech collection service
   """
   service = SpeechCollectionService(unit_of_work)
   
   # Register plugins if provided
   if plugins:
       for plugin in plugins:
           try:
               service.register_plugin(plugin)
           except Exception as e:
               logger.error(f"Failed to register plugin {plugin.get_institution_code()}: {e}")
   
   return service


# Health check function
async def check_speech_collection_service_health(
   service: SpeechCollectionService
) -> Dict[str, Any]:
   """
   Check the health of the speech collection service.
   
   Args:
       service: Speech collection service to check
       
   Returns:
       Health status dictionary
   """
   try:
       # Check plugin health
       plugin_health = await service.get_plugin_health_status()
       healthy_plugins = sum(1 for status in plugin_health.values() if status['is_healthy'])
       total_plugins = len(plugin_health)
       
       # Check service metrics
       metrics = await service.get_service_metrics()
       
       # Determine overall health
       is_healthy = (
           healthy_plugins > 0 and  # At least one plugin healthy
           len(service.events) < 10000  # Event queue not overflowing
       )
       
       return {
           'status': 'healthy' if is_healthy else 'unhealthy',
           'healthy_plugins': healthy_plugins,
           'total_plugins': total_plugins,
           'plugin_health_percentage': (healthy_plugins / max(total_plugins, 1)) * 100,
           'service_metrics': metrics,
           'timestamp': datetime.now().isoformat()
       }
       
   except Exception as e:
       return {
           'status': 'unhealthy',
           'error': str(e),
           'timestamp': datetime.now().isoformat()
       }


# Export public interface
__all__ = [
   'SpeechCollectionService',
   'SpeechCollectionEvent',
   'SpeechCollectionEventType',
   'SpeechCollectionError',
   'PluginRegistrationError',
   'CollectionTimeoutError',
   'PluginHealthError',
   'BatchProcessingError',
   'create_speech_collection_service',
   'check_speech_collection_service_health'
]