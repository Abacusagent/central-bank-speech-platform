# application/services/analysis_service.py

"""
Speech Analysis Service - Production-Ready Application Layer

This service orchestrates NLP analysis operations with enterprise-grade reliability,
performance optimization, and comprehensive monitoring. Implements advanced patterns
for batch processing, error recovery, progress tracking, and result validation.

Key Features:
- Advanced batch processing with streaming support
- Circuit breaker protection for NLP pipeline
- Comprehensive retry logic with exponential backoff
- Real-time progress tracking and cancellation support
- Analysis result validation and quality assurance
- Event-driven architecture with domain events
- Performance monitoring and optimization
- Multiple analysis types and configuration support

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Callable
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from domain.entities import (
    CentralBankSpeech, SentimentAnalysis, SpeechStatus, PolicyStance
)
from domain.value_objects import ConfidenceLevel, DateRange
from domain.repositories import (
    UnitOfWork, RepositoryError, EntityNotFoundError, DuplicateEntityError
)
from infrastructure.nlp.pipeline import NLPProcessingPipeline, NLPAnalysis, ProcessorResult
from interfaces.plugin_interfaces import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


# Analysis Domain Events
class AnalysisEventType(Enum):
    """Types of analysis events."""
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    BATCH_STARTED = "batch_started"
    BATCH_PROGRESS = "batch_progress"
    BATCH_COMPLETED = "batch_completed"
    BATCH_CANCELLED = "batch_cancelled"
    PIPELINE_ERROR = "pipeline_error"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class AnalysisEvent:
    """Domain event for analysis operations."""
    event_type: AnalysisEventType
    timestamp: datetime
    correlation_id: str
    data: Dict[str, Any] = field(default_factory=dict)


# Exception Hierarchy
class AnalysisError(Exception):
    """Base exception for analysis operations."""
    pass


class PipelineError(AnalysisError):
    """Raised when NLP pipeline fails."""
    pass


class ValidationError(AnalysisError):
    """Raised when analysis validation fails."""
    pass


class AnalysisTimeoutError(AnalysisError):
    """Raised when analysis operations timeout."""
    pass


class BatchProcessingError(AnalysisError):
    """Raised when batch processing fails."""
    pass


# Analysis Configuration
@dataclass
class AnalysisConfig:
    """Configuration for analysis operations."""
    timeout_seconds: int = 300
    max_concurrent: int = 5
    batch_size: int = 10
    retry_attempts: int = 3
    enable_validation: bool = True
    save_intermediate_results: bool = True
    analysis_types: List[str] = field(default_factory=lambda: ['sentiment', 'topics', 'uncertainty'])


# Progress Tracking
@dataclass
class AnalysisProgress:
    """Tracks progress of analysis operations."""
    total_speeches: int
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    current_operation: str = ""
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_speeches == 0:
            return 100.0
        return (self.completed + self.failed + self.skipped) / self.total_speeches * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total_processed = self.completed + self.failed
        if total_processed == 0:
            return 100.0
        return self.completed / total_processed * 100


# Circuit Breaker for NLP Pipeline
class PipelineCircuitBreaker:
    """Circuit breaker specifically for NLP pipeline operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        self.logger = logging.getLogger(f"{__name__}.PipelineCircuitBreaker")
    
    async def call(self, operation: Callable, *args, **kwargs):
        """Execute operation with circuit breaker protection."""
        if self.is_open:
            if self._should_attempt_reset():
                self.is_open = False
                self.logger.info("Pipeline circuit breaker reset - attempting operation")
            else:
                raise PipelineError("NLP pipeline circuit breaker is open")
        
        try:
            result = await operation(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should be reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            self.logger.warning("Pipeline circuit breaker opened due to failures")


# Analysis Result Validator
class AnalysisResultValidator:
    """Validates analysis results for quality assurance."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AnalysisResultValidator")
    
    def validate_sentiment_analysis(self, analysis: SentimentAnalysis) -> ValidationResult:
        """Validate sentiment analysis results."""
        issues = []
        confidence = 1.0
        
        # Check hawkish-dovish score range
        if analysis.hawkish_dovish_score < -1.0 or analysis.hawkish_dovish_score > 1.0:
            issues.append("Hawkish-dovish score out of valid range [-1.0, 1.0]")
            confidence *= 0.5
        
        # Check confidence score
        if analysis.confidence_score < 0.0 or analysis.confidence_score > 1.0:
            issues.append("Confidence score out of valid range [0.0, 1.0]")
            confidence *= 0.5
        
        # Check uncertainty score
        if analysis.uncertainty_score < 0.0 or analysis.uncertainty_score > 1.0:
            issues.append("Uncertainty score out of valid range [0.0, 1.0]")
            confidence *= 0.5
        
        # Check policy stance consistency
        if abs(analysis.hawkish_dovish_score) > 0.3:
            expected_stance = PolicyStance.HAWKISH if analysis.hawkish_dovish_score > 0 else PolicyStance.DOVISH
            if analysis.policy_stance == PolicyStance.NEUTRAL:
                issues.append("Policy stance inconsistent with hawkish-dovish score")
                confidence *= 0.8
        
        # Check minimum confidence threshold
        if analysis.confidence_score < 0.3:
            issues.append("Analysis confidence below minimum threshold")
            confidence *= 0.7
        
        status = ValidationStatus.VALID if not issues else (
            ValidationStatus.QUESTIONABLE if confidence > 0.5 else ValidationStatus.INVALID
        )
        
        return ValidationResult(
            status=status,
            confidence=confidence,
            issues=issues,
            metadata={'validator': 'AnalysisResultValidator'}
        )
    
    def validate_nlp_analysis(self, analysis: NLPAnalysis) -> ValidationResult:
        """Validate complete NLP analysis results."""
        issues = []
        confidence = 1.0
        
        # Check overall confidence
        if analysis.overall_confidence < 0.2:
            issues.append("Overall analysis confidence very low")
            confidence *= 0.6
        
        # Check processing time (flag unusually long processing)
        if analysis.processing_time > 120:  # 2 minutes
            issues.append("Processing time unusually long")
            confidence *= 0.9
        
        # Check processor results
        successful_processors = sum(1 for result in analysis.processor_results if result.success)
        total_processors = len(analysis.processor_results)
        
        if successful_processors < total_processors * 0.5:
            issues.append("More than half of NLP processors failed")
            confidence *= 0.4
        
        # Check for required processors
        required_processors = ['HawkDoveAnalyzer']
        for required in required_processors:
            if not any(r.processor_name == required and r.success for r in analysis.processor_results):
                issues.append(f"Required processor {required} failed or missing")
                confidence *= 0.5
        
        status = ValidationStatus.VALID if not issues else (
            ValidationStatus.QUESTIONABLE if confidence > 0.5 else ValidationStatus.INVALID
        )
        
        return ValidationResult(
            status=status,
            confidence=confidence,
            issues=issues,
            metadata={'validator': 'AnalysisResultValidator', 'processor_count': total_processors}
        )


# Main Service Implementation
class SpeechAnalysisService:
    """
    Production-ready speech analysis service with comprehensive error handling,
    progress tracking, and performance optimization.
    """
    
    def __init__(
        self, 
        unit_of_work: UnitOfWork, 
        nlp_pipeline: NLPProcessingPipeline,
        config: Optional[AnalysisConfig] = None
    ):
        """Initialize the speech analysis service."""
        self.uow = unit_of_work
        self.nlp_pipeline = nlp_pipeline
        self.config = config or AnalysisConfig()
        self.circuit_breaker = PipelineCircuitBreaker()
        self.validator = AnalysisResultValidator()
        self.events: List[AnalysisEvent] = []
        self.active_operations: Dict[str, AnalysisProgress] = {}
        self._setup_logging()
        self._setup_metrics()
    
    def _setup_logging(self) -> None:
        """Configure structured logging for the service."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
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
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'pipeline_failures': 0,
            'validation_failures': 0,
            'circuit_breaker_trips': 0
        }
    
    async def analyze_speech(
        self,
        speech: CentralBankSpeech,
        persist: bool = True,
        validate_results: bool = True,
        timeout_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single speech with comprehensive error handling and validation.
        
        Args:
            speech: Speech to analyze
            persist: Whether to save results to database
            validate_results: Whether to validate analysis results
            timeout_seconds: Override default timeout
            
        Returns:
            Dictionary with analysis results and metadata
        """
        correlation_id = str(uuid4())
        start_time = datetime.now()
        timeout = timeout_seconds or self.config.timeout_seconds
        
        self.logger.info(f"Starting analysis for speech {speech.id} [{correlation_id}]")
        
        # Emit analysis started event
        self._emit_event(
            AnalysisEventType.ANALYSIS_STARTED,
            correlation_id,
            {
                'speech_id': str(speech.id),
                'speech_title': getattr(speech.metadata, 'title', '') if speech.metadata else '',
                'timeout_seconds': timeout
            }
        )
        
        try:
            # Ensure pipeline is initialized
            await self._ensure_pipeline_initialized()
            
            # Check if speech has content
            if not speech.content or not speech.content.cleaned_text:
                raise AnalysisError("Speech has no content to analyze")
            
            # Run analysis with circuit breaker and timeout protection
            nlp_analysis = await asyncio.wait_for(
                self.circuit_breaker.call(self._analyze_speech_protected, speech, correlation_id),
                timeout=timeout
            )
            
            # Validate analysis results if requested
            if validate_results:
                validation_result = self.validator.validate_nlp_analysis(nlp_analysis)
                if validation_result.status == ValidationStatus.INVALID:
                    raise ValidationError(f"Analysis validation failed: {validation_result.issues}")
            
            # Convert to domain entity
            sentiment_analysis = self._convert_to_sentiment_analysis(nlp_analysis)
            
            # Validate sentiment analysis
            if validate_results:
                sentiment_validation = self.validator.validate_sentiment_analysis(sentiment_analysis)
                if sentiment_validation.status == ValidationStatus.INVALID:
                    raise ValidationError(f"Sentiment analysis validation failed: {sentiment_validation.issues}")
            
            # Update speech entity
            speech.set_sentiment_analysis(sentiment_analysis)
            speech.status = SpeechStatus.ANALYZED
            
            # Persist if requested
            if persist:
                async with self.uow:
                    await self.uow.speeches.save(speech)
                    await self.uow.commit()
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update metrics
            self.metrics['total_analyses'] += 1
            self.metrics['successful_analyses'] += 1
            self.metrics['total_processing_time'] += processing_time
            self.metrics['average_processing_time'] = (
                self.metrics['total_processing_time'] / self.metrics['total_analyses']
            )
            
            result = {
                'speech_id': str(speech.id),
                'correlation_id': correlation_id,
                'nlp_analysis': nlp_analysis,
                'sentiment_analysis': sentiment_analysis,
                'processing_time_seconds': processing_time,
                'success': True,
                'errors': [],
                'warnings': [],
                'validation_results': {
                    'nlp_validation': validation_result if validate_results else None,
                    'sentiment_validation': sentiment_validation if validate_results else None
                }
            }
            
            # Emit success event
            self._emit_event(
                AnalysisEventType.ANALYSIS_COMPLETED,
                correlation_id,
                {
                    'speech_id': str(speech.id),
                    'processing_time': processing_time,
                    'confidence': nlp_analysis.overall_confidence
                }
            )
            
            self.logger.info(
                f"Analysis completed for speech {speech.id} "
                f"(confidence: {nlp_analysis.overall_confidence:.3f}, "
                f"time: {processing_time:.2f}s) [{correlation_id}]"
            )
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Analysis timeout after {timeout} seconds"
            self.logger.error(f"Analysis timeout for speech {speech.id} [{correlation_id}]")
            return self._create_error_result(speech.id, correlation_id, AnalysisTimeoutError(error_msg))
            
        except (PipelineError, ValidationError, AnalysisError) as e:
            self.logger.error(f"Analysis failed for speech {speech.id} [{correlation_id}]: {e}")
            return self._create_error_result(speech.id, correlation_id, e)
            
        except Exception as e:
            self.logger.error(f"Unexpected error analyzing speech {speech.id} [{correlation_id}]: {e}")
            return self._create_error_result(speech.id, correlation_id, AnalysisError(f"Unexpected error: {e}"))
    
    async def analyze_speeches_batch(
        self,
        speeches: List[CentralBankSpeech],
        persist: bool = True,
        validate_results: bool = True,
        max_concurrent: Optional[int] = None,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[AnalysisProgress], None]] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple speeches with advanced batch processing and progress tracking.
        
        Args:
            speeches: List of speeches to analyze
            persist: Whether to save results to database
            validate_results: Whether to validate analysis results
            max_concurrent: Override default concurrency limit
            batch_size: Override default batch size
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with batch results and comprehensive metrics
        """
        if not speeches:
            return {
                'correlation_id': str(uuid4()),
                'results': [],
                'aggregate': self._create_empty_aggregate(),
                'errors': ['No speeches provided for analysis']
            }
        
        correlation_id = str(uuid4())
        start_time = datetime.now()
        concurrent_limit = max_concurrent or self.config.max_concurrent
        batch_limit = batch_size or self.config.batch_size
        
        self.logger.info(
            f"Starting batch analysis for {len(speeches)} speeches "
            f"(concurrent: {concurrent_limit}, batch_size: {batch_limit}) [{correlation_id}]"
        )
        
        # Initialize progress tracking
        progress = AnalysisProgress(total_speeches=len(speeches))
        self.active_operations[correlation_id] = progress
        
        # Emit batch started event
        self._emit_event(
            AnalysisEventType.BATCH_STARTED,
            correlation_id,
            {
                'total_speeches': len(speeches),
                'max_concurrent': concurrent_limit,
                'batch_size': batch_limit
            }
        )
        
        try:
            # Ensure pipeline is initialized
            await self._ensure_pipeline_initialized()
            
            # Process speeches in batches
            all_results = []
            semaphore = asyncio.Semaphore(concurrent_limit)
            
            for i in range(0, len(speeches), batch_limit):
                batch = speeches[i:i + batch_limit]
                progress.current_operation = f"Processing batch {i//batch_limit + 1}"
                
                # Process batch concurrently
                batch_results = await self._process_speech_batch(
                    batch, semaphore, persist, validate_results, correlation_id, progress
                )
                
                all_results.extend(batch_results)
                
                # Update progress and call callback
                if progress_callback:
                    progress_callback(progress)
                
                # Emit progress event
                self._emit_event(
                    AnalysisEventType.BATCH_PROGRESS,
                    correlation_id,
                    {
                        'completed': progress.completed,
                        'failed': progress.failed,
                        'completion_percentage': progress.completion_percentage
                    }
                )
                
                # Small delay between batches to prevent overwhelming the system
                if i + batch_limit < len(speeches):
                    await asyncio.sleep(0.1)
            
            # Compile final results
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            aggregate = self._compile_batch_aggregate(all_results, total_duration, len(speeches))
            
            # Emit batch completed event
            self._emit_event(
                AnalysisEventType.BATCH_COMPLETED,
                correlation_id,
                {
                    'total_speeches': len(speeches),
                    'successful': aggregate['successfully_analyzed'],
                    'failed': aggregate['failed'],
                    'duration': total_duration
                }
            )
            
            self.logger.info(
                f"Batch analysis completed [{correlation_id}]: "
                f"{aggregate['successfully_analyzed']}/{len(speeches)} successful "
                f"({aggregate['success_rate']:.1f}% success rate, {total_duration:.2f}s)"
            )
            
            return {
                'correlation_id': correlation_id,
                'results': all_results,
                'aggregate': aggregate,
                'progress': progress,
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed [{correlation_id}]: {e}")
            
            # Emit batch failed event
            self._emit_event(
                AnalysisEventType.BATCH_CANCELLED,
                correlation_id,
                {'error': str(e)}
            )
            
            return {
                'correlation_id': correlation_id,
                'results': [],
                'aggregate': self._create_empty_aggregate(),
                'progress': progress,
                'errors': [str(e)]
            }
            
        finally:
            # Clean up progress tracking
            if correlation_id in self.active_operations:
                del self.active_operations[correlation_id]
    
    async def analyze_unprocessed_speeches(
        self,
        limit: Optional[int] = None,
        institution_codes: Optional[List[str]] = None,
        date_range: Optional[DateRange] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze speeches that haven't been processed yet.
        
        Args:
            limit: Maximum number of speeches to process
            institution_codes: Filter by specific institutions
            date_range: Filter by date range
            **kwargs: Additional arguments passed to analyze_speeches_batch
            
        Returns:
            Dictionary with analysis results
        """
        correlation_id = str(uuid4())
        
        try:
            async with self.uow:
                # Get unprocessed speeches
                unprocessed_speeches = await self.uow.speeches.get_unprocessed_speeches(limit=limit)
                
                # Apply additional filters if specified
                if institution_codes:
                    unprocessed_speeches = [
                        speech for speech in unprocessed_speeches
                        if speech.institution and speech.institution.code in institution_codes
                    ]
                
                if date_range and hasattr(unprocessed_speeches[0], 'metadata'):
                    unprocessed_speeches = [
                        speech for speech in unprocessed_speeches
                        if speech.metadata and 
                        date_range.start_date <= speech.metadata.date <= date_range.end_date
                    ]
                
                if not unprocessed_speeches:
                    self.logger.info("No unprocessed speeches found")
                    return {
                        'correlation_id': correlation_id,
                        'results': [],
                        'aggregate': self._create_empty_aggregate(),
                        'message': 'No unprocessed speeches found'
                    }
                
                self.logger.info(f"Found {len(unprocessed_speeches)} unprocessed speeches")
                
                # Analyze speeches
                return await self.analyze_speeches_batch(
                    unprocessed_speeches,
                    persist=True,
                    **kwargs
                )
                
        except Exception as e:
            self.logger.error(f"Error analyzing unprocessed speeches [{correlation_id}]: {e}")
            return {
                'correlation_id': correlation_id,
                'results': [],
                'aggregate': self._create_empty_aggregate(),
                'errors': [str(e)]
            }
    
    async def reanalyze_speeches(
        self,
        speech_ids: List[UUID],
        force: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Re-analyze specific speeches (useful for testing new models).
        
        Args:
            speech_ids: List of speech IDs to re-analyze
            force: Whether to re-analyze even if already analyzed
            **kwargs: Additional arguments passed to analyze_speeches_batch
            
        Returns:
            Dictionary with analysis results
        """
        correlation_id = str(uuid4())
        
        try:
            async with self.uow:
                speeches = []
                
                for speech_id in speech_ids:
                    speech = await self.uow.speeches.get_by_id(speech_id)
                    if speech:
                        if force or not speech.sentiment_analysis:
                            speeches.append(speech)
                        else:
                            self.logger.debug(f"Skipping already analyzed speech {speech_id}")
                    else:
                        self.logger.warning(f"Speech not found: {speech_id}")
                
                if not speeches:
                    return {
                        'correlation_id': correlation_id,
                        'results': [],
                        'aggregate': self._create_empty_aggregate(),
                        'message': 'No speeches to re-analyze'
                    }
                
                self.logger.info(f"Re-analyzing {len(speeches)} speeches")
                
                return await self.analyze_speeches_batch(speeches, persist=True, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Error re-analyzing speeches [{correlation_id}]: {e}")
            return {
                'correlation_id': correlation_id,
                'results': [],
                'aggregate': self._create_empty_aggregate(),
                'errors': [str(e)]
            }
    
    async def stream_analyze_speeches(
        self,
        speech_iterator: AsyncIterator[CentralBankSpeech],
        persist: bool = True,
        validate_results: bool = True,
        max_concurrent: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream analysis of speeches for memory-efficient processing of large datasets.
        
        Args:
            speech_iterator: Async iterator of speeches
            persist: Whether to save results to database
            validate_results: Whether to validate analysis results
            max_concurrent: Override default concurrency limit
            
        Yields:
            Analysis results for each speech
        """
        correlation_id = str(uuid4())
        concurrent_limit = max_concurrent or self.config.max_concurrent
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        self.logger.info(f"Starting streaming analysis [{correlation_id}]")
        
        # Ensure pipeline is initialized
        await self._ensure_pipeline_initialized()
        
        async def analyze_single(speech: CentralBankSpeech) -> Dict[str, Any]:
            """Analyze a single speech with semaphore protection."""
            async with semaphore:
                return await self.analyze_speech(
                    speech, persist=persist, validate_results=validate_results
                )
        
        # Process speeches as they come in
        speech_count = 0
        async for speech in speech_iterator:
            speech_count += 1
            
            try:
                result = await analyze_single(speech)
                yield result
                
            except Exception as e:
                self.logger.error(f"Error in streaming analysis [{correlation_id}]: {e}")
                yield self._create_error_result(speech.id, correlation_id, e)
        
        self.logger.info(f"Streaming analysis completed: {speech_count} speeches processed [{correlation_id}]")
    
    # Protected and helper methods
    
    async def _analyze_speech_protected(self, speech: CentralBankSpeech, correlation_id: str) -> NLPAnalysis:
        """Analyze speech with retry protection."""
        for attempt in range(self.config.retry_attempts):
            try:
                return await self.nlp_pipeline.process_speech(speech)
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise PipelineError(f"Pipeline failed after {self.config.retry_attempts} attempts: {e}")
                
                self.logger.warning(
                    f"Pipeline attempt {attempt + 1} failed [{correlation_id}]: {e}. Retrying..."
                )
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    async def _process_speech_batch(
        self,
        batch: List[CentralBankSpeech],
        semaphore: asyncio.Semaphore,
        persist: bool,
        validate_results: bool,
        correlation_id: str,
        progress: AnalysisProgress
    ) -> List[Dict[str, Any]]:
        """Process a batch of speeches concurrently."""
        
        async def analyze_with_semaphore(speech: CentralBankSpeech) -> Dict[str, Any]:
            """Analyze speech with semaphore protection."""
            async with semaphore:
                result = await self.analyze_speech(
                    speech, persist=persist, validate_results=validate_results
                )
                
                # Update progress
                if result['success']:
                    progress.completed += 1
                else:
                    progress.failed += 1
                
                return result
        
        # Process batch concurrently
        tasks = [analyze_with_semaphore(speech) for speech in batch]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def _ensure_pipeline_initialized(self) -> None:
        """Ensure NLP pipeline is initialized."""
        if not getattr(self.nlp_pipeline, 'is_initialized', False):
            self.logger.info("Initializing NLP pipeline...")
            await self.nlp_pipeline.initialize()
            self.logger.info("NLP pipeline initialized successfully")
    
def _convert_to_sentiment_analysis(self, nlp_analysis: NLPAnalysis) -> SentimentAnalysis:
       """Convert NLP analysis to domain sentiment analysis entity."""
       # Extract sentiment information from NLP analysis
       hawkish_dovish_score = 0.0
       policy_stance = PolicyStance.NEUTRAL
       uncertainty_score = 0.0
       topic_classifications = []
       
       # Extract data from processor results
       for result in nlp_analysis.processor_results:
           if result.processor_name == "HawkDoveAnalyzer" and result.success:
               hawkish_dovish_score = result.results.get('hawkish_dovish_score', 0.0)
               stance_str = result.results.get('policy_stance', 'neutral')
               try:
                   policy_stance = PolicyStance(stance_str.lower())
               except ValueError:
                   policy_stance = PolicyStance.NEUTRAL
                   
           elif result.processor_name == "UncertaintyQuantifier" and result.success:
               uncertainty_score = result.results.get('uncertainty_score', 0.0)
               
           elif result.processor_name == "TopicModelingProcessor" and result.success:
               topics = result.results.get('topics', [])
               keyword_topics = result.results.get('keyword_topics', [])
               topic_classifications.extend(topics + keyword_topics)
       
       # Create sentiment analysis entity
       return SentimentAnalysis(
           speech_id=nlp_analysis.speech_id,
           hawkish_dovish_score=hawkish_dovish_score,
           policy_stance=policy_stance,
           uncertainty_score=uncertainty_score,
           confidence_score=nlp_analysis.overall_confidence,
           analysis_timestamp=nlp_analysis.analysis_timestamp,
           analyzer_version=nlp_analysis.pipeline_version,
           topic_classifications=topic_classifications[:10]  # Limit to top 10 topics
       )
   
def _create_error_result(self, speech_id: UUID, correlation_id: str, error: Exception) -> Dict[str, Any]:
       """Create standardized error result."""
       self.metrics['total_analyses'] += 1
       self.metrics['failed_analyses'] += 1
       
       # Classify error type for metrics
       if isinstance(error, PipelineError):
           self.metrics['pipeline_failures'] += 1
       elif isinstance(error, ValidationError):
           self.metrics['validation_failures'] += 1
       elif isinstance(error, AnalysisTimeoutError):
           self.metrics['circuit_breaker_trips'] += 1
       
       # Emit failure event
       self._emit_event(
           AnalysisEventType.ANALYSIS_FAILED,
           correlation_id,
           {
               'speech_id': str(speech_id),
               'error_type': type(error).__name__,
               'error_message': str(error)
           }
       )
       
       return {
           'speech_id': str(speech_id),
           'correlation_id': correlation_id,
           'nlp_analysis': None,
           'sentiment_analysis': None,
           'processing_time_seconds': 0.0,
           'success': False,
           'errors': [str(error)],
           'warnings': [],
           'error_type': type(error).__name__
       }
   
def _create_empty_aggregate(self) -> Dict[str, Any]:
       """Create empty aggregate results."""
       return {
           'total_speeches': 0,
           'successfully_analyzed': 0,
           'failed': 0,
           'skipped': 0,
           'success_rate': 0.0,
           'total_duration_seconds': 0.0,
           'average_processing_time': 0.0,
           'confidence_distribution': {},
           'error_distribution': {}
       }
   
def _compile_batch_aggregate(self, results: List[Dict[str, Any]], total_duration: float, total_count: int) -> Dict[str, Any]:
       """Compile comprehensive aggregate statistics from batch results."""
       successful_results = [r for r in results if r['success']]
       failed_results = [r for r in results if not r['success']]
       
       # Basic counts
       successful_count = len(successful_results)
       failed_count = len(failed_results)
       
       # Processing time statistics
       processing_times = [r['processing_time_seconds'] for r in successful_results]
       avg_processing_time = sum(processing_times) / max(len(processing_times), 1)
       
       # Confidence distribution
       confidences = []
       for result in successful_results:
           if result['nlp_analysis']:
               confidences.append(result['nlp_analysis'].overall_confidence)
       
       confidence_distribution = self._calculate_confidence_distribution(confidences)
       
       # Error distribution
       error_types = [r.get('error_type', 'Unknown') for r in failed_results]
       error_distribution = {}
       for error_type in error_types:
           error_distribution[error_type] = error_distribution.get(error_type, 0) + 1
       
       # Policy stance distribution
       stance_distribution = {}
       for result in successful_results:
           if result['sentiment_analysis']:
               stance = result['sentiment_analysis'].policy_stance.value
               stance_distribution[stance] = stance_distribution.get(stance, 0) + 1
       
       return {
           'total_speeches': total_count,
           'successfully_analyzed': successful_count,
           'failed': failed_count,
           'skipped': 0,  # Not applicable for batch processing
           'success_rate': (successful_count / max(total_count, 1)) * 100,
           'total_duration_seconds': total_duration,
           'average_processing_time': avg_processing_time,
           'confidence_distribution': confidence_distribution,
           'error_distribution': error_distribution,
           'stance_distribution': stance_distribution,
           'throughput_speeches_per_second': total_count / max(total_duration, 0.001)
       }
   
def _calculate_confidence_distribution(self, confidences: List[float]) -> Dict[str, int]:
       """Calculate confidence level distribution."""
       if not confidences:
           return {}
       
       distribution = {
           'very_low': 0,    # 0.0 - 0.2
           'low': 0,         # 0.2 - 0.4
           'medium': 0,      # 0.4 - 0.6
           'high': 0,        # 0.6 - 0.8
           'very_high': 0    # 0.8 - 1.0
       }
       
       for confidence in confidences:
           if confidence < 0.2:
               distribution['very_low'] += 1
           elif confidence < 0.4:
               distribution['low'] += 1
           elif confidence < 0.6:
               distribution['medium'] += 1
           elif confidence < 0.8:
               distribution['high'] += 1
           else:
               distribution['very_high'] += 1
       
       return distribution
   
def _emit_event(self, event_type: AnalysisEventType, correlation_id: str, data: Dict[str, Any]) -> None:
       """Emit an analysis event."""
       event = AnalysisEvent(
           event_type=event_type,
           timestamp=datetime.now(),
           correlation_id=correlation_id,
           data=data
       )
       self.events.append(event)
       
       # Keep only last 1000 events to prevent memory issues
       if len(self.events) > 1000:
           self.events = self.events[-1000:]
   
   # Public API Methods
   
async def get_analysis_statistics(self) -> Dict[str, Any]:
       """Get comprehensive analysis statistics."""
       async with self.uow:
           # Get analysis summary from repository
           analysis_summary = await self.uow.analyses.get_analysis_summary()
           
           # Combine with service metrics
           return {
               'service_metrics': self.metrics,
               'repository_summary': analysis_summary,
               'pipeline_status': {
                   'is_initialized': getattr(self.nlp_pipeline, 'is_initialized', False),
                   'circuit_breaker_open': self.circuit_breaker.is_open,
                   'circuit_breaker_failures': self.circuit_breaker.failure_count
               },
               'active_operations': len(self.active_operations),
               'recent_events_count': len(self.events),
               'last_updated': datetime.now().isoformat()
           }
   
async def get_operation_progress(self, correlation_id: str) -> Optional[AnalysisProgress]:
       """Get progress of a specific operation."""
       return self.active_operations.get(correlation_id)
   
async def cancel_operation(self, correlation_id: str) -> bool:
       """Cancel an active operation (best effort)."""
       if correlation_id in self.active_operations:
           # Mark operation as cancelled
           progress = self.active_operations[correlation_id]
           progress.current_operation = "Cancelling..."
           
           # Emit cancellation event
           self._emit_event(
               AnalysisEventType.BATCH_CANCELLED,
               correlation_id,
               {'reason': 'user_requested'}
           )
           
           self.logger.info(f"Operation cancellation requested [{correlation_id}]")
           return True
       
       return False
   
async def get_recent_events(self, limit: int = 100, event_types: Optional[List[AnalysisEventType]] = None) -> List[Dict[str, Any]]:
       """Get recent analysis events for monitoring."""
       events = self.events[-limit:] if len(self.events) > limit else self.events
       
       # Filter by event types if specified
       if event_types:
           events = [e for e in events if e.event_type in event_types]
       
       return [
           {
               'event_type': event.event_type.value,
               'timestamp': event.timestamp.isoformat(),
               'correlation_id': event.correlation_id,
               'data': event.data
           }
           for event in events
       ]
   
async def reset_circuit_breaker(self) -> None:
       """Manually reset the pipeline circuit breaker."""
       self.circuit_breaker.is_open = False
       self.circuit_breaker.failure_count = 0
       self.circuit_breaker.last_failure_time = None
       self.logger.info("Pipeline circuit breaker manually reset")
   
async def update_configuration(self, config: AnalysisConfig) -> None:
       """Update service configuration."""
       self.config = config
       self.logger.info("Analysis service configuration updated")
   
async def health_check(self) -> Dict[str, Any]:
       """Perform comprehensive health check."""
       try:
           # Check pipeline initialization
           pipeline_healthy = getattr(self.nlp_pipeline, 'is_initialized', False)
           
           # Check circuit breaker status
           circuit_breaker_healthy = not self.circuit_breaker.is_open
           
           # Check recent error rate
           recent_analyses = self.metrics['total_analyses']
           recent_failures = self.metrics['failed_analyses']
           error_rate = (recent_failures / max(recent_analyses, 1)) * 100
           
           # Check active operations
           active_ops_count = len(self.active_operations)
           
           # Overall health determination
           is_healthy = (
               pipeline_healthy and
               circuit_breaker_healthy and
               error_rate < 50 and  # Less than 50% error rate
               active_ops_count < 10  # Not overwhelmed with operations
           )
           
           return {
               'status': 'healthy' if is_healthy else 'unhealthy',
               'pipeline_initialized': pipeline_healthy,
               'circuit_breaker_open': not circuit_breaker_healthy,
               'error_rate_percentage': error_rate,
               'active_operations': active_ops_count,
               'metrics': self.metrics,
               'timestamp': datetime.now().isoformat()
           }
           
       except Exception as e:
           return {
               'status': 'unhealthy',
               'error': str(e),
               'timestamp': datetime.now().isoformat()
           }


# Context manager for batch analysis with progress tracking
@asynccontextmanager
async def batch_analysis_context(
   service: SpeechAnalysisService,
   speeches: List[CentralBankSpeech],
   **kwargs
):
   """Context manager for batch analysis with automatic cleanup."""
   correlation_id = str(uuid4())
   
   try:
       # Start batch analysis
       result = await service.analyze_speeches_batch(speeches, **kwargs)
       yield result
       
   except Exception as e:
       logger.error(f"Batch analysis context error [{correlation_id}]: {e}")
       raise
       
   finally:
       # Cleanup if needed
       if correlation_id in service.active_operations:
           del service.active_operations[correlation_id]


# Factory function for creating service with dependencies
async def create_analysis_service(
   unit_of_work: UnitOfWork,
   nlp_pipeline: NLPProcessingPipeline,
   config: Optional[AnalysisConfig] = None
) -> SpeechAnalysisService:
   """
   Factory function to create a fully configured analysis service.
   
   Args:
       unit_of_work: Unit of work for database operations
       nlp_pipeline: NLP processing pipeline
       config: Optional service configuration
       
   Returns:
       Configured analysis service
   """
   service = SpeechAnalysisService(unit_of_work, nlp_pipeline, config)
   
   # Initialize pipeline
   await service._ensure_pipeline_initialized()
   
   return service


# Health check function
async def check_analysis_service_health(service: SpeechAnalysisService) -> Dict[str, Any]:
   """
   Check the health of the analysis service.
   
   Args:
       service: Analysis service to check
       
   Returns:
       Health status dictionary
   """
   return await service.health_check()


# Export public interface
__all__ = [
   'SpeechAnalysisService',
   'AnalysisConfig',
   'AnalysisProgress',
   'AnalysisEvent',
   'AnalysisEventType',
   'AnalysisError',
   'PipelineError',
   'ValidationError',
   'AnalysisTimeoutError',
   'BatchProcessingError',
   'batch_analysis_context',
   'create_analysis_service',
   'check_analysis_service_health'
]