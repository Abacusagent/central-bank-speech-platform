#!/usr/bin/env python3
"""
NLP Processing Interfaces for Central Bank Speech Analysis Platform

This module defines the core interfaces and contracts for all NLP processing components.
Following the plugin-first architecture from the BaselineObjective, these interfaces
ensure consistent behavior across all processors while enabling infinite extensibility.

Key Principles:
- Plugin-First Architecture: Every NLP processor implements standardized interfaces
- Domain-Driven Design: Interfaces are built around monetary policy analysis concepts
- Dependency Inversion: High-level modules depend on abstractions, not concretions
- Single Responsibility: Each interface has exactly one reason to change
- Extensive Type Safety: Full typing with no Any types in production interfaces

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Dict, List, Optional, Any, Union, Tuple, Set, AsyncIterator, Protocol,
    TypeVar, Generic, Callable, Awaitable
)
from uuid import UUID, uuid4

import numpy as np

# Domain imports
from domain.entities import CentralBankSpeech, PolicyStance
from domain.value_objects import SentimentScore, ConfidenceLevel, Version


# Type Variables for Generic Interfaces
T = TypeVar('T')
ProcessorResultType = TypeVar('ProcessorResultType', bound='ProcessorResult')
AnalysisResultType = TypeVar('AnalysisResultType', bound='ProcessorResult')


class ProcessingStatus(Enum):
    """Status of NLP processing operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AnalysisType(Enum):
    """Types of NLP analysis supported by the platform."""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    STANCE_DETECTION = "stance_detection"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    KEYWORD_EXTRACTION = "keyword_extraction"


class ProcessorCapability(Enum):
    """Capabilities that processors can declare."""
    REAL_TIME = "real_time"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"
    MULTILINGUAL = "multilingual"
    CONTEXTUAL_AWARENESS = "contextual_awareness"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    INTERPRETABILITY = "interpretability"
    CONFIDENCE_SCORING = "confidence_scoring"


@dataclass
class ProcessingMetrics:
    """Comprehensive metrics for NLP processing operations."""
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    tokens_processed: Optional[int] = None
    characters_processed: Optional[int] = None
    throughput_tokens_per_second: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    errors_encountered: int = 0
    warnings_generated: int = 0
    
    def complete(self, end_time: Optional[datetime] = None) -> None:
        """Mark processing as complete and calculate derived metrics."""
        self.end_time = end_time or datetime.now()
        self.processing_time_seconds = (self.end_time - self.start_time).total_seconds()
        
        if self.tokens_processed and self.processing_time_seconds:
            self.throughput_tokens_per_second = self.tokens_processed / self.processing_time_seconds


@dataclass
class ProcessorResult:
    """
    Standard result format for all NLP processors.
    
    This is the core contract that all processors must implement to ensure
    consistent behavior and interoperability across the platform.
    """
    processor_name: str
    processor_version: str
    analysis_type: AnalysisType
    status: ProcessingStatus
    confidence: float  # 0.0 to 1.0
    results: Dict[str, Any]
    
    # Metadata and diagnostics
    processing_metrics: ProcessingMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    # Unique identifier for this specific analysis
    result_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_successful(self) -> bool:
        """Check if processing completed successfully."""
        return self.status == ProcessingStatus.COMPLETED and not self.errors
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def add_warning(self, message: str) -> None:
        """Add a warning message to the result."""
        self.warnings.append(f"{datetime.now().isoformat()}: {message}")
    
    def add_error(self, message: str) -> None:
        """Add an error message to the result."""
        self.errors.append(f"{datetime.now().isoformat()}: {message}")
        if self.status == ProcessingStatus.COMPLETED:
            self.status = ProcessingStatus.FAILED


@dataclass
class AnalysisRequest:
    """
    Request object for NLP analysis operations.
    
    Encapsulates all information needed to perform analysis while providing
    extensibility for future requirements.
    """
    text: str
    analysis_types: List[AnalysisType]
    
    # Optional context and constraints
    metadata: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Request identification and tracking
    request_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5  # 1-10, higher is more urgent
    
    # Processing options
    enable_caching: bool = True
    require_explanation: bool = False
    max_processing_time_seconds: Optional[int] = None
    
    def __post_init__(self):
        """Validate the analysis request."""
        if not self.text or not self.text.strip():
            raise ValueError("Text content cannot be empty")
        
        if not self.analysis_types:
            raise ValueError("At least one analysis type must be specified")
        
        if not 1 <= self.priority <= 10:
            raise ValueError("Priority must be between 1 and 10")


@dataclass
class BatchAnalysisRequest:
    """Request for batch processing of multiple texts."""
    texts: List[str]
    analysis_types: List[AnalysisType]
    
    # Batch-specific options
    batch_id: UUID = field(default_factory=uuid4)
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    preserve_order: bool = True
    
    # Common metadata for all texts in batch
    common_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Individual metadata for each text (must match length of texts)
    individual_metadata: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Validate the batch analysis request."""
        if not self.texts:
            raise ValueError("Batch must contain at least one text")
        
        if self.individual_metadata and len(self.individual_metadata) != len(self.texts):
            raise ValueError("Individual metadata must match the number of texts")


@dataclass
class ProcessorCapabilities:
    """Describes the capabilities and limitations of an NLP processor."""
    processor_name: str
    processor_version: str
    supported_analysis_types: Set[AnalysisType]
    supported_capabilities: Set[ProcessorCapability]
    supported_languages: Set[str]
    
    # Performance characteristics
    max_text_length: Optional[int] = None
    optimal_text_length_range: Optional[Tuple[int, int]] = None
    average_processing_time_per_1k_chars: Optional[float] = None
    memory_requirements_mb: Optional[int] = None
    
    # Quality metrics
    typical_confidence_range: Optional[Tuple[float, float]] = None
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    
    # Dependencies and requirements
    requires_internet: bool = False
    requires_gpu: bool = False
    requires_models: List[str] = field(default_factory=list)
    
    def supports_analysis_type(self, analysis_type: AnalysisType) -> bool:
        """Check if processor supports a specific analysis type."""
        return analysis_type in self.supported_analysis_types
    
    def supports_capability(self, capability: ProcessorCapability) -> bool:
        """Check if processor supports a specific capability."""
        return capability in self.supported_capabilities
    
    def supports_language(self, language: str) -> bool:
        """Check if processor supports a specific language."""
        return language.lower() in {lang.lower() for lang in self.supported_languages}


# Core Processor Interface

class NLPProcessor(ABC):
    """
    Abstract base class for all NLP processors in the platform.
    
    This is the core interface that all processors must implement to be
    compatible with the processing pipeline. It follows the plugin-first
    architecture and ensures consistent behavior across all processors.
    
    Key Responsibilities:
    - Provide standardized analysis interface
    - Declare capabilities and limitations
    - Handle initialization and resource management
    - Provide comprehensive error handling
    - Support both synchronous and asynchronous processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor with optional configuration.
        
        Args:
            config: Processor-specific configuration dictionary
        """
        self.config = config or {}
        self._is_initialized = False
        self._capabilities: Optional[ProcessorCapabilities] = None
        self._version = Version.from_string("1.0.0")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the processor name."""
        pass
    
    @property
    def version(self) -> Version:
        """Get the processor version."""
        return self._version
    
    @property
    def is_initialized(self) -> bool:
        """Check if processor has been initialized."""
        return self._is_initialized
    
    @property
    def capabilities(self) -> ProcessorCapabilities:
        """Get processor capabilities."""
        if self._capabilities is None:
            self._capabilities = self._build_capabilities()
        return self._capabilities
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the processor (load models, connect to services, etc.).
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> ProcessorResult:
        """
        Perform analysis on a single text.
        
        Args:
            request: Analysis request containing text and parameters
            
        Returns:
            ProcessorResult with analysis results and metadata
        """
        pass
    
    async def analyze_batch(self, request: BatchAnalysisRequest) -> List[ProcessorResult]:
        """
        Perform batch analysis on multiple texts.
        
        Default implementation processes texts individually. Processors can
        override this for optimized batch processing.
        
        Args:
            request: Batch analysis request
            
        Returns:
            List of ProcessorResult objects
        """
        results = []
        
        for i, text in enumerate(request.texts):
            # Build individual analysis request
            individual_metadata = request.common_metadata.copy()
            if request.individual_metadata and i < len(request.individual_metadata):
                individual_metadata.update(request.individual_metadata[i])
            
            analysis_request = AnalysisRequest(
                text=text,
                analysis_types=request.analysis_types,
                metadata=individual_metadata,
                enable_caching=True  # Default for batch processing
            )
            
            try:
                result = await self.analyze(analysis_request)
                results.append(result)
            except Exception as e:
                # Create error result for failed analysis
                error_result = ProcessorResult(
                    processor_name=self.name,
                    processor_version=self.version.to_string(),
                    analysis_type=request.analysis_types[0] if request.analysis_types else AnalysisType.SENTIMENT_ANALYSIS,
                    status=ProcessingStatus.FAILED,
                    confidence=0.0,
                    results={},
                    processing_metrics=ProcessingMetrics(start_time=datetime.now()),
                    errors=[str(e)]
                )
                results.append(error_result)
        
        return results
    
    @abstractmethod
    def _build_capabilities(self) -> ProcessorCapabilities:
        """
        Build and return processor capabilities.
        
        This method should declare what the processor can do, its limitations,
        and performance characteristics.
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Clean up resources when processor is no longer needed.
        
        Default implementation does nothing. Processors should override
        if they need to clean up models, connections, etc.
        """
        pass
    
    def validate_request(self, request: AnalysisRequest) -> List[str]:
        """
        Validate an analysis request against processor capabilities.
        
        Args:
            request: Analysis request to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check supported analysis types
        for analysis_type in request.analysis_types:
            if not self.capabilities.supports_analysis_type(analysis_type):
                errors.append(f"Analysis type {analysis_type.value} not supported")
        
        # Check text length limits
        if self.capabilities.max_text_length and len(request.text) > self.capabilities.max_text_length:
            errors.append(f"Text length {len(request.text)} exceeds maximum {self.capabilities.max_text_length}")
        
        return errors


# Specialized Processor Interfaces

class SentimentAnalysisProcessor(NLPProcessor):
    """
    Specialized interface for sentiment analysis processors.
    
    Extends the base processor interface with sentiment-specific methods
    and standardized result formats for hawk-dove analysis.
    """
    
    @abstractmethod
    async def analyze_sentiment(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> 'SentimentAnalysisResult':
        """
        Perform sentiment analysis with standardized result format.
        
        Args:
            text: Text to analyze
            metadata: Optional metadata about the text
            
        Returns:
            SentimentAnalysisResult with hawk-dove scores and confidence
        """
        pass
    
    @abstractmethod
    def get_sentiment_explanation(self, text: str, result: 'SentimentAnalysisResult') -> Dict[str, Any]:
        """
        Provide explanation for sentiment analysis results.
        
        Args:
            text: Original text that was analyzed
            result: Sentiment analysis result to explain
            
        Returns:
            Dictionary containing explanation details
        """
        pass


class TopicModelingProcessor(NLPProcessor):
    """
    Specialized interface for topic modeling processors.
    
    Provides standardized methods for topic discovery, classification,
    and temporal topic analysis.
    """
    
    @abstractmethod
    async def discover_topics(self, texts: List[str], num_topics: Optional[int] = None) -> 'TopicModelResult':
        """
        Discover topics from a collection of texts.
        
        Args:
            texts: Collection of texts to analyze
            num_topics: Optional number of topics to discover
            
        Returns:
            TopicModelResult with discovered topics
        """
        pass
    
    @abstractmethod
    async def classify_topics(self, text: str, topic_model: Optional[Any] = None) -> 'TopicClassificationResult':
        """
        Classify text against existing topic model.
        
        Args:
            text: Text to classify
            topic_model: Pre-trained topic model (optional)
            
        Returns:
            TopicClassificationResult with topic probabilities
        """
        pass


class UncertaintyQuantificationProcessor(NLPProcessor):
    """
    Specialized interface for uncertainty quantification processors.
    
    Focuses on measuring and quantifying uncertainty in central bank
    communications.
    """
    
    @abstractmethod
    async def quantify_uncertainty(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> 'UncertaintyResult':
        """
        Quantify uncertainty expressed in text.
        
        Args:
            text: Text to analyze for uncertainty
            metadata: Optional metadata about the text
            
        Returns:
            UncertaintyResult with uncertainty scores and indicators
        """
        pass
    
    @abstractmethod
    def get_uncertainty_indicators(self, text: str) -> List[str]:
        """
        Extract specific uncertainty indicators from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of uncertainty indicators found in the text
        """
        pass


# Result Type Interfaces

@dataclass
class SentimentAnalysisResult:
    """Standardized result format for sentiment analysis."""
    hawkish_dovish_score: float  # -1.0 (very dovish) to +1.0 (very hawkish)
    policy_stance: PolicyStance
    confidence: float
    
    # Detailed breakdown
    method_scores: Dict[str, float] = field(default_factory=dict)
    contributing_factors: Dict[str, Any] = field(default_factory=dict)
    uncertainty_indicators: List[str] = field(default_factory=list)
    
    # Temporal aspects
    temporal_consistency: Optional[float] = None
    forward_guidance_strength: Optional[float] = None
    
    # Interpretability
    key_phrases: List[Tuple[str, float]] = field(default_factory=list)
    attention_weights: Optional[Dict[str, float]] = None


@dataclass
class TopicModelResult:
    """Result format for topic modeling operations."""
    topics: List[Dict[str, Any]]
    topic_words: Dict[int, List[Tuple[str, float]]]
    coherence_score: Optional[float] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicClassificationResult:
    """Result format for topic classification."""
    topic_probabilities: Dict[str, float]
    dominant_topic: str
    confidence: float
    topic_words: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class UncertaintyResult:
    """Result format for uncertainty quantification."""
    uncertainty_score: float  # 0.0 (certain) to 1.0 (very uncertain)
    uncertainty_level: str  # "low", "medium", "high"
    uncertainty_indicators: List[str]
    
    # Detailed breakdown
    linguistic_uncertainty: float
    semantic_uncertainty: float
    temporal_uncertainty: Optional[float] = None
    
    # Context
    uncertainty_sources: Dict[str, float] = field(default_factory=dict)


# Pipeline and Orchestration Interfaces

class NLPPipeline(ABC):
    """
    Abstract interface for NLP processing pipelines.
    
    Defines the contract for orchestrating multiple processors to perform
    comprehensive analysis on central bank speeches.
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the pipeline and all its processors."""
        pass
    
    @abstractmethod
    async def process(self, speech: CentralBankSpeech) -> 'PipelineResult':
        """
        Process a speech through the entire pipeline.
        
        Args:
            speech: Central bank speech to analyze
            
        Returns:
            PipelineResult with comprehensive analysis
        """
        pass
    
    @abstractmethod
    async def process_batch(self, speeches: List[CentralBankSpeech]) -> List['PipelineResult']:
        """
        Process multiple speeches in batch.
        
        Args:
            speeches: List of speeches to analyze
            
        Returns:
            List of PipelineResult objects
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive pipeline capabilities."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        pass


@dataclass
class PipelineResult:
    """Comprehensive result from NLP pipeline processing."""
    speech_id: UUID
    processing_timestamp: datetime
    
    # Aggregated results
    overall_confidence: float
    processing_time_seconds: float
    
    # Individual processor results
    processor_results: Dict[str, ProcessorResult]
    
    # Derived insights
    hawkish_dovish_score: Optional[float] = None
    policy_stance: Optional[PolicyStance] = None
    uncertainty_score: Optional[float] = None
    complexity_score: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    
    # Quality metrics
    consensus_level: float = 0.0
    result_stability: Optional[float] = None
    
    # Warnings and diagnostics
    warnings: List[str] = field(default_factory=list)
    processing_notes: List[str] = field(default_factory=list)


# Processor Registry and Discovery

class ProcessorRegistry(ABC):
    """
    Abstract interface for processor registry and discovery.
    
    Manages the lifecycle of processors and provides discovery mechanisms
    for finding appropriate processors for specific analysis tasks.
    """
    
    @abstractmethod
    async def register_processor(self, processor: NLPProcessor) -> bool:
        """
        Register a processor with the registry.
        
        Args:
            processor: Processor to register
            
        Returns:
            True if registration successful
        """
        pass
    
    @abstractmethod
    async def unregister_processor(self, processor_name: str) -> bool:
        """
        Unregister a processor from the registry.
        
        Args:
            processor_name: Name of processor to unregister
            
        Returns:
            True if unregistration successful
        """
        pass
    
    @abstractmethod
    def find_processors(self, analysis_type: AnalysisType, 
                       capabilities: Optional[Set[ProcessorCapability]] = None) -> List[NLPProcessor]:
        """
        Find processors that support specific analysis type and capabilities.
        
        Args:
            analysis_type: Type of analysis needed
            capabilities: Optional required capabilities
            
        Returns:
            List of matching processors
        """
        pass
    
    @abstractmethod
    def get_best_processor(self, analysis_type: AnalysisType, 
                          criteria: Optional[Dict[str, Any]] = None) -> Optional[NLPProcessor]:
        """
        Get the best processor for a specific analysis type based on criteria.
        
        Args:
            analysis_type: Type of analysis needed
            criteria: Selection criteria (performance, accuracy, etc.)
            
        Returns:
            Best matching processor or None
        """
        pass
    
    @abstractmethod
    def list_all_processors(self) -> List[ProcessorCapabilities]:
        """List capabilities of all registered processors."""
        pass


# Caching and Optimization Interfaces

class AnalysisCache(ABC):
    """
    Abstract interface for caching analysis results.
    
    Provides mechanisms for caching and retrieving analysis results to
    improve performance and reduce redundant computations.
    """
    
    @abstractmethod
    async def get(self, cache_key: str) -> Optional[ProcessorResult]:
        """
        Retrieve cached analysis result.
        
        Args:
            cache_key: Unique cache key for the analysis
            
        Returns:
            Cached result or None if not found
        """
        pass
    
    @abstractmethod
    async def put(self, cache_key: str, result: ProcessorResult, ttl_seconds: Optional[int] = None) -> bool:
        """
        Store analysis result in cache.
        
        Args:
            cache_key: Unique cache key for the analysis
            result: Analysis result to cache
            ttl_seconds: Time-to-live in seconds (optional)
            
        Returns:
            True if caching successful
        """
        pass
    
    @abstractmethod
    async def invalidate(self, pattern: str) -> int:
        """
        Invalidate cached results matching pattern.
        
        Args:
            pattern: Pattern to match for invalidation
            
        Returns:
            Number of invalidated entries
        """
        pass
    
    @abstractmethod
    def generate_cache_key(self, request: AnalysisRequest, processor_name: str) -> str:
        """
        Generate cache key for analysis request.
        
        Args:
            request: Analysis request
            processor_name: Name of processor
            
        Returns:
            Unique cache key
        """
        pass


# Quality Assurance and Validation Interfaces

class ResultValidator(ABC):
    """
    Abstract interface for validating analysis results.
    
    Provides mechanisms for ensuring result quality and consistency
    across different processors and analysis types.
    """
    
    @abstractmethod
    def validate_result(self, result: ProcessorResult) -> List[str]:
        """
        Validate analysis result.
        
        Args:
            result: Analysis result to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass
    
    @abstractmethod
    def validate_consistency(self, results: List[ProcessorResult]) -> Dict[str, Any]:
        """
        Validate consistency across multiple results.
        
        Args:
            results: List of results to check for consistency
            
        Returns:
            Consistency validation report
        """
        pass
    
    @abstractmethod
    def calculate_quality_score(self, result: ProcessorResult) -> float:
        """
        Calculate quality score for analysis result.
        
        Args:
            result: Analysis result to score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        pass


# Factory and Builder Interfaces

class ProcessorFactory(ABC):
    """
    Abstract factory for creating NLP processors.
    
    Provides a standardized way to create and configure processors
    based on requirements and available resources.
    """
    
    @abstractmethod
    def create_processor(self, processor_type: str, config: Dict[str, Any]) -> NLPProcessor:
        """
        Create processor instance.
        
        Args:
            processor_type: Type of processor to create
            config: Processor configuration
            
        Returns:
            Configured processor instance
        """
        pass
    
    @abstractmethod
    def list_available_processors(self) -> List[str]:
        """List all available processor types."""
        pass
    
    @abstractmethod
    def get_default_config(self, processor_type: str) -> Dict[str, Any]:
        """Get default configuration for processor type."""
        pass


# Monitoring and Observability Interfaces

class ProcessorMonitor(ABC):
    """
    Abstract interface for monitoring processor performance and health.
    
    Provides mechanisms for tracking processor performance, detecting
    issues, and providing operational insights.
    """
    
    @abstractmethod
    def record_processing_event(self, processor_name: str, event_type: str, 
                               metadata: Dict[str, Any]) -> None:
        """
        Record a processing event for monitoring.
        
        Args:
            processor_name: Name of processor
            event_type: Type of event (success, failure, timeout, etc.)
            metadata: Event metadata
        """
        pass
    
    @abstractmethod
    def get_processor_metrics(self, processor_name: str, 
                             time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a processor.
        
        Args:
            processor_name: Name of processor
            time_range: Optional time range for metrics
            
        Returns:
            Performance metrics dictionary
        """
        pass
    
    @abstractmethod
    def get_health_status(self, processor_name: str) -> Dict[str, Any]:
        """
        Get health status for a processor.
        
        Args:
            processor_name: Name of processor
            
        Returns:
            Health status information
        """
        pass
    
    @abstractmethod
    async def run_health_check(self, processor: NLPProcessor) -> bool:
        """
        Run health check on processor.
        
        Args:
            processor: Processor to check
            
        Returns:
            True if processor is healthy
        """
        pass


# Extensibility and Plugin Support

class ProcessorPlugin(Protocol):
    """
    Protocol for processor plugins.
    
    Defines the interface that all processor plugins must implement
    to be compatible with the platform's plugin system.
    """
    
    def get_processor_class(self) -> type:
        """Get the processor class implemented by this plugin."""
        ...
    
    def get_plugin_metadata(self) -> Dict[str, Any]:
        """Get metadata about this plugin."""
        ...
    
    def validate_dependencies(self) -> List[str]:
        """Validate plugin dependencies."""
        ...


class PluginManager(ABC):
    """
    Abstract interface for managing processor plugins.
    
    Handles the lifecycle of plugins including discovery, loading,
    validation, and unloading.
    """
    
    @abstractmethod
    async def discover_plugins(self, plugin_directory: str) -> List[ProcessorPlugin]:
        """
        Discover plugins in a directory.
        
        Args:
            plugin_directory: Directory to search for plugins
            
        Returns:
            List of discovered plugins
        """
        pass
    
    @abstractmethod
    async def load_plugin(self, plugin: ProcessorPlugin) -> bool:
        """
        Load a plugin into the system.
        
        Args:
            plugin: Plugin to load
            
        Returns:
            True if loading successful
        """
        pass
    
    @abstractmethod
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin from the system.
        
        Args:
            plugin_name: Name of plugin to unload
            
        Returns:
            True if unloading successful
        """
        pass
    
    @abstractmethod
    def list_loaded_plugins(self) -> List[Dict[str, Any]]:
        """List all currently loaded plugins."""
        pass


# Type Aliases and Helper Types

# Function type for custom analysis functions
# Type Aliases and Helper Types

# Function type for custom analysis functions (async)
AnalysisFunction = Callable[
    [str, Optional[Dict[str, Any]]],  # text and optional metadata
    Awaitable[ProcessorResult]
]

# Batch function for batch analysis (async)
BatchAnalysisFunction = Callable[
    [List[str], Optional[List[Dict[str, Any]]]],  # texts and optional metadata
    Awaitable[List[ProcessorResult]]
]

# (Optional) Synchronous custom function type
SyncAnalysisFunction = Callable[
    [str, Optional[Dict[str, Any]]],
    ProcessorResult
]

# Factory for processor instantiation
ProcessorFactoryFunction = Callable[
    [Dict[str, Any]],
    NLPProcessor
]
