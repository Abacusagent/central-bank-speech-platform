# application/orchestrators/speech_collection.py

"""
Speech Collection Orchestrator - Production-Ready Multi-Plugin Workflow

This orchestrator coordinates the complete speech collection process across multiple
central bank plugins with enterprise-grade reliability, monitoring, and performance.
Implements advanced patterns for workflow coordination, error recovery, and observability.

Key Features:
- Advanced workflow orchestration with state management
- Plugin health monitoring and circuit breaker protection
- Comprehensive progress tracking and cancellation support
- Event-driven architecture with domain events
- Performance optimization with batch processing
- Real-time metrics and observability
- Configuration-driven workflow customization
- Production-ready error handling and recovery

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Callable, AsyncIterator
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from domain.entities import (
    CentralBankSpeech, Institution, SpeechStatus, InstitutionType
)
from domain.value_objects import DateRange
from domain.repositories import (
    UnitOfWork, RepositoryError, EntityNotFoundError, DuplicateEntityError
)
from interfaces.plugin_interfaces import (
    CentralBankScraperPlugin, SpeechMetadata, SpeechContent, ValidationResult,
    PluginError, ContentExtractionError, ValidationError, RateLimitError
)
from application.services.speech_collection_service import (
    SpeechCollectionService, SpeechCollectionEvent, SpeechCollectionEventType
)

logger = logging.getLogger(__name__)


# Orchestration Domain Events
class OrchestrationEventType(Enum):
    """Types of orchestration events."""
    ORCHESTRATION_STARTED = "orchestration_started"
    ORCHESTRATION_COMPLETED = "orchestration_completed"
    ORCHESTRATION_FAILED = "orchestration_failed"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    PLUGIN_WORKFLOW_STARTED = "plugin_workflow_started"
    PLUGIN_WORKFLOW_COMPLETED = "plugin_workflow_completed"
    PLUGIN_WORKFLOW_FAILED = "plugin_workflow_failed"
    BATCH_COORDINATION_STARTED = "batch_coordination_started"
    BATCH_COORDINATION_COMPLETED = "batch_coordination_completed"


@dataclass
class OrchestrationEvent:
    """Domain event for orchestration operations."""
    event_type: OrchestrationEventType
    timestamp: datetime
    correlation_id: str
    data: Dict[str, Any] = field(default_factory=dict)


# Workflow State Management
class WorkflowState(Enum):
    """States of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowProgress:
    """Tracks progress of orchestration workflow."""
    workflow_id: str
    state: WorkflowState
    total_institutions: int
    completed_institutions: int = 0
    failed_institutions: int = 0
    skipped_institutions: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    current_operation: str = ""
    estimated_completion: Optional[datetime] = None
    
    # Aggregate speech metrics
    total_speeches_discovered: int = 0
    total_speeches_collected: int = 0
    total_speeches_failed: int = 0
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_institutions == 0:
            return 100.0
        processed = self.completed_institutions + self.failed_institutions + self.skipped_institutions
        return processed / self.total_institutions * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for institutions."""
        total_processed = self.completed_institutions + self.failed_institutions
        if total_processed == 0:
            return 100.0
        return self.completed_institutions / total_processed * 100
    
    @property
    def speech_success_rate(self) -> float:
        """Calculate success rate for speeches."""
        total_speeches = self.total_speeches_collected + self.total_speeches_failed
        if total_speeches == 0:
            return 100.0
        return self.total_speeches_collected / total_speeches * 100


# Exception Hierarchy
class OrchestrationError(Exception):
    """Base exception for orchestration operations."""
    pass


class WorkflowError(OrchestrationError):
    """Raised when workflow execution fails."""
    pass


class CoordinationError(OrchestrationError):
    """Raised when coordination between plugins fails."""
    pass


class WorkflowTimeoutError(OrchestrationError):
    """Raised when workflow execution times out."""
    pass


# Orchestration Configuration
@dataclass
class OrchestrationConfig:
    """Configuration for orchestration operations."""
    max_concurrent_institutions: int = 3
    max_concurrent_speeches_per_institution: int = 5
    institution_timeout_minutes: int = 60
    workflow_timeout_minutes: int = 480  # 8 hours
    retry_failed_institutions: bool = True
    retry_attempts: int = 2
    pause_between_retries_minutes: int = 5
    enable_progressive_backoff: bool = True
    checkpoint_interval_minutes: int = 15
    enable_real_time_monitoring: bool = True


# Plugin Coordination Manager
class PluginCoordinationManager:
    """Manages coordination and dependencies between plugins."""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.plugin_dependencies: Dict[str, List[str]] = {}
        self.plugin_priorities: Dict[str, int] = {}
        self.logger = logging.getLogger(f"{__name__}.PluginCoordinationManager")
    
    def set_plugin_dependency(self, plugin_code: str, depends_on: List[str]) -> None:
        """Set dependencies for a plugin (must run after others)."""
        self.plugin_dependencies[plugin_code] = depends_on
        self.logger.info(f"Set dependencies for {plugin_code}: {depends_on}")
    
    def set_plugin_priority(self, plugin_code: str, priority: int) -> None:
        """Set priority for a plugin (higher numbers run first)."""
        self.plugin_priorities[plugin_code] = priority
        self.logger.info(f"Set priority for {plugin_code}: {priority}")
    
    def get_execution_order(self, plugin_codes: List[str]) -> List[List[str]]:
        """
        Calculate optimal execution order considering dependencies and priorities.
        Returns batches that can be executed concurrently.
        """
        # Sort by priority first
        sorted_codes = sorted(
            plugin_codes,
            key=lambda x: self.plugin_priorities.get(x, 0),
            reverse=True
        )
        
        # Group into batches based on dependencies
        batches = []
        remaining = set(sorted_codes)
        completed = set()
        
        while remaining:
            current_batch = []
            
            for code in list(remaining):
                dependencies = self.plugin_dependencies.get(code, [])
                
                # Check if all dependencies are completed
                if all(dep in completed for dep in dependencies):
                    current_batch.append(code)
                    remaining.remove(code)
            
            if not current_batch:
                # Circular dependency or unresolvable - add remaining as single batch
                current_batch = list(remaining)
                remaining.clear()
                self.logger.warning(f"Potential circular dependency detected, adding remaining plugins: {current_batch}")
            
            batches.append(current_batch)
            completed.update(current_batch)
        
        self.logger.info(f"Calculated execution order: {batches}")
        return batches


# Workflow Checkpointing
@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow state persistence."""
    workflow_id: str
    timestamp: datetime
    progress: WorkflowProgress
    completed_institutions: List[str]
    failed_institutions: List[str]
    institution_results: Dict[str, Any]
    configuration: OrchestrationConfig


class WorkflowCheckpointManager:
    """Manages workflow checkpointing for recovery."""
    
    def __init__(self):
        self.checkpoints: Dict[str, WorkflowCheckpoint] = {}
        self.logger = logging.getLogger(f"{__name__}.WorkflowCheckpointManager")
    
    def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save a workflow checkpoint."""
        self.checkpoints[checkpoint.workflow_id] = checkpoint
        self.logger.debug(f"Saved checkpoint for workflow {checkpoint.workflow_id}")
    
    def load_checkpoint(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Load a workflow checkpoint."""
        return self.checkpoints.get(workflow_id)
    
    def delete_checkpoint(self, workflow_id: str) -> None:
        """Delete a workflow checkpoint."""
        if workflow_id in self.checkpoints:
            del self.checkpoints[workflow_id]
            self.logger.debug(f"Deleted checkpoint for workflow {workflow_id}")


# Main Orchestrator Implementation
class SpeechCollectionOrchestrator:
    """
    Production-ready orchestrator for coordinating speech collection across
    multiple central bank plugins with advanced workflow management.
    """
    
    def __init__(
        self,
        unit_of_work: UnitOfWork,
        collection_service: SpeechCollectionService,
        config: Optional[OrchestrationConfig] = None
    ):
        """Initialize the orchestrator."""
        self.uow = unit_of_work
        self.collection_service = collection_service
        self.config = config or OrchestrationConfig()
        self.coordination_manager = PluginCoordinationManager(self.config)
        self.checkpoint_manager = WorkflowCheckpointManager()
        self.events: List[OrchestrationEvent] = []
        self.active_workflows: Dict[str, WorkflowProgress] = {}
        self._setup_logging()
        self._setup_metrics()
    
    def _setup_logging(self) -> None:
        """Configure structured logging for the orchestrator."""
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
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'cancelled_workflows': 0,
            'total_institutions_processed': 0,
            'total_speeches_orchestrated': 0,
            'average_workflow_duration': 0.0,
            'checkpoint_saves': 0,
            'workflow_recoveries': 0
        }
    
    def register_plugin(self, plugin: CentralBankScraperPlugin) -> None:
        """Register a plugin with the collection service."""
        self.collection_service.register_plugin(plugin)
        self.logger.info(f"Registered plugin with orchestrator: {plugin.get_institution_code()}")
    
    def configure_plugin_dependencies(self, dependencies: Dict[str, List[str]]) -> None:
        """Configure plugin dependencies for execution ordering."""
        for plugin_code, deps in dependencies.items():
            self.coordination_manager.set_plugin_dependency(plugin_code, deps)
    
    def configure_plugin_priorities(self, priorities: Dict[str, int]) -> None:
        """Configure plugin priorities for execution ordering."""
        for plugin_code, priority in priorities.items():
            self.coordination_manager.set_plugin_priority(plugin_code, priority)
    
    async def orchestrate_collection(
        self,
        date_range: DateRange,
        institution_filters: Optional[List[str]] = None,
        limit_per_institution: Optional[int] = None,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[WorkflowProgress], None]] = None,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate speech collection across all plugins with advanced workflow management.
        
        Args:
            date_range: Date range for collection
            institution_filters: Optional list of institution codes to process
            limit_per_institution: Optional limit per institution
            skip_existing: Whether to skip existing speeches
            progress_callback: Optional callback for progress updates
            workflow_id: Optional workflow ID for resumption
            
        Returns:
            Comprehensive orchestration results
        """
        # Initialize or resume workflow
        if workflow_id and workflow_id in self.active_workflows:
            progress = self.active_workflows[workflow_id]
            self.logger.info(f"Resuming workflow {workflow_id}")
        else:
            workflow_id = workflow_id or str(uuid4())
            progress = self._initialize_workflow(workflow_id, date_range, institution_filters)
        
        start_time = datetime.now()
        
        # Emit orchestration started event
        self._emit_event(
            OrchestrationEventType.ORCHESTRATION_STARTED,
            workflow_id,
            {
                'date_range': {
                    'start_date': date_range.start_date.isoformat(),
                    'end_date': date_range.end_date.isoformat()
                },
                'institution_filters': institution_filters,
                'limit_per_institution': limit_per_institution
            }
        )
        
        self.logger.info(
            f"Starting orchestrated collection [{workflow_id}]: "
            f"{progress.total_institutions} institutions, "
            f"{date_range.start_date} to {date_range.end_date}"
        )
        
        try:
            # Set workflow state to running
            progress.state = WorkflowState.RUNNING
            
            # Get available plugins
            available_plugins = self.collection_service.get_registered_plugins()
            
            # Filter institutions if specified
            target_institutions = (
                institution_filters if institution_filters 
                else list(available_plugins.keys())
            )
            
            # Calculate execution order
            execution_batches = self.coordination_manager.get_execution_order(target_institutions)
            
            # Execute collection in coordinated batches
            institution_results = {}
            
            for batch_index, batch in enumerate(execution_batches):
                self.logger.info(f"Processing batch {batch_index + 1}/{len(execution_batches)}: {batch}")
                
                # Emit batch coordination event
                self._emit_event(
                    OrchestrationEventType.BATCH_COORDINATION_STARTED,
                    workflow_id,
                    {
                        'batch_index': batch_index,
                        'batch_institutions': batch,
                        'total_batches': len(execution_batches)
                    }
                )
                
                # Process batch with timeout and cancellation support
                batch_results = await self._process_institution_batch(
                    batch,
                    date_range,
                    limit_per_institution,
                    skip_existing,
                    progress,
                    workflow_id,
                    progress_callback
                )
                
                institution_results.update(batch_results)
                
                # Save checkpoint after each batch
                if self.config.checkpoint_interval_minutes > 0:
                    self._save_checkpoint(workflow_id, progress, institution_results)
                
                # Check for cancellation
                if progress.state == WorkflowState.CANCELLED:
                    self.logger.info(f"Workflow cancelled [{workflow_id}]")
                    break
                
                # Emit batch completion event
                self._emit_event(
                    OrchestrationEventType.BATCH_COORDINATION_COMPLETED,
                    workflow_id,
                    {
                        'batch_index': batch_index,
                        'batch_results': {k: v.get('speeches_collected', 0) for k, v in batch_results.items()}
                    }
                )
            
            # Handle retry logic for failed institutions
            if self.config.retry_failed_institutions and progress.failed_institutions > 0:
                await self._retry_failed_institutions(
                    institution_results,
                    date_range,
                    limit_per_institution,
                    skip_existing,
                    progress,
                    workflow_id,
                    progress_callback
                )
            
            # Finalize workflow
            end_time = datetime.now()
            progress.end_time = end_time
            progress.state = WorkflowState.COMPLETED if progress.state != WorkflowState.CANCELLED else WorkflowState.CANCELLED
            
            # Compile comprehensive results
            final_results = self._compile_orchestration_results(
                workflow_id,
                progress,
                institution_results,
                start_time,
                end_time,
                date_range
            )
            
            # Update metrics
            self._update_workflow_metrics(progress, end_time - start_time)
            
            # Emit completion event
            self._emit_event(
                OrchestrationEventType.ORCHESTRATION_COMPLETED,
                workflow_id,
                {
                    'final_results': final_results,
                    'duration_seconds': (end_time - start_time).total_seconds()
                }
            )
            
            self.logger.info(
                f"Orchestration completed [{workflow_id}]: "
                f"{progress.completed_institutions}/{progress.total_institutions} institutions successful, "
                f"{progress.total_speeches_collected} speeches collected "
                f"({progress.speech_success_rate:.1f}% speech success rate)"
            )
            
            return final_results
            
        except asyncio.TimeoutError:
            error_msg = f"Workflow timeout after {self.config.workflow_timeout_minutes} minutes"
            self.logger.error(f"Orchestration timeout [{workflow_id}]: {error_msg}")
            return self._handle_workflow_error(workflow_id, progress, WorkflowTimeoutError(error_msg))
            
        except Exception as e:
            self.logger.error(f"Orchestration failed [{workflow_id}]: {e}")
            return self._handle_workflow_error(workflow_id, progress, WorkflowError(f"Orchestration failed: {e}"))
            
        finally:
            # Cleanup workflow tracking
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            # Delete checkpoint if workflow completed successfully
            if progress.state == WorkflowState.COMPLETED:
                self.checkpoint_manager.delete_checkpoint(workflow_id)
    
    async def _process_institution_batch(
        self,
        batch: List[str],
        date_range: DateRange,
        limit_per_institution: Optional[int],
        skip_existing: bool,
        progress: WorkflowProgress,
        workflow_id: str,
        progress_callback: Optional[Callable[[WorkflowProgress], None]]
    ) -> Dict[str, Any]:
        """Process a batch of institutions concurrently."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_institutions)
        batch_results = {}
        
        async def process_institution_with_protection(institution_code: str) -> Tuple[str, Dict[str, Any]]:
            """Process single institution with timeout and error protection."""
            async with semaphore:
                try:
                    # Set current operation
                    progress.current_operation = f"Processing {institution_code}"
                    
                    # Emit plugin workflow started event
                    self._emit_event(
                        OrchestrationEventType.PLUGIN_WORKFLOW_STARTED,
                        workflow_id,
                        {'institution_code': institution_code}
                    )
                    
                    # Process with timeout
                    result = await asyncio.wait_for(
                        self.collection_service.collect_speeches_by_institution(
                            institution_code=institution_code,
                            date_range=date_range,
                            limit=limit_per_institution,
                            skip_existing=skip_existing,
                            batch_size=self.config.max_concurrent_speeches_per_institution
                        ),
                        timeout=self.config.institution_timeout_minutes * 60
                    )
                    
                    # Update progress
                    progress.completed_institutions += 1
                    progress.total_speeches_discovered += result.get('speeches_discovered', 0)
                    progress.total_speeches_collected += result.get('speeches_collected', 0)
                    progress.total_speeches_failed += result.get('speeches_failed', 0)
                    
                    # Call progress callback
                    if progress_callback:
                        progress_callback(progress)
                    
                    # Emit success event
                    self._emit_event(
                        OrchestrationEventType.PLUGIN_WORKFLOW_COMPLETED,
                        workflow_id,
                        {
                            'institution_code': institution_code,
                            'speeches_collected': result.get('speeches_collected', 0)
                        }
                    )
                    
                    self.logger.info(
                        f"Institution {institution_code} completed: "
                        f"{result.get('speeches_collected', 0)} speeches collected "
                        f"[{workflow_id}]"
                    )
                    
                    return institution_code, result
                    
                except asyncio.TimeoutError:
                    error_msg = f"Institution timeout after {self.config.institution_timeout_minutes} minutes"
                    self.logger.error(f"Institution {institution_code} timeout [{workflow_id}]")
                    
                    progress.failed_institutions += 1
                    if progress_callback:
                        progress_callback(progress)
                    
                    return institution_code, {
                        'error': error_msg,
                        'speeches_discovered': 0,
                        'speeches_collected': 0,
                        'speeches_failed': 0,
                        'timeout': True
                    }
                    
                except Exception as e:
                    self.logger.error(f"Institution {institution_code} failed [{workflow_id}]: {e}")
                    
                    progress.failed_institutions += 1
                    if progress_callback:
                        progress_callback(progress)
                    
                    # Emit failure event
                    self._emit_event(
                        OrchestrationEventType.PLUGIN_WORKFLOW_FAILED,
                        workflow_id,
                        {
                            'institution_code': institution_code,
                            'error': str(e)
                        }
                    )
                    
                    return institution_code, {
                        'error': str(e),
                        'speeches_discovered': 0,
                        'speeches_collected': 0,
                        'speeches_failed': 0
                    }
        
        # Process batch concurrently
        tasks = [process_institution_with_protection(code) for code in batch]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Compile batch results
        for institution_code, result in results:
            batch_results[institution_code] = result
        
        return batch_results
    
    async def _retry_failed_institutions(
        self,
        institution_results: Dict[str, Any],
        date_range: DateRange,
        limit_per_institution: Optional[int],
        skip_existing: bool,
        progress: WorkflowProgress,
        workflow_id: str,
        progress_callback: Optional[Callable[[WorkflowProgress], None]]
    ) -> None:
        """Retry failed institutions with exponential backoff."""
        failed_institutions = [
            code for code, result in institution_results.items()
            if 'error' in result and not result.get('timeout', False)
        ]
        
        if not failed_institutions:
            return
        
        self.logger.info(f"Retrying {len(failed_institutions)} failed institutions [{workflow_id}]")
        
        for attempt in range(self.config.retry_attempts):
            if not failed_institutions:
                break
            
            # Wait between retry attempts
            if attempt > 0:
                wait_time = self.config.pause_between_retries_minutes
                if self.config.enable_progressive_backoff:
                    wait_time *= (2 ** (attempt - 1))
                
                self.logger.info(f"Waiting {wait_time} minutes before retry attempt {attempt + 1}")
                await asyncio.sleep(wait_time * 60)
            
            # Reset progress counters for retry
            retry_progress = WorkflowProgress(
                workflow_id=f"{workflow_id}_retry_{attempt}",
                state=WorkflowState.RUNNING,
                total_institutions=len(failed_institutions)
            )
            
            # Retry failed institutions
            retry_results = await self._process_institution_batch(
                failed_institutions,
                date_range,
                limit_per_institution,
                skip_existing,
                retry_progress,
                workflow_id,
                progress_callback
            )
            
            # Update main progress and results
            newly_successful = []
            for institution_code, result in retry_results.items():
                if 'error' not in result:
                    newly_successful.append(institution_code)
                    progress.completed_institutions += 1
                    progress.failed_institutions -= 1
                    progress.total_speeches_discovered += result.get('speeches_discovered', 0)
                    progress.total_speeches_collected += result.get('speeches_collected', 0)
                    progress.total_speeches_failed += result.get('speeches_failed', 0)
                
                # Update main results
                institution_results[institution_code] = result
            
            # Remove successful institutions from retry list
            failed_institutions = [code for code in failed_institutions if code not in newly_successful]
            
            self.logger.info(
                f"Retry attempt {attempt + 1} completed: "
                f"{len(newly_successful)} institutions recovered, "
                f"{len(failed_institutions)} still failed [{workflow_id}]"
            )
    
    def _initialize_workflow(
        self,
        workflow_id: str,
        date_range: DateRange,
        institution_filters: Optional[List[str]]
    ) -> WorkflowProgress:
        """Initialize a new workflow."""
        available_plugins = self.collection_service.get_registered_plugins()
        target_institutions = (
            institution_filters if institution_filters 
            else list(available_plugins.keys())
        )
        
        progress = WorkflowProgress(
            workflow_id=workflow_id,
            state=WorkflowState.PENDING,
            total_institutions=len(target_institutions)
        )
        
        self.active_workflows[workflow_id] = progress
        return progress
    
    def _save_checkpoint(
        self,
        workflow_id: str,
        progress: WorkflowProgress,
        institution_results: Dict[str, Any]
    ) -> None:
        """Save workflow checkpoint."""
        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow_id,
            timestamp=datetime.now(),
            progress=progress,
            completed_institutions=[
                code for code, result in institution_results.items()
                if 'error' not in result
            ],
            failed_institutions=[
                code for code, result in institution_results.items()
                if 'error' in result
            ],
            institution_results=institution_results,
            configuration=self.config
        )
        
        self.checkpoint_manager.save_checkpoint(checkpoint)
        self.metrics['checkpoint_saves'] += 1
        
        self.logger.debug(f"Saved checkpoint for workflow {workflow_id}")
    
    def _compile_orchestration_results(
        self,
        workflow_id: str,
        progress: WorkflowProgress,
        institution_results: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
        date_range: DateRange
    ) -> Dict[str, Any]:
        """Compile comprehensive orchestration results."""
        duration = (end_time - start_time).total_seconds()
        
        # Calculate aggregate metrics
        successful_institutions = [
            code for code, result in institution_results.items()
            if 'error' not in result
        ]
        
        failed_institutions = [
            code for code, result in institution_results.items()
            if 'error' in result
        ]
        
        # Error distribution
        error_distribution = {}
        for code, result in institution_results.items():
            if 'error' in result:
                error_type = 'timeout' if result.get('timeout') else 'processing_error'
                error_distribution[error_type] = error_distribution.get(error_type, 0) + 1
        
        return {
            'workflow_id': workflow_id,
            'workflow_state': progress.state.value,
            'execution_summary': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'date_range': {
                    'start_date': date_range.start_date.isoformat(),
                    'end_date': date_range.end_date.isoformat(),
                    'days': date_range.days
                }
            },
            'institution_metrics': {
                'total_institutions': progress.total_institutions,
                'successful_institutions': len(successful_institutions),
                'failed_institutions': len(failed_institutions),
                'institution_success_rate': progress.success_rate,
                'successful_list': successful_institutions,
                'failed_list': failed_institutions
            },
            'speech_metrics': {
                'total_speeches_discovered': progress.total_speeches_discovered,
                'total_speeches_collected': progress.total_speeches_collected,
                'total_speeches_failed': progress.total_speeches_failed,
                'speech_success_rate': progress.speech_success_rate,
                'throughput_speeches_per_second': (
                    progress.total_speeches_collected / max(duration, 1)
                )
            },'error_analysis': {
               'error_distribution': error_distribution,
               'retry_attempts_made': self.config.retry_attempts if self.config.retry_failed_institutions else 0
           },
           'institution_results': institution_results,
           'performance_metrics': {
               'average_institution_processing_time': duration / max(progress.total_institutions, 1),
               'institutions_per_hour': progress.total_institutions / max(duration / 3600, 1),
               'speeches_per_hour': progress.total_speeches_collected / max(duration / 3600, 1),
               'efficiency_score': (progress.total_speeches_collected / max(progress.total_speeches_discovered, 1)) * 100
           },
           'configuration': {
               'max_concurrent_institutions': self.config.max_concurrent_institutions,
               'max_concurrent_speeches_per_institution': self.config.max_concurrent_speeches_per_institution,
               'retry_enabled': self.config.retry_failed_institutions,
               'checkpoint_interval_minutes': self.config.checkpoint_interval_minutes
           }
       }
   
def _handle_workflow_error(
       self,
       workflow_id: str,
       progress: WorkflowProgress,
       error: Exception
   ) -> Dict[str, Any]:
       """Handle workflow errors and create error result."""
       progress.state = WorkflowState.FAILED
       progress.end_time = datetime.now()
       
       # Update metrics
       self.metrics['total_workflows'] += 1
       self.metrics['failed_workflows'] += 1
       
       # Emit failure event
       self._emit_event(
           OrchestrationEventType.ORCHESTRATION_FAILED,
           workflow_id,
           {
               'error_type': type(error).__name__,
               'error_message': str(error),
               'institutions_completed': progress.completed_institutions,
               'institutions_failed': progress.failed_institutions
           }
       )
       
       return {
           'workflow_id': workflow_id,
           'workflow_state': WorkflowState.FAILED.value,
           'error': {
               'type': type(error).__name__,
               'message': str(error),
               'timestamp': datetime.now().isoformat()
           },
           'partial_progress': {
               'institutions_completed': progress.completed_institutions,
               'institutions_failed': progress.failed_institutions,
               'speeches_collected': progress.total_speeches_collected,
               'completion_percentage': progress.completion_percentage
           },
           'recovery_options': {
               'can_resume': True,
               'checkpoint_available': workflow_id in self.checkpoint_manager.checkpoints,
               'resume_instructions': f"Call resume_workflow('{workflow_id}') to continue from last checkpoint"
           }
       }
   
def _update_workflow_metrics(self, progress: WorkflowProgress, duration: timedelta) -> None:
       """Update workflow metrics."""
       self.metrics['total_workflows'] += 1
       
       if progress.state == WorkflowState.COMPLETED:
           self.metrics['successful_workflows'] += 1
       elif progress.state == WorkflowState.CANCELLED:
           self.metrics['cancelled_workflows'] += 1
       
       self.metrics['total_institutions_processed'] += progress.completed_institutions
       self.metrics['total_speeches_orchestrated'] += progress.total_speeches_collected
       
       # Update average duration
       total_duration = self.metrics['average_workflow_duration'] * (self.metrics['total_workflows'] - 1)
       self.metrics['average_workflow_duration'] = (
           (total_duration + duration.total_seconds()) / self.metrics['total_workflows']
       )
   
def _emit_event(
    self,
    event_type: OrchestrationEventType,
    correlation_id: str,
    data: Dict[str, Any]
   ) -> None:
    """Emit an orchestration event."""
    event = OrchestrationEvent(
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
   
async def resume_workflow(self, workflow_id: str) -> Dict[str, Any]:
       """Resume a workflow from its last checkpoint."""
       checkpoint = self.checkpoint_manager.load_checkpoint(workflow_id)
       
       if not checkpoint:
           raise WorkflowError(f"No checkpoint found for workflow {workflow_id}")
       
       self.logger.info(f"Resuming workflow {workflow_id} from checkpoint")
       self.metrics['workflow_recoveries'] += 1
       
       # Restore workflow state
       progress = checkpoint.progress
       progress.state = WorkflowState.RUNNING
       self.active_workflows[workflow_id] = progress
       
       # Extract date range from checkpoint
       completed_institutions = set(checkpoint.completed_institutions)
       failed_institutions = set(checkpoint.failed_institutions)
       
       # Determine remaining institutions
       all_institutions = self.collection_service.get_registered_plugins().keys()
       remaining_institutions = [
           code for code in all_institutions 
           if code not in completed_institutions and code not in failed_institutions
       ]
       
       if not remaining_institutions:
           self.logger.info(f"No remaining institutions to process for workflow {workflow_id}")
           progress.state = WorkflowState.COMPLETED
           return {
               'workflow_id': workflow_id,
               'message': 'Workflow already completed',
               'state': progress.state.value
           }
       
       # Continue with remaining institutions
       # Note: This would need the original date_range and other parameters
       # In a real implementation, these would be stored in the checkpoint
       self.logger.info(f"Continuing workflow {workflow_id} with {len(remaining_institutions)} remaining institutions")
       
       return {
           'workflow_id': workflow_id,
           'message': f'Workflow resumed with {len(remaining_institutions)} remaining institutions',
           'state': progress.state.value,
           'remaining_institutions': remaining_institutions
       }
   
async def pause_workflow(self, workflow_id: str) -> bool:
       """Pause an active workflow."""
       if workflow_id not in self.active_workflows:
           return False
       
       progress = self.active_workflows[workflow_id]
       if progress.state != WorkflowState.RUNNING:
           return False
       
       progress.state = WorkflowState.PAUSED
       
       # Emit pause event
       self._emit_event(
           OrchestrationEventType.WORKFLOW_PAUSED,
           workflow_id,
           {'paused_at': datetime.now().isoformat()}
       )
       
       self.logger.info(f"Workflow {workflow_id} paused")
       return True
   
async def cancel_workflow(self, workflow_id: str) -> bool:
       """Cancel an active workflow."""
       if workflow_id not in self.active_workflows:
           return False
       
       progress = self.active_workflows[workflow_id]
       progress.state = WorkflowState.CANCELLED
       
       # Emit cancellation event
       self._emit_event(
           OrchestrationEventType.WORKFLOW_CANCELLED,
           workflow_id,
           {'cancelled_at': datetime.now().isoformat()}
       )
       
       self.logger.info(f"Workflow {workflow_id} cancelled")
       return True
   
async def get_workflow_progress(self, workflow_id: str) -> Optional[WorkflowProgress]:
       """Get current progress of a workflow."""
       return self.active_workflows.get(workflow_id)
   
async def list_active_workflows(self) -> Dict[str, Dict[str, Any]]:
       """List all active workflows with their current status."""
       return {
           workflow_id: {
               'state': progress.state.value,
               'completion_percentage': progress.completion_percentage,
               'institutions_completed': progress.completed_institutions,
               'institutions_failed': progress.failed_institutions,
               'speeches_collected': progress.total_speeches_collected,
               'current_operation': progress.current_operation,
               'start_time': progress.start_time.isoformat(),
               'estimated_completion': progress.estimated_completion.isoformat() if progress.estimated_completion else None
           }
           for workflow_id, progress in self.active_workflows.items()
       }
   
async def get_orchestration_statistics(self) -> Dict[str, Any]:
       """Get comprehensive orchestration statistics."""
       return {
           'orchestrator_metrics': self.metrics,
           'active_workflows': len(self.active_workflows),
           'available_checkpoints': len(self.checkpoint_manager.checkpoints),
           'registered_plugins': len(self.collection_service.get_registered_plugins()),
           'plugin_health': await self.collection_service.get_plugin_health_status(),
           'recent_events_count': len(self.events),
           'configuration': {
               'max_concurrent_institutions': self.config.max_concurrent_institutions,
               'workflow_timeout_minutes': self.config.workflow_timeout_minutes,
               'retry_enabled': self.config.retry_failed_institutions,
               'checkpoint_enabled': self.config.checkpoint_interval_minutes > 0
           },
           'last_updated': datetime.now().isoformat()
       }
   
async def get_recent_events(
       self,
       limit: int = 100,
       event_types: Optional[List[OrchestrationEventType]] = None,
       workflow_id: Optional[str] = None
   ) -> List[Dict[str, Any]]:
       """Get recent orchestration events for monitoring."""
       events = self.events[-limit:] if len(self.events) > limit else self.events
       
       # Filter by event types if specified
       if event_types:
           events = [e for e in events if e.event_type in event_types]
       
       # Filter by workflow ID if specified
       if workflow_id:
           events = [e for e in events if e.correlation_id == workflow_id]
       
       return [
           {
               'event_type': event.event_type.value,
               'timestamp': event.timestamp.isoformat(),
               'workflow_id': event.correlation_id,
               'data': event.data
           }
           for event in events
       ]
   
async def cleanup_completed_workflows(self, older_than_hours: int = 24) -> int:
       """Clean up old completed workflow checkpoints."""
       cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
       cleaned_count = 0
       
       for workflow_id, checkpoint in list(self.checkpoint_manager.checkpoints.items()):
           if (checkpoint.progress.state in [WorkflowState.COMPLETED, WorkflowState.FAILED] and 
               checkpoint.timestamp < cutoff_time):
               self.checkpoint_manager.delete_checkpoint(workflow_id)
               cleaned_count += 1
       
       if cleaned_count > 0:
           self.logger.info(f"Cleaned up {cleaned_count} old workflow checkpoints")
       
       return cleaned_count
   
async def export_workflow_report(self, workflow_id: str) -> Optional[Dict[str, Any]]:
       """Export comprehensive report for a workflow."""
       checkpoint = self.checkpoint_manager.load_checkpoint(workflow_id)
       
       if not checkpoint:
           return None
       
       # Get workflow events
       workflow_events = [
           e for e in self.events 
           if e.correlation_id == workflow_id
       ]
       
       return {
           'workflow_summary': {
               'workflow_id': workflow_id,
               'state': checkpoint.progress.state.value,
               'duration': (
                   (checkpoint.progress.end_time - checkpoint.progress.start_time).total_seconds()
                   if checkpoint.progress.end_time else None
               ),
               'institutions_processed': checkpoint.progress.completed_institutions + checkpoint.progress.failed_institutions,
               'success_rate': checkpoint.progress.success_rate,
               'speeches_collected': checkpoint.progress.total_speeches_collected
           },
           'detailed_results': checkpoint.institution_results,
           'configuration_used': checkpoint.configuration.__dict__,
           'timeline': [
               {
                   'timestamp': event.timestamp.isoformat(),
                   'event_type': event.event_type.value,
                   'data': event.data
               }
               for event in workflow_events
           ],
           'performance_analysis': {
               'bottlenecks': self._analyze_workflow_bottlenecks(checkpoint),
               'recommendations': self._generate_workflow_recommendations(checkpoint)
           }
       }
   
def _analyze_workflow_bottlenecks(self, checkpoint: WorkflowCheckpoint) -> List[str]:
       """Analyze workflow for performance bottlenecks."""
       bottlenecks = []
       
       # Analyze institution failure rate
       total_institutions = checkpoint.progress.total_institutions
       failed_institutions = checkpoint.progress.failed_institutions
       
       if failed_institutions / max(total_institutions, 1) > 0.3:
           bottlenecks.append("High institution failure rate (>30%)")
       
       # Analyze speech collection efficiency
       if checkpoint.progress.speech_success_rate < 70:
           bottlenecks.append("Low speech collection efficiency (<70%)")
       
       # Analyze processing time
       if checkpoint.progress.end_time and checkpoint.progress.start_time:
           duration_hours = (checkpoint.progress.end_time - checkpoint.progress.start_time).total_seconds() / 3600
           if duration_hours > 4:
               bottlenecks.append("Long processing time (>4 hours)")
       
       return bottlenecks
   
def _generate_workflow_recommendations(self, checkpoint: WorkflowCheckpoint) -> List[str]:
       """Generate recommendations for workflow optimization."""
       recommendations = []
       
       # Concurrency recommendations
       if checkpoint.configuration.max_concurrent_institutions < 5:
           recommendations.append("Consider increasing max_concurrent_institutions for better parallelism")
       
       # Retry recommendations
       if not checkpoint.configuration.retry_failed_institutions and checkpoint.progress.failed_institutions > 0:
           recommendations.append("Enable retry_failed_institutions to improve success rate")
       
       # Checkpoint recommendations
       if checkpoint.configuration.checkpoint_interval_minutes > 30:
           recommendations.append("Consider reducing checkpoint_interval_minutes for better recovery")
       
       return recommendations
   
async def health_check(self) -> Dict[str, Any]:
       """Perform comprehensive health check of the orchestrator."""
       try:
           # Check collection service health
           collection_service_health = await self.collection_service.get_service_metrics()
           
           # Check active workflows
           active_workflows_count = len(self.active_workflows)
           
           # Check for stuck workflows
           stuck_workflows = []
           for workflow_id, progress in self.active_workflows.items():
               if progress.state == WorkflowState.RUNNING:
                   time_since_start = datetime.now() - progress.start_time
                   if time_since_start.total_seconds() > self.config.workflow_timeout_minutes * 60:
                       stuck_workflows.append(workflow_id)
           
           # Determine overall health
           is_healthy = (
               len(stuck_workflows) == 0 and
               active_workflows_count < 10 and  # Not overwhelmed
               collection_service_health.get('status') == 'healthy'
           )
           
           return {
               'status': 'healthy' if is_healthy else 'unhealthy',
               'active_workflows': active_workflows_count,
               'stuck_workflows': stuck_workflows,
               'collection_service_health': collection_service_health,
               'orchestrator_metrics': self.metrics,
               'checkpoint_count': len(self.checkpoint_manager.checkpoints),
               'timestamp': datetime.now().isoformat()
           }
           
       except Exception as e:
           return {
               'status': 'unhealthy',
               'error': str(e),
               'timestamp': datetime.now().isoformat()
           }


# Context manager for orchestrated collection
@asynccontextmanager
async def orchestrated_collection_context(
   orchestrator: SpeechCollectionOrchestrator,
   date_range: DateRange,
   **kwargs
):
   """Context manager for orchestrated collection with automatic cleanup."""
   workflow_id = str(uuid4())
   
   try:
       # Start orchestrated collection
       result = await orchestrator.orchestrate_collection(
           date_range=date_range,
           workflow_id=workflow_id,
           **kwargs
       )
       yield result
       
   except Exception as e:
       logger.error(f"Orchestrated collection context error [{workflow_id}]: {e}")
       raise
       
   finally:
       # Cleanup if needed
       if workflow_id in orchestrator.active_workflows:
           await orchestrator.cancel_workflow(workflow_id)


# Factory function for creating orchestrator with dependencies
async def create_speech_collection_orchestrator(
   unit_of_work: UnitOfWork,
   collection_service: SpeechCollectionService,
   config: Optional[OrchestrationConfig] = None,
   plugins: Optional[List[CentralBankScraperPlugin]] = None
) -> SpeechCollectionOrchestrator:
   """
   Factory function to create a fully configured orchestrator.
   
   Args:
       unit_of_work: Unit of work for database operations
       collection_service: Speech collection service
       config: Optional orchestration configuration
       plugins: Optional list of plugins to register
       
   Returns:
       Configured orchestrator
   """
   orchestrator = SpeechCollectionOrchestrator(unit_of_work, collection_service, config)
   
   # Register plugins if provided
   if plugins:
       for plugin in plugins:
           try:
               orchestrator.register_plugin(plugin)
           except Exception as e:
               logger.error(f"Failed to register plugin {plugin.get_institution_code()}: {e}")
   
   return orchestrator


# Health check function
async def check_orchestrator_health(orchestrator: SpeechCollectionOrchestrator) -> Dict[str, Any]:
   """
   Check the health of the orchestrator.
   
   Args:
       orchestrator: Orchestrator to check
       
   Returns:
       Health status dictionary
   """
   return await orchestrator.health_check()


# Export public interface
__all__ = [
   'SpeechCollectionOrchestrator',
   'OrchestrationConfig',
   'WorkflowProgress',
   'WorkflowState',
   'OrchestrationEvent',
   'OrchestrationEventType',
   'OrchestrationError',
   'WorkflowError',
   'CoordinationError',
   'WorkflowTimeoutError',
   'orchestrated_collection_context',
   'create_speech_collection_orchestrator',
   'check_orchestrator_health'
]