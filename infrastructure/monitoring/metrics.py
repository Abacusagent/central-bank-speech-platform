# infrastructure/monitoring/metrics.py

"""
Enhanced Metrics and Observability for Central Bank Speech Analysis Platform

Production-grade Prometheus metrics with comprehensive coverage of all system
components including NLP pipeline performance, plugin health, and business metrics.

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from functools import wraps
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Enum, Info,
    start_http_server, generate_latest, CollectorRegistry,
    multiprocess, values
)
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# Create custom registry for isolation
REGISTRY = CollectorRegistry()

# -------------------------
# Core Business Metrics
# -------------------------

SPEECH_COLLECTION_SUCCESS = Counter(
    "speech_collection_success_total",
    "Number of successfully collected speeches",
    ["institution", "plugin_version"],
    registry=REGISTRY
)

SPEECH_COLLECTION_FAILURE = Counter(
    "speech_collection_failure_total",
    "Number of failed speech collection attempts",
    ["institution", "failure_reason", "plugin_version"],
    registry=REGISTRY
)

SPEECH_PROCESSING_DURATION = Histogram(
    "speech_processing_duration_seconds",
    "Time spent processing a speech end-to-end",
    ["institution", "processing_stage"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=REGISTRY
)

SPEECH_VALIDATION_FAILURE = Counter(
    "speech_validation_failure_total",
    "Number of speech validation failures",
    ["institution", "validation_type", "failure_reason"],
    registry=REGISTRY
)

# -------------------------
# NLP Pipeline Metrics
# -------------------------

NLP_ANALYSIS_DURATION = Histogram(
    "nlp_analysis_duration_seconds",
    "Time spent on NLP analysis per speech",
    ["institution", "processor_type"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY
)

NLP_ANALYSIS_SUCCESS = Counter(
    "nlp_analysis_success_total",
    "Number of successful NLP analyses",
    ["institution", "processor_type"],
    registry=REGISTRY
)

NLP_ANALYSIS_FAILURE = Counter(
    "nlp_analysis_failure_total",
    "Number of failed NLP analyses",
    ["institution", "processor_type", "error_type"],
    registry=REGISTRY
)

SENTIMENT_SCORE_DISTRIBUTION = Histogram(
    "sentiment_score_distribution",
    "Distribution of sentiment scores",
    ["institution", "sentiment_type"],
    buckets=[-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0],
    registry=REGISTRY
)

# -------------------------
# Plugin Health Metrics
# -------------------------

PLUGIN_HEALTH_STATUS = Enum(
    "plugin_health_status",
    "Health status of each plugin",
    ["institution"],
    states=["healthy", "degraded", "unhealthy", "unknown"],
    registry=REGISTRY
)

PLUGIN_RATE_LIMIT_HITS = Counter(
    "plugin_rate_limit_hits_total",
    "Number of times plugin hit rate limits",
    ["institution", "endpoint"],
    registry=REGISTRY
)

PLUGIN_HTTP_REQUESTS = Counter(
    "plugin_http_requests_total",
    "Total HTTP requests made by plugins",
    ["institution", "method", "status_code"],
    registry=REGISTRY
)

PLUGIN_HTTP_DURATION = Histogram(
    "plugin_http_request_duration_seconds",
    "HTTP request duration for plugins",
    ["institution", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=REGISTRY
)

# -------------------------
# Database Metrics
# -------------------------

DATABASE_OPERATIONS = Counter(
    "database_operations_total",
    "Total database operations",
    ["operation_type", "table", "status"],
    registry=REGISTRY
)

DATABASE_QUERY_DURATION = Histogram(
    "database_query_duration_seconds",
    "Database query execution time",
    ["operation_type", "table"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=REGISTRY
)

DATABASE_CONNECTION_POOL = Gauge(
    "database_connection_pool_size",
    "Current database connection pool size",
    ["pool_name", "status"],
    registry=REGISTRY
)

# -------------------------
# System Health Metrics
# -------------------------

SYSTEM_HEALTH_STATUS = Gauge(
    "system_health_status",
    "Overall system health (1=healthy, 0=unhealthy)",
    registry=REGISTRY
)

DATA_FRESHNESS_SECONDS = Gauge(
    "data_freshness_seconds",
    "Age of newest data per institution",
    ["institution", "data_type"],
    registry=REGISTRY
)

ACTIVE_COLLECTION_JOBS = Gauge(
    "active_collection_jobs",
    "Number of currently active collection jobs",
    ["institution"],
    registry=REGISTRY
)

# -------------------------
# Application Metrics
# -------------------------

APPLICATION_INFO = Info(
    "application_info",
    "Application information",
    registry=REGISTRY
)

UPTIME_SECONDS = Gauge(
    "uptime_seconds",
    "Application uptime in seconds",
    registry=REGISTRY
)

CONCURRENT_REQUESTS = Gauge(
    "concurrent_requests",
    "Number of concurrent requests being processed",
    ["endpoint"],
    registry=REGISTRY
)

# -------------------------
# Helper Functions
# -------------------------

def record_speech_collection_success(institution: str, plugin_version: str = "unknown"):
    """Record successful speech collection."""
    SPEECH_COLLECTION_SUCCESS.labels(
        institution=institution,
        plugin_version=plugin_version
    ).inc()

def record_speech_collection_failure(institution: str, failure_reason: str, plugin_version: str = "unknown"):
    """Record failed speech collection."""
    SPEECH_COLLECTION_FAILURE.labels(
        institution=institution,
        failure_reason=failure_reason,
        plugin_version=plugin_version
    ).inc()

def record_speech_processing_duration(institution: str, processing_stage: str, duration: float):
    """Record speech processing duration."""
    SPEECH_PROCESSING_DURATION.labels(
        institution=institution,
        processing_stage=processing_stage
    ).observe(duration)

def record_speech_validation_failure(institution: str, validation_type: str, failure_reason: str):
    """Record speech validation failure."""
    SPEECH_VALIDATION_FAILURE.labels(
        institution=institution,
        validation_type=validation_type,
        failure_reason=failure_reason
    ).inc()

def record_nlp_analysis_duration(institution: str, processor_type: str, duration: float):
    """Record NLP analysis duration."""
    NLP_ANALYSIS_DURATION.labels(
        institution=institution,
        processor_type=processor_type
    ).observe(duration)

def record_nlp_analysis_success(institution: str, processor_type: str):
    """Record successful NLP analysis."""
    NLP_ANALYSIS_SUCCESS.labels(
        institution=institution,
        processor_type=processor_type
    ).inc()

def record_nlp_analysis_failure(institution: str, processor_type: str, error_type: str):
    """Record failed NLP analysis."""
    NLP_ANALYSIS_FAILURE.labels(
        institution=institution,
        processor_type=processor_type,
        error_type=error_type
    ).inc()

def record_sentiment_score(institution: str, sentiment_type: str, score: float):
    """Record sentiment score distribution."""
    SENTIMENT_SCORE_DISTRIBUTION.labels(
        institution=institution,
        sentiment_type=sentiment_type
    ).observe(score)

def set_plugin_health_status(institution: str, status: str):
    """Set plugin health status."""
    PLUGIN_HEALTH_STATUS.labels(institution=institution).state(status)

def record_plugin_rate_limit_hit(institution: str, endpoint: str):
    """Record plugin rate limit hit."""
    PLUGIN_RATE_LIMIT_HITS.labels(
        institution=institution,
        endpoint=endpoint
    ).inc()

def record_plugin_http_request(institution: str, method: str, status_code: int, duration: float, endpoint: str):
    """Record plugin HTTP request metrics."""
    PLUGIN_HTTP_REQUESTS.labels(
        institution=institution,
        method=method,
        status_code=str(status_code)
    ).inc()
    
    PLUGIN_HTTP_DURATION.labels(
        institution=institution,
        endpoint=endpoint
    ).observe(duration)

def record_database_operation(operation_type: str, table: str, status: str, duration: float):
    """Record database operation metrics."""
    DATABASE_OPERATIONS.labels(
        operation_type=operation_type,
        table=table,
        status=status
    ).inc()
    
    DATABASE_QUERY_DURATION.labels(
        operation_type=operation_type,
        table=table
    ).observe(duration)

def set_database_connection_pool_size(pool_name: str, status: str, size: int):
    """Set database connection pool size."""
    DATABASE_CONNECTION_POOL.labels(
        pool_name=pool_name,
        status=status
    ).set(size)

def set_system_health_status(is_healthy: bool):
    """Set overall system health status."""
    SYSTEM_HEALTH_STATUS.set(1 if is_healthy else 0)

def set_data_freshness(institution: str, data_type: str, seconds: float):
    """Set data freshness metrics."""
    DATA_FRESHNESS_SECONDS.labels(
        institution=institution,
        data_type=data_type
    ).set(seconds)

def increment_active_collection_jobs(institution: str):
    """Increment active collection jobs counter."""
    ACTIVE_COLLECTION_JOBS.labels(institution=institution).inc()

def decrement_active_collection_jobs(institution: str):
    """Decrement active collection jobs counter."""
    ACTIVE_COLLECTION_JOBS.labels(institution=institution).dec()

def set_application_info(version: str, build_date: str, commit_hash: str):
    """Set application information."""
    APPLICATION_INFO.info({
        'version': version,
        'build_date': build_date,
        'commit_hash': commit_hash
    })

def set_uptime_seconds(seconds: float):
    """Set application uptime."""
    UPTIME_SECONDS.set(seconds)

def increment_concurrent_requests(endpoint: str):
    """Increment concurrent requests counter."""
    CONCURRENT_REQUESTS.labels(endpoint=endpoint).inc()

def decrement_concurrent_requests(endpoint: str):
    """Decrement concurrent requests counter."""
    CONCURRENT_REQUESTS.labels(endpoint=endpoint).dec()

# -------------------------
# Decorators
# -------------------------

def track_processing_time(institution: str, processing_stage: str):
    """Decorator to track processing time."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                record_speech_processing_duration(institution, processing_stage, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_speech_processing_duration(institution, f"{processing_stage}_error", duration)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                record_speech_processing_duration(institution, processing_stage, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_speech_processing_duration(institution, f"{processing_stage}_error", duration)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def track_database_operation(operation_type: str, table: str):
    """Decorator to track database operations."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                record_database_operation(operation_type, table, "success", duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_database_operation(operation_type, table, "error", duration)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                record_database_operation(operation_type, table, "success", duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_database_operation(operation_type, table, "error", duration)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

@asynccontextmanager
async def track_concurrent_requests(endpoint: str):
    """Context manager to track concurrent requests."""
    increment_concurrent_requests(endpoint)
    try:
        yield
    finally:
        decrement_concurrent_requests(endpoint)

# -------------------------
# Metrics Server
# -------------------------

class MetricsServer:
    """Production-grade metrics server with health checks."""
    
    def __init__(self, port: int = 9100, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.server = None
        self.start_time = time.time()
        
    def start(self):
        """Start the Prometheus metrics server."""
        try:
            # Set application info
            import os
            set_application_info(
                version=os.getenv("APP_VERSION", "unknown"),
                build_date=os.getenv("BUILD_DATE", "unknown"),
                commit_hash=os.getenv("COMMIT_HASH", "unknown")
            )
            
            # Start server
            start_http_server(self.port, self.host, registry=REGISTRY)
            logger.info(f"Prometheus metrics server started on {self.host}:{self.port}")
            
            # Update uptime periodically
            asyncio.create_task(self._update_uptime())
            
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")
            raise
    
    async def _update_uptime(self):
        """Update uptime metrics periodically."""
        while True:
            uptime = time.time() - self.start_time
            set_uptime_seconds(uptime)
            await asyncio.sleep(60)  # Update every minute
    
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format."""
        return generate_latest(REGISTRY).decode('utf-8')
    
    def health_check(self) -> Dict[str, Any]:
        """Perform metrics system health check."""
        try:
            # Test metric collection
            test_metric = Counter('test_metric', 'Test metric', registry=REGISTRY)
            test_metric.inc()
            
            return {
                "status": "healthy",
                "uptime_seconds": time.time() - self.start_time,
                "metrics_count": len(list(REGISTRY._collector_to_names.keys())),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Metrics health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# -------------------------
# Initialization
# -------------------------

# Global metrics server instance
metrics_server = MetricsServer()

def start_metrics_server(port: int = 9100, host: str = "0.0.0.0"):
    """Start the metrics server."""
    global metrics_server
    metrics_server = MetricsServer(port, host)
    metrics_server.start()

def get_metrics_server() -> MetricsServer:
    """Get the global metrics server instance."""
    return metrics_server

# -------------------------
# Usage Examples
# -------------------------

# Example usage in application code:
#
# # Basic metrics
# record_speech_collection_success("ECB", "v1.2.0")
# record_nlp_analysis_duration("ECB", "sentiment", 2.5)
# set_plugin_health_status("ECB", "healthy")
#
# # Using decorators
# @track_processing_time("ECB", "extraction")
# async def extract_speech_content(url: str) -> str:
#     # Speech extraction logic
#     pass
#
# @track_database_operation("insert", "speeches")
# async def save_speech(speech: CentralBankSpeech) -> None:
#     # Database save logic
#     pass
#
# # Using context managers
# async with track_concurrent_requests("/api/speeches"):
#     # Handle API request
#     pass