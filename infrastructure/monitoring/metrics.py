# infrastructure/monitoring/metrics.py

"""
Metrics and Observability Utilities for Central Bank Speech Analysis Platform

Provides Prometheus-compatible metrics for speech collection, processing,
plugin events, error rates, and system health.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import logging
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

logger = logging.getLogger(__name__)

# --------------------
# Metric Definitions
# --------------------

COLLECTION_SUCCESS = Counter(
    "speech_collection_success_total",
    "Number of successfully collected speeches",
    ["institution"]
)

COLLECTION_FAILURE = Counter(
    "speech_collection_failure_total",
    "Number of failed speech collection attempts",
    ["institution", "reason"]
)

PROCESSING_TIME = Histogram(
    "speech_processing_time_seconds",
    "Time spent processing a speech (including NLP pipeline)",
    ["institution"]
)

VALIDATION_FAILURE = Counter(
    "speech_validation_failure_total",
    "Number of speech validation failures",
    ["institution", "failure_type"]
)

PLUGIN_FAILURE = Counter(
    "plugin_failure_total",
    "Number of plugin failures/errors",
    ["institution", "plugin_error"]
)

DATA_FRESHNESS = Gauge(
    "speech_data_freshness_seconds",
    "Age (seconds) of newest collected speech per institution",
    ["institution"]
)

SYSTEM_HEALTH = Gauge(
    "system_health",
    "1 if platform healthy, 0 otherwise"
)

# ----------------------
# Metric Helper Methods
# ----------------------

def record_collection_success(institution: str):
    COLLECTION_SUCCESS.labels(institution=institution).inc()

def record_collection_failure(institution: str, reason: str):
    COLLECTION_FAILURE.labels(institution=institution, reason=reason).inc()

def record_processing_time(institution: str, seconds: float):
    PROCESSING_TIME.labels(institution=institution).observe(seconds)

def record_validation_failure(institution: str, failure_type: str):
    VALIDATION_FAILURE.labels(institution=institution, failure_type=failure_type).inc()

def record_plugin_failure(institution: str, plugin_error: str):
    PLUGIN_FAILURE.labels(institution=institution, plugin_error=plugin_error).inc()

def set_data_freshness(institution: str, seconds: float):
    DATA_FRESHNESS.labels(institution=institution).set(seconds)

def set_system_health(is_healthy: bool):
    SYSTEM_HEALTH.set(1 if is_healthy else 0)

def start_metrics_server(port: int = 9100):
    """Starts the Prometheus HTTP metrics server (exposes /metrics endpoint)."""
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {e}")

# ----------------------
# Usage Example:
# ----------------------
# from infrastructure.monitoring.metrics import record_collection_success, start_metrics_server
# start_metrics_server(9100)
# record_collection_success("ECB")
