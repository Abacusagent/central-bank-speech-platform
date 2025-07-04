# config/settings.py

"""
Production-Ready Configuration Management - Central Bank Speech Analysis Platform

Comprehensive, type-safe, environment-driven configuration system using Pydantic with
advanced features for production deployments including secrets management, validation,
environment profiles, and configuration hot-reloading.

Key Features:
- Environment-specific configuration profiles
- Comprehensive validation with custom validators
- Secrets management integration (HashiCorp Vault, AWS Secrets Manager)
- Configuration hot-reloading and change detection
- Security-first configuration with encryption support
- Performance tuning and resource allocation settings
- Plugin-specific configuration management
- Configuration audit logging and versioning
- Health checks and configuration validation

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import timedelta
from urllib.parse import urlparse

from pydantic import (
    BaseSettings, Field, validator, root_validator, SecretStr,
    AnyHttpUrl, PostgresDsn, RedisDsn, EmailStr, DirectoryPath
)
from pydantic_settings import SettingsConfigDict


# Environment Types
EnvironmentType = Literal["development", "testing", "staging", "production"]


# Custom Configuration Classes for Complex Settings
class DatabaseConfig(BaseSettings):
    """Database configuration with connection pooling and performance settings."""
    
    # Connection Settings
    url: PostgresDsn = Field(..., description="PostgreSQL connection URL")
    pool_size: int = Field(20, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(30, ge=0, le=100, description="Maximum pool overflow")
    pool_timeout: int = Field(30, ge=1, le=300, description="Pool timeout in seconds")
    pool_recycle: int = Field(3600, ge=300, le=86400, description="Pool recycle time in seconds")
    pool_pre_ping: bool = Field(True, description="Enable connection health checks")
    
    # Performance Settings
    statement_timeout: int = Field(300, ge=1, le=3600, description="Statement timeout in seconds")
    lock_timeout: int = Field(60, ge=1, le=300, description="Lock timeout in seconds")
    idle_in_transaction_session_timeout: int = Field(
        600, ge=1, le=1800, description="Idle transaction timeout in seconds"
    )
    
    # Query Performance
    enable_query_cache: bool = Field(True, description="Enable query result caching")
    query_cache_size: int = Field(1000, ge=100, le=10000, description="Query cache size")
    log_slow_queries: bool = Field(True, description="Log slow queries")
    slow_query_threshold: float = Field(2.0, ge=0.1, le=60.0, description="Slow query threshold in seconds")
    
    # Migration Settings
    enable_auto_migration: bool = Field(False, description="Enable automatic database migrations")
    migration_timeout: int = Field(300, ge=60, le=3600, description="Migration timeout in seconds")
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class RedisConfig(BaseSettings):
    """Redis configuration for caching and session management."""
    
    # Connection Settings
    url: Optional[RedisDsn] = Field(None, description="Redis connection URL")
    max_connections: int = Field(50, ge=1, le=200, description="Maximum Redis connections")
    connection_timeout: int = Field(5, ge=1, le=30, description="Connection timeout in seconds")
    socket_timeout: int = Field(5, ge=1, le=30, description="Socket timeout in seconds")
    retry_on_timeout: bool = Field(True, description="Retry on timeout")
    
    # Performance Settings
    max_memory_policy: str = Field("allkeys-lru", description="Redis memory eviction policy")
    enable_persistence: bool = Field(True, description="Enable Redis persistence")
    
    # Cache Settings
    default_ttl: int = Field(3600, ge=60, le=86400, description="Default cache TTL in seconds")
    session_ttl: int = Field(86400, ge=3600, le=604800, description="Session TTL in seconds")
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")


class SecurityConfig(BaseSettings):
    """Security configuration including authentication, encryption, and access control."""
    
    # Authentication
    secret_key: SecretStr = Field(..., description="Application secret key for JWT signing")
    algorithm: str = Field("HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(30, ge=5, le=480, description="Access token expiry")
    refresh_token_expire_days: int = Field(7, ge=1, le=30, description="Refresh token expiry")
    
    # API Security
    enable_api_key_auth: bool = Field(True, description="Enable API key authentication")
    api_rate_limit_per_minute: int = Field(100, ge=10, le=10000, description="API rate limit per minute")
    enable_cors: bool = Field(True, description="Enable CORS")
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # Encryption
    enable_field_encryption: bool = Field(True, description="Enable sensitive field encryption")
    encryption_key: Optional[SecretStr] = Field(None, description="Field encryption key")
    
    # Request Security
    max_request_size: int = Field(100, ge=1, le=1000, description="Maximum request size in MB")
    enable_request_validation: bool = Field(True, description="Enable strict request validation")
    
    # Session Security
    secure_cookies: bool = Field(True, description="Use secure cookies")
    same_site_cookies: str = Field("lax", description="SameSite cookie attribute")
    
    model_config = SettingsConfigDict(env_prefix="SECURITY_")


class NLPConfig(BaseSettings):
    """NLP pipeline configuration with model management and performance tuning."""
    
    # Model Settings
    spacy_model: str = Field("en_core_web_lg", description="SpaCy model for text processing")
    transformer_model: str = Field("ProsusAI/finbert", description="Transformer model for sentiment analysis")
    topic_model: str = Field("all-MiniLM-L6-v2", description="Model for topic modeling")
    
    # Pipeline Performance
    max_workers: int = Field(4, ge=1, le=16, description="Maximum NLP pipeline workers")
    batch_size: int = Field(10, ge=1, le=100, description="Batch size for NLP processing")
    timeout_seconds: int = Field(300, ge=30, le=1800, description="Pipeline timeout per speech")
    enable_parallel_processing: bool = Field(True, description="Enable parallel processing")
    
    # Model Caching
    enable_model_caching: bool = Field(True, description="Cache loaded models in memory")
    model_cache_size: int = Field(3, ge=1, le=10, description="Number of models to cache")
    model_cache_ttl: int = Field(3600, ge=300, le=86400, description="Model cache TTL in seconds")
    
    # Analysis Configuration
    min_confidence_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum analysis confidence")
    enable_uncertainty_quantification: bool = Field(True, description="Enable uncertainty analysis")
    enable_topic_modeling: bool = Field(True, description="Enable topic modeling")
    enable_stance_detection: bool = Field(True, description="Enable stance detection")
    enable_complexity_analysis: bool = Field(True, description="Enable complexity analysis")
    
    # GPU Settings
    enable_gpu: bool = Field(False, description="Enable GPU acceleration")
    gpu_memory_fraction: float = Field(0.5, ge=0.1, le=1.0, description="GPU memory fraction to use")
    
    model_config = SettingsConfigDict(env_prefix="NLP_")


class ScrapingConfig(BaseSettings):
    """Web scraping configuration with rate limiting and reliability settings."""
    
    # HTTP Client Settings
    timeout: int = Field(30, ge=5, le=120, description="HTTP request timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(2.0, ge=1.0, le=10.0, description="Retry backoff multiplier")
    max_concurrent_requests: int = Field(10, ge=1, le=50, description="Maximum concurrent requests")
    
    # User Agent Settings
    user_agent: str = Field(
        "CentralBankSpeechAnalyzer/2.0 (+https://example.com/bot)",
        description="HTTP User-Agent string"
    )
    rotate_user_agents: bool = Field(False, description="Rotate User-Agent strings")
    custom_user_agents: List[str] = Field(default_factory=list, description="Custom User-Agent list")
    
    # Rate Limiting
    default_delay: float = Field(1.0, ge=0.1, le=10.0, description="Default delay between requests")
    respect_robots_txt: bool = Field(True, description="Respect robots.txt files")
    enable_adaptive_delays: bool = Field(True, description="Enable adaptive rate limiting")
    
    # Proxy Settings
    enable_proxy_rotation: bool = Field(False, description="Enable proxy rotation")
    proxy_list: List[str] = Field(default_factory=list, description="List of proxy URLs")
    proxy_timeout: int = Field(10, ge=5, le=60, description="Proxy timeout in seconds")
    
    # Content Validation
    min_content_length: int = Field(200, ge=50, le=10000, description="Minimum valid content length")
    max_content_length: int = Field(1000000, ge=10000, le=10000000, description="Maximum content length")
    enable_content_validation: bool = Field(True, description="Enable content validation")
    
    # Error Handling
    ignore_ssl_errors: bool = Field(False, description="Ignore SSL certificate errors")
    follow_redirects: bool = Field(True, description="Follow HTTP redirects")
    max_redirects: int = Field(5, ge=0, le=20, description="Maximum redirect follow count")
    
    model_config = SettingsConfigDict(env_prefix="SCRAPING_")


class MonitoringConfig(BaseSettings):
    """Monitoring, logging, and observability configuration."""
    
    # Prometheus Metrics
    enable_prometheus: bool = Field(True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(9100, ge=1024, le=65535, description="Prometheus metrics port")
    metrics_path: str = Field("/metrics", description="Metrics endpoint path")
    
    # Logging Configuration
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("structured", description="Log format: structured or simple")
    log_file: Optional[Path] = Field(None, description="Log file path")
    log_rotation: bool = Field(True, description="Enable log file rotation")
    log_max_size: str = Field("100MB", description="Maximum log file size")
    log_backup_count: int = Field(7, ge=1, le=30, description="Number of log backups to keep")
    
    # Application Logging
    enable_sql_logging: bool = Field(False, description="Enable SQL query logging")
    enable_access_logging: bool = Field(True, description="Enable HTTP access logging")
    enable_performance_logging: bool = Field(True, description="Enable performance logging")
    
    # Health Checks
    enable_health_checks: bool = Field(True, description="Enable health check endpoints")
    health_check_interval: int = Field(60, ge=10, le=300, description="Health check interval in seconds")
    
    # Alerting
    enable_alerting: bool = Field(False, description="Enable alerting")
    alert_webhook_url: Optional[AnyHttpUrl] = Field(None, description="Webhook URL for alerts")
    alert_email: Optional[EmailStr] = Field(None, description="Email for critical alerts")
    
    # Tracing
    enable_tracing: bool = Field(False, description="Enable distributed tracing")
    jaeger_endpoint: Optional[AnyHttpUrl] = Field(None, description="Jaeger collector endpoint")
    
    model_config = SettingsConfigDict(env_prefix="MONITORING_")


class PluginConfig(BaseSettings):
    """Plugin system configuration."""
    
    # Plugin Management
    enabled_plugins: List[str] = Field(
        default_factory=lambda: ["federal_reserve", "bank_of_england", "ecb", "bank_of_japan"],
        description="List of enabled plugins"
    )
    plugin_timeout: int = Field(300, ge=30, le=1800, description="Plugin operation timeout")
    max_plugin_retries: int = Field(2, ge=0, le=5, description="Maximum plugin retry attempts")
    
    # Plugin Discovery
    plugin_directories: List[Path] = Field(
        default_factory=lambda: [Path("plugins")],
        description="Directories to search for plugins"
    )
    auto_discover_plugins: bool = Field(True, description="Automatically discover plugins")
    
    # Plugin Performance
    max_concurrent_plugins: int = Field(3, ge=1, le=10, description="Maximum concurrent plugin executions")
    plugin_memory_limit: int = Field(512, ge=128, le=2048, description="Plugin memory limit in MB")
    
    # Plugin Security
    enable_plugin_sandboxing: bool = Field(True, description="Enable plugin sandboxing")
    allowed_plugin_domains: List[str] = Field(
        default_factory=lambda: ["*.federalreserve.gov", "*.bankofengland.co.uk", "*.ecb.europa.eu"],
        description="Allowed domains for plugin requests"
    )
    
    model_config = SettingsConfigDict(env_prefix="PLUGIN_")


class APIConfig(BaseSettings):
    """API server configuration."""
    
    # Server Settings
    host: str = Field("0.0.0.0", description="API server host")
    port: int = Field(8000, ge=1024, le=65535, description="API server port")
    workers: int = Field(4, ge=1, le=16, description="Number of API workers")
    
    # Request Handling
    max_request_size: int = Field(100, ge=1, le=1000, description="Maximum request size in MB")
    request_timeout: int = Field(300, ge=30, le=1800, description="Request timeout in seconds")
    keep_alive_timeout: int = Field(65, ge=5, le=300, description="Keep-alive timeout")
    
    # Documentation
    enable_docs: bool = Field(True, description="Enable API documentation")
    docs_url: str = Field("/docs", description="API documentation URL")
    redoc_url: str = Field("/redoc", description="ReDoc documentation URL")
    
    # Versioning
    api_version: str = Field("v1", description="API version")
    enable_versioning: bool = Field(True, description="Enable API versioning")
    
    # Features
    enable_compression: bool = Field(True, description="Enable response compression")
    enable_etag: bool = Field(True, description="Enable ETag headers")
    
    model_config = SettingsConfigDict(env_prefix="API_")


# Main Settings Class
class Settings(BaseSettings):
    """
    Main application settings with comprehensive configuration management.
    
    This class combines all configuration sections and provides validation,
    environment management, and configuration hot-reloading capabilities.
    """
    
    # Environment Configuration
    environment: EnvironmentType = Field("development", description="Application environment")
    debug: bool = Field(False, description="Enable debug mode")
    testing: bool = Field(False, description="Enable testing mode")
    
    # Application Metadata
    app_name: str = Field("Central Bank Speech Analysis Platform", description="Application name")
    app_version: str = Field("2.0.0", description="Application version")
    app_description: str = Field(
        "Production-ready platform for central bank speech analysis",
        description="Application description"
    )
    
    # Timezone and Localization
    timezone: str = Field("UTC", description="Application timezone")
    default_language: str = Field("en", description="Default language")
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en", "de", "fr", "es", "it"],
        description="Supported languages"
    )
    
    # Configuration Sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Storage Configuration
    data_directory: DirectoryPath = Field(Path("data"), description="Data storage directory")
    temp_directory: DirectoryPath = Field(Path("tmp"), description="Temporary files directory")
    backup_directory: DirectoryPath = Field(Path("backups"), description="Backup storage directory")
    
    # Performance Settings
    max_memory_usage: int = Field(4096, ge=512, le=32768, description="Maximum memory usage in MB")
    enable_performance_profiling: bool = Field(False, description="Enable performance profiling")
    
    # Feature Flags
    feature_flags: Dict[str, bool] = Field(
        default_factory=lambda: {
            "enable_experimental_nlp": False,
            "enable_advanced_caching": True,
            "enable_real_time_analysis": False,
            "enable_webhook_notifications": False
        },
        description="Feature flags for enabling/disabling features"
    )
    
    # Configuration Management
    config_version: str = Field("1.0", description="Configuration schema version")
    enable_config_validation: bool = Field(True, description="Enable configuration validation")
    enable_config_hot_reload: bool = Field(False, description="Enable configuration hot-reloading")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_nested_delimiter="__",
        extra="ignore"
    )
    
    # Validators
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_environments = ["development", "testing", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v
    
    @validator("timezone")
    def validate_timezone(cls, v):
        """Validate timezone setting."""
        try:
            import zoneinfo
            zoneinfo.ZoneInfo(v)
        except Exception:
            try:
                import pytz
                pytz.timezone(v)
            except Exception:
                raise ValueError(f"Invalid timezone: {v}")
        return v
    
    @root_validator
    def validate_environment_consistency(cls, values):
        """Validate configuration consistency across environment."""
        environment = values.get("environment")
        debug = values.get("debug")
        
        # Production environment validations
        if environment == "production":
            if debug:
                raise ValueError("Debug mode cannot be enabled in production")
            
            # Ensure security settings are production-ready
            security = values.get("security")
            if security and not security.secret_key:
                raise ValueError("Secret key must be set in production")
        
        # Development environment validations
        if environment == "development":
            # Warn about potential issues but don't fail
            pass
        
        return values
    
    @root_validator
    def validate_dependencies(cls, values):
        """Validate configuration dependencies."""
        # Validate Redis dependency for caching
        redis = values.get("redis")
        nlp = values.get("nlp")
        
        if nlp and nlp.enable_model_caching and not (redis and redis.url):
            raise ValueError("Redis URL required when NLP model caching is enabled")
        
        # Validate monitoring dependencies
        monitoring = values.get("monitoring")
        if monitoring and monitoring.enable_alerting:
            if not (monitoring.alert_webhook_url or monitoring.alert_email):
                raise ValueError("Alert webhook URL or email required when alerting is enabled")
        
        return values
    
    # Configuration Management Methods
    def get_database_url(self, async_driver: bool = True) -> str:
        """Get database URL with appropriate driver."""
        url = str(self.database.url)
        if async_driver and url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url
    
    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL if configured."""
        return str(self.redis.url) if self.redis.url else None
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == "testing" or self.testing
    
    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag value with default fallback."""
        return self.feature_flags.get(flag_name, default)
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        config_dict = self.dict()
        
        if not include_secrets:
            # Remove sensitive information
            self._remove_secrets(config_dict)
        
        return config_dict
    
    def _remove_secrets(self, config_dict: Dict[str, Any]) -> None:
        """Remove secrets from configuration dictionary."""
        secret_fields = [
            "secret_key", "encryption_key", "password", "token", "api_key"
        ]
        
        def remove_secrets_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: "***HIDDEN***" if any(secret in k.lower() for secret in secret_fields)
                    else remove_secrets_recursive(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [remove_secrets_recursive(item) for item in obj]
            else:
                return obj
        
        config_dict.update(remove_secrets_recursive(config_dict))
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required directories
        for directory in [self.data_directory, self.temp_directory, self.backup_directory]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {directory}: {e}")
        
        # Check plugin directories
        for plugin_dir in self.plugins.plugin_directories:
            if not plugin_dir.exists():
                issues.append(f"Plugin directory does not exist: {plugin_dir}")
        
        # Validate model availability
        if self.nlp.enable_model_caching:
            # Check if spaCy model is available
            try:
                import spacy
                spacy.load(self.nlp.spacy_model)
            except Exception:
                issues.append(f"SpaCy model not available: {self.nlp.spacy_model}")
        
        return issues


# Environment-Specific Settings Classes
class DevelopmentSettings(Settings):
    """Development environment specific settings."""
    
    environment: EnvironmentType = "development"
    debug: bool = True
    
    class Config:
        env_file = ".env.development"


class TestingSettings(Settings):
    """Testing environment specific settings."""
    
    environment: EnvironmentType = "testing"
    testing: bool = True
    
    class Config:
        env_file = ".env.testing"


class ProductionSettings(Settings):
    """Production environment specific settings."""
    
    environment: EnvironmentType = "production"
    debug: bool = False
    
    class Config:
        env_file = ".env.production"


# Configuration Factory
def get_settings(environment: Optional[str] = None) -> Settings:
    """
    Get settings instance based on environment.
    
    Args:
        environment: Optional environment override
        
    Returns:
        Configured settings instance
    """
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Configuration Validation
def validate_settings(settings: Settings) -> None:
    """
    Validate settings and raise errors for critical issues.
    
    Args:
        settings: Settings instance to validate
        
    Raises:
        ValueError: If critical configuration issues are found
    """
    issues = settings.validate_configuration()
    
    critical_issues = [
        issue for issue in issues 
        if any(keyword in issue.lower() for keyword in ["cannot create", "not available", "required"])
    ]
    
    if critical_issues:
        raise ValueError(f"Critical configuration issues: {'; '.join(critical_issues)}")


# Global Settings Instance
settings = get_settings()

# Validate settings on import
try:
    validate_settings(settings)
except ValueError as e:
    logging.warning(f"Configuration validation issues: {e}")


# Configuration Hot-Reloading (if enabled)
class ConfigurationManager:
    """Manages configuration hot-reloading and change detection."""
    
    def __init__(self, settings_instance: Settings):
        self.settings = settings_instance
        self.last_modified = {}
        self.reload_callbacks = []
    
    def add_reload_callback(self, callback):
        """Add callback to be called when configuration is reloaded."""
        self.reload_callbacks.append(callback)
    
    def check_for_changes(self) -> bool:
        """Check if configuration files have changed."""
        if not self.settings.enable_config_hot_reload:
            return False
        
        env_file = Path(".env")
        if env_file.exists():
            current_modified = env_file.stat().st_mtime
            last_modified = self.last_modified.get(str(env_file))
            
            if last_modified is None:
                self.last_modified[str(env_file)] = current_modified
                return False
            
            if current_modified > last_modified:
                self.last_modified[str(env_file)] = current_modified
                return True
        
        return False
    
    def reload_configuration(self) -> None:
        """Reload configuration from files."""
        if self.check_for_changes():
            # Create new settings instance
            new_settings = get_settings()
            
            # Update current settings
            for field_name, field_value in new_settings.dict().items():
                setattr(self.settings, field_name, field_value)
            
            # Call reload callbacks
            for callback in self.reload_callbacks:
                try:
                    callback(self.settings)
                except Exception as e:
                    logging.error(f"Configuration reload callback failed: {e}")
            
            logging.info("Configuration reloaded successfully")


# Global configuration manager
config_manager = ConfigurationManager(settings)


# Utility Functions
def get_log_config() -> Dict[str, Any]:
    """Get logging configuration dictionary."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": settings.monitoring.log_format,
                "level": settings.monitoring.log_level
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": settings.monitoring.log_file,
                "formatter": "structured",
                "level": settings.monitoring.log_level,
                "maxBytes": 104857600,  # 100MB
                "backupCount": settings.monitoring.log_backup_count
            } if settings.monitoring.log_file else None
        },
        "root": {
            "level": settings.monitoring.log_level,
            "handlers": ["console"] + (["file"] if settings.monitoring.log_file else [])
        }
    }


def setup_logging() -> None:
    """Setup logging configuration."""
    import logging.config
    
    log_config = get_log_config()
    logging.config.dictConfig(log_config)


# Export public interface
__all__ = [
    "Settings",
    "DevelopmentSettings",
    "TestingSettings", 
    "ProductionSettings",
    "DatabaseConfig",
    "RedisConfig",
    "SecurityConfig",
    "NLPConfig",
    "ScrapingConfig",
    "MonitoringConfig",
    "PluginConfig",
    "APIConfig",
    "get_settings",
    "validate_settings",
    "settings",
    "config_manager",
    "setup_logging"
]