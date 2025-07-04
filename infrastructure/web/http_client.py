# infrastructure/web/http_client.py

"""
Production-Grade HTTP Client for Central Bank Speech Analysis Platform

Provides robust, async HTTP access with comprehensive rate limiting, retry logic,
session management, metrics integration, and plugin-aware request handling.

Key Features:
- Adaptive rate limiting with per-domain policies
- Connection pooling and session management
- Comprehensive metrics integration
- User-Agent rotation and proxy support
- Circuit breaker pattern for reliability
- Robots.txt compliance
- Content validation and filtering

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import asyncio
import logging
import time
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple, AsyncIterator
from urllib.parse import urljoin, urlparse, robots
from urllib.robotparser import RobotFileParser

import httpx
import aiofiles
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)

# Import metrics for integration
from infrastructure.monitoring.metrics import (
    record_plugin_http_request, record_plugin_rate_limit_hit,
    record_plugin_failure
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
DEFAULT_RATE_LIMIT = 1.0  # seconds between requests
DEFAULT_MAX_CONCURRENT = 10
DEFAULT_USER_AGENTS = [
    'CentralBankSpeechAnalyzer/2.0 (+https://example.com/bot)',
    'Mozilla/5.0 (compatible; CentralBankBot/2.0; +https://example.com)',
    'CentralBankResearch/2.0 (Academic Research; contact@example.com)'
]

# Custom exceptions
class HttpClientError(Exception):
    """Base exception for HTTP client errors."""
    pass

class RateLimitExceededError(HttpClientError):
    """Raised when rate limit is exceeded."""
    pass

class ContentValidationError(HttpClientError):
    """Raised when content validation fails."""
    pass

class CircuitBreakerOpenError(HttpClientError):
    """Raised when circuit breaker is open."""
    pass

class RobotsDisallowedError(HttpClientError):
    """Raised when robots.txt disallows access."""
    pass

@dataclass
class RateLimitPolicy:
    """Rate limiting policy for a domain."""
    requests_per_second: float = 1.0
    burst_size: int = 5
    adaptive: bool = True
    respect_retry_after: bool = True
    backoff_factor: float = 1.5
    max_delay: float = 300.0

@dataclass 
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half-open
    failure_threshold: int = 5
    recovery_timeout: float = 60.0

@dataclass
class RequestMetrics:
    """Request metrics tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: float = 0.0

class DomainRateLimiter:
    """Per-domain rate limiter with adaptive behavior."""
    
    def __init__(self, policy: RateLimitPolicy):
        self.policy = policy
        self.request_times: List[float] = []
        self.last_request_time = 0.0
        self.current_delay = 1.0 / policy.requests_per_second
        self.consecutive_errors = 0
        
    async def acquire(self, institution: str = "unknown") -> None:
        """Acquire permission to make a request."""
        now = time.time()
        
        # Clean old request times (outside burst window)
        burst_window = self.policy.burst_size / self.policy.requests_per_second
        self.request_times = [t for t in self.request_times if now - t < burst_window]
        
        # Check burst limit
        if len(self.request_times) >= self.policy.burst_size:
            sleep_time = self.request_times[0] + burst_window - now
            if sleep_time > 0:
                logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s for burst control")
                await asyncio.sleep(sleep_time)
                record_plugin_rate_limit_hit(institution, "burst_limit")
        
        # Check minimum delay between requests
        time_since_last = now - self.last_request_time
        if time_since_last < self.current_delay:
            sleep_time = self.current_delay - time_since_last
            logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s for rate control")
            await asyncio.sleep(sleep_time)
            record_plugin_rate_limit_hit(institution, "rate_limit")
        
        # Record this request
        self.last_request_time = time.time()
        self.request_times.append(self.last_request_time)
    
    def adapt_for_error(self, status_code: int, retry_after: Optional[int] = None) -> None:
        """Adapt rate limiting based on error response."""
        if not self.policy.adaptive:
            return
            
        self.consecutive_errors += 1
        
        if status_code == 429:  # Too Many Requests
            if retry_after and self.policy.respect_retry_after:
                self.current_delay = min(retry_after, self.policy.max_delay)
            else:
                self.current_delay = min(
                    self.current_delay * self.policy.backoff_factor,
                    self.policy.max_delay
                )
        elif status_code >= 500:  # Server errors
            self.current_delay = min(
                self.current_delay * 1.2,  # Gentle increase
                self.policy.max_delay
            )
    
    def adapt_for_success(self) -> None:
        """Adapt rate limiting based on successful response."""
        if not self.policy.adaptive:
            return
            
        self.consecutive_errors = 0
        
        # Gradually decrease delay if we've been successful
        if self.consecutive_errors == 0:
            base_delay = 1.0 / self.policy.requests_per_second
            self.current_delay = max(
                self.current_delay * 0.95,  # Gentle decrease
                base_delay
            )

class RobotsChecker:
    """Robots.txt compliance checker with caching."""
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache: Dict[str, Tuple[RobotFileParser, float]] = {}
        self.cache_ttl = cache_ttl
    
    async def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        # Check cache
        if robots_url in self.cache:
            rp, cached_time = self.cache[robots_url]
            if time.time() - cached_time < self.cache_ttl:
                return rp.can_fetch(user_agent, url)
        
        # Fetch and parse robots.txt
        try:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            
            # Use basic httpx for robots.txt to avoid circular dependency
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(robots_url)
                if response.status_code == 200:
                    rp.set_content(response.text)
                else:
                    # If no robots.txt, allow all
                    return True
            
            self.cache[robots_url] = (rp, time.time())
            return rp.can_fetch(user_agent, url)
            
        except Exception as e:
            logger.warning(f"Failed to check robots.txt for {robots_url}: {e}")
            return True  # Allow if can't check

class EnhancedHttpClient:
    """
    Production-grade async HTTP client with comprehensive features for web scraping.
    
    Features:
    - Per-domain rate limiting with adaptive behavior
    - Circuit breaker pattern for reliability
    - Session management and connection pooling
    - User-Agent rotation and proxy support
    - Robots.txt compliance checking
    - Content validation and filtering
    - Comprehensive metrics integration
    - Retry logic with exponential backoff
    """
    
    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        user_agents: Optional[List[str]] = None,
        proxies: Optional[List[str]] = None,
        respect_robots: bool = True,
        default_rate_limit: float = DEFAULT_RATE_LIMIT,
        enable_metrics: bool = True
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.user_agents = user_agents or DEFAULT_USER_AGENTS
        self.proxies = proxies or []
        self.respect_robots = respect_robots
        self.enable_metrics = enable_metrics
        
        # Per-domain state management
        self.rate_limiters: Dict[str, DomainRateLimiter] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.request_metrics: Dict[str, RequestMetrics] = {}
        
        # Global state
        self.robots_checker = RobotsChecker() if respect_robots else None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.current_ua_index = 0
        self.current_proxy_index = 0
        
        # Session management
        self.session: Optional[httpx.AsyncClient] = None
        self.session_lock = asyncio.Lock()
        
        # Default rate limit policy
        self.default_policy = RateLimitPolicy(requests_per_second=1.0/default_rate_limit)
        
        logger.info(f"Initialized HTTP client with {max_concurrent} max concurrent requests")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is initialized."""
        if self.session is None:
            async with self.session_lock:
                if self.session is None:
                    limits = httpx.Limits(
                        max_keepalive_connections=20,
                        max_connections=100,
                        keepalive_expiry=30.0
                    )
                    
                    self.session = httpx.AsyncClient(
                        timeout=httpx.Timeout(self.timeout),
                        limits=limits,
                        follow_redirects=True,
                        verify=True
                    )
    
    async def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc
    
    def _get_rate_limiter(self, domain: str) -> DomainRateLimiter:
        """Get or create rate limiter for domain."""
        if domain not in self.rate_limiters:
            self.rate_limiters[domain] = DomainRateLimiter(self.default_policy)
        return self.rate_limiters[domain]
    
    def _get_circuit_breaker(self, domain: str) -> CircuitBreakerState:
        """Get or create circuit breaker for domain."""
        if domain not in self.circuit_breakers:
            self.circuit_breakers[domain] = CircuitBreakerState()
        return self.circuit_breakers[domain]
    
    def _get_request_metrics(self, domain: str) -> RequestMetrics:
        """Get or create request metrics for domain."""
        if domain not in self.request_metrics:
            self.request_metrics[domain] = RequestMetrics()
        return self.request_metrics[domain]
    
    def _get_next_user_agent(self) -> str:
        """Get next User-Agent in rotation."""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua
    
    def _get_next_proxy(self) -> Optional[str]:
        """Get next proxy in rotation."""
        if not self.proxies:
            return None
        proxy = self.proxies[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        return proxy
    
    async def _check_circuit_breaker(self, domain: str) -> None:
        """Check circuit breaker state."""
        cb = self._get_circuit_breaker(domain)
        
        if cb.state == "open":
            if time.time() - cb.last_failure_time > cb.recovery_timeout:
                cb.state = "half-open"
                logger.info(f"Circuit breaker half-open for domain {domain}")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker open for domain {domain}")
    
    def _update_circuit_breaker(self, domain: str, success: bool) -> None:
        """Update circuit breaker state based on request result."""
        cb = self._get_circuit_breaker(domain)
        
        if success:
            if cb.state == "half-open":
                cb.state = "closed"
                cb.failure_count = 0
                logger.info(f"Circuit breaker closed for domain {domain}")
        else:
            cb.failure_count += 1
            cb.last_failure_time = time.time()
            
            if cb.failure_count >= cb.failure_threshold and cb.state == "closed":
                cb.state = "open"
                logger.warning(f"Circuit breaker opened for domain {domain}")
    
    async def _validate_content(self, content: Union[str, bytes], url: str) -> None:
        """Validate response content."""
        content_length = len(content)
        
        # Basic length validation
        if content_length < 200:  # Configurable minimum
            raise ContentValidationError(f"Content too short ({content_length} bytes) for {url}")
        
        if content_length > 10_000_000:  # 10MB max
            raise ContentValidationError(f"Content too large ({content_length} bytes) for {url}")
        
        # Additional validation can be added here
        if isinstance(content, str):
            # Check for common error patterns
            error_indicators = [
                "404 not found", "403 forbidden", "500 internal server error",
                "access denied", "rate limit exceeded", "temporarily unavailable"
            ]
            content_lower = content.lower()
            for indicator in error_indicators:
                if indicator in content_lower and len(content) < 1000:
                    raise ContentValidationError(f"Content appears to be error page for {url}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        institution: str = "unknown",
        validate_content: bool = True,
        return_bytes: bool = False,
        **kwargs
    ) -> Union[str, bytes]:
        """
        Perform GET request with comprehensive error handling and metrics.
        
        Args:
            url: Target URL
            params: Query parameters
            headers: Additional headers
            institution: Institution name for metrics
            validate_content: Whether to validate response content
            return_bytes: Return bytes instead of text
            **kwargs: Additional httpx parameters
        
        Returns:
            Response content as string or bytes
            
        Raises:
            HttpClientError: For various HTTP-related errors
        """
        await self._ensure_session()
        domain = self._get_domain(url)
        
        # Check circuit breaker
        await self._check_circuit_breaker(domain)
        
        # Check robots.txt
        if self.robots_checker:
            user_agent = headers.get('User-Agent', self._get_next_user_agent())
            if not await self.robots_checker.can_fetch(url, user_agent):
                raise RobotsDisallowedError(f"Robots.txt disallows access to {url}")
        
        # Rate limiting
        rate_limiter = self._get_rate_limiter(domain)
        await rate_limiter.acquire(institution)
        
        # Prepare request
        request_headers = {
            'User-Agent': self._get_next_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        if headers:
            request_headers.update(headers)
        
        # Proxy selection
        proxy = self._get_next_proxy()
        if proxy:
            kwargs['proxies'] = {"http://": proxy, "https://": proxy}
        
        start_time = time.time()
        response = None
        
        try:
            async with self.semaphore:
                response = await self.session.get(
                    url,
                    params=params,
                    headers=request_headers,
                    **kwargs
                )
                
                # Handle rate limiting responses
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    retry_after_int = int(retry_after) if retry_after else None
                    rate_limiter.adapt_for_error(429, retry_after_int)
                    
                    if retry_after_int:
                        logger.warning(f"Rate limited, waiting {retry_after_int}s")
                        await asyncio.sleep(retry_after_int)
                    
                    raise RateLimitExceededError(f"Rate limited for {url}")
                
                response.raise_for_status()
                
                # Get content
                content = response.content if return_bytes else response.text
                
                # Validate content
                if validate_content:
                    await self._validate_content(content, url)
                
                # Update metrics and state
                duration = time.time() - start_time
                if self.enable_metrics:
                    record_plugin_http_request(
                        institution, "GET", response.status_code, duration, domain
                    )
                
                rate_limiter.adapt_for_success()
                self._update_circuit_breaker(domain, True)
                
                # Update request metrics
                metrics = self._get_request_metrics(domain)
                metrics.total_requests += 1
                metrics.successful_requests += 1
                metrics.average_response_time = (
                    (metrics.average_response_time * (metrics.total_requests - 1) + duration) /
                    metrics.total_requests
                )
                metrics.last_request_time = time.time()
                
                return content
                
        except Exception as e:
            duration = time.time() - start_time
            
            # Update error metrics
            if self.enable_metrics:
                status_code = response.status_code if response else 0
                record_plugin_http_request(
                    institution, "GET", status_code, duration, domain
                )
                record_plugin_failure(institution, f"http_error_{type(e).__name__}")
            
            # Update rate limiter and circuit breaker
            if response:
                rate_limiter.adapt_for_error(response.status_code)
            self._update_circuit_breaker(domain, False)
            
            # Update request metrics
            metrics = self._get_request_metrics(domain)
            metrics.total_requests += 1
            metrics.failed_requests += 1
            
            # Re-raise with context
            if isinstance(e, httpx.HTTPStatusError):
                raise HttpClientError(f"HTTP {e.response.status_code} for {url}: {e}")
            elif isinstance(e, httpx.TimeoutException):
                raise HttpClientError(f"Timeout for {url}: {e}")
            elif isinstance(e, (RateLimitExceededError, ContentValidationError, RobotsDisallowedError)):
                raise
            else:
                raise HttpClientError(f"Request failed for {url}: {e}")
    
    async def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        institution: str = "unknown",
        **kwargs
    ) -> Any:
        """Get and parse JSON response."""
        text = await self.get(url, params, headers, institution, **kwargs)
        try:
            import json
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise HttpClientError(f"JSON parse error for {url}: {e}")
    
    async def download_file(
        self,
        url: str,
        dest_path: Union[str, Path],
        institution: str = "unknown",
        chunk_size: int = 8192,
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> int:
        """
        Download file with progress tracking and validation.
        
        Args:
            url: Source URL
            dest_path: Destination file path
            institution: Institution name for metrics
            chunk_size: Download chunk size
            progress_callback: Optional progress callback function
            **kwargs: Additional request parameters
        
        Returns:
            Number of bytes downloaded
        """
        await self._ensure_session()
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        domain = self._get_domain(url)
        await self._check_circuit_breaker(domain)
        
        rate_limiter = self._get_rate_limiter(domain)
        await rate_limiter.acquire(institution)
        
        start_time = time.time()
        bytes_downloaded = 0
        
        try:
            async with self.semaphore:
                async with self.session.stream("GET", url, **kwargs) as response:
                    response.raise_for_status()
                    
                    async with aiofiles.open(dest_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size):
                            await f.write(chunk)
                            bytes_downloaded += len(chunk)
                            
                            if progress_callback:
                                progress_callback(bytes_downloaded)
            
            duration = time.time() - start_time
            if self.enable_metrics:
                record_plugin_http_request(
                    institution, "GET", response.status_code, duration, domain
                )
            
            self._update_circuit_breaker(domain, True)
            return bytes_downloaded
            
        except Exception as e:
            duration = time.time() - start_time
            
            if self.enable_metrics:
                status_code = getattr(response, 'status_code', 0) if 'response' in locals() else 0
                record_plugin_http_request(
                    institution, "GET", status_code, duration, domain
                )
                record_plugin_failure(institution, f"download_error_{type(e).__name__}")
            
            self._update_circuit_breaker(domain, False)
            
            # Clean up partial download
            if dest_path.exists():
                dest_path.unlink()
            
            raise HttpClientError(f"Download failed for {url}: {e}")
    
    def set_rate_limit_policy(self, domain: str, policy: RateLimitPolicy) -> None:
        """Set custom rate limiting policy for a domain."""
        self.rate_limiters[domain] = DomainRateLimiter(policy)
        logger.info(f"Set custom rate limit policy for {domain}: {policy.requests_per_second} req/s")
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a specific domain."""
        metrics = self.request_metrics.get(domain, RequestMetrics())
        cb = self.circuit_breakers.get(domain, CircuitBreakerState())
        
        return {
            "domain": domain,
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "success_rate": (
                metrics.successful_requests / metrics.total_requests 
                if metrics.total_requests > 0 else 0.0
            ),
            "average_response_time": metrics.average_response_time,
            "circuit_breaker_state": cb.state,
            "circuit_breaker_failures": cb.failure_count,
            "last_request_time": metrics.last_request_time
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all domains."""
        return {
            "global": {
                "total_domains": len(self.request_metrics),
                "active_rate_limiters": len(self.rate_limiters),
                "open_circuit_breakers": sum(
                    1 for cb in self.circuit_breakers.values() 
                    if cb.state == "open"
                )
            },
            "domains": {
                domain: self.get_domain_stats(domain)
                for domain in self.request_metrics.keys()
            }
        }

# Factory functions for common use cases

def create_plugin_http_client(
    institution: str,
    rate_limit: float = 1.0,
    max_concurrent: int = 5,
    **kwargs
) -> EnhancedHttpClient:
    """Create HTTP client optimized for plugin use."""
    return EnhancedHttpClient(
        default_rate_limit=rate_limit,
        max_concurrent=max_concurrent,
        **kwargs
    )

def create_bulk_download_client(
    max_concurrent: int = 20,
    rate_limit: float = 0.5,
    **kwargs
) -> EnhancedHttpClient:
    """Create HTTP client optimized for bulk downloads."""
    return EnhancedHttpClient(
        max_concurrent=max_concurrent,
        default_rate_limit=rate_limit,
        timeout=60.0,
        **kwargs
    )

# Global default client instance
default_http_client = EnhancedHttpClient()

# Context manager for automatic session management
@asynccontextmanager
async def http_client_session(**kwargs) -> AsyncIterator[EnhancedHttpClient]:
    """Context manager for HTTP client with automatic cleanup."""
    client = EnhancedHttpClient(**kwargs)
    try:
        async with client:
            yield client
    finally:
        await client.close()

# Example usage:
#
# # Basic usage
# async with http_client_session() as client:
#     content = await client.get("https://www.ecb.europa.eu/speeches", institution="ECB")
#
# # Plugin-specific client
# plugin_client = create_plugin_http_client("ECB", rate_limit=2.0)
# async with plugin_client:
#     content = await plugin_client.get(url, institution="ECB")
#
# # Custom rate limiting
# client.set_rate_limit_policy("ecb.europa.eu", RateLimitPolicy(
#     requests_per_second=0.5,
#     burst_size=3,
#     adaptive=True
# ))