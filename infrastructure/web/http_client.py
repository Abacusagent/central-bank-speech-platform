# infrastructure/web/http_client.py

"""
HTTP Client Utilities for Central Bank Speech Analysis Platform

Provides robust, async HTTP access with retry logic, timeouts,
and structured error handling, suitable for scraping and API consumption.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import logging
from typing import Optional, Dict, Any, Tuple, List, Union
import httpx
import asyncio

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 20.0  # seconds
DEFAULT_RETRIES = 3
DEFAULT_HEADERS = {
    'User-Agent': 'CentralBankSpeechBot/1.0 (https://your-domain.org)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}

class HttpRequestError(Exception):
    pass

class HttpClient:
    """
    Async HTTP client with built-in retry logic and structured error handling.
    """

    def __init__(self,
                 timeout: float = DEFAULT_TIMEOUT,
                 retries: int = DEFAULT_RETRIES,
                 headers: Optional[Dict[str, str]] = None):
        self.timeout = timeout
        self.retries = retries
        self.headers = headers or DEFAULT_HEADERS

    async def get(self,
                  url: str,
                  params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None,
                  allow_redirects: bool = True,
                  return_text: bool = True,
                  encoding: Optional[str] = None,
                  ) -> Union[str, bytes]:
        """
        GET request with retries and timeout.

        Returns: str (decoded text) or bytes (raw) depending on return_text.
        Raises: HttpRequestError on repeated failure.
        """
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
                    response = await client.get(
                        url,
                        params=params,
                        headers=headers,
                        follow_redirects=allow_redirects
                    )
                    response.raise_for_status()
                    if encoding:
                        response.encoding = encoding
                    if return_text:
                        return response.text
                    return response.content
            except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(f"[HTTP] Attempt {attempt}/{self.retries} failed for {url}: {e}")
                await asyncio.sleep(min(attempt, 3))  # Simple backoff
        raise HttpRequestError(f"Failed to GET {url} after {self.retries} attempts: {last_exc}")

    async def get_json(self,
                       url: str,
                       params: Optional[Dict[str, Any]] = None,
                       headers: Optional[Dict[str, str]] = None,
                       ) -> Any:
        """
        GET request, parse JSON response.

        Returns: parsed JSON object.
        Raises: HttpRequestError on repeated failure.
        """
        text = await self.get(url, params=params, headers=headers, return_text=True)
        import json
        try:
            return json.loads(text)
        except Exception as e:
            logger.error(f"Failed to parse JSON from {url}: {e}")
            raise HttpRequestError(f"JSON parse error for {url}: {e}")

    async def download_file(self,
                           url: str,
                           dest_path: str,
                           headers: Optional[Dict[str, str]] = None,
                           chunk_size: int = 8192,
                           ) -> int:
        """
        Download a file from a URL to a local destination.

        Returns: number of bytes written.
        Raises: HttpRequestError on repeated failure.
        """
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
                    with open(dest_path, "wb") as f:
                        async with client.stream("GET", url, headers=headers) as response:
                            response.raise_for_status()
                            bytes_written = 0
                            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                                f.write(chunk)
                                bytes_written += len(chunk)
                            return bytes_written
            except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(f"[HTTP] Download attempt {attempt}/{self.retries} failed for {url}: {e}")
                await asyncio.sleep(min(attempt, 3))
        raise HttpRequestError(f"Failed to download {url} after {self.retries} attempts: {last_exc}")

# Convenience instance for project-wide use
default_http_client = HttpClient()

# Example usage:
# from infrastructure.web.http_client import default_http_client
# content = await default_http_client.get("https://www.ecb.europa.eu/speeches")
