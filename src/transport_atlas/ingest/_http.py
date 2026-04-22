"""Shared rate-limited HTTP session."""
from __future__ import annotations

import time
from threading import Lock

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RateLimiter:
    """Simple token-bucket-ish limiter — enforces max N req/s across all callers."""

    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / max(rate_per_sec, 0.01)
        self._lock = Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval


def make_session(user_agent: str = "transport-atlas/0.1 (chois@umn.edu)") -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def get_json(session: requests.Session, url: str, *, params=None, headers=None, timeout: int = 30) -> dict:
    r = session.get(url, params=params, headers=headers, timeout=timeout)
    if r.status_code == 429:
        retry_after = int(r.headers.get("Retry-After", "30"))
        time.sleep(min(retry_after, 120))
        r = session.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def get_text(session: requests.Session, url: str, *, headers=None, timeout: int = 60) -> tuple[int, str]:
    r = session.get(url, headers=headers, timeout=timeout)
    return r.status_code, r.text
