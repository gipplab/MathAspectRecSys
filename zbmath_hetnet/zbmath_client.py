"""Thin HTTP client for the zbMATH Open REST API (https://api.zbmath.org/v1)."""
import logging
import random
import threading
import time

import requests

try:  # use the OS certificate store if available (helps behind corporate TLS proxies)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

API_BASE = "https://api.zbmath.org/v1"
MAX_RESULTS_PER_PAGE = 500  # hard limit enforced by the API
DEFAULT_RESULTS_PER_PAGE = 200  # large pages take ~30 ms/record server-side and risk gateway timeouts
MAX_RESULT_WINDOW = 10_000  # page * results_per_page must stay within this window

log = logging.getLogger(__name__)


class ZbMathClient:
    def __init__(self, min_interval=0.0, max_retries=8, timeout=120):
        """
        min_interval: minimum seconds between two requests (shared across threads), to stay polite.
        """
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._local = threading.local()
        self._lock = threading.Lock()
        self._last_request = 0.0

    @property
    def session(self):
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers["User-Agent"] = "zbmath-hetnet-crawler (research; python-requests)"
            self._local.session = s
        return self._local.session

    def _throttle(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def search(self, endpoint, query, page=0, results_per_page=DEFAULT_RESULTS_PER_PAGE):
        """
        Run `/{endpoint}/_search`. Returns (results, total). A query without hits returns ([], 0).
        endpoint: 'document' or 'author'.
        """
        url = f"{API_BASE}/{endpoint}/_search"
        params = {"search_string": query, "page": page, "results_per_page": results_per_page}
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                err = repr(e)
            else:
                if r.status_code == 404:  # the API answers "no results" with 404
                    return [], 0
                if r.status_code == 200:
                    try:
                        payload = r.json()
                    except ValueError as e:
                        err = f"invalid JSON: {e}"
                    else:
                        status = payload.get("status") or {}
                        return payload.get("result") or [], status.get("nr_total_results") or 0
                elif r.status_code in (429, 500, 502, 503, 504):
                    err = f"HTTP {r.status_code}"
                else:
                    raise RuntimeError(f"HTTP {r.status_code} for {endpoint} query {query!r} page {page}: {r.text[:300]}")
            delay = min(300, 2 ** attempt) + random.random()
            log.warning("%s for %r page %d, retrying in %.0fs", err, query, page, delay)
            time.sleep(delay)
        raise RuntimeError(f"giving up on {endpoint} query {query!r} page {page} after {self.max_retries} attempts")

    def count(self, endpoint, query):
        return self.search(endpoint, query, page=0, results_per_page=1)[1]
