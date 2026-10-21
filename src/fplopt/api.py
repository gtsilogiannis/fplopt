"""HTTP access to the public FPL API (no login needed).

FplClient is the only place that talks to the network. The rest of the package receives a
client object as an argument, so tests pass a fake client that answers from a synthetic
season, and nothing has to be patched globally.

Transient failures are retried with a growing wait (backoff): timeouts, connection errors,
HTTP 5xx (server errors) and HTTP 429 (too many requests, whose Retry-After header is
respected). A 404 or any other 4xx is not retried: asking again gives the same answer.

The per-gameweek live data (/event/{gw}/live/, the input of the xP model) of FINAL gameweeks
can be kept on disk (cache_dir) and read from there in later runs; see event_live().
"""
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .errors import ApiError, NotFoundError

log = logging.getLogger(__name__)

FPL_API = "https://fantasy.premierleague.com/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
TIMEOUT = 12  # seconds per request (unchanged since chat 5)
# [assumption] retry policy: 4 attempts, waiting 1, 2 and 4 s before attempts 2-4 (at most 7 s
# plus the timeouts). Enough for a short server hiccup; not tuned to measured FPL outages.
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = (1.0, 2.0, 4.0)
MAX_RETRY_AFTER = 60.0   # [assumption] longest wait accepted from a Retry-After header (seconds)
CACHE_VERSION = 1        # format of a cache file; a file with another version is ignored and replaced


def _retry_after(response):
    """Seconds asked by a Retry-After header (the number form), or None."""
    try:
        return max(0.0, float(response.headers.get("Retry-After", "")))
    except ValueError:
        return None          # missing, or the HTTP-date form (not used by the FPL API as far as known)


class FplClient:
    """GET requests to the FPL API, returning the decoded JSON.

    Endpoints used (paths relative to FPL_API):
      /bootstrap-static/           players, teams, gameweeks, chip windows
      /fixtures/                   every fixture of the season
      /event/{gw}/live/            per-player stats of one gameweek
      /entry/{id}/                 manager (team) details
      /entry/{id}/history/         per-GW history and chips played
      /entry/{id}/event/{gw}/picks/  squad and bank of one gameweek
      /entry/{id}/transfers/       every transfer of the manager
      /element-summary/{id}/       per-match history of one player
    """

    def __init__(self, base_url=FPL_API, session=None, sleep=time.sleep, cache_dir=None, refresh_cache=False):
        """cache_dir: folder for the live data of final gameweeks (None = no cache).
        refresh_cache: download again and overwrite the cached files of this run."""
        self.base_url = base_url.rstrip('/')
        self.session = session or requests.Session()   # one connection reused for all requests
        self.session.headers.update(HEADERS)
        self.sleep = sleep                             # replaced in tests, so they do not wait
        self.cache_dir = Path(cache_dir) / "fpl_live" if cache_dir else None
        self.refresh_cache = refresh_cache
        self.stats = {'requests': 0, 'retries': 0, 'cache_hits': 0, 'cache_writes': 0}

    def get(self, path):
        """JSON of one endpoint. Raises NotFoundError for 404 and ApiError for any other failure
        (after MAX_ATTEMPTS attempts for the transient ones)."""
        url = self.base_url + path
        for attempt in range(1, MAX_ATTEMPTS + 1):
            retry_after = None
            self.stats['requests'] += 1
            try:
                response = self.session.get(url, timeout=TIMEOUT)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                reason, cause = type(e).__name__, e            # transient: retry
            except requests.exceptions.RequestException as e:  # invalid URL, too many redirects, ...
                raise ApiError(f"Network/FPL API Handshake Failure: {e}") from e
            else:
                status = response.status_code
                if status == 404:
                    raise NotFoundError(f"404 Not Found: Verify Team ID or Target Gameweek.\nEndpoint: {url}")
                if status == 429 or status >= 500:
                    reason, cause = f"HTTP {status}", None     # transient: retry
                    retry_after = _retry_after(response) if status == 429 else None
                elif status >= 400:
                    raise ApiError(f"FPL API answered HTTP {status} for {url}")
                else:
                    try:
                        return response.json()
                    except ValueError as e:                    # not JSON (e.g. a maintenance page)
                        raise ApiError(f"FPL API returned invalid JSON for {url}: {e}") from e
            if attempt == MAX_ATTEMPTS:
                raise ApiError(f"Network/FPL API Handshake Failure after {MAX_ATTEMPTS} attempts "
                               f"({reason}): {url}") from cause
            wait = BACKOFF_SECONDS[attempt - 1]
            if retry_after is not None:
                wait = max(wait, min(retry_after, MAX_RETRY_AFTER))
            log.warning(f"[WARNING] FPL API: {reason} for {path}; retry {attempt}/{MAX_ATTEMPTS - 1} "
                        f"in {wait:.0f} s", extra={'code': 'api_retry'})
            self.stats['retries'] += 1
            self.sleep(wait)

    # --- endpoints -------------------------------------------------------------------------
    def bootstrap(self):
        return self.get("/bootstrap-static/")

    def fixtures(self):
        return self.get("/fixtures/")

    def event_live(self, gw, event=None):
        """Live stats of gameweek gw. event = its entry of bootstrap-static; when it says the
        gameweek is final (is_final), the reply is cached and later runs read it from disk."""
        path = self._cache_path(gw, event)
        if path is not None and not self.refresh_cache:
            data = self._read_cache(path, gw)
            if data is not None:
                self.stats['cache_hits'] += 1
                return data
        data = self.get(f"/event/{gw}/live/")
        if path is not None:
            self._write_cache(path, gw, event, data)
        return data

    # --- cache of the live data of final gameweeks -------------------------------------------
    def _cache_path(self, gw, event):
        """data_cache/fpl_live/event_<gw>_<deadline date>.json, or None if not cacheable. The
        deadline date is part of the name, so GW8 of another season is another file."""
        if self.cache_dir is None or not is_final(event):
            return None
        return self.cache_dir / f"event_{gw:02d}_{event['deadline_time'][:10]}.json"

    def _read_cache(self, path, gw):
        """The cached reply, or None (missing, unreadable, other format or other gameweek)."""
        try:
            with open(path, encoding='utf-8') as fh:
                doc = json.load(fh)
            if doc.get('cache_version') == CACHE_VERSION and doc.get('event_id') == gw:
                return doc['data']
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, AttributeError) as e:
            log.warning(f"[WARNING] Ignoring unreadable cache file {path}: {e}", extra={'code': 'cache_unreadable'})
            return None
        log.warning(f"[WARNING] Ignoring cache file {path}: other format or gameweek", extra={'code': 'cache_unreadable'})
        return None

    def _write_cache(self, path, gw, event, data):
        """Writes atomically: to a temporary file in the same folder, then os.replace() (atomic on
        Linux and Windows), so an interrupted run never leaves a half-written file. A cache that
        cannot be written only costs time, so failures are warnings."""
        doc = {'cache_version': CACHE_VERSION, 'event_id': gw, 'deadline_time': event['deadline_time'],
               'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'data': data}
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(doc, fh, ensure_ascii=False)
            os.replace(tmp, path)
            self.stats['cache_writes'] += 1
        except OSError as e:
            log.warning(f"[WARNING] Could not write cache file {path}: {e}", extra={'code': 'cache_write_failed'})
            if tmp is not None and os.path.exists(tmp):
                os.remove(tmp)

    def entry(self, team_id):
        return self.get(f"/entry/{team_id}/")

    def entry_history(self, team_id):
        return self.get(f"/entry/{team_id}/history/")

    def picks(self, team_id, gw):
        return self.get(f"/entry/{team_id}/event/{gw}/picks/")

    def transfers(self, team_id):
        return self.get(f"/entry/{team_id}/transfers/")

    def element_summary(self, player_id):
        return self.get(f"/element-summary/{player_id}/")


def is_final(event):
    """True if bootstrap-static marks the gameweek as final: 'finished' (all its matches played)
    AND 'data_checked' (FPL has checked the data; after that the points and stats are no longer
    changed, premierleague.com "Why players' Fantasy points could change within a Gameweek").
    A missing field counts as not final, so nothing is cached that could still change.
    [assumption] the meaning of data_checked is taken from library type definitions and the
    behaviour above; it is checked on real data in the Arch run of chat 9a(i)."""
    return (isinstance(event, dict) and event.get('finished') is True and event.get('data_checked') is True
            and bool(event.get('deadline_time')))
