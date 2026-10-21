"""HTTP access to the public FPL API (no login needed).

FplClient is the only place that talks to the network. The rest of the package receives a
client object as an argument, so tests pass a fake client that answers from a synthetic
season, and nothing has to be patched globally.

Transient failures are retried with a growing wait (backoff): timeouts, connection errors,
HTTP 5xx (server errors) and HTTP 429 (too many requests, whose Retry-After header is
respected). A 404 or any other 4xx is not retried: asking again gives the same answer.
"""
import logging
import time

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

    def __init__(self, base_url=FPL_API, session=None, sleep=time.sleep):
        self.base_url = base_url.rstrip('/')
        self.session = session or requests.Session()   # one connection reused for all requests
        self.session.headers.update(HEADERS)
        self.sleep = sleep                             # replaced in tests, so they do not wait
        self.stats = {'requests': 0, 'retries': 0}

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
        """Live stats of gameweek gw. event = its entry of bootstrap-static (unused for now)."""
        return self.get(f"/event/{gw}/live/")

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
