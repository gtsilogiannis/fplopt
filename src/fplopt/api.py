"""HTTP access to the public FPL API (no login needed).

FplClient is the only place that talks to the network. The rest of the package receives a
client object as an argument, so tests pass a fake client that answers from a synthetic
season, and nothing has to be patched globally.
"""
import requests

from .errors import ApiError, NotFoundError

FPL_API = "https://fantasy.premierleague.com/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
TIMEOUT = 12  # seconds per request (unchanged since chat 5)


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

    def __init__(self, base_url=FPL_API, session=None):
        self.base_url = base_url.rstrip('/')
        self.session = session or requests.Session()   # one connection reused for all requests
        self.session.headers.update(HEADERS)

    def get(self, path):
        """JSON of one endpoint. Raises NotFoundError for 404 and ApiError for any other failure."""
        url = self.base_url + path
        try:
            response = self.session.get(url, timeout=TIMEOUT)
            if response.status_code == 404:
                raise NotFoundError(f"404 Not Found: Verify Team ID or Target Gameweek.\nEndpoint: {url}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:   # includes an invalid JSON body
            raise ApiError(f"Network/FPL API Handshake Failure: {e}") from e

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
