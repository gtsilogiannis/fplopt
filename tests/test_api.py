"""FplClient against a local HTTP server: retries with backoff for transient failures only."""
import socket

import pytest

from fplopt.api import BACKOFF_SECONDS, MAX_ATTEMPTS, FplClient
from fplopt.errors import ApiError, NotFoundError
from fpl_fake import FakeApi, build_manager
from fake_server import FakeServer


@pytest.fixture
def api(season):
    return FakeApi(season, build_manager(season))


def client_for(url):
    waits = []
    return FplClient(base_url=url, sleep=waits.append), waits     # records the waits instead of sleeping


def test_retries_server_errors_then_succeeds(api, messages):
    with FakeServer(api, failures={'/fixtures/': [503, 502]}) as url:
        client, waits = client_for(url)
        assert len(client.fixtures()) == 380
        assert waits == [1.0, 2.0]
        assert (client.stats['requests'], client.stats['retries']) == (3, 2)
    assert messages() == ["[WARNING] FPL API: HTTP 503 for /fixtures/; retry 1/3 in 1 s",
                          "[WARNING] FPL API: HTTP 502 for /fixtures/; retry 2/3 in 2 s"]


def test_too_many_requests_is_retried(api):
    with FakeServer(api, failures={'/fixtures/': [429]}) as url:   # the server sends Retry-After: 0
        client, waits = client_for(url)
        client.fixtures()
        assert waits == [1.0]                                          # never shorter than the backoff


def test_gives_up_after_the_last_attempt(api):
    with FakeServer(api, failures={'/fixtures/': [500] * MAX_ATTEMPTS}) as url:
        client, waits = client_for(url)
        with pytest.raises(ApiError, match=f"after {MAX_ATTEMPTS} attempts"):
            client.fixtures()
        assert waits == list(BACKOFF_SECONDS)


@pytest.mark.parametrize("status, error", [(404, NotFoundError), (403, ApiError)])
def test_client_errors_are_not_retried(api, status, error):
    with FakeServer(api, failures={'/fixtures/': [status]}) as server_url:
        client, waits = client_for(server_url)
        with pytest.raises(error):
            client.fixtures()
        assert waits == [] and client.stats['requests'] == 1


def test_unknown_path_is_not_found(api):
    with FakeServer(api) as url:
        client, _ = client_for(url)
        with pytest.raises(NotFoundError, match="Verify Team ID"):
            client.get('/no-such-endpoint/')


def test_unreachable_server_is_retried():
    with socket.socket() as s:                    # a port that nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    client, waits = client_for(f"http://127.0.0.1:{port}/api")
    with pytest.raises(ApiError, match="ConnectionError"):
        client.fixtures()
    assert waits == list(BACKOFF_SECONDS)


class _HtmlSession:
    """A session whose reply is an HTML page with status 200 (e.g. a maintenance page)."""
    headers = {}

    def get(self, url, timeout):
        class Reply:
            status_code, headers = 200, {}

            def json(self):
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return Reply()


def test_invalid_json_is_an_api_error():
    client = FplClient(session=_HtmlSession(), sleep=lambda s: None)
    with pytest.raises(ApiError, match="invalid JSON"):
        client.fixtures()
