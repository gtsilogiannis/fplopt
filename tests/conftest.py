"""Shared fixtures: the synthetic season and a fake FPL client (no network)."""
import logging

import pytest

from fplopt.api import FplClient
from fpl_fake import FakeApi, Season, build_manager


class FakeClient(FplClient):
    """An FplClient whose requests are answered by a FakeApi instead of the network. Only get()
    is replaced, so everything built on it (endpoint methods, cache) runs as in production."""

    def __init__(self, api, **kw):
        super().__init__(**kw)
        self.api = api

    def get(self, path):
        return self.api.respond(path)


@pytest.fixture(scope="session")
def season():
    """The small synthetic season (360 players); built once, never modified by the tests."""
    return Season()


@pytest.fixture
def fake_client(season):
    """Factory: fake_client(**manager options) -> FakeClient for the test manager."""
    def make(**manager_kw):
        return FakeClient(FakeApi(season, build_manager(season, **manager_kw)))
    return make


@pytest.fixture(autouse=True)
def restore_fplopt_logger():
    """cli.main() points the 'fplopt' logger at the sys.stderr of the test that calls it, and pytest
    closes that stream when the test ends; a later test that logs through the engine then printed
    'Logging error ... I/O operation on closed file'. Every test gets back the logger it started with."""
    logger = logging.getLogger('fplopt')
    handlers, level = logger.handlers[:], logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


@pytest.fixture
def messages(caplog):
    """The messages the package logged during the test, in order (text as printed on stderr)."""
    caplog.set_level(logging.INFO, logger='fplopt')
    return lambda: [r.getMessage() for r in caplog.records if r.name.startswith('fplopt')]
