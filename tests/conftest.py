"""Shared fixtures: the synthetic season and a fake FPL API (no network)."""
import pytest

import fpl_v2 as F
from fpl_fake import FakeApi, Season, build_manager


@pytest.fixture(scope="session")
def season():
    """The small synthetic season (360 players); built once, never modified by the tests."""
    return Season()


@pytest.fixture
def fake_api(monkeypatch, season):
    """Factory: fake_api(**manager options) -> FakeApi, and every API request of the optimizer
    is answered by it for the rest of the test."""
    def make(**manager_kw):
        api = FakeApi(season, build_manager(season, **manager_kw))
        monkeypatch.setattr(F, 'fetch_fpl_api', lambda url: api.respond(url.replace(F.FPL_API, '')))
        return api
    return make
