"""Disk cache of /event/{gw}/live/ for final gameweeks (finished AND data_checked)."""
import json

import pytest

from fplopt.api import CACHE_VERSION, is_final
from fplopt.data import fetch_history
from fpl_fake import FakeApi, Season, build_manager
from conftest import FakeClient


@pytest.mark.parametrize("event, final", [
    ({'finished': True, 'data_checked': True, 'deadline_time': '2026-10-17T10:00:00Z'}, True),
    ({'finished': True, 'data_checked': False, 'deadline_time': '2026-10-17T10:00:00Z'}, False),
    ({'finished': False, 'data_checked': False, 'deadline_time': '2026-10-17T10:00:00Z'}, False),
    ({'finished': True, 'deadline_time': '2026-10-17T10:00:00Z'}, False),      # field missing: not final
    ({'finished': True, 'data_checked': True}, False),                         # no deadline: no cache key
    (None, False),
])
def test_is_final(event, final):
    assert is_final(event) is final


@pytest.fixture
def checked_season():
    return Season(data_checked=True)


def client(season, cache_dir, **kw):
    return FakeClient(FakeApi(season, build_manager(season)), cache_dir=cache_dir, **kw)


def live_calls(c):
    return [p for p in c.api.calls if p.startswith('/event/')]


def test_second_run_reads_the_cache(checked_season, tmp_path, messages):
    boot = checked_season.bootstrap()
    first = client(checked_season, tmp_path)
    hist1, gws = fetch_history(first, boot, checked_season.fixtures, 8)
    assert gws == [1, 2, 3, 4, 5, 6, 7] and len(live_calls(first)) == 7
    assert first.stats['cache_writes'] == 7
    files = sorted(p.name for p in (tmp_path / "fpl_live").iterdir())
    assert files[0] == "event_01_2026-08-15.json" and len(files) == 7

    second = client(checked_season, tmp_path)
    hist2, _ = fetch_history(second, boot, checked_season.fixtures, 8)
    assert live_calls(second) == []                            # nothing downloaded
    assert second.stats['cache_hits'] == 7
    assert hist2.equals(hist1)                                 # same input for the xP model
    assert messages()[-1] == "[*] Live data of 7 final GW(s) read from the cache (--refresh-cache downloads them again)"

    third = client(checked_season, tmp_path, refresh_cache=True)
    fetch_history(third, boot, checked_season.fixtures, 8)
    assert len(live_calls(third)) == 7 and third.stats['cache_writes'] == 7


def test_gameweeks_not_checked_are_not_cached(tmp_path):
    season = Season(data_checked=False)                         # finished but not data_checked
    c = client(season, tmp_path)
    fetch_history(c, season.bootstrap(), season.fixtures, 8)
    assert not (tmp_path / "fpl_live").exists()
    c2 = client(season, tmp_path)
    fetch_history(c2, season.bootstrap(), season.fixtures, 8)
    assert len(live_calls(c2)) == 7


def test_no_cache_dir_no_files(checked_season, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = client(checked_season, None)
    fetch_history(c, checked_season.bootstrap(), checked_season.fixtures, 8)
    assert list(tmp_path.iterdir()) == []


def test_damaged_or_other_format_files_are_replaced(checked_season, tmp_path, messages):
    event = checked_season.events[0]
    c = client(checked_season, tmp_path)
    data = c.event_live(1, event)
    path = tmp_path / "fpl_live" / "event_01_2026-08-15.json"
    path.write_text("{ not json", encoding='utf-8')            # e.g. a run killed while writing (old format)
    c2 = client(checked_season, tmp_path)
    assert c2.event_live(1, event) == data and live_calls(c2) == ['/event/1/live/']
    assert any(m.startswith("[WARNING] Ignoring unreadable cache file") for m in messages())
    doc = json.loads(path.read_text(encoding='utf-8'))          # rewritten with the current format
    assert (doc['cache_version'], doc['event_id'], doc['deadline_time']) == (CACHE_VERSION, 1, event['deadline_time'])
    doc['cache_version'] = CACHE_VERSION + 1
    path.write_text(json.dumps(doc), encoding='utf-8')
    c3 = client(checked_season, tmp_path)
    assert c3.event_live(1, event) == data and len(live_calls(c3)) == 1
    assert list((tmp_path / "fpl_live").glob("*.tmp")) == []     # no temporary file left behind
