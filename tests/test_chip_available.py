"""--chip must be playable in the current GW.

A chip whose sets are all played, expired or not open yet is an input error (exit 2) before any
solve; a chip already active for the GW (played before its deadline) is still allowed, and so is NONE.
Synthetic season, GW8 next; its manager played WC in GW5 and FH in GW7 (set 1 of both), BB and TC are
unplayed."""
import pytest

from fplopt.engine import Options, _check_chip_available, optimize
from fplopt.errors import InputError
from fplopt.squad import chips_active_now
from fpl_fake import FakeApi, build_manager
from conftest import FakeClient

BASE = dict(team_id=1, lookahead=1, chip_report=False, write_log=False, transfer_gain=False, formations=False,
            chip_plan=False)


def run(season, manager=None, **kw):
    client = FakeClient(FakeApi(season, manager or build_manager(season)))
    return optimize(client, Options(**dict(BASE, **kw)))


def no_chips(season):
    """The manager without its WC (GW5) and FH (GW7); --ft 3 because the history still has the GW5 transfers."""
    m = build_manager(season, fh_last=False)
    m['chips'] = []
    return m


# The rule by hand
ROWS = [('WC', 2, 19, 'played GW5'), ('WC', 20, 38, 'available'), ('TC', 1, 19, 'available')]


def test_chips_active_now():
    h = {'chips': [{'name': 'wildcard', 'event': 8}, {'name': 'freehit', 'event': 7}, {'name': 'manager', 'event': 8}]}
    assert chips_active_now(h, 8) == {'WC'}
    assert chips_active_now(h, 7) == {'FH'}
    assert chips_active_now({}, 8) == set() and chips_active_now(None, 8) == set()


@pytest.mark.parametrize("gw", [8, 19])
def test_window_must_cover_the_gw(gw):
    """Windows as available_chips builds them: (max(start, t0), stop). The second set (GW20-38) is not
    open before GW20, so it does not cover GW8 or GW19."""
    with pytest.raises(InputError) as e:
        _check_chip_available('WC', gw, {'WC': [(20, 38)]}, ROWS, set(), False)
    assert f"WC cannot be played in GW{gw}: GW2-19 played GW5, GW20-38 available" in str(e.value)


@pytest.mark.parametrize("gw", [20, 38])
def test_window_that_covers_the_gw_passes(gw):
    _check_chip_available('WC', gw, {'WC': [(20, 38)]}, ROWS, set(), False)


def test_none_and_active_chip_always_pass():
    _check_chip_available('NONE', 8, {}, ROWS, set(), False)
    _check_chip_available('WC', 8, {}, ROWS, {'WC'}, False)


def test_new_team_wildcard_and_free_hit():
    for chip in ('WC', 'FH'):
        with pytest.raises(InputError, match="before the team's first deadline"):
            _check_chip_available(chip, 1, {}, [], set(), True)
    _check_chip_available('BB', 1, {'BB': [(1, 19)]}, [], set(), True)


# Through the engine
@pytest.mark.parametrize("chip, played", [('WC', 'GW2-19 played GW5'), ('FH', 'GW2-19 played GW7')])
def test_played_chip_is_an_input_error(season, chip, played):
    with pytest.raises(InputError) as e:
        run(season, chip=chip)
    assert f"{chip} cannot be played in GW8: {played}, GW20-38 available" in str(e.value)


@pytest.mark.parametrize("chip", ['TC', 'BB'])
def test_unplayed_chip_is_allowed(season, chip):
    assert run(season, chip=chip).options.chip == chip


@pytest.mark.parametrize("chip", ['WC', 'FH'])
def test_available_chip_is_allowed(season, chip):
    assert run(season, no_chips(season), chip=chip, ft=3).options.chip == chip


def test_chip_active_for_this_gw_is_allowed(season):
    """The user activated the Wildcard for GW8 before its deadline: the history shows it as played in GW8,
    its set is no longer available, and --chip WC (what the warning asks for) must still work."""
    m = no_chips(season)
    m['chips'] = [{'name': 'wildcard', 'event': 8}]
    r = run(season, m, chip='WC', ft=3)
    assert r.options.chip == 'WC'
    assert any("WC is already active for GW8" in w['message'] for w in r.warnings)


def test_new_team_chip_errors(season):
    m = build_manager(season)
    m.update(current=[], chips=[], picks={}, banks={}, transfers=[])
    for chip in ('WC', 'FH'):
        with pytest.raises(InputError, match="first deadline"):
            run(season, m, chip=chip)
    assert run(season, m, chip='BB').options.chip == 'BB'
