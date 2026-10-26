"""Decision check through the engine (engine.optimize with Options.check), on the synthetic season
(GW8 next, lookahead 1, no chip analysis, no log): argument errors, the rule 'chip still
available', the user's captain outside the squad or unavailable, and no change without a check."""
import pytest

from fplopt.engine import Options, optimize
from fplopt.errors import InputError
from fpl_fake import FakeApi, Season, build_manager
from conftest import FakeClient

BASE = dict(team_id=1, lookahead=1, chip_report=False, write_log=False)


def run(season, manager=None, **kw):
    client = FakeClient(FakeApi(season, manager or build_manager(season)))
    return optimize(client, Options(**dict(BASE, **kw)))


@pytest.fixture(scope="module")
def plain(season):
    """The same run without a check (the main plan the check is compared with)."""
    return run(season)


def test_no_check_no_section(plain):
    assert plain.decision is None


def test_triple_captain_check(season, plain):
    """TC on the owned player with the highest xP in GW8: computed, G + R = gain of the chip, R >= 0
    within the gap, timing over the whole first window (GW8-GW19), and the main plan is the same as
    without the check (same pool, same model). An owned player, so no 'captain_bought' note."""
    cap = max((s for s in plain.squad if s.action == 'KEEP' and s.available), key=lambda s: s.xp_now).id
    r = run(season, check='TC', check_captain=cap)
    d = r.decision
    assert (d.chip, d.gw, d.captain_id, d.status) == ('TC', 8, cap, 'computed')
    assert d.violations == [] and d.notes == []
    x = d.result
    assert x['g'] + x['r'] == pytest.approx(x['gain_action'], abs=1e-9)
    assert x['r'] >= -x['gaps']['z_decision'] - 1e-9
    assert x['plan_decision']['captain'] == cap and x['plan_decision']['chip'] == 'TC'
    assert x['timing']['window_end'] == 19
    assert sorted(x['timing']['estimates']) == list(range(8, 20))
    assert isinstance(x['timing']['is_best'], bool)
    assert d.names[cap] == d.captain_name
    assert r.solver.objective == pytest.approx(plain.solver.objective, abs=1e-6)
    assert [s.id for s in r.squad] == [s.id for s in plain.squad]


def test_bench_boost_check(season):
    d = run(season, check='BB').decision
    assert d.status == 'computed' and d.captain_id is None
    assert d.result['r'] == 0.0
    assert d.result['g'] == pytest.approx(d.result['gain_action'], abs=1e-9)


def test_captain_unavailable_and_not_owned():
    """A forward outside the squad with chance 0 (filtered out of the pool without a check): he
    stays in the pool for the check, is bought in GW8 and captained with xP 0 -> R > 0."""
    season = Season()
    owned = set(build_manager(season)['truth'][0])
    # the cheapest forward not owned, so that the budget cannot rule the decision out
    cap = min((e for e in season.elements if e['element_type'] == 4 and e['id'] not in owned),
              key=lambda e: (e['now_cost'], e['id']))['id']
    next(e for e in season.elements if e['id'] == cap)['chance_of_playing_next_round'] = 0
    d = run(season, check='TC', check_captain=cap).decision
    assert d.status == 'computed'
    assert [n['code'] for n in d.notes] == ['captain_unavailable', 'captain_bought']
    assert d.result['captain_bought'] == 8
    assert d.result['plan_decision']['captain'] == cap
    assert d.result['r'] > 1.0


def test_chip_already_played_is_invalid(season):
    """TC played in GW3 (first set used): a TC in GW8 breaks a rule of the game -> 'invalid',
    with the reason, a warning, and no solve; the main plan is still given."""
    manager = build_manager(season)
    manager['chips'].append({'name': '3xc', 'event': 3})
    r = run(season, manager, check='TC', check_captain=1)
    d = r.decision
    assert d.status == 'invalid' and d.result is None
    assert d.violations[0]['code'] == 'chip_unavailable'
    assert 'played GW3' in d.violations[0]['message']
    assert any(w['code'] == 'decision_invalid' for w in r.warnings)
    assert len(r.squad) == 15


@pytest.mark.parametrize("kw, message", [
    (dict(check='WC'), "only TC, BB can be checked"),
    (dict(check='TC'), "needs --captain"),
    (dict(check='BB', check_captain=1), "only with --check TC"),
    (dict(check='TC', check_captain=999999), "no player with this id"),
    (dict(check='TC', check_captain=1, check_gw=9), "outside the horizon GW8-GW8"),
    (dict(check='TC', check_captain=1, chip='TC'), "cannot be combined with --chip"),
])
def test_argument_errors(season, kw, message):
    with pytest.raises(InputError, match=message):
        run(season, **kw)
