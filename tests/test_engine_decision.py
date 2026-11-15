"""Decision check through the engine (engine.optimize with Options.check), on the synthetic season
(GW8 next, lookahead 1, no chip analysis, no log): argument errors, the rule 'chip still
available', the user's captain outside the squad or unavailable, and no change without a check."""
import pytest

from fplopt import engine
from fplopt.engine import DecisionSection, Options, optimize
from fplopt.errors import InputError
from fpl_fake import FakeApi, Season, build_manager
from conftest import FakeClient
from toy_pools import base_squad, fill, toy_pool, weights

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
    (dict(check='NONE'), "only TC, BB, TRANSFERS, WC, FH can be checked"),
    (dict(check='TC'), "needs --captain"),
    (dict(check='BB', check_captain=1), "only with --check TC"),
    (dict(check='TC', check_captain=999999), "no player with this id"),
    (dict(check='TC', check_captain=1, check_gw=9), "outside the horizon GW8-GW8"),
    (dict(check='TC', check_captain=1, chip='TC'), "cannot be combined with --chip"),
])
def test_argument_errors(season, kw, message):
    with pytest.raises(InputError, match=message):
        run(season, **kw)


# --------------------------------------------------------------------------- transfers (D3)
def _owned(season):
    return build_manager(season)['truth'][0]


def _elem(season):
    return {e['id']: e for e in season.elements}


def test_transfers_check_with_the_models_own_transfers(season, plain):
    """The main plan's own GW8 transfers checked as the user's: R = 0 within the gaps, G + R = gain."""
    moves = plain.plan[0].transfers
    assert moves                                           # the synthetic manager has 3 FT and transfers
    r = run(season, check='TRANSFERS', check_out=[t.out_id for t in moves], check_in=[t.in_id for t in moves])
    d = r.decision
    assert d.status == 'computed' and d.violations == []
    x = d.result
    assert x['r_low'] - 1e-9 <= 0.0 <= x['r_high'] + 1e-9
    assert x['g'] + x['r'] == pytest.approx(x['gain_action'], abs=1e-9)
    assert sorted(x['plan_decision']['in']) == sorted(t.in_id for t in moves)
    assert d.transfers_out == [t.out_id for t in moves] and set(d.names) >= {t.in_id for t in moves}


def test_transfers_rules(season):
    """Rules of the game: each broken rule gives its code, no solve, and the main plan is still given."""
    owned, el = _owned(season), _elem(season)
    pos = {i: el[i]['element_type'] for i in el}
    gk = next(i for i in owned if pos[i] == 1)
    other_mid = next(i for i in el if pos[i] == 3 and i not in owned)
    other_def = next(i for i in el if pos[i] == 2 and i not in owned)
    owned_def = [i for i in owned if pos[i] == 2]
    club = el[owned_def[0]]['team']                       # 3 more DEFs of the club of an owned DEF
    same_club = [i for i in el if pos[i] == 2 and el[i]['team'] == club and i not in owned][:3]
    cases = [
        (dict(check_out=[gk], check_in=[other_mid]), 'positions'),
        (dict(check_out=[other_mid], check_in=[other_def]), 'not_owned'),
        (dict(check_out=[owned_def[0]], check_in=[owned_def[1]]), 'already_owned'),
        (dict(check_out=owned_def[1:4], check_in=same_club), 'club_limit'),
    ]
    for kw, code in cases:
        r = run(season, check='TRANSFERS', **kw)
        d = r.decision
        assert d.status == 'invalid' and d.result is None, code
        assert code in [v['code'] for v in d.violations], (code, d.violations)
        assert len(r.squad) == 15


def test_transfers_over_budget(season):
    """Two cheap owned midfielders for the two dearest ones not owned: over the budget (in tenths)."""
    owned, el = _owned(season), _elem(season)
    mids = sorted((i for i in owned if el[i]['element_type'] == 3), key=lambda i: el[i]['now_cost'])[:2]
    dear = sorted((i for i in el if el[i]['element_type'] == 3 and i not in owned),
                  key=lambda i: -el[i]['now_cost'])[:2]
    # a sale returns at most the current price, so this difference above the bank (2.9) is enough
    assert sum(el[i]['now_cost'] for i in dear) - sum(el[i]['now_cost'] for i in mids) > 29
    d = run(season, check='TRANSFERS', check_out=mids, check_in=dear).decision
    assert d.status == 'invalid'
    assert 'over_budget' in [v['code'] for v in d.violations]


@pytest.mark.parametrize("kw, message", [
    (dict(check='TRANSFERS', check_out=[1]), "needs --out and --in with the same number"),
    (dict(check='TRANSFERS', check_out=[1, 2], check_in=[3]), "needs --out and --in with the same number"),
    (dict(check='TRANSFERS', check_out=[1], check_in=[1]), "each player once, and no player in both"),
    (dict(check='TRANSFERS', check_out=[1], check_in=[999999]), "no player with id 999999"),
    (dict(check='TRANSFERS', check_out=[1], check_in=[3], check_gw=9, lookahead=2), "is for the current GW"),
    (dict(check='TC', check_captain=1, check_out=[1], check_in=[3]), "used only with --check TRANSFERS"),
])
def test_transfers_argument_errors(season, kw, message):
    with pytest.raises(InputError, match=message):
        run(season, **kw)


# --------------------------------------------------------------------------- 15b: Wildcard and Free Hit
def _no_chips(season):
    """The synthetic manager without the chips it played (WC GW5, FH GW7), so that WC and FH are available
    in GW8. --ft 3 is passed: the history still has the GW5 transfers, which the rebuild would now count."""
    m = build_manager(season, fh_last=False)
    m['chips'] = []
    return m, dict(ft=3)


def test_wildcard_check_with_the_models_squad(season):
    """15b K2/K4': the squad of the main plan of --chip WC checked as the user's Wildcard: R = 0 within the
    gaps, the notes of 15b O1 (measured over 1 GW) and O3 (no timing), and the default time of a Wildcard (O4)."""
    m, kw = _no_chips(season)
    best = [s.id for s in run(season, m, chip='WC', **kw).squad]
    r = run(season, m, check='WC', check_squad=best, **kw)
    d = r.decision
    assert (d.chip, d.status, d.violations, d.squad) == ('WC', 'computed', [], best)
    assert [n['code'] for n in d.notes] == ['changes', 'wc_horizon', 'wc_no_timing']
    x = d.result
    assert x['r_low'] - 1e-9 <= 0.0 <= x['r_high'] + 1e-9
    assert x['g'] + x['r'] == pytest.approx(x['gain_action'], abs=1e-9)
    assert x['timing'] is None and x['gws_measured'] == 1
    assert x['squad_diff']['only_decision'] == []
    assert r.options.time_limit == 900
    assert set(d.names) >= set(best)


def test_free_hit_check_with_the_current_squad(season):
    """Decision c: a Free Hit with no change. 0 changes, timing over the first window, G <= 0 within the gaps
    (the plan without the chip can do anything this one does, with the same squad), and the main plan is the
    same as without a check (the squad's players are in the pool anyway)."""
    m, kw = _no_chips(season)
    sq = list(m['truth'][0])
    r = run(season, m, check='FH', check_squad=sq, **kw)
    d = r.decision
    assert d.status == 'computed' and [n['code'] for n in d.notes] == ['changes']
    assert d.notes[0]['message'].startswith('0 change(s)')
    x = d.result
    assert x['g_low'] <= 0.0 + 1e-9
    assert x['timing']['window_end'] == 19 and sorted(x['timing']['estimates']) == list(range(8, 20))
    assert r.options.time_limit is None                               # FH keeps the general default
    assert r.solver.objective == pytest.approx(run(season, m, **kw).solver.objective, abs=1e-6)


def test_squad_chip_with_two_unavailable_goalkeepers(season):
    """15b K8: the current squad with both goalkeepers replaced by the two cheapest ones not owned, both with
    chance 0 (dropped from the pool without a check): they stay in the pool, the plan exists (a GK with xP 0
    starts), and the note lists them."""
    season = Season()
    m, kw = _no_chips(season)
    owned, el = list(m['truth'][0]), _elem(season)
    gks = sorted((i for i in el if el[i]['element_type'] == 1 and i not in owned),
                 key=lambda i: (el[i]['now_cost'], i))[:2]
    for i in gks:
        el[i]['chance_of_playing_next_round'] = 0
    sq = [i for i in owned if el[i]['element_type'] != 1] + gks
    d = run(season, m, check='FH', check_squad=sq, **kw).decision
    assert d.status == 'computed'
    assert 'player_unavailable' in [n['code'] for n in d.notes]
    xi_gk = next(i for i in d.result['plan_decision']['xi'] if el[i]['element_type'] == 1)
    assert xi_gk in gks


def test_squad_chip_rules(season):
    """15b K7: each broken rule gives its code and no solve; the main plan is still given."""
    m, kw = _no_chips(season)
    owned, el = list(m['truth'][0]), _elem(season)
    pos = {i: el[i]['element_type'] for i in el}
    gk = next(i for i in owned if pos[i] == 1)
    other_mid = next(i for i in el if pos[i] == 3 and i not in owned)
    owned_def = [i for i in owned if pos[i] == 2]
    club = el[owned_def[0]]['team']
    same_club = [i for i in el if pos[i] == 2 and el[i]['team'] == club and i not in owned][:3]
    mids = sorted((i for i in owned if pos[i] == 3), key=lambda i: el[i]['now_cost'])[:2]
    dear = sorted((i for i in el if pos[i] == 3 and i not in owned), key=lambda i: -el[i]['now_cost'])[:2]
    cases = [
        ([i for i in owned if i != gk] + [other_mid], 'positions'),
        ([i for i in owned if i not in owned_def[1:4]] + same_club, 'club_limit'),
        ([i for i in owned if i not in mids] + dear, 'over_budget'),
    ]
    for chip in ('WC', 'FH'):
        for sq, code in cases:
            r = run(season, m, check=chip, check_squad=sq, **kw)
            d = r.decision
            assert d.status == 'invalid' and d.result is None, (chip, code)
            assert code in [v['code'] for v in d.violations], (chip, code, d.violations)
            assert len(r.squad) == 15


@pytest.mark.parametrize("chip", ['WC', 'FH'])
def test_squad_chip_already_played_is_invalid(season, chip):
    """The synthetic manager played WC in GW5 and FH in GW7 (first sets): a check in GW8 -> chip_unavailable."""
    sq = list(build_manager(season)['truth'][0])
    d = run(season, check=chip, check_squad=sq).decision
    assert d.status == 'invalid' and d.violations[0]['code'] == 'chip_unavailable'


def _toy_section(chip, gw, horizon, budget=75.0):
    """A toy pool and the arguments of engine._squad_chip_section (no API)."""
    pool = toy_pool(); fill(pool, horizon); sq = base_squad(pool)
    sec = DecisionSection(chip=chip, gw=gw, captain_id=None, captain_name=None, status='invalid')
    opts = Options(team_id=1, check=chip, check_squad=list(sq), lookahead=len(horizon), max_hits=None)
    names = {i: str(i) for i in pool['ids']}
    return sec, opts, pool, sq, names


def test_free_hit_after_a_free_hit_is_invalid():
    """fh_consecutive: FH played in GW19 (first set), a FH checked in GW20 (second set, available): the game
    allows no Free Hit in two GWs in a row. No solve (the main plan 'base' is not even needed)."""
    horizon = [20, 21]
    sec, opts, pool, sq, names = _toy_section('FH', 20, horizon)
    rows = [('FH', 2, 19, 'played GW19'), ('FH', 20, 38, 'available')]
    out = engine._squad_chip_section(sec, opts, 38, rows, pool, horizon, weights(horizon), sq, 75.0, 1, {},
                                     names, pool['avail'], 0.0)
    assert out.status == 'invalid' and out.result is None
    assert [v['code'] for v in out.violations] == ['fh_consecutive']


def test_squad_budget_at_the_limit():
    """15b K7: the budget in whole tenths: a squad that costs exactly the budget is valid, GBP 0.1m less is
    'over_budget' by 0.1. Toy: the 15 owned at 5.0 (sell) and one bought at 5.3 instead of an owned 5.0."""
    pool = toy_pool(); fill(pool, [8]); sq = base_squad(pool)
    new = next(i for i in pool['ids'] if pool['pos'][i] == 3 and i not in sq)
    pool['price'][new] = 5.3
    squad = [i for i in sq if i != sq[7]] + [new]                 # sq[7] is a MID
    names = {i: str(i) for i in pool['ids']}
    assert engine._squad_violations(squad, sq, pool, names, 75.3) == []
    v = engine._squad_violations(squad, sq, pool, names, 75.2)
    assert [x['code'] for x in v] == ['over_budget'] and '£0.1m over' in v[0]['message']


@pytest.mark.parametrize("kw, message", [
    (dict(check='WC'), "needs --squad"),
    (dict(check='FH', check_squad=list(range(1, 15))), "15 different players"),
    (dict(check='FH', check_squad=[1] + list(range(1, 15))), "15 different players"),
    (dict(check='WC', check_squad=list(range(1, 15)) + [999999]), "no player with id 999999"),
    (dict(check='TC', check_captain=1, check_squad=list(range(1, 16))), "used with --check WC or FH"),
    (dict(check='FH', check_captain=1, check_squad=list(range(1, 16))), "only with --check TC"),
])
def test_squad_chip_argument_errors(season, kw, message):
    with pytest.raises(InputError, match=message):
        run(season, **kw)
