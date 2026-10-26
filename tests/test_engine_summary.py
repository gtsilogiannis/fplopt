"""15c step 2: the facts the engine adds for the summary. The gain of the transfers over making
none (decision.transfer_gain) by hand on the toy pool and against the decision check of the same
transfers; the deadline of the current GW; the upper bound of the selling value."""
import pytest

from fplopt.decision import transfer_gain
from fplopt.engine import Options, optimize
from fplopt.milp import plan_squad
from fplopt.summary import build_summary
from fpl_fake import FakeApi, build_manager
from toy_pools import add, base_squad, fill, toy_pool, weights
from conftest import FakeClient

H3 = [8, 9, 10]
W3 = weights(H3)


def run(season, manager=None, **kw):
    opts = dict(team_id=1, lookahead=1, chip_report=False, write_log=False)
    return optimize(FakeClient(FakeApi(season, manager or build_manager(season))), Options(**dict(opts, **kw)))


@pytest.fixture(scope="module")
def plain(season):
    return run(season)


# --------------------------------------------------------------------------- transfer_gain by hand
def test_transfer_gain_by_hand():
    """Toy pool (as test_milp_force): a DEF with 4 xP in every GW, the owned DEF sq[2] has 2.
    Main plan: buy him in GW8 with the free transfer. No transfer now: the same move in GW9
    (2 FT then). Unweighted: GW8 +2 (4 - 2), GW9 and GW10 equal -> this GW +2, over GW8-10 +2."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    add(pool, 101, 2, 5.0, {g: 4.0 for g in H3})
    base = plan_squad(pool, H3, W3, sq, 75.0, 1, 'NONE', None, False, False)
    assert base['plan'][8]['in'] == [101]
    gain, seconds = transfer_gain(pool, H3, W3, sq, 75.0, 1, base, use_vice=False, use_bench=False)
    assert gain == {'this_gw': pytest.approx(2.0, abs=1e-6), 'next_gws': pytest.approx(2.0, abs=1e-6), 'n_gws': 3}
    assert seconds >= 0


def test_transfer_gain_needs_a_complete_main_plan():
    """C12: no number from an incomplete solve."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    base = plan_squad(pool, H3, W3, sq, 75.0, 1, 'NONE', None, False, False)
    gain, _ = transfer_gain(pool, H3, W3, sq, 75.0, 1, dict(base, complete=False), use_vice=False, use_bench=False)
    assert gain is None


# --------------------------------------------------------------------------- through the engine
def test_engine_gain_equals_the_decision_check(season, plain):
    """Lookahead 1 (weight 1): the gain of the main plan's transfers over none is the G of the
    decision check of the same transfers (same two solves), within their gaps and the tie-breaks."""
    pg = plain.plan[0]
    assert pg.transfers, "the synthetic plan makes transfers in GW8"
    nt = plain.no_transfer
    assert nt is not None and nt['n_gws'] == 1
    assert nt['this_gw'] == pytest.approx(nt['next_gws'], abs=1e-9)          # one GW
    r = run(season, check='TRANSFERS', check_out=[t.out_id for t in pg.transfers],
            check_in=[t.in_id for t in pg.transfers])
    x = r.decision.result
    tol = sum(v for v in x['gaps'].values() if v) + 0.001 * len(pg.transfers) + 1e-6
    assert nt['this_gw'] == pytest.approx(x['g'], abs=tol)
    assert plain.timings['no_transfer_s'] > 0


def test_no_gain_without_the_option(season):
    assert run(season, transfer_gain=False).no_transfer is None


def test_deadline_of_the_current_gw(season, plain):
    event = next(e for e in season.events if e['id'] == plain.current_gw)
    assert plain.deadline == event['deadline_time']
    assert build_summary(plain)['text'][1].startswith("Deadline: ")


def test_selling_value_range(season, plain, messages):
    """Exact purchase prices: no range, no alert. A team created mid-season (purchase price of P57
    not exactly known, test_squad): £96.2m-£96.3m -> the alert, and the code in the warnings
    ('messages' sets the logger to INFO, as the command line does: coded INFO lines are collected)."""
    assert plain.squad_value_high == pytest.approx(plain.squad_value)
    assert 'selling_value_range' not in [a['code'] for a in build_summary(plain)['alerts']]
    r = run(season, manager=build_manager(season, first_gw=2))
    assert r.squad_value_high == pytest.approx(r.squad_value + 0.1)
    assert 'selling_value_range' in [w['code'] for w in r.warnings]
    s = build_summary(r)
    assert any(a['code'] == 'selling_value_range' and '£0.1m' in a['message'] for a in s['alerts'])
    r = run(season, manager=build_manager(season, first_gw=2), override_budget=100.0)
    assert r.squad_value_high is None                    # an exact budget from the user


def test_summary_text_has_the_gain(plain):
    nt = plain.no_transfer
    x = " ".join("\n".join(build_summary(plain)['text']).split())
    assert f"Gain vs no transfer: {nt['this_gw']:+.1f} points this gameweek" in x


@pytest.mark.parametrize("chip, expected", [
    ('TC', ["You are playing Triple Captain in GW8: the plan above uses it.", "tripled (Triple Captain)"]),
    ('BB', ["Bench Boost: the bench players score too."]),
    ('WC', ["Free transfers: unlimited (Wildcard)", "Sell: ", "Buy: "]),
    ('FH', ["Free transfers: unlimited (Free Hit)", "After GW8 your squad goes back to the one you have now."]),
])
def test_summary_with_a_chip_played(season, chip, expected):
    """--chip: the summary says the plan uses it; WC/FH list sales and buys, no 'no transfer' solve."""
    r = run(season, chip=chip, chip_report=True)
    x = " ".join("\n".join(build_summary(r)['text']).split())
    for e in expected:
        assert e in x
    if chip in ('WC', 'FH'):
        assert r.no_transfer is None
