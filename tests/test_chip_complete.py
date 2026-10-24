"""C12: chip numbers are either computed in full or left out, so they do not depend on the speed
of the machine. The time limit is simulated (a solve marked as stopped by time, a pass that is not
done), so these tests do not depend on the speed of the machine either."""
import pytest

import fplopt.chips as chips
import fplopt.milp as milp
from fplopt.chips import advise, chip_report
from fplopt.milp import plan_squad
from toy_pools import base_squad, fill, toy_pool, weights

H3 = [8, 9, 10]


def toy():
    pool = toy_pool()
    fill(pool, H3, papp=0.9)
    sq = base_squad(pool)
    return pool, sq


def test_plan_squad_complete_flag(monkeypatch):
    pool, sq = toy()
    w = weights(H3)
    assert plan_squad(pool, H3, w, sq, 75.0, 1)['complete'] is True
    # the passes stop by the rule (here: the pass cap, the weights never count as converged)
    monkeypatch.setattr(milp, 'BENCH_BETA_TOL', -1.0)
    res = plan_squad(pool, H3, w, sq, 75.0, 1)
    assert res['passes'] == milp.BENCH_MAX_PASSES and res['complete'] is True
    # a pass the rule wants is not done: no time left for it
    monkeypatch.setattr(milp, 'MIN_PASS_SECONDS', 10 ** 9)
    res = plan_squad(pool, H3, w, sq, 75.0, 1, time_limit=60)
    assert res['passes'] == 1 and res['complete'] is False
    monkeypatch.undo()
    # a pass stopped by the time limit before the gap target (its plan is kept, but not complete)
    original = milp.solve_plan

    def stopped(*args, **kwargs):
        res = original(*args, **kwargs)
        res['proven'] = False
        return res
    monkeypatch.setattr(milp, 'solve_plan', stopped)
    res = plan_squad(pool, H3, w, sq, 75.0, 1)
    assert 'plan' in res and res['complete'] is False


def test_chip_report_leaves_out_cut_solves(monkeypatch):
    pool, sq = toy()
    w = weights(H3)
    plan = plan_squad(pool, H3, w, sq, 75.0, 1)['plan']
    windows = {'TC': [(2, 10)], 'BB': [(2, 10)]}
    args = (pool, H3, w, plan, sq, 75.0, 1, windows, None, True, True, 10 ** 6)
    full = chip_report(*args)
    assert set(full['exact']) == {'TC', 'BB'} and not full['skipped']
    # the Bench Boost solves are cut by the time limit: every BB number is left out, TC is unchanged
    original = chips.plan_squad

    def cut_bb(*a, **k):
        res = original(*a, **k)
        if a[6] == 'BB':                                       # chip argument of plan_squad
            res['complete'] = False
        return res
    monkeypatch.setattr(chips, 'plan_squad', cut_bb)
    part = chip_report(*args)
    assert 'BB' not in part['exact'] and 'BB' not in part['est']
    assert part['exact']['TC'] == pytest.approx(full['exact']['TC'], abs=1e-9)
    assert part['est']['TC'] == pytest.approx(full['est']['TC'], abs=1e-9)
    assert 'exact BB' in part['skipped'] and 'estimate BB GW8' in part['skipped']
    adv = advise(part, 8)
    assert adv['incomplete'] == ['BB'] and 'BB' not in [h[0] for h in adv['holds']]


def test_advise_needs_every_number_of_the_window():
    """TC: exact Delta and an estimate for every GW to the end of the window, best now -> play.
    BB: best estimate now and positive exact Delta, but GW10 is missing -> no suggestion at all
    (with the missing GW it could be a 'hold'). FH: exact Delta missing -> no suggestion."""
    rep = {'window': {'TC': 10, 'BB': 10, 'FH': 10},
           'exact': {'TC': 5.0, 'BB': 9.0},
           'est': {'TC': {8: 4.0, 9: 3.0, 10: 3.5}, 'BB': {8: 6.0, 9: 2.0}, 'FH': {8: 1.0, 9: 1.0, 10: 1.0}}}
    adv = advise(rep, 8)
    assert adv['pick'] == 'TC'
    assert sorted(adv['incomplete']) == ['BB', 'FH']
    assert adv['holds'] == [] and adv['also'] == []
