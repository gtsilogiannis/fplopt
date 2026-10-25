"""M11: the bench-weight fixed point keeps the pass with the highest value by its own bench weights."""
import pytest

import fplopt.milp as milp
from fplopt.chips import gw_points
from fplopt.milp import TRANSFER_TIEBREAK, bench_weights, plan_squad
from toy_pools import add, base_squad, fill, toy_pool, weights

H3 = [8, 9, 10]


def toy():
    """Toy squad with doubtful starters (P(app) 0.7) and one better target, so that the XI, and with
    it the bench weights, can change between passes."""
    pool = toy_pool()
    add(pool, 101, 3, 5.0, {8: 4.0, 9: 4.0, 10: 4.0}, {8: 0.6, 9: 0.6, 10: 0.6})
    fill(pool, H3, papp=0.7)
    sq = base_squad(pool)
    return pool, sq


def own_weights_value(res, pool, horizon, weight):
    """V by definition: unweighted points of every GW with the bench weights of the plan's own XI,
    weighted, minus the transfer tie-break (the only other term of the objective here)."""
    v = 0.0
    for g in horizon:
        pg = res['plan'][g]
        beta = bench_weights(pg['xi'], pool['pos'], {i: pool['papp'][(i, g)] for i in pool['ids']})
        v += weight[g] * gw_points(pg, g, pool, beta)['total'] - TRANSFER_TIEBREAK * len(pg['in'])
    return v


def test_self_consistent_value_from_the_objective(monkeypatch):
    """The value computed from the solver's objective equals the definition, in every pass."""
    pool, sq = toy()
    w = weights(H3)
    monkeypatch.setattr(milp, 'BENCH_BETA_TOL', -1.0)            # always the full number of passes
    seen = []
    original = milp.self_consistent_value

    def check(res, pool_, horizon, weight, beta, implied):
        v = original(res, pool_, horizon, weight, beta, implied)
        assert v == pytest.approx(own_weights_value(res, pool_, horizon, weight), abs=1e-6)
        seen.append(v)
        return v
    monkeypatch.setattr(milp, 'self_consistent_value', check)
    res = plan_squad(pool, H3, w, sq, 80.0, 1)
    assert len(seen) == milp.BENCH_MAX_PASSES == res['passes']
    # the plan kept is the best one (the latest of equal ones), and its weights are its own
    assert res['value'] == pytest.approx(max(seen), abs=1e-12)
    assert res['pass'] == max(k + 1 for k, v in enumerate(seen) if v >= max(seen) - 1e-9)
    for g in H3:
        own = bench_weights(res['plan'][g]['xi'], pool['pos'], {i: pool['papp'][(i, g)] for i in pool['ids']})
        assert res['beta'][g] == pytest.approx(own, abs=1e-12)


def test_an_earlier_better_pass_is_kept(monkeypatch):
    """The passes are made to get worse: the value of the first pass is kept, not the last one."""
    pool, sq = toy()
    w = weights(H3)
    monkeypatch.setattr(milp, 'BENCH_BETA_TOL', -1.0)
    values = iter([10.0, 9.0, 9.5])
    monkeypatch.setattr(milp, 'self_consistent_value', lambda *a: next(values))
    res = plan_squad(pool, H3, w, sq, 80.0, 1)
    assert res['passes'] == 3 and res['pass'] == 1 and res['value'] == 10.0
    assert res['beta_used'] == res['beta_history'][0]            # it was solved with the first weights
