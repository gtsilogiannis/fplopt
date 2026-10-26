"""Decisions of the user held fixed in the MILP (decision check, D1): a captain forced in one GW.
Expected values are computed by hand on the toy pool (toy_pools.py): 15 owned players, the last
forward (sq[14]) has 20 xP in every GW, every other player 2 xP, price 5.0, budget 75.0, 1 FT."""
import pytest

from toy_pools import base_squad, fill, toy_pool, weights
from fplopt.milp import TRANSFER_TIEBREAK, plan_squad, solve_plan

H3 = [8, 9, 10]
W3 = weights(H3)                                   # 1, 0.85, 0.7225
NO_TERMS = dict(use_vice=False, use_bench=False)


def solve(pool, sq, **kw):
    """solve_plan on the toy pool without the vice-captain and bench terms."""
    return solve_plan(pool, H3, W3, sq, 75.0, 1, **dict(NO_TERMS, **kw))


def test_forced_tc_captain_by_hand():
    """GW8 = XI (captain counted once) + extra captain points; GW9-10 are the same in all three.
      Z0   (no chip)                 : XI 40 + 20       = 60
      Z+   (TC, the model's captain) : XI 40 + 2 * 20   = 80
      Z(D) (TC on a 2-xP forward)    : XI 40 + 2 * 2    = 44
    -> G = Z(D) - Z0 = -16, R = Z+ - Z(D) = 2 * (20 - 2) = 36, G + R = 20 = Delta_TC(GW8).
    (The XI is 40 in all three because both forwards start either way; only the multiplier moves.)"""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    z0 = solve(pool, sq)
    zp = solve(pool, sq, chip_fix={('TC', 8): 1})
    zd = solve(pool, sq, chip_fix={('TC', 8): 1}, force_captain={8: sq[13]})
    assert zd['plan'][8]['captain'] == sq[13]
    assert zd['plan'][8]['chip'] == 'TC'
    assert zd['plan'][9]['captain'] == sq[14]                        # only GW8 is forced
    assert zp['objective'] - z0['objective'] == pytest.approx(20.0, abs=1e-6)    # Delta_TC(8)
    assert zd['objective'] - z0['objective'] == pytest.approx(-16.0, abs=1e-6)   # G
    assert zp['objective'] - zd['objective'] == pytest.approx(36.0, abs=1e-6)    # R


def test_forcing_the_models_captain_changes_nothing():
    """R = 0 when the user's captain is the one the model picks (K2), with the vice-captain and
    bench terms on (P(app) 0.9, fixed bench weights)."""
    pool = toy_pool(); fill(pool, H3, papp=0.9); sq = base_squad(pool)
    kw = dict(use_vice=True, use_bench=True, beta={g: [0.5, 0.15, 0.03] for g in H3}, chip_fix={('TC', 9): 1})
    free = solve_plan(pool, H3, W3, sq, 75.0, 1, **kw)
    cap = free['plan'][9]['captain']
    forced = solve_plan(pool, H3, W3, sq, 75.0, 1, force_captain={9: cap}, **kw)
    assert forced['plan'][9]['captain'] == cap
    assert forced['objective'] == pytest.approx(free['objective'], abs=1e-6)


def test_forced_captain_never_beats_the_free_plan():
    """R >= 0 (K3): each of the 15 owned players forced as the GW8 captain, xP and P(app) varied
    by player and GW, all terms on. Forcing only removes plans, so no forced plan can be better."""
    pool = toy_pool()
    for i in pool['ids']:
        for g in H3:
            pool['xp'][(i, g)] = 1.0 + ((7 * i + 3 * g) % 11) / 2    # 1.0 to 6.0
            pool['papp'][(i, g)] = 0.6 + ((5 * i + g) % 5) / 10     # 0.6 to 1.0
    sq = base_squad(pool)                                            # sq[14]: 20 xP
    kw = dict(use_vice=True, use_bench=True, beta={g: [0.5, 0.15, 0.03] for g in H3})
    free = solve_plan(pool, H3, W3, sq, 75.0, 1, **kw)['objective']
    for i in sq:
        res = solve_plan(pool, H3, W3, sq, 75.0, 1, force_captain={8: i}, **kw)
        assert res['plan'][8]['captain'] == i
        assert i in res['plan'][8]['xi']
        assert res['objective'] <= free + 1e-6


def test_forced_captain_outside_the_squad_is_bought():
    """A forced captain who is not owned is bought with the free transfer (no hit), selling one of
    the two 2-xP forwards: GW8 = 44 as in the first test, minus one transfer tie-break."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    new_fwd = next(i for i in pool['ids'] if pool['pos'][i] == 4 and i not in sq)
    z0 = solve(pool, sq)
    zd = solve(pool, sq, chip_fix={('TC', 8): 1}, force_captain={8: new_fwd})
    assert zd['plan'][8]['in'] == [new_fwd]
    assert zd['plan'][8]['hits'] == 0
    assert zd['plan'][8]['out'][0] in (sq[12], sq[13])
    assert zd['objective'] - z0['objective'] == pytest.approx(-16.0 - TRANSFER_TIEBREAK, abs=1e-6)


def test_unavailable_forced_captain_may_start():
    """FPL lets an injured captain start (he scores 0); the decision must still have a plan.
    sq[13] unavailable, 0 xP in GW8.  Z0: XI of the 14 others = 20 + 10 * 2 = 40, captain +20 -> 60.
    Z(D), TC on sq[13]: XI = 20 + 0 + 9 * 2 = 38, captain extra 2 * 0 = 0 -> 38.  G = -22."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    pool['avail'][sq[13]] = False
    pool['xp'][(sq[13], 8)] = 0.0
    pool['papp'][(sq[13], 8)] = 0.0
    z0 = solve(pool, sq)
    zd = solve(pool, sq, chip_fix={('TC', 8): 1}, force_captain={8: sq[13]})
    assert zd['plan'][8]['captain'] == sq[13]
    assert sq[13] not in z0['plan'][8]['xi']                         # A still applies without forcing
    assert zd['objective'] - z0['objective'] == pytest.approx(-22.0, abs=1e-6)


def test_plan_squad_keeps_the_forced_captain():
    """The bench-weight fixed point passes the forced captain to every pass (here a goalkeeper,
    which the formation allows), and the result is complete."""
    pool = toy_pool(); fill(pool, H3, papp=0.9); sq = base_squad(pool)
    res = plan_squad(pool, H3, W3, sq, 75.0, 1, 'NONE', None, True, True, force_captain={10: sq[0]})
    assert res['plan'][10]['captain'] == sq[0]
    assert res['plan'][8]['captain'] == sq[14]
    assert res['complete']


def test_forced_captain_outside_the_pool_is_an_error():
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    with pytest.raises(ValueError):
        solve(pool, sq, force_captain={8: 999})
