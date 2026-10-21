"""Multi-period MILP (chat 6b): free transfers, hits, chips fixed in the first GW, budget with
selling prices, vice-captain, GW1. Expected values are computed by hand in each test
(reference output: harness_6b.py in baseline_09a.txt).

Players with exactly equal xP are interchangeable for the solver, and HiGHS may pick a
different one between runs; the tests therefore check counts, objectives and properties,
and player ids only where the choice is unique."""
import pytest

from toy_pools import add, base_squad, fill, toy_pool, weights
from fpl_v2 import solve_plan

H3 = [8, 9, 10]
W3 = weights(H3)                       # 1, 0.85, 0.7225
NO_TERMS = dict(use_vice=False, use_bench=False)


def n_in(res, g):
    return len(res['plan'][g]['in'])


def test_waits_for_the_next_free_transfer():
    """Two targets each +2 xP per GW, 1 FT: one transfer now and one with next GW's free
    transfer beats two now with a hit: 2*2.5725 + 2*1.5725 = 8.29 (minus 2 x 0.001 tie-break)."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    add(pool, 101, 3, 5.0, {g: 4.0 for g in H3})
    add(pool, 102, 3, 5.0, {g: 4.0 for g in H3})
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, **NO_TERMS)
    assert [n_in(res, g) for g in H3] == [1, 1, 0]
    assert [res['plan'][g]['hits'] for g in H3] == [0, 0, 0]
    assert set(res['plan'][8]['in'] + res['plan'][9]['in']) == {101, 102}
    keep = sum(W3[g] * (2.0 * 10 + 20 * 2) for g in H3)      # XI 10 x 2 + captain 20 x 2
    assert res['objective'] - keep == pytest.approx(2 * 2.5725 + 2 * 1.5725 - 0.002, abs=1e-6)
    # lookahead 1 sees only GW8: one transfer, no hit (a second gains 2 < 4)
    res1 = solve_plan(pool, [8], {8: 1.0}, sq, 75.0, 1, **NO_TERMS)
    assert (n_in(res1, 8), res1['plan'][8]['hits']) == (1, 0)


def test_takes_a_hit_when_it_pays():
    """Two targets +5 per GW: hit now 5*2.5725 - 4 = 8.86 > waiting 5*1.5725 = 7.86."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    add(pool, 101, 3, 5.0, {g: 7.0 for g in H3})
    add(pool, 102, 3, 5.0, {g: 7.0 for g in H3})
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, **NO_TERMS)
    assert sorted(res['plan'][8]['in']) == [101, 102]
    assert [res['plan'][g]['hits'] for g in H3] == [1, 0, 0]
    assert [res['plan'][g]['ft'] for g in H3] == [1, 1, 2]


def test_saves_free_transfers():
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, **NO_TERMS)
    assert all(not pg['in'] for pg in res['plan'].values())
    assert [res['plan'][g]['ft'] for g in H3] == [1, 2, 3]
    res5 = solve_plan(pool, H3, W3, sq, 75.0, 5, **NO_TERMS)      # cap: no 6th free transfer
    assert [res5['plan'][g]['ft'] for g in H3] == [5, 5, 5]


def test_target_good_only_in_the_last_gw():
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    add(pool, 101, 3, 5.0, {8: 2.0, 9: 2.0, 10: 5.0})
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, **NO_TERMS)
    assert [n_in(res, g) for g in H3] == [1, 0, 0]          # the GW8 FT would expire unused otherwise
    assert res['plan'][8]['in'] == [101]
    assert [res['plan'][g]['ft'] for g in H3] == [1, 1, 2]


def chips_pool():
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    for k in (101, 102, 103):                               # good in every GW
        add(pool, k, 3, 5.0, {g: 4.0 for g in H3})
    add(pool, 104, 4, 5.0, {8: 9.0, 9: 0.0, 10: 0.0})        # good only in GW8
    return pool, sq


def test_wildcard_in_the_first_gw():
    pool, sq = chips_pool()
    res = solve_plan(pool, H3, W3, sq, 75.0, 2, chip='WC', **NO_TERMS)
    assert sorted(res['plan'][8]['in']) == [101, 102, 103, 104]
    assert res['plan'][8]['hits'] == 0
    assert [res['plan'][g]['ft'] for g in H3] == [2, 2, 3]   # FTs kept, none added (F9 = F8)


def test_free_hit_in_the_first_gw():
    pool, sq = chips_pool()
    res = solve_plan(pool, H3, W3, sq, 75.0, 2, chip='FH', **NO_TERMS)
    assert sorted(res['plan'][8]['in']) == [101, 102, 103, 104]
    assert set(res['plan'][9]['out']) <= set(sq)            # GW9 transfers start from the reverted squad
    assert [n_in(res, g) for g in H3] == [4, 2, 1]
    assert [res['plan'][g]['ft'] for g in H3] == [2, 2, 1]


def test_budget_selling_price_and_rebuy():
    """Owned player A: bought 7.0, now 7.3 -> sells for 7.1. Bank 0.1. A blanks in GW9, a 7.2
    target is strong only in GW9. Selling A in GW9 is possible; buying him back in GW10 costs
    7.3 > 7.2 available, so the plan must not do it."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    a = sq[5]
    pool['price'][a] = 7.3
    pool['sell'][a] = 7.1
    for g in H3:
        pool['xp'][(a, g)] = 6.0 if g != 9 else 0.0
    budget = sum(pool['sell'][i] for i in sq) + 0.1
    add(pool, 201, 3, 7.2, {8: 0.0, 9: 6.0, 10: 0.0})
    res = solve_plan(pool, H3, W3, sq, budget, 2, **NO_TERMS)
    for g, pg in res['plan'].items():
        kept = lambda i: i in sq and all(i in res['plan'][h]['squad'] for h in H3 if h <= g)
        spend = sum(pool['sell'][i] if kept(i) else pool['price'][i] for i in pg['squad'])
        assert spend <= budget + 1e-9
    assert a in res['plan'][9]['out'] and 201 in res['plan'][9]['in']
    assert a not in res['plan'][10]['squad']
    assert [n_in(res, g) for g in H3] == [0, 2, 0]
    # the error the original-copy variable k prevents: if re-buying cost the selling price,
    # the plan would buy A back in GW10
    pool['price'][a] = 7.1
    res = solve_plan(pool, H3, W3, sq, budget, 2, **NO_TERMS)
    assert a in res['plan'][10]['in']


def test_vice_captain():
    """A 75% captain with a vice-captain: Haaland C = 5.62 + 0.25*5.79 = 7.0675 beats
    Bruno C = 5.79 + 0.03*5.62 = 5.9586. Other 9 starters 2.0 each."""
    pool = toy_pool(); fill(pool, [8]); sq = base_squad(pool)
    pool['xp'][(sq[14], 8)] = 2.0
    hal, bru = sq[12], sq[8]
    pool['xp'][(hal, 8)], pool['papp'][(hal, 8)] = 5.62, 0.75
    pool['xp'][(bru, 8)], pool['papp'][(bru, 8)] = 5.79, 0.97
    res = solve_plan(pool, [8], {8: 1.0}, sq, 75.0, 1, use_vice=False, use_bench=False)
    assert (res['plan'][8]['captain'], res['plan'][8]['vice']) == (bru, None)
    assert res['objective'] == pytest.approx(9 * 2.0 + 5.62 + 2 * 5.79, abs=1e-6)       # 35.20
    res = solve_plan(pool, [8], {8: 1.0}, sq, 75.0, 1, use_vice=True, use_bench=False)
    assert (res['plan'][8]['captain'], res['plan'][8]['vice']) == (hal, bru)
    assert res['objective'] == pytest.approx(9 * 2.0 + 5.79 + 5.62 + 5.62 + 0.25 * 5.79, abs=1e-6)  # 36.4775


def test_before_the_first_deadline():
    """GW1: empty squad, unlimited free transfers, then 1 FT."""
    h = [1, 2, 3]
    pool = toy_pool(); fill(pool, h)
    add(pool, 101, 3, 5.0, {1: 2.0, 2: 6.0, 3: 6.0})
    add(pool, 102, 3, 5.0, {1: 2.0, 2: 6.0, 3: 6.0})
    res = solve_plan(pool, h, weights(h), [], 75.0, None, use_vice=True, use_bench=True,
                     beta={g: [0.5, 0.15, 0.03] for g in h})
    assert res['status'] == 'Optimal'
    assert n_in(res, 1) == 15 and res['plan'][1]['hits'] == 0
    assert {101, 102} <= set(res['plan'][1]['squad'])
    assert [res['plan'][g]['ft'] for g in h] == [None, 1, 2]
    assert res['plan'][2]['captain'] in (101, 102)
