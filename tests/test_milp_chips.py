"""Chip variables in the MILP (chat 7, C2-C5) and the one-GW chip estimates (C8-C9).
Expected values are computed by hand (reference output: harness_7.py in baseline_09a.txt)."""
import pytest

from toy_pools import add, base_squad, chip_gws, fill, toy_pool, weights
from fpl_v2 import (CHIP_TIEBREAK, CHIPS, plan_points, plan_squad, poisson_binomial_tail,
                    solve_plan, squad_value)

H3 = [8, 9, 10]
W3 = weights(H3)                                   # 1, 0.85, 0.7225
NO_TERMS = dict(use_vice=False, use_bench=False)


def test_triple_captain():
    """Captain 20 xP, 30 xP in GW10: Delta_TC = 20, 0.85*20 = 17, 0.7225*30 = 21.675 -> GW10."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    pool['xp'][(sq[14], 10)] = 30.0
    z0 = solve_plan(pool, H3, W3, sq, 75.0, 1, **NO_TERMS)['objective']
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, chip='AUTO', chips_avail={'TC': [(1, 19)]}, **NO_TERMS)
    assert chip_gws(res) == [10]
    assert res['objective'] - z0 == pytest.approx(0.7225 * 30 - CHIP_TIEBREAK, abs=1e-6)       # 21.674
    # the vice-captain term is tripled too: captain P(app) 0.5 in GW10, vice-captain 10 xP
    pool['papp'][(sq[14], 10)] = 0.5
    pool['xp'][(sq[13], 10)] = 10.0
    z0 = solve_plan(pool, H3, W3, sq, 75.0, 1, use_vice=True, use_bench=False)['objective']
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, chip='AUTO', chips_avail={'TC': [(1, 19)]},
                     use_vice=True, use_bench=False)
    assert chip_gws(res) == [10]
    assert res['objective'] - z0 == pytest.approx(0.7225 * (30 + 0.5 * 10) - CHIP_TIEBREAK, abs=1e-6)  # 25.2865


def test_bench_boost():
    """Bench of the kept squad = 4 x 2 xP = 8 in every GW: Delta_BB = 8, 6.8, 5.78 -> GW8."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    pool['xp'][(sq[1], 9)] = 10.0                 # 2nd GK starts in GW9, the 1st (2 xP) is benched
    kw = dict(NO_TERMS, max_hits=0)
    z0 = solve_plan(pool, H3, W3, sq, 75.0, 1, **kw)['objective']
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, chip='AUTO', chips_avail={'BB': [(1, 19)]}, **kw)
    assert chip_gws(res) == [8]
    assert res['objective'] - z0 == pytest.approx(8.0 - CHIP_TIEBREAK, abs=1e-6)                 # 7.999
    # every player +1 in GW9 -> bench 4 x 3 = 12: Delta(GW9) = 0.85*12 = 10.2 > 8
    for i in sq:
        pool['xp'][(i, 9)] += 1.0
    z0 = solve_plan(pool, H3, W3, sq, 75.0, 1, **kw)['objective']
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, chip='AUTO', chips_avail={'BB': [(1, 19)]}, **kw)
    assert sorted(pool['xp'][(i, 9)] for i in sq)[:4] == [3.0, 3.0, 3.0, 3.0]
    assert chip_gws(res) == [9]
    assert res['objective'] - z0 == pytest.approx(0.85 * 12 - CHIP_TIEBREAK, abs=1e-6)           # 10.199
    # with autosub value, BB gains the bench xP minus the autosub value it replaces:
    # GW9: 0.85*(12 - 3*(0.5 + 0.15 + 0.03)) = 8.466 > GW8: 8 - 2*0.68 = 6.64
    beta = {g: [0.5, 0.15, 0.03] for g in H3}
    kw = dict(use_vice=False, use_bench=True, beta=beta, max_hits=0)
    z0 = solve_plan(pool, H3, W3, sq, 75.0, 1, **kw)['objective']
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, chip='AUTO', chips_avail={'BB': [(1, 19)]}, **kw)
    assert chip_gws(res) == [9]
    assert res['objective'] - z0 == pytest.approx(0.85 * (12 - 3 * 0.68) - CHIP_TIEBREAK, abs=1e-6)  # 8.465


def test_wildcard():
    """3 target MIDs +2 xP per GW, 1 FT, no hits.
    No WC: one per GW: 2*(1 + .85 + .7225) + 2*(.85 + .7225) + 2*.7225 = 9.735.
    WC in GW8: all three now: 3*2*2.5725 = 15.435 -> Delta 5.70 (GW9: 1.70, GW10: 0)."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    for k in (101, 102, 103):
        add(pool, k, 3, 5.0, {g: 4.0 for g in H3})
    kw = dict(NO_TERMS, max_hits=0)
    z0 = solve_plan(pool, H3, W3, sq, 75.0, 1, **kw)
    assert [len(z0['plan'][g]['in']) for g in H3] == [1, 1, 1]
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, chip='AUTO', chips_avail={'WC': [(2, 19)]}, **kw)
    assert chip_gws(res) == [8]
    assert sorted(res['plan'][8]['in']) == [101, 102, 103]
    assert res['objective'] - z0['objective'] == pytest.approx(15.435 - 9.735 - CHIP_TIEBREAK, abs=1e-6)
    assert res['plan'][9]['ft'] == 1                       # F9 = F8: FTs kept, none added
    for g, delta in ((8, 5.7), (9, 1.7), (10, 0.0)):
        zf = solve_plan(pool, H3, W3, sq, 75.0, 1, chip_fix={('WC', g): 1}, **kw)['objective']
        assert zf - z0['objective'] == pytest.approx(delta, abs=1e-6)


def test_free_hit():
    """Four MIDs with 10 xP in GW9 only, 1 FT, no hits.
    No FH: bank the GW8 FT, buy 2 in GW9: 2*8*0.85 = 13.6. FH in GW9: 4 for one GW: 27.2, squad
    reverts in GW10. Delta = 13.6 - 0.002 (two more transfers) - 0.001 (chip) = 13.597."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    for k in (101, 102, 103, 104):
        add(pool, k, 3, 5.0, {8: 0.0, 9: 10.0, 10: 0.0})
    kw = dict(NO_TERMS, max_hits=0)
    z0 = solve_plan(pool, H3, W3, sq, 75.0, 1, **kw)
    res = solve_plan(pool, H3, W3, sq, 75.0, 1, chip='AUTO', chips_avail={'FH': [(2, 19)]}, **kw)
    assert chip_gws(res) == [9]
    assert res['objective'] - z0['objective'] == pytest.approx(27.2 - 13.6 - 0.002 - CHIP_TIEBREAK, abs=1e-6)
    assert set(res['plan'][10]['squad']) == set(res['plan'][8]['squad'])   # permanent squad of GW8
    assert res['plan'][10]['in'] == []
    assert res['plan'][10]['ft'] == res['plan'][9]['ft'] == 1             # F10 = F9
    # equivalent optimal plans: 1 target bought in GW8 + 3 with the FH, or 4 with the FH
    assert len(set(res['plan'][8]['in'] + res['plan'][9]['in']) & {101, 102, 103, 104}) == 4


def test_chip_rules():
    """K1 at most one chip per GW, K2 each chip once per set, K3 no Free Hit in consecutive GWs."""
    h = [18, 19, 20, 21]
    pool = toy_pool(); fill(pool, h); sq = base_squad(pool)
    for i in sq:                                  # TC and BB both best in GW19
        pool['xp'][(i, 19)] = 3.0
    pool['xp'][(sq[14], 19)] = 40.0
    res = solve_plan(pool, h, weights(h), sq, 75.0, 1, chip='AUTO',
                     chips_avail={'TC': [(1, 19), (20, 38)], 'BB': [(1, 19)]}, max_hits=0, **NO_TERMS)
    per_gw = {g: sum(1 for c in CHIPS if (res['values'].get(f"chip_{c}_{g}") or 0) > 0.5) for g in h}
    assert per_gw == {18: 1, 19: 1, 20: 1, 21: 0}
    assert {g: p['chip'] for g, p in res['plan'].items()} == {18: 'BB', 19: 'TC', 20: 'TC', 21: None}
    pool = toy_pool(); fill(pool, h); sq = base_squad(pool)
    for k in (101, 102, 103, 104):                # Free Hit spike in GW19 and GW20
        add(pool, k, 3, 5.0, {18: 0.0, 19: 10.0, 20: 10.0, 21: 0.0})
    res = solve_plan(pool, h, weights(h), sq, 75.0, 1, chip='AUTO', chips_avail={'FH': [(2, 19), (20, 38)]},
                     max_hits=0, **NO_TERMS)
    assert [g for g, p in res['plan'].items() if p['chip'] == 'FH'] == [19]


# Reference objectives of the toy equivalence check (baseline_09a.txt, harness_7.py equiv):
# (chip, GW, vice/bench terms) -> (variable formulation, fixed chip, no chip)
EQUIV = {
    ('WC', 8, False): (180.1790, 180.1800, 168.4800), ('WC', 8, True): (183.6766, 183.6776, 171.8842),
    ('WC', 9, False): (174.1790, 174.1800, 168.4800), ('WC', 9, True): (177.6766, 177.6776, 171.8842),
    ('WC', 10, False): (169.0790, 169.0800, 168.4800), ('WC', 10, True): (172.5256, 172.5266, 171.8842),
    ('FH', 8, False): (172.1880, 172.1890, 168.4800), ('FH', 8, True): (175.6866, 175.6876, 171.8842),
    ('FH', 9, False): (171.2880, 171.2890, 168.4800), ('FH', 9, True): (174.7422, 174.7432, 171.8842),
    ('FH', 10, False): (169.0790, 169.0800, 168.4800), ('FH', 10, True): (172.5256, 172.5266, 171.8842),
    ('BB', 8, False): (176.4790, 176.4800, 168.4800), ('BB', 8, True): (178.5232, 178.5242, 171.8842),
    ('BB', 9, False): (173.5790, 173.5800, 168.4800), ('BB', 9, True): (175.8782, 175.8792, 171.8842),
    ('BB', 10, False): (173.2355, 173.2365, 168.4800), ('BB', 10, True): (175.7005, 175.7015, 171.8842),
    ('TC', 8, False): (188.4790, 188.4800, 168.4800), ('TC', 8, True): (191.8832, 191.8842, 171.8842),
    ('TC', 9, False): (185.4790, 185.4800, 168.4800), ('TC', 9, True): (188.8832, 188.8842, 171.8842),
    ('TC', 10, False): (182.9290, 182.9300, 168.4800), ('TC', 10, True): (186.3332, 186.3342, 171.8842),
}


@pytest.mark.parametrize("chip, gw, terms", sorted(EQUIV, key=lambda k: (CHIPS.index(k[0]), k[1], k[2])))
def test_chip_variable_equals_fixed_formulation(chip, gw, terms):
    """With the chip allowed only in GW g, the variable formulation (C2-C5) must give
    max(Z(chip fixed in g) - 0.001, Z(no chip)); fixed chips use the separate code paths of M3."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    for k in (101, 102, 103):
        add(pool, k, 3, 5.0, {g: 4.0 for g in H3})
    add(pool, 104, 4, 5.0, {8: 9.0, 9: 0.0, 10: 0.0})
    add(pool, 105, 1, 4.0, {8: 1.0, 9: 6.0, 10: 1.0})            # bench GK strong in GW9 (BB)
    kw = dict(use_vice=terms, use_bench=terms, beta={h: [0.5, 0.15, 0.03] for h in H3} if terms else None)
    z_none = solve_plan(pool, H3, W3, sq, 75.0, 1, **kw)['objective']
    z_fix = solve_plan(pool, H3, W3, sq, 75.0, 1, chip_fix={(chip, gw): 1}, **kw)['objective']
    z_var = solve_plan(pool, H3, W3, sq, 75.0, 1, chip='AUTO', chips_avail={chip: [(gw, gw)]}, **kw)['objective']
    assert z_var == pytest.approx(max(z_none, z_fix - CHIP_TIEBREAK), abs=1e-6)
    assert (z_var, z_fix, z_none) == pytest.approx(EQUIV[(chip, gw, terms)], abs=5e-5)


def test_squad_value_and_chip_estimates():
    """One-GW value of a fixed squad: 10 starters x 2 + captain 20 (x2) = 60; TC +20; BB +8.
    With P(app) 0.9 for all: beta = 1 - 0.9^10 = 0.6513, 0.2639, 0.0702; autosubs = bench GK
    0.1*2 + 2*(0.6513 + 0.2639 + 0.0702) = 2.1708; vice 0.1*2 = 0.2; BB 8 - 2.1708; TC 20.2."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    v0 = squad_value(pool, sq, 8, 'NONE', use_vice=False, use_bench=False)
    assert v0 == pytest.approx(60.0, abs=1e-6)
    assert squad_value(pool, sq, 8, 'TC', use_vice=False, use_bench=False) - v0 == pytest.approx(20.0, abs=1e-6)
    assert squad_value(pool, sq, 8, 'BB', use_vice=False, use_bench=False) - v0 == pytest.approx(8.0, abs=1e-6)
    for i in pool['ids']:
        for g in H3:
            pool['papp'][(i, g)] = 0.9
    b = poisson_binomial_tail([0.1] * 10)
    assert b == pytest.approx([0.6513, 0.2639, 0.0702], abs=5e-5)
    auto = 0.1 * 2 + 2 * sum(b)
    v0 = squad_value(pool, sq, 8, 'NONE')
    assert v0 == pytest.approx(60 + 0.2 + auto, abs=1e-6)                         # 62.3708
    assert squad_value(pool, sq, 8, 'TC') - v0 == pytest.approx(20.2, abs=1e-6)
    assert squad_value(pool, sq, 8, 'BB') - v0 == pytest.approx(8 - auto, abs=1e-6)   # 5.8292
    # Free Hit estimate: four MIDs with 10 xP in GW9 -> one-GW gain 4*(10 - 2) = 32
    for k in (101, 102, 103, 104):
        add(pool, k, 3, 5.0, {8: 0.0, 9: 10.0, 10: 0.0})
    v0 = squad_value(pool, sq, 9, 'NONE', use_vice=False, use_bench=False)
    sub = dict(pool)
    sub['sell'] = {i: pool['price'][i] for i in sq}
    r = plan_squad(sub, [9], {9: 1.0}, sq, 75.0, 1, 'FH', None, False, False)
    assert plan_points(r, sub, [9], False) - v0 == pytest.approx(32.0, abs=1e-6)
