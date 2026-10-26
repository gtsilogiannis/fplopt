"""Decision check (decision.py): G, R and the gain of the action for Triple Captain and Bench Boost.
Toy pool (toy_pools.py): 15 owned players, the last forward (sq[14]) has 20 xP in every GW, every
other player 2 xP, price 5.0, budget 75.0, 1 FT. Expected values are computed by hand."""
import pytest

from toy_pools import add, base_squad, fill, toy_pool, weights
import fplopt.decision
from fplopt.chips import chip_report, estimate_gw
from fplopt.decision import VERDICTS, check_decision, plan_value, verdict
from fplopt.milp import TRANSFER_TIEBREAK, plan_squad

H3 = [8, 9, 10]
W3 = weights(H3)                                   # 1, 0.85, 0.7225
NO_TERMS = dict(use_vice=False, use_bench=False)


def main_plan(pool, sq, weight=W3, **kw):
    """Z0: the main plan as the engine solves it (no chip, bench-weight fixed point)."""
    kw = dict(NO_TERMS, **kw)
    return plan_squad(pool, H3, weight, sq, 75.0, 1, 'NONE', kw.pop('max_hits', None),
                      kw.pop('use_vice'), kw.pop('use_bench'))


def check(pool, sq, base, chip, gw, captain=None, weight=W3, **kw):
    return check_decision(pool, H3, weight, sq, 75.0, 1, base, chip, gw, captain, **dict(NO_TERMS, **kw))


def varied_pool():
    """xP 1.0-6.0 and P(app) 0.6-1.0 varied by player and GW; sq[14] keeps 20 xP (always captain)."""
    pool = toy_pool()
    for i in pool['ids']:
        for g in H3:
            pool['xp'][(i, g)] = 1.0 + ((7 * i + 3 * g) % 11) / 2
            pool['papp'][(i, g)] = 0.6 + ((5 * i + g) % 5) / 10
    return pool, base_squad(pool)


def test_plan_value_is_the_objective_without_tie_breaks():
    """J = objective + 0.001 per transfer (no chip, FT_END_VALUE 0), with every term of the
    objective on: vice-captain, autosubs with the bench weights of the solve, hits, three target
    MIDs (+2 xP per GW) bought over the horizon."""
    pool, sq = varied_pool()
    for k in (101, 102, 103):
        add(pool, k, 3, 5.0, {g: 8.0 for g in H3}, {g: 0.9 for g in H3})
    res = plan_squad(pool, H3, W3, sq, 75.0, 1, 'NONE', None, True, True)
    n_in = sum(len(res['plan'][g]['in']) for g in H3)
    assert n_in >= 1
    assert plan_value(res, pool, H3, W3, True) == pytest.approx(res['objective'] + TRANSFER_TIEBREAK * n_in, abs=1e-6)


def test_triple_captain_by_hand():
    """GW8 (weight 1), GW9-10 equal in all three plans:
      Z0 = XI 40 + captain 20 = 60;  Z+ (TC, model's captain sq[14]) = 40 + 2*20 = 80;
      Z(D) (TC on sq[13], 2 xP) = 40 + 2*2 = 44.
    G = 44 - 60 = -16, R = 80 - 44 = 36 = 2*(20 - 2), gain of the action = 20 = G + R."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    base = main_plan(pool, sq)
    dc = check(pool, sq, base, 'TC', 8, sq[13])
    assert dc['status'] == 'computed' and dc['not_computed'] == []
    assert dc['g'] == pytest.approx(-16.0, abs=1e-6)
    assert dc['r'] == pytest.approx(36.0, abs=1e-6)
    assert dc['gain_action'] == pytest.approx(20.0, abs=1e-6)
    assert dc['g'] + dc['r'] == pytest.approx(dc['gain_action'], abs=1e-9)
    assert dc['points_gw'] == pytest.approx({'z0': 60.0, 'z_best': 80.0, 'z_decision': 44.0}, abs=1e-6)
    assert dc['plan_decision']['captain'] == sq[13] and dc['plan_best']['captain'] == sq[14]
    assert dc['captain_bought'] is None
    assert dc['g_low'] <= dc['g'] <= dc['g_high'] and dc['r_low'] <= dc['r'] <= dc['r_high']


def test_triple_captain_later_gw_is_weighted():
    """The same decision in GW10 (weight 0.7225): G = -16 * 0.7225 = -11.56, R = 36 * 0.7225 = 26.01;
    the points of GW10 itself are unweighted (D2)."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    dc = check(pool, sq, main_plan(pool, sq), 'TC', 10, sq[13])
    assert dc['g'] == pytest.approx(-16.0 * 0.7225, abs=1e-6)
    assert dc['r'] == pytest.approx(36.0 * 0.7225, abs=1e-6)
    assert dc['points_gw']['z_decision'] - dc['points_gw']['z0'] == pytest.approx(-16.0, abs=1e-6)


def test_triple_captain_on_the_models_captain():
    """K2: the user's captain is the model's -> R = 0 within the gaps (all terms on)."""
    pool, sq = varied_pool()
    beta_kw = dict(use_vice=True, use_bench=True)
    dc = check(pool, sq, main_plan(pool, sq, **beta_kw), 'TC', 9, sq[14], **beta_kw)
    assert dc['plan_best']['captain'] == sq[14]
    # both solves stop within their gaps (0.05%), so R = 0 holds only up to them
    assert dc['r_low'] - 1e-9 <= 0.0 <= dc['r_high'] + 1e-9


def test_triple_captain_never_beats_the_best_version():
    """K3: R >= 0 within the gaps for every owned player as the TC captain in GW8 (all terms on)."""
    pool, sq = varied_pool()
    beta_kw = dict(use_vice=True, use_bench=True)
    base = main_plan(pool, sq, **beta_kw)
    for i in sq:
        dc = check(pool, sq, base, 'TC', 8, i, **beta_kw)
        assert dc['plan_decision']['captain'] == i
        assert dc['r_high'] >= -1e-9                       # the interval of R reaches 0 or above
        assert dc['r'] >= -dc['gaps']['z_decision'] - 1e-9


def test_triple_captain_on_a_player_not_owned():
    """O1: the captain is bought with the free transfer in GW8 (no hit), selling a 2-xP forward:
    G = -16 as above (plan_value has no tie-breaks), and the purchase GW is reported."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    new_fwd = next(i for i in pool['ids'] if pool['pos'][i] == 4 and i not in sq)
    dc = check(pool, sq, main_plan(pool, sq), 'TC', 8, new_fwd)
    assert dc['g'] == pytest.approx(-16.0, abs=1e-6)
    assert dc['captain_bought'] == 8
    assert dc['plan_decision']['hits'] == 0


def test_bench_boost_by_hand():
    """Bench of the kept squad = 4 x 2 xP = 8 in GW8 (weight 1): G = gain = 8, R = 0 (no choice)."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    dc = check(pool, sq, main_plan(pool, sq, max_hits=0), 'BB', 8, max_hits=0)
    assert dc['g'] == pytest.approx(8.0, abs=1e-6)
    assert dc['r'] == 0.0
    assert dc['gain_action'] == pytest.approx(8.0, abs=1e-6)
    assert dc['plan_decision'] is dc['plan_best']


@pytest.mark.parametrize("chip", ['TC', 'BB'])
def test_gain_matches_the_chip_analysis(chip):
    """K4: with every weight 1, the gain of the action in the current GW (Z+ - Z0) equals the
    exact Delta of the chip analysis (chips.chip_report: other code path, the chip passed as
    'chip' instead of chip_fix, points summed by plan_points), within the solver gaps."""
    pool, sq = varied_pool()
    w1 = {g: 1.0 for g in H3}
    beta_kw = dict(use_vice=True, use_bench=True)
    base = main_plan(pool, sq, weight=w1, **beta_kw)
    rep = chip_report(pool, H3, w1, base['plan'], sq, 75.0, 1, {chip: [(8, 10)]}, None, True, True, 600)
    dc = check(pool, sq, base, chip, 8, sq[14] if chip == 'TC' else None, weight=w1, **beta_kw)
    tol = rep['gap'][chip] + dc['gaps']['z_best'] + dc['gaps']['z0'] + 1e-6
    assert dc['gain_action'] == pytest.approx(rep['exact'][chip], abs=tol)


def test_incomplete_main_plan_gives_no_numbers_that_need_it():
    """C12: a main plan cut by the time limit -> no G and no gain of the action; R does not use
    Z0, so it is still given."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    base = dict(main_plan(pool, sq), complete=False)
    dc = check(pool, sq, base, 'TC', 8, sq[13])
    assert dc['status'] == 'not_computed'
    assert dc['not_computed'] == ['Z0 (main plan)']
    assert dc['g'] is None and dc['gain_action'] is None and dc['z0'] is None
    assert dc['r'] == pytest.approx(36.0, abs=1e-6)


@pytest.mark.parametrize("chip, gw, captain", [('WC', 8, None), ('TC', 11, 15), ('TC', 8, None), ('BB', 8, 15)])
def test_invalid_requests(chip, gw, captain):
    """Unsupported chip, GW outside the horizon, TC without a captain, a captain with BB."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    with pytest.raises(ValueError):
        check(pool, sq, {'plan': {}}, chip, gw, captain)


# --------------------------------------------------------------------------- step 3: timing
def test_estimate_gw_by_hand():
    """One-GW estimates on the kept squad (chips.estimate_gw, moved out of chip_report):
    TC = the captain's 20 xP once more, BB = bench 4 x 2 = 8."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    assert estimate_gw(pool, sq, 8, 8, ['TC', 'BB'], 75.0, None, False, False) == pytest.approx(
        {'TC': 20.0, 'BB': 8.0}, abs=1e-6)


def test_timing_finds_a_better_gw_later():
    """Captain 30 xP in GW10: estimates TC = 20, 20, 30 for GW8-10. A TC in GW8 gains (G = +20,
    R = 0) but GW10 is better: is_best False. A TC in GW10 is the best GW."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    pool['xp'][(sq[14], 10)] = 30.0
    base = main_plan(pool, sq)
    dc = check(pool, sq, base, 'TC', 8, sq[14], window_end=10)
    assert dc['g'] == pytest.approx(20.0, abs=1e-6) and dc['r'] == pytest.approx(0.0, abs=1e-6)
    t = dc['timing']
    assert t['estimates'] == pytest.approx({8: 20.0, 9: 20.0, 10: 30.0}, abs=1e-6)
    assert (t['at_gw'], t['best_gw'], t['best'], t['is_best']) == pytest.approx((20.0, 10, 30.0, False), abs=1e-6)
    dc = check(pool, sq, base, 'TC', 10, sq[14], window_end=10)
    assert dc['timing']['is_best'] is True
    assert dc['status'] == 'computed'


def test_timing_tie_counts_as_best():
    """Bench Boost = 8 in every GW: GW9 ties with GW8 (the earliest is reported as best_gw), and a
    tie counts as the best GW, as in the chip suggestions."""
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    dc = check(pool, sq, main_plan(pool, sq, max_hits=0), 'BB', 9, max_hits=0, window_end=10)
    assert dc['timing']['best_gw'] == 8
    assert dc['timing']['is_best'] is True


def test_timing_with_a_missing_estimate(monkeypatch):
    """C12: one GW of the window not estimated -> no verdict on the timing, the GW is listed;
    G and R do not depend on it and are still given."""
    real = fplopt.decision.estimate_gw

    def cut_gw9(pool, q, t, *args, **kw):
        out = real(pool, q, t, *args, **kw)
        return {cp: (None if t == 9 else v) for cp, v in out.items()}

    monkeypatch.setattr(fplopt.decision, 'estimate_gw', cut_gw9)
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    dc = check(pool, sq, main_plan(pool, sq), 'TC', 8, sq[13], window_end=10)
    assert dc['timing']['estimates'] == pytest.approx({8: 20.0, 10: 20.0}, abs=1e-6)
    assert dc['timing']['best_gw'] is None and dc['timing']['is_best'] is None
    assert dc['not_computed'] == ['estimate TC GW9'] and dc['status'] == 'not_computed'
    assert dc['g'] == pytest.approx(-16.0, abs=1e-6)


def test_gw_after_the_window_is_an_error():
    pool = toy_pool(); fill(pool, H3); sq = base_squad(pool)
    with pytest.raises(ValueError):
        check(pool, sq, {'plan': {}}, 'TC', 10, sq[14], window_end=9)


# --------------------------------------------------------------------------- verdict (spec 0.4)
def _res(g, g_lo, g_hi, r=0.0, r_lo=-0.1, r_hi=0.1, is_best=True):
    return {'g': g, 'g_low': g_lo, 'g_high': g_hi, 'r': r, 'r_low': r_lo, 'r_high': r_hi,
            'timing': {'is_best': is_best}}


@pytest.mark.parametrize("res, codes", [
    (_res(-3.0, -3.1, -2.9), ['worse_than_none']),                       # G below 0 beyond the gaps
    (_res(0.05, -0.05, 0.10), ['no_clear_gain']),                       # G within the gaps around 0
    (_res(5.0, 4.9, 5.1), ['agrees']),
    (_res(5.0, 4.9, 5.1, is_best=False), ['better_gw']),
    (_res(5.0, 4.9, 5.1, r=2.0, r_lo=1.9, r_hi=2.1), ['better_version']),
    (_res(5.0, 4.9, 5.1, r=2.0, r_lo=1.9, r_hi=2.1, is_best=False), ['better_gw', 'better_version']),
    (_res(5.0, 4.9, 5.1, is_best=None), []),                            # timing incomplete (C12)
    (_res(None, None, None), []),                                       # G not computed (C12)
    (dict(_res(5.0, 4.9, 5.1), timing=None), ['agrees']),               # no timing asked
])
def test_verdict(res, codes):
    """The verdict uses only the gap bounds and the timing rule; every code it returns is defined."""
    assert verdict(res) == codes
    assert set(codes) <= set(VERDICTS)
