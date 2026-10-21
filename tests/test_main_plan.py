"""Checks on the main plan of the synthetic season (GW8, lookahead 3, vice-captain and bench
terms with the real bench weights). Slow: pytest -m slow"""
import io
import contextlib

import pytest

import fpl_v2 as F
from fpl_fake import FakeApi, build_manager

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def main_plan(season, tmp_path_factory):
    """One optimizer run without the chip analysis; returns the arguments and result of the
    main plan_squad call and the printed output."""
    captured = {}
    with pytest.MonkeyPatch.context() as mp:
        api = FakeApi(season, build_manager(season))
        mp.setattr(F, 'fetch_fpl_api', lambda url: api.respond(url.replace(F.FPL_API, '')))
        mp.setattr(F, 'chip_report', lambda *a, **k: {'exact': {}, 'gap': {}, 'est': {}, 'window': {}, 'skipped': []})
        orig = F.plan_squad

        def spy(pool, horizon, weight, *a, **kw):
            res = orig(pool, horizon, weight, *a, **kw)
            captured.update(pool=pool, res=res, horizon=horizon, weight=weight, args=a)
            return res
        mp.setattr(F, 'plan_squad', spy)
        mp.chdir(tmp_path_factory.mktemp("run"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            F.optimize_team(1, None, 'NONE', 75, 3, None, None, None)
    captured['out'] = buf.getvalue()
    return captured


def test_gw_points_equals_expected_points_line(main_plan):
    res, pool = main_plan['res'], main_plan['pool']
    line = [ln for ln in main_plan['out'].splitlines() if ln.startswith('Expected points')][0]
    assert line == "Expected points GW8: XI 45.95 + captain 6.52 + vice 0.79 + autosubs 4.16 - hits 8 = 49.42"
    assert round(F.gw_points(res['plan'][8], 8, pool, res['beta'][8])['total'], 2) == 49.42


# baseline_09a.txt, harness_7.py equiv_big: (variable formulation, fixed chip); no chip: 141.832
EQUIV_BIG = {
    ('WC', 8): (164.694, 164.671), ('WC', 9): (152.345, 152.346), ('WC', 10): (146.137, 146.138),
    ('FH', 8): (148.199, 148.200), ('FH', 9): (146.280, 146.281), ('FH', 10): (146.137, 146.138),
    ('BB', 8): (144.681, 144.682), ('BB', 9): (144.590, 144.591), ('BB', 10): (144.166, 144.167),
    ('TC', 8): (149.146, 149.147), ('TC', 9): (148.143, 148.144), ('TC', 10): (147.027, 147.028),
}


@pytest.mark.parametrize("chip, gw", list(EQUIV_BIG))
def test_chip_variable_equals_fixed_formulation_big(main_plan, chip, gw):
    """Same check as the toy equivalence, on 360 players with vice-captain and bench terms. The
    solver stops at the relative gap MIP_REL_GAP, so equality holds within 2 x gap."""
    pool, res0 = main_plan['pool'], main_plan['res']
    hz, w = main_plan['horizon'], main_plan['weight']
    start, budget, ft0 = main_plan['args'][:3]
    beta = res0['beta']
    z_none = F.solve_plan(pool, hz, w, start, budget, ft0, beta=beta)['objective']
    z_fix = F.solve_plan(pool, hz, w, start, budget, ft0, beta=beta, chip_fix={(chip, gw): 1})['objective']
    res = F.solve_plan(pool, hz, w, start, budget, ft0, beta=beta, chip='AUTO', chips_avail={chip: [(gw, gw)]})
    expected = max(z_none, z_fix - F.CHIP_TIEBREAK)
    tol = 2 * F.MIP_REL_GAP * abs(expected) + 1e-6
    assert abs(res['objective'] - expected) <= tol
    assert res['plan'][gw]['chip'] == chip
    ref_var, ref_fix = EQUIV_BIG[(chip, gw)]
    assert z_none == pytest.approx(141.832, abs=tol)
    assert (res['objective'], z_fix) == pytest.approx((ref_var, ref_fix), abs=tol)
