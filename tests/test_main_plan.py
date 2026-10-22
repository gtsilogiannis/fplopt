"""Checks on the main plan of the synthetic season (GW8, lookahead 3, vice-captain and bench
terms with the real bench weights). Slow: pytest -m slow"""
import pytest

from fplopt.chips import gw_points
from fplopt.engine import Options, optimize
from fplopt.milp import CHIP_TIEBREAK, MIP_REL_GAP, solve_plan
from fpl_fake import FakeApi, build_manager
from conftest import FakeClient

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def main_plan(season):
    """One optimizer run without the chip analysis and without writing the log."""
    client = FakeClient(FakeApi(season, build_manager(season)))
    return optimize(client, Options(team_id=1, lookahead=3, chip_report=False, write_log=False))


def test_gw_points_equals_expected_points(main_plan):
    st = main_plan.state
    ep = main_plan.expected_now
    assert [round(x, 2) for x in (ep.xi, ep.captain, ep.vice, ep.autosubs, ep.total)] == \
        [45.95, 6.52, 0.79, 4.16, 49.42]
    assert ep.hits == 8
    res = st.plan_result
    assert gw_points(res['plan'][8], 8, st.pool, res['beta'][8])['total'] == pytest.approx(ep.total, abs=1e-9)


# harness_7.py equiv_big (first commit): (variable formulation, fixed chip); no chip: 141.832
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
    st = main_plan.state
    pool, hz, w = st.pool, st.horizon, st.weights
    beta = st.plan_result['beta']
    args = (pool, hz, w, st.start_squad, st.budget, st.ft0)
    z_none = solve_plan(*args, beta=beta)['objective']
    z_fix = solve_plan(*args, beta=beta, chip_fix={(chip, gw): 1})['objective']
    res = solve_plan(*args, beta=beta, chip='AUTO', chips_avail={chip: [(gw, gw)]})
    expected = max(z_none, z_fix - CHIP_TIEBREAK)
    tol = 2 * MIP_REL_GAP * abs(expected) + 1e-6
    assert abs(res['objective'] - expected) <= tol
    assert res['plan'][gw]['chip'] == chip
    ref_var, ref_fix = EQUIV_BIG[(chip, gw)]
    assert z_none == pytest.approx(141.832, abs=tol)
    assert (res['objective'], z_fix) == pytest.approx((ref_var, ref_fix), abs=tol)
