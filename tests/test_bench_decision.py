"""scripts/bench_decision.py on the synthetic season (GW8 next, lookahead 2).

The squads made by rule are valid and change what they should, the decision 'a' (the model's own chip
squad) gives R = 0 within the gaps, and the static budget check is exact to the tenth."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_decision as BD                                          # noqa: E402

from fplopt.engine import Options, optimize                          # noqa: E402
from fpl_fake import FakeApi, build_manager                          # noqa: E402
from conftest import FakeClient                                      # noqa: E402

OPTS = dict(team_id=1, lookahead=2, chip_report=False, write_log=False, time_limit=600)


@pytest.fixture(scope="module")
def engine_run(season):
    client = FakeClient(FakeApi(season, build_manager(season)))
    opts = Options(**OPTS)
    return optimize(client, opts), opts


@pytest.mark.parametrize("chip", ['FH', 'WC'])
def test_decisions_by_rule(engine_run, chip):
    r, opts = engine_run
    st = r.state
    g = st.horizon[0]
    rows = {row['solve']: row for row in BD.run_decisions(st, chip, g, opts, 600)}
    assert set(rows) == {'z0', 'z_best', 'z_a', 'z_b', 'z_c'}
    assert all(row['complete'] for row in rows.values())
    assert rows['z_a']['changed_vs_best'] == 0 and rows['z_a']['k2_ok'] is True     # R = 0 within the gaps
    assert rows['z_b']['changed_vs_best'] == 1                                        # exactly one change
    assert rows['z_c']['changed_vs_current'] == 0                                     # the current squad
    for d in ('z_a', 'z_b', 'z_c'):
        row = rows[d]
        assert row['G'] + row['R'] == pytest.approx(rows['z_best']['J'] - rows['z0']['J'], abs=2e-3)


def test_budget_check_to_the_tenth(engine_run):
    """A squad that costs exactly the budget is valid; the same squad with the budget 0.1 lower is not."""
    st = engine_run[0].state
    squad = list(st.start_squad)
    cost = BD.squad_cost(st.pool, squad, st.start_squad) / 10
    assert BD.violations(st.pool, squad, st.start_squad, cost) == []
    assert BD.violations(st.pool, squad, st.start_squad, cost - 0.1) == ['over_budget']
