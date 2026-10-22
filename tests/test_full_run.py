"""Whole optimizer on the synthetic season (GW8, lookahead 3, chip suggestions).

The text must equal the reference run of the original version (tests/data/full_run_L3.txt)
except for the solve times and the optimality gaps. Since the refactor the progress
lines ('[*] ...', '[WARNING] ...', '[INFO] ...') are logged to stderr and the rest is printed to
stdout, so each stream is compared with its own lines of the reference.

The gaps ('proven within X', 'error <= X') are not compared exactly: the solver stops as soon as
the gap is below MIP_REL_GAP, and the bound it has proven at that moment depends on its search
path, which differs between machines and Python builds through last-digit floating-point
differences (Arch, Python 3.14: error <= 0.12 / 0.14; reference, Python 3.13: 0.14 / 0.13, with
the same plan, objective and Deltas). They are checked against their guarantee instead.
Slow (~1.5 minutes): pytest -m slow"""
import io
import re
from pathlib import Path

import pytest

from fplopt import report
from fplopt.engine import Options, optimize
from fplopt.milp import MIP_REL_GAP

GOLDEN = Path(__file__).parent / "data" / "full_run_L3.txt"
TIME = re.compile(r"(in |Chip analysis: )\d+\.\d s")      # solve times vary between runs and machines
GAP = re.compile(r"(proven within |error <= )\d+\.\d+")  # optimality gaps vary between machines (see above)
LOG_PREFIXES = ('[*]', '[WARNING]', '[INFO]')


def mask(lines):
    return [GAP.sub(r"\1<gap>", TIME.sub(r"\1<t> s", ln)) for ln in lines]


@pytest.mark.slow
def test_full_run_equals_reference(fake_client, messages, tmp_path):
    log_path = tmp_path / "predictions_log.csv"
    # 9a(ii): time limit far above the ~2.5 minutes the run needs. With the default 180 s each solve
    # of the chip analysis gets a share of the time LEFT, so on a slow or busy machine the bench-weight
    # fixed point of the BB solve stopped after 2 passes instead of 3 (BB +3.01 instead of +2.85).
    # The reference was recorded with every pass done; the time-sharing rule itself
    # is not what this test checks.
    result = optimize(fake_client(), Options(team_id=1, lookahead=3, log_path=str(log_path), time_limit=600))
    out = io.StringIO()
    report.print_optimize(result, out)

    golden = GOLDEN.read_text(encoding='utf-8').splitlines()
    golden_log = [ln for ln in golden if ln.startswith(LOG_PREFIXES)]
    golden_out = [ln for ln in golden if not ln.startswith(LOG_PREFIXES)]
    golden_log[-1] = golden_log[-1].replace("predictions_log.csv", str(log_path))   # written to tmp
    assert mask(out.getvalue().splitlines()) == mask(golden_out)
    assert mask(messages()) == mask(golden_log)

    # the same numbers as data (what --json returns)
    assert result.current_gw == 8 and result.horizon == [8, 9, 10]
    assert (result.ft, result.ft_source) == (3, 'history')
    assert round(result.budget, 1) == 99.1
    assert round(result.solver.objective, 2) == 141.83
    assert round(result.expected_now.total, 2) == 49.42
    assert [pg.hits for pg in result.plan] == [2, 0, 0]
    assert [len(pg.transfers) for pg in result.plan] == [5, 1, 1]
    assert [r.role for r in result.squad].count('Captain') == 1
    assert result.chips.advice['pick'] is None
    assert result.warnings == []

    # gaps: within the guarantee of the solver. The main plan is proven within MIP_REL_GAP of its
    # objective; each chip Delta is the difference of two such solves over the same 3 GWs (no chip,
    # objective = the main plan's; with the chip, objective <= that + Delta), so its error bound
    # is at most MIP_REL_GAP x (2 x objective + Delta).
    obj = result.solver.objective
    assert result.solver.proven and 0.0 <= result.solver.gap <= MIP_REL_GAP * obj + 1e-9
    assert set(result.chips.exact) == {'BB', 'TC'}
    assert [round(result.chips.exact[c], 2) for c in ('BB', 'TC')] == [2.85, 7.31]
    for c, err in result.chips.error.items():
        assert 0.0 <= err <= MIP_REL_GAP * (2 * obj + result.chips.exact[c]) + 1e-9

    log = log_path.read_text(encoding='utf-8').splitlines()
    assert log[0] == "Gameweek,Player_ID,Name,Position,Model_xP,FPL_ep_next,Role,Source"
    assert len(log) == 1 + 360                      # one row per player of the pool
