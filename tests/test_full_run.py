"""Whole optimizer on the synthetic season (GW8, lookahead 3, chip suggestions).

The text must equal the reference run of chat 7 (tests/data/full_run_L3.txt, from
baseline_09a.txt) except for the solve times. Since 9a(i) the progress lines ('[*] ...',
'[WARNING] ...', '[INFO] ...') are logged to stderr and the rest is printed to stdout, so each
stream is compared with its own lines of the reference. Slow (~1.5 minutes): pytest -m slow"""
import io
import re
from pathlib import Path

import pytest

from fplopt import report
from fplopt.engine import Options, optimize

GOLDEN = Path(__file__).parent / "data" / "full_run_L3.txt"
TIME = re.compile(r"(in |Chip analysis: )\d+\.\d s")      # solve times vary between runs and machines
LOG_PREFIXES = ('[*]', '[WARNING]', '[INFO]')


def mask(lines):
    return [TIME.sub(r"\1<t> s", ln) for ln in lines]


@pytest.mark.slow
def test_full_run_equals_reference(fake_client, messages, tmp_path):
    log_path = tmp_path / "predictions_log.csv"
    result = optimize(fake_client(), Options(team_id=1, lookahead=3, log_path=str(log_path)))
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

    log = log_path.read_text(encoding='utf-8').splitlines()
    assert log[0] == "Gameweek,Player_ID,Name,Position,Model_xP,FPL_ep_next,Role,Source"
    assert len(log) == 1 + 360                      # one row per player of the pool
