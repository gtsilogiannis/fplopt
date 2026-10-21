"""Whole optimizer on the synthetic season (GW8, lookahead 3, chip suggestions): the text
output must equal the reference run of chat 7 (tests/data/full_run_L3.txt, from
baseline_09a.txt) except for the solve times. Slow (~1.5 minutes): pytest -m slow"""
import re
from pathlib import Path

import pytest

import fpl_v2 as F

GOLDEN = Path(__file__).parent / "data" / "full_run_L3.txt"
TIME = re.compile(r"(in |Chip analysis: )\d+\.\d s")      # solve times vary between runs and machines


def mask(lines):
    return [TIME.sub(r"\1<t> s", ln) for ln in lines]


@pytest.mark.slow
def test_full_run_equals_reference(fake_api, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                    # predictions_log.csv goes to a temporary folder
    fake_api()
    F.optimize_team(1, None, 'NONE', 75, 3, None, None, None)
    out = capsys.readouterr().out
    assert mask(out.splitlines()) == mask(GOLDEN.read_text(encoding='utf-8').splitlines())
    log = (tmp_path / "predictions_log.csv").read_text(encoding='utf-8').splitlines()
    assert log[0] == "Gameweek,Player_ID,Name,Position,Model_xP,FPL_ep_next,Role,Source"
    assert len(log) == 1 + 360                      # one row per player of the pool
