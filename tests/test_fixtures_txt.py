"""X11: the Football.TXT parser (scripts/fixtures_txt.py) and the club-name map of the congestion study."""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fixtures_txt import parse                      # noqa: E402  (import after the path change)
from check_congestion import normalise              # noqa: E402

SAMPLE = """\
= English FA Cup 2024/25

# Date       Sat Dec 28 2024 - Sun Jan 12 2025 (16d)
# Teams      4

▪ Round 3
  Sat Dec 28
    12:15  Arsenal FC              v Manchester United FC     1-1 a.e.t. (0-0)
           Bodø/Glimt (NOR)        v Tottenham Hotspur (ENG)  0-2   @ Aspmyra
  Sun Jan 5
    16:30  AFC Bournemouth         v West Ham United FC       5-4 pen. 2-2 a.e.t.
  Sun Jan 12 2025
           Nottingham Forest FC    v Wolverhampton Wanderers FC
"""


def write(tmp_path, text):
    p = tmp_path / "x.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_dates_teams_and_year_rollover(tmp_path):
    m = parse(write(tmp_path, SAMPLE))
    assert m == [
        (date(2024, 12, 28), "Arsenal FC", "Manchester United FC"),
        (date(2024, 12, 28), "Bodø/Glimt (NOR)", "Tottenham Hotspur (ENG)"),   # no time: same date
        (date(2025, 1, 5), "AFC Bournemouth", "West Ham United FC"),              # Dec -> Jan: next year
        (date(2025, 1, 12), "Nottingham Forest FC", "Wolverhampton Wanderers FC"),  # no score yet
    ]


def test_match_before_any_date_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        parse(write(tmp_path, "= X\n    20:00  A FC  v B FC  1-0\n"))


@pytest.mark.parametrize("raw, fpl", [
    ("Arsenal FC (ENG)", "Arsenal"), ("AFC Bournemouth", "Bournemouth"),
    ("Manchester United (ENG)", "Man Utd"), ("Manchester United FC", "Man Utd"),
    ("Nottingham Forest FC", "Nott'm Forest"), ("Tottenham Hotspur (ENG)", "Spurs"),
    ("Brighton & Hove Albion FC", "Brighton"), ("Crystal Palace FC", "Crystal Palace"),
])
def test_normalise_to_fpl_names(raw, fpl):
    assert normalise(raw) == fpl
