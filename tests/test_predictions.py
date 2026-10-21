"""Prediction log (V1) and evaluation (V2) with hand-computed errors."""
import pandas as pd
import pytest

from fplopt.errors import DataError
from fplopt.predictions import evaluate, log_predictions


def player(pid, xp, ep_next, gw=8):
    return {'id': pid, 'name': f"P{pid}", 'position': 3, 'xp_gw': {gw: xp}, 'ep_next': ep_next, 'source': 'model'}


class LiveClient:
    """Answers /event/{gw}/live/ with the given actual points."""

    def __init__(self, points):
        self.points = points

    def event_live(self, gw, event=None):
        return {'elements': [{'id': i, 'stats': {'total_points': p}} for i, p in self.points.items()]}


def test_log_replaces_the_same_gameweek(tmp_path, messages):
    path = tmp_path / "log.csv"
    log_predictions(8, [player(1, 2.0, 1.0), player(2, 3.0, 2.0)], {1: 'Captain'}, str(path))
    log_predictions(8, [player(1, 5.0, 1.0)], {}, str(path))            # re-run of GW8
    log_predictions(9, [player(1, 4.0, 1.0, gw=9)], {}, str(path))
    df = pd.read_csv(path)
    assert df[['Gameweek', 'Player_ID', 'Model_xP', 'Role']].values.tolist() == [[8, 1, 5.0, 'Pool'], [9, 1, 4.0, 'Pool']]
    assert messages()[-1] == f"[*] 1 predictions for GW9 written to {path}"


def test_evaluate(tmp_path):
    path = tmp_path / "log.csv"
    players = [player(1, 6.0, 4.0), player(2, 2.0, 3.0), player(3, 1.0, 1.0)]
    log_predictions(8, players, {1: 'Captain', 2: 'Start'}, str(path))
    r = evaluate(LiveClient({1: 10, 2: 1}), 8, str(path))        # player 3 missing -> 0 points
    rows = {row.label: row for row in r.rows}
    # model errors: 6-10 = -4, 2-1 = 1, 1-0 = 1 -> MAE 2, bias -2/3; FPL: -6, 2, 1 -> MAE 3, bias -1
    allr = rows["All logged"]
    assert (allr.n, allr.mae, allr.mae_fpl) == (3, pytest.approx(2.0), pytest.approx(3.0))
    assert (allr.bias, allr.bias_fpl) == (pytest.approx(-2 / 3), pytest.approx(-1.0))
    assert rows["Relevant (model or FPL >= 2)"].n == 2
    assert rows["Your starting XI"].n == 2
    assert (r.xi_predicted, r.xi_actual) == (pytest.approx(6 + 2 + 6), pytest.approx(10 + 1 + 10))   # captain twice


def test_evaluate_without_log(tmp_path):
    with pytest.raises(DataError, match="not found"):
        evaluate(LiveClient({}), 8, str(tmp_path / "missing.csv"))


def test_evaluate_gameweek_not_logged(tmp_path, messages):
    path = tmp_path / "log.csv"
    log_predictions(8, [player(1, 2.0, 1.0)], {}, str(path))
    r = evaluate(LiveClient({1: 3}), 9, str(path))
    assert r.rows == []
    assert messages()[-1] == f"[WARNING] {path} has no predictions for GW9."
