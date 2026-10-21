"""Prediction log (V1) and its evaluation against the actual points (V2).

Before each deadline the optimizer appends one row per player of the pool to
predictions_log.csv; after the gameweek, `--mode evaluate --gw N` compares them with the
actual points and with FPL's own ep_next."""
import logging
import os
from dataclasses import dataclass, field

import pandas as pd

from .errors import DataError

log = logging.getLogger(__name__)

# V1: new log file (one row per player per GW, see log_predictions). The old
# predictions_history.csv has a different schema and is no longer read.
PREDICTIONS_FILE = "predictions_log.csv"
XI_ROLES = ['Start', 'Captain', 'Vice']   # M6: logged roles of the starting XI ('Vice' from chat 6b)


def log_predictions(gw, players, roles, path=PREDICTIONS_FILE):
    """V1: one row per player in the pool for this GW (not only the XI), so the
    evaluation is not limited to the players the optimizer already liked.
    Re-running for the same GW replaces that GW's rows (no duplicates).
    roles: {player id: 'Captain' / 'Vice' / 'Start' / 'Bench'} of the chosen squad."""
    new = pd.DataFrame([{
        'Gameweek': gw, 'Player_ID': p['id'], 'Name': p['name'], 'Position': p['position'],
        'Model_xP': p['xp_gw'][gw], 'FPL_ep_next': p['ep_next'],
        'Role': roles.get(p['id'], 'Pool'),  # 'Pool' = not in the chosen squad ('None' reads back as NaN)
        'Source': p['source'],
    } for p in players if gw in p['xp_gw']])
    if os.path.isfile(path):
        old = pd.read_csv(path)
        old = old[old['Gameweek'] != gw]           # drop the previous run for this GW
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    log.info(f"[*] {len(players)} predictions for GW{gw} written to {path}")


@dataclass
class EvalRow:
    """Errors of one group of players: model vs FPL's ep_next."""
    label: str
    n: int
    mae: float
    mae_fpl: float
    bias: float
    bias_fpl: float


@dataclass
class EvaluateResult:
    gw: int
    log_path: str
    rows: list = field(default_factory=list)       # [EvalRow]; empty if the log has no row for gw
    xi_predicted: float = None                      # XI + captain (captain counted twice)
    xi_actual: float = None


def evaluate(client, gw, path=PREDICTIONS_FILE):
    """V2: compares logged predictions for one GW with the actual points.
    Per-player errors use unmultiplied points (the captain is not doubled), and
    FPL's own ep_next for the same players is the baseline."""
    if not os.path.isfile(path):
        raise DataError(f"{path} not found. Run an optimization before the GW deadline first.")
    result = EvaluateResult(gw=gw, log_path=path)
    df = pd.read_csv(path)
    df = df[df['Gameweek'] == gw]
    if df.empty:
        log.warning(f"[WARNING] {path} has no predictions for GW{gw}.", extra={'code': 'no_predictions'})
        return result

    live_data = client.event_live(gw)
    actual = {el['id']: el['stats']['total_points'] for el in live_data['elements']}
    df = df.assign(Actual=df['Player_ID'].map(actual).fillna(0.0))

    def group(label, d):
        if d.empty:
            return
        e_m = d['Model_xP'] - d['Actual']
        e_f = d['FPL_ep_next'] - d['Actual']
        result.rows.append(EvalRow(label, len(d), float(e_m.abs().mean()), float(e_f.abs().mean()),
                                   float(e_m.mean()), float(e_f.mean())))

    group("All logged", df)
    group("Relevant (model or FPL >= 2)", df[(df['Model_xP'] >= 2) | (df['FPL_ep_next'] >= 2)])
    group("Your starting XI", df[df['Role'].isin(XI_ROLES)])

    # Team total, captain counted twice on both sides. Autosubs are ignored.
    xi = df[df['Role'].isin(XI_ROLES)]
    if not xi.empty:
        cap = xi[xi['Role'] == 'Captain']
        result.xi_predicted = float(xi['Model_xP'].sum() + cap['Model_xP'].sum())
        result.xi_actual = float(xi['Actual'].sum() + cap['Actual'].sum())
    return result
