"""Acceptance check of a change to the xP model (roadmap, permanent criterion, decision 2026-10-20).
Usage (from the repository root):
    cp src/fplopt/xp_model.py xp_model_base.py           (before the change)
    python scripts/check_acceptance.py xp_model_base.py [<candidate>] [--season 2024-25]
A version is a .py file that defines the xP model (a copy of src/fplopt/xp_model.py, or an older
single-file version such as fpl_v2.py) or 'current' (the installed fplopt.xp_model), optionally
with constants overridden: current@PRICE_PRIOR_EXPONENT=1.0,... The candidate defaults to 'current'.
Criterion, in EACH season (2024-25 and 2025-26):
  (1) RMSE (calibrate_xp tuning score) of the candidate < RMSE of the baseline, and
  (2) |gap| of the best XI + captain (mean over GWs of predicted - actual, calibrate_xp.top11)
      of the candidate <= |gap| of the baseline + 2 standard errors of the PAIRED difference
      (same GWs; the gap of one season alone has a standard error of ~2.5-3.5 points, the
      paired difference of two versions ~0.4-1.1, notes_08c §6.3).
Runtime: two backtests per season (~1-2 minutes each)."""
import argparse
import importlib.util
import os
import sys
import numpy as np
import calibrate_xp as C

SEASONS = ('2024-25', '2025-26')


def load_version(spec):
    """'<file.py>' / 'current', optionally '@NAME=value,NAME=value' -> an independent copy of that module."""
    name, _, overrides = spec.partition('@')
    if name == 'current':
        path = importlib.util.find_spec('fplopt.xp_model').origin     # the file of the installed package
    else:
        path = name if name.endswith('.py') else name + '.py'
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] {path} not found")
    loader = importlib.util.spec_from_file_location(f"version_{abs(hash(spec))}", path)
    mod = importlib.util.module_from_spec(loader)   # a new module object: overrides stay local to it
    loader.loader.exec_module(mod)                   # runs the file (its __main__ block does not run)
    for item in filter(None, overrides.split(',')):
        key, value = item.split('=')
        if not hasattr(mod, key.strip()):
            sys.exit(f"[ERROR] {name} has no constant {key.strip()}")
        setattr(mod, key.strip(), float(value))
    return mod


def backtest(mod, data):
    """RMSE (tuning score) and the per-GW top-11 gaps of one version on one season."""
    C.F = mod                                        # calibrate_xp reads the model through its global F
    d, hist, pos_of, prices = data
    p = C.predict(d, hist, pos_of, range(2, 39), prices)
    rmse, _ = C.score(p)
    tops = C.top11(p)
    return rmse, tops[:, 0] - tops[:, 1]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Acceptance check: candidate vs baseline version of the xP model")
    ap.add_argument('baseline')
    ap.add_argument('candidate', nargs='?', default='current')
    ap.add_argument('--season', choices=SEASONS, default=None, help="one season only (default: both)")
    args = ap.parse_args()
    base, cand = load_version(args.baseline), load_version(args.candidate)
    verdict = []
    for season in ([args.season] if args.season else SEASONS):
        C.F = cand
        d, hist, pos_of = C.load(season)
        d = d.rename(columns={'GW': 'gw'})
        data = (d, hist, pos_of, C.price_table(d))
        rb, gb = backtest(base, data)
        rc, gc = backtest(cand, data)
        diff = gc - gb                               # paired: same GWs, two versions
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        ok_rmse = rc < rb
        ok_gap = abs(gc.mean()) <= abs(gb.mean()) + 2 * se
        verdict.append(ok_rmse and ok_gap)
        print(f"{season}: RMSE {rb:.4f} -> {rc:.4f} ({'pass' if ok_rmse else 'FAIL'}) | top-11 gap "
              f"{gb.mean():+.2f} -> {gc.mean():+.2f}, paired difference {diff.mean():+.2f} +/- {se:.2f}, "
              f"allowed |gap| <= {abs(gb.mean()) + 2 * se:.2f} ({'pass' if ok_gap else 'FAIL'})", flush=True)
    print("ACCEPTED" if all(verdict) else "NOT ACCEPTED",
          "" if args.season is None else f"(season {args.season} only; the criterion needs both)")
