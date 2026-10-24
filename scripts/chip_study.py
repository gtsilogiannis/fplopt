"""W3: counterfactual test of the chip suggestion rule on past seasons.

Comparing chip rules by replaying whole seasons does not work: the same setting can differ by
60-100 points over a season only because one early decision changed the rest of the path, and a
chip is worth 8-20 points. So the chips are judged on ONE path, the season played without chips
(scripts/backtest_season.py, same fake API and scoring):

  1. collect: at every GW t of that path the optimizer runs with the chip analysis. For every chip c
     it can play at t this records the exact gain Delta_c(t) (best plan with c minus best plan
     without, next 3 GWs, expected points), the estimates Delta_c(t') up to the end of the chip's
     window, and the REAL gain at t: real points of the GW-t decisions of the plan with c minus
     those of the plan without (both from the same exact solves, autosubs and hits included).
  2. evaluate: every rule is replayed on these records, window by window (GW2-19, GW20-38), with
     one chip per GW; a chip a rule never plays gains 0. The oracle (best real gains, one GW per
     chip) is the upper bound.

    python scripts/chip_study.py collect --season 2024-25 --out chips_2425.jsonl
    python scripts/chip_study.py evaluate chips_2425.jsonl chips_2526.jsonl

[assumption] the path after a chip is the path without it: Triple Captain and Bench Boost leave the
squad as it is, a Free Hit squad reverts at the next deadline; only the transfers made in the chip's
GW (and with a Free Hit the free transfer it keeps) can differ. The Wildcard changes the squad for
good, so it is recorded (exact Delta, gain in its own GW) but not part of the rules, as in the code
(it has no estimate after the current GW).
"""
import argparse
import itertools
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_season as B                                   # noqa: E402

from fplopt.engine import Options, optimize                   # noqa: E402

RULE_CHIPS = ('TC', 'BB', 'FH')                               # chips the rules decide
WINDOWS = [(2, 19), (20, 38)]                                 # 2026/27 sets (GW1 is not simulated)


# --------------------------------------------------------------------------- collect
def as_gw(pg):
    """A plan dict of milp.solve_plan (one GW) in the form score_gw expects."""
    return SimpleNamespace(squad=pg['squad'], xi=pg['xi'], bench=pg['bench'], captain_id=pg['captain'],
                           vice_id=pg['vice'], chip=None if pg['chip'] == 'NONE' else pg['chip'],
                           hits=pg['hits'])


def real_points(data, t, pg):
    ids = pg.squad
    return B.score_gw(pg, {i: data.pos[i] for i in ids}, {i: data.minutes.get((i, t), 0) for i in ids},
                      {i: data.points.get((i, t), 0) for i in ids})[0]


def collect(season, lookahead, out_path, first_gw=2, last_gw=38):
    data = B.SeasonData(season)
    m = B.Manager()
    with open(out_path, 'w', encoding='utf-8') as fh:
        for t in range(first_gw, last_gw + 1):
            t0 = time.time()
            r = optimize(B.Client(B.VaastavApi(data, m, t)),
                         Options(team_id=1, lookahead=lookahead, write_log=False, time_limit=B.NO_LIMIT))
            plans = r.state.chip_plans
            base = real_points(data, t, as_gw(plans['NONE'])) if 'NONE' in plans else None
            rec = {'season': season, 'gw': t, 'window_end': r.chips.window,
                   'exact': r.chips.exact,
                   'est': {c: {str(g): v for g, v in e.items()} for c, e in r.chips.estimates.items()},
                   'real_none': base,
                   'gain': {c: real_points(data, t, as_gw(p)) - base for c, p in plans.items() if c != 'NONE'},
                   'path_points': real_points(data, t, r.plan[0]), 'seconds': None}
            m.apply(data, t, r.plan[0])                       # the path itself plays no chip
            rec['seconds'] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            print(f"GW{t:2d}  exact " + ' '.join(f"{c} {v:+5.2f}" for c, v in rec['exact'].items())
                  + "  real " + ' '.join(f"{c} {v:+3d}" for c, v in rec['gain'].items())
                  + f"  {rec['seconds']:6.1f} s", flush=True)


# --------------------------------------------------------------------------- rules
def rule_now_ok(rec, c, tol, kind):
    """Would the rule play chip c at rec['gw'] (before the one-chip-per-GW choice)?"""
    ex = rec['exact'].get(c)
    if ex is None:
        return False
    if kind == 'first':                       # play as soon as the exact gain is positive
        return ex > 0
    est = rec['est'].get(c, {})
    now = est.get(str(rec['gw']))
    if now is None:
        return False
    return now >= max(est.values()) - tol - 1e-9 and ex > 0   # the code's rule: tol = 0


def replay(recs, kind, tol=0.0, force=False):
    """Replays one rule over one season. Returns {(window, chip): (gw or None, real gain, exact)}."""
    by_gw = {r['gw']: r for r in recs}
    out = {}
    for w0, w1 in WINDOWS:
        left = [c for c in RULE_CHIPS]
        for t in range(w0, w1 + 1):
            rec = by_gw.get(t)
            if rec is None or not left:
                continue
            ok = [c for c in left if rule_now_ok(rec, c, tol, kind)]
            if force and not ok and len(left) >= w1 - t + 1:   # as many chips left as GWs: play one now
                ok = [c for c in left if c in rec['exact']]
            if kind == 'last':
                ok = [c for c in left if c in rec['exact']] if len(left) >= w1 - t + 1 else []
            if not ok:
                continue
            c = max(ok, key=lambda x: rec['exact'][x])        # one chip per GW: the largest exact gain
            out[((w0, w1), c)] = (t, rec['gain'][c], rec['exact'][c])
            left.remove(c)
        for c in left:
            out[((w0, w1), c)] = (None, 0, 0.0)
    return out


def oracle(recs):
    """Best real total per window with one GW per chip (exhaustive)."""
    by_gw = {r['gw']: r for r in recs}
    out = {}
    for w0, w1 in WINDOWS:
        gws = [t for t in range(w0, w1 + 1) if t in by_gw]
        best, arg = -math.inf, None
        for combo in itertools.permutations(gws + [None] * len(RULE_CHIPS), len(RULE_CHIPS)):
            used = [g for g in combo if g is not None]
            if len(used) != len(set(used)):
                continue
            tot = sum(by_gw[g]['gain'].get(c, 0) for c, g in zip(RULE_CHIPS, combo) if g is not None)
            if tot > best:
                best, arg = tot, combo
        for c, g in zip(RULE_CHIPS, arg):
            out[((w0, w1), c)] = (g, by_gw[g]['gain'].get(c, 0) if g else 0, by_gw[g]['exact'].get(c, 0.0) if g else 0.0)
    return out


RULES = [('code rule (tol 0)', dict(kind='rule', tol=0.0)),
         ('code rule + forced end', dict(kind='rule', tol=0.0, force=True)),
         ('tol 1', dict(kind='rule', tol=1.0, force=True)),
         ('tol 2', dict(kind='rule', tol=2.0, force=True)),
         ('tol 3', dict(kind='rule', tol=3.0, force=True)),
         ('first positive', dict(kind='first')),
         ('last GWs of window', dict(kind='last'))]


def evaluate(paths):
    seasons = {}
    for p in paths:
        recs = [json.loads(ln) for ln in open(p, encoding='utf-8')]
        seasons[recs[0]['season']] = recs
    # calibration: exact expected gain against the real gain, every GW and chip
    print("Exact Delta (expected) vs real gain, all GWs of the no-chip path:")
    for c in ('TC', 'BB', 'FH', 'WC'):
        pairs = [(r['exact'][c], r['gain'][c]) for recs in seasons.values() for r in recs
                 if c in r['exact'] and c in r['gain']]
        if pairs:
            d = [g - e for e, g in pairs]
            sd = (sum((x - sum(d) / len(d)) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else float('nan')
            print(f"  {c}: n={len(pairs):3d}  mean exact {sum(e for e, _ in pairs) / len(pairs):6.2f}  "
                  f"mean real {sum(g for _, g in pairs) / len(pairs):6.2f}  real - exact {sum(d) / len(d):+.2f} "
                  f"+- {sd / math.sqrt(len(d)):.2f}")
    results = {name: {s: replay(recs, **kw) for s, recs in seasons.items()} for name, kw in RULES}
    results['oracle (upper bound)'] = {s: oracle(recs) for s, recs in seasons.items()}
    base = results['code rule (tol 0)']
    print("\nReal gain of the chips TC + BB + FH, per season and window (sum over the 3 chips):")
    head = ''.join(f"{s} {w0}-{w1}".rjust(16) for s in seasons for w0, w1 in WINDOWS)
    print(f"{'rule':26s}{head}{'total':>9s}{'vs code':>10s}")
    for name, res in results.items():
        cells, diffs = [], []
        for s in seasons:
            for w in WINDOWS:
                v = sum(res[s][(w, c)][1] for c in RULE_CHIPS)
                cells.append(v)
                diffs.append(v - sum(base[s][(w, c)][1] for c in RULE_CHIPS))
        n = len(diffs)
        mean = sum(diffs) / n
        se = (sum((x - mean) ** 2 for x in diffs) / (n - 1)) ** 0.5 / math.sqrt(n) if n > 1 else float('nan')
        print(f"{name:26s}" + ''.join(f"{v:16d}" for v in cells) + f"{sum(cells):9d}"
              + f"{mean:+7.1f}±{se:.1f}")
    print("\nWhen each rule played each chip (GW, real gain):")
    for name, res in results.items():
        line = []
        for s in seasons:
            for w in WINDOWS:
                line.append(' '.join(f"{c}:{'-' if res[s][(w, c)][0] is None else res[s][(w, c)][0]}"
                                     f"({res[s][(w, c)][1]:+d})" for c in RULE_CHIPS))
        print(f"  {name:26s} " + ' | '.join(line))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--season', required=True)
    c.add_argument('--lookahead', type=int, default=3)
    c.add_argument('--first-gw', type=int, default=2)
    c.add_argument('--last-gw', type=int, default=38)
    c.add_argument('--out', required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('files', nargs='+')
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    if args.cmd == 'collect':
        collect(args.season, args.lookahead, args.out, args.first_gw, args.last_gw)
    else:
        evaluate(args.files)


if __name__ == '__main__':
    main()
