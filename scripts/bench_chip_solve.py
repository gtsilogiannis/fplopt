"""W4: timing of the main-plan solve when a Wildcard or Free Hit is played, on real past squads.

The squads are GWs of a saved backtest path (scripts/backtest_season.py --save-gws ...), so every
case is a real pool of the vaastav data and a squad the optimizer itself built during a season.

    python scripts/backtest_season.py --season 2025-26 --last-gw 33 --save-gws 8,15,24,33 --save-dir st
    python scripts/bench_chip_solve.py --states st --cases 2025-26:8 2025-26:15 --chips WC FH \
        --lookahead 3 5 --time-limit 1000000 --out bench_chips.csv

Each row: case, chip, lookahead, time limit, total seconds, passes, proven, the seconds / proven /
gap of every bench-weight pass, the final change of the bench weights, and
  value = weighted expected points of the final plan with the bench weights implied by its OWN XI
          (the fixed point the passes aim for). Two runs with different bench weights have
          objectives that cannot be compared; their 'value' can.
Times depend on the machine and on its load; run nothing else meanwhile.
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backtest_season as B                                         # noqa: E402

import fplopt.milp as milp                                          # noqa: E402
from fplopt.chips import gw_points                                  # noqa: E402
from fplopt.engine import Options, optimize                         # noqa: E402

FIELDS = ['case', 'chip', 'lookahead', 'time_limit', 'seconds', 'passes', 'proven', 'gap', 'beta_change',
          'value', 'objective', 'first_seconds', 'pass_seconds', 'pass_proven', 'pass_gap', 'transfers', 'captain']


def record_passes():
    """Wraps milp.solve_plan (called once per bench-weight pass by plan_squad) and returns the list
    it appends to: (horizon length, seconds, proven, gap) of every solve."""
    log, original = [], milp.solve_plan

    def timed(*args, **kwargs):
        t = time.time()
        res = original(*args, **kwargs)
        log.append((len(args[1]), time.time() - t, res.get('proven'), res.get('gap')))
        return res
    milp.solve_plan = timed
    return log


def self_consistent_value(state):
    """Weighted expected points of the final plan, bench weights from its own XI in every GW."""
    pool, res = state.pool, state.plan_result
    total = 0.0
    for g in state.horizon:
        pg = res['plan'][g]
        beta = milp.bench_weights(pg['xi'], pool['pos'], {i: pool['papp'][(i, g)] for i in pool['ids']})
        total += state.weights[g] * gw_points(pg, g, pool, beta)['total']
    return total


def run_case(states, case, chip, lookahead, time_limit, passes):
    season, gw = case.split(':')
    _, client = B.load_state(season, int(gw), states)
    passes.clear()
    t = time.time()
    r = optimize(client, Options(team_id=1, lookahead=lookahead, write_log=False, time_limit=time_limit,
                                 chip=chip, chip_report=False))
    pg = r.plan[0]
    first = [p for p in passes if p[0] < lookahead]        # one-GW solves before the passes (if any)
    full = [p for p in passes if p[0] == lookahead]
    return {'case': case, 'chip': chip, 'lookahead': lookahead, 'time_limit': time_limit,
            'seconds': round(time.time() - t, 1), 'passes': r.solver.passes, 'proven': r.solver.proven,
            'gap': None if r.solver.gap is None else round(r.solver.gap, 3),
            'beta_change': round(r.solver.beta_change, 3), 'value': round(self_consistent_value(r.state), 3),
            'objective': round(r.solver.objective, 3),
            'first_seconds': round(sum(p[1] for p in first), 1),
            'pass_seconds': ' '.join(f"{s:.1f}" for _, s, _, _ in full),
            'pass_proven': ' '.join(str(int(bool(p))) for _, _, p, _ in full),
            'pass_gap': ' '.join('-' if gp is None else f"{gp:.3f}" for _, _, _, gp in full),
            'transfers': len(pg.transfers), 'captain': next(s.name for s in r.squad if s.role == 'Captain')}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--states', required=True, help="folder of the saved teams")
    ap.add_argument('--cases', nargs='+', required=True, help="season:gw, e.g. 2025-26:8")
    ap.add_argument('--chips', nargs='+', default=['WC', 'FH'], choices=['WC', 'FH'])
    ap.add_argument('--lookahead', nargs='+', type=int, default=[3, 5])
    ap.add_argument('--time-limit', type=int, default=B.NO_LIMIT)
    ap.add_argument('--out')
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    passes = record_passes()
    rows = []
    for case in args.cases:
        for chip in args.chips:
            for la in args.lookahead:
                row = run_case(args.states, case, chip, la, args.time_limit, passes)
                rows.append(row)
                print(' | '.join(f"{k} {row[k]}" for k in FIELDS), flush=True)
                if args.out:
                    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
                        w = csv.DictWriter(fh, fieldnames=FIELDS)
                        w.writeheader()
                        w.writerows(rows)


if __name__ == '__main__':
    main()
