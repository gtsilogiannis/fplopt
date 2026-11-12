"""W6: cost and effect of the solver's gap target (milp.MIP_REL_GAP) on the default run, on real past squads.

The default run is what `fplopt --team ID` does: main plan (L5), 'no transfer' for the summary, chip report,
plan with the suggested chip. The squads come from a saved backtest path (scripts/backtest_season.py
--save-gws ...), as in scripts/bench_chip_solve.py.

    python scripts/bench_gap.py --states st --cases 2025-26:8 2025-26:15 --gaps 0.0005 0.0001 \
        --time-limit 180 --out gap_180.csv

Each row: case, gap target, time limit, total seconds, seconds per stage, how many numbers were
'not computed' (C12: chips Delta, 'no transfer', chip plan), and what the user sees (objective, transfers,
captain, chip Deltas, suggested chip) to compare the gap targets case by case.
--gaps changes milp.MIP_REL_GAP for the measurement only; the program itself uses the value in milp.py.
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
from fplopt.engine import Options, optimize                         # noqa: E402

FIELDS = ['case', 'gap_target', 'time_limit', 'seconds', 'solve_s', 'no_transfer_s', 'chips_s', 'chip_plan_s',
          'not_computed', 'proven', 'gap', 'objective', 'transfers', 'captain', 'exact', 'pick', 'chip_plan']


def run_case(states, case, gap, time_limit):
    season, gw = case.split(':')
    _, client = B.load_state(season, int(gw), states)
    milp.MIP_REL_GAP = gap        # run_solver reads the module value at every solve
    t = time.time()
    r = optimize(client, Options(team_id=1, write_log=False, time_limit=time_limit))
    seconds = time.time() - t
    pg = r.plan[0]
    cp = r.chip_plan
    # C12: every number the run could not finish in the time limit
    tried_no_transfer = bool(pg.transfers) and pg.chip not in ('WC', 'FH')   # when the engine solves it
    nc = len(r.chips.not_computed) + (tried_no_transfer and r.no_transfer is None) \
        + (cp is not None and cp.status == 'not_computed')
    return {'case': case, 'gap_target': gap, 'time_limit': time_limit, 'seconds': round(seconds, 1),
            **{k: round(r.timings.get(k, 0.0), 1) for k in ('solve_s', 'no_transfer_s', 'chips_s', 'chip_plan_s')},
            'not_computed': int(nc), 'proven': r.solver.proven,
            'gap': None if r.solver.gap is None else round(r.solver.gap, 3),
            'objective': round(r.solver.objective, 3),
            'transfers': ' '.join(f"{m.out_name}>{m.in_name}" for m in pg.transfers) or '-',
            'captain': pg.captain_name,
            'exact': ' '.join(f"{c}{v:+.2f}" for c, v in sorted(r.chips.exact.items())) or '-',
            'pick': (r.chips.advice or {}).get('pick') or '-',
            'chip_plan': '-' if cp is None else f"{cp.chip} {cp.status}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--states', required=True, help="folder of the saved teams")
    ap.add_argument('--cases', nargs='+', required=True, help="season:gw, e.g. 2025-26:8")
    ap.add_argument('--gaps', nargs='+', type=float, default=[milp.MIP_REL_GAP], help="relative gap targets")
    ap.add_argument('--time-limit', type=int, default=B.NO_LIMIT)
    ap.add_argument('--out')
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    rows = []
    for case in args.cases:
        for gap in args.gaps:
            row = run_case(args.states, case, gap, args.time_limit)
            rows.append(row)
            print(' | '.join(f"{k} {row[k]}" for k in FIELDS), flush=True)
            if args.out:
                with open(args.out, 'w', newline='', encoding='utf-8') as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    w.writeheader()
                    w.writerows(rows)


if __name__ == '__main__':
    main()
