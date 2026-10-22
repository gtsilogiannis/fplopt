"""Solver timing on the 700-player synthetic season: one main-plan solve of the
optimizer per setting, without the chip analysis, on the same data for every PuLP version.

    python scripts/bench_solver.py                         # HiGHS, lookahead 3 5 8
    python scripts/bench_solver.py --start partial --lookahead 5 --out bench.csv

Settings:
  --solver   highs; cbc only with an older PuLP 3 checkout of this repository (its bundled CBC): CBC was
             removed from milp.py after measurement, and with the current code 'cbc' is refused
  --start    full    = starting solution with every variable set (0 where none was given);
                       what the PuLP 3 version of milp.py passed to HiGHS
             partial = only the variables that have a value (PuLP 4's own HiGHS behaviour);
                       PuLP 4 only (milp.fill_start switched off)
             none    = no starting solution at all
Each row: PuLP and highspy versions, setting, solve seconds (all bench-weight passes), passes,
proven (gap target reached before the time limit), gap and objective (weighted points), and a
short fingerprint of the GW8 decision (transfers, hits, captain) to see whether plans differ.
The synthetic season comes from tests/fpl_fake.py (Season(big=True), seed 7), so the input is
identical in every run. Times depend on the machine and on the load; run nothing else meanwhile.
"""
import argparse
import csv
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))          # the synthetic season of the tests

import pulp                                              # noqa: E402

import fplopt.milp as milp                               # noqa: E402
from fplopt.engine import Options, optimize              # noqa: E402
from fpl_fake import FakeApi, Season, build_manager      # noqa: E402
from conftest import FakeClient                          # noqa: E402



def pkg_version(name):
    """Installed version of a package, or None."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


FIELDS = ['pulp', 'highspy', 'python', 'solver', 'start', 'lookahead', 'seconds', 'passes', 'proven',
          'gap', 'objective', 'gw8_transfers', 'gw8_hits', 'gw8_captain']


def set_start(mode):
    """Switches the starting-solution behaviour of milp.run_solver for this process."""
    run_solver = milp.run_solver
    if mode == 'partial':
        if not hasattr(milp, 'fill_start'):
            return False                                 # PuLP 3 version: no partial start
        milp.fill_start = lambda prob: None              # PuLP 4: pass only the variables with a value
    elif mode == 'none':
        def no_start(prob, solver, time_limit, warm_start):
            return run_solver(prob, solver, time_limit, False)   # warm_start=False in every solve
        milp.run_solver = no_start
    return True


def run(season, solver, lookahead, time_limit):
    client = FakeClient(FakeApi(season, build_manager(season)))
    r = optimize(client, Options(team_id=1, lookahead=lookahead, chip_report=False, write_log=False,
                                 solver=solver, time_limit=time_limit))
    now = r.plan[0]
    return {'seconds': round(r.solver.seconds, 1), 'passes': r.solver.passes, 'proven': r.solver.proven,
            'gap': None if r.solver.gap is None else round(r.solver.gap, 3),
            'objective': round(r.solver.objective, 3), 'gw8_transfers': len(now.transfers),
            'gw8_hits': now.hits, 'gw8_captain': now.captain_name}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--solver', nargs='+', default=['highs'], choices=['highs', 'cbc'])
    ap.add_argument('--start', default='full', choices=['full', 'partial', 'none'])
    ap.add_argument('--lookahead', nargs='+', type=int, default=[3, 5, 8])
    ap.add_argument('--time-limit', type=int, default=milp.SOLVER_TIME_LIMIT)
    ap.add_argument('--out', default=None, help="append the rows to this CSV file")
    args = ap.parse_args()
    if not set_start(args.start):
        print(f"[skip] --start {args.start} does not exist with PuLP {pulp.__version__}")
        return
    season = Season(big=True)
    rows = []
    for solver in args.solver:
        for la in args.lookahead:
            row = {'pulp': pulp.__version__, 'highspy': pkg_version('highspy'), 'python': platform.python_version(),
                   'solver': solver, 'start': args.start, 'lookahead': la}
            row.update(run(season, solver, la, args.time_limit))
            rows.append(row)
            print(" | ".join(f"{k}={row[k]}" for k in FIELDS), flush=True)
    if args.out:
        path = Path(args.out)
        new = not path.exists()
        with open(path, 'a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
