"""Command line: python -m fplopt (or the 'fplopt' command after pip install).

Results go to stdout, progress and warnings to stderr. Errors raised by the library
(errors.FplError) are printed as '[ERROR] ...' and end the program with a non-zero status.
"""
import argparse
import logging
import sys

from . import report
from .api import FplClient
from .engine import Options, optimize
from .errors import FplError, InputError, UserActionError
from .milp import SOLVER_TIME_LIMIT, highspy
from .predictions import evaluate
from .rules import MAX_FT

log = logging.getLogger('fplopt')


def build_parser():
    parser = argparse.ArgumentParser(prog="fplopt", description="FPL squad optimizer (multi-period MILP)")
    parser.add_argument("--mode", type=str, choices=['optimize', 'evaluate'], default='optimize',
                        help="Execution mode (Default: optimize)")
    parser.add_argument("--team", type=int, help="Official FPL Team ID (Required for optimize)")
    parser.add_argument("--ft", type=int, default=None,
                        help="Free transfers available now (1-5). Default: rebuilt from your GW history")  # M8
    # C6: the chip to play NOW (override). The suggestion comes from the "Chip suggestions" section;
    # the model does not choose chips inside the main plan (notes_07 §4).
    parser.add_argument("--chip", type=str, choices=['WC', 'FH', 'BB', 'TC', 'NONE'], default='NONE',
                        help="Chip played in the current GW (Default: NONE; see the Chip suggestions section)")
    parser.add_argument("--no-chip-report", action="store_true",
                        help="Skip the Chip suggestions section (saves its solve time)")   # C10
    parser.add_argument("--min-chance", type=int, default=75,
                        help="Minimum availability probability percentage (Default: 75)")
    # M7: default 5. Timing on the offline harness (700 players, 2 cores): lookahead 3 ->
    # 10 s, 5 -> 88 s (proven optimal), 8 -> 244 s with a 0.45-point gap left (notes_06b §9).
    parser.add_argument("--lookahead", type=int, default=5,
                        help="Gameweeks planned, including the current one (Default: 5)")
    parser.add_argument("--override-budget", type=float, default=None,
                        help="Total budget in GBPm = selling value of your squad + bank, as shown in the app "
                             "(e.g. 100.9). Default: computed from your transfers")  # S2
    parser.add_argument("--gw", type=int, default=None, help="Gameweek number to evaluate (Required for evaluate)")
    # M7: the multi-period model values waiting for a free transfer itself, so the cap
    # is no longer needed for correctness; it stays as an optional risk limit
    # (a hit is a certain -4, its gain is an estimate).
    parser.add_argument("--max-hits", type=int, default=None,
                        help="Optional cap on paid transfers (-4 each) per GW (Default: no cap)")
    parser.add_argument("--explain", type=str, default=None,
                        help="Comma-separated web names; print their xP components (e.g. Gabriel,Greaves)")  # X7
    parser.add_argument("--time-limit", type=int, default=SOLVER_TIME_LIMIT,
                        help=f"Seconds for the whole solve; the optimality gap is always printed "
                             f"(Default: {SOLVER_TIME_LIMIT})")  # M7/M10
    parser.add_argument("--solver", type=str, choices=['highs', 'cbc'], default=None,
                        help="MIP solver (Default: highs if highspy is installed, else cbc)")  # M10
    return parser


def check_args(args):
    """Checks that argparse cannot express (ranges, required-by-mode). Raises InputError."""
    if args.mode == 'evaluate':
        if not args.gw:
            raise InputError("Evaluation mode requires target Gameweek. Example: --mode evaluate --gw 5")
        return
    if not args.team:
        raise InputError("Optimization mode requires --team ID. Example: --team 10248663")
    if args.max_hits is not None and args.max_hits < 0:  # H: negative cap -> infeasible
        raise InputError("--max-hits must be 0 or greater.")
    if args.ft is not None and not 1 <= args.ft <= MAX_FT:  # M8: FPL stores 1 to 5
        raise InputError(f"--ft must be between 1 and {MAX_FT}.")
    if args.lookahead < 1:
        raise InputError("--lookahead must be 1 or greater.")
    if args.solver == 'highs' and highspy is None:  # M10
        raise UserActionError("--solver highs needs the highspy package: pip install highspy")


def configure_logging():
    """Every message of the package goes to stderr, as it is (the text carries its own prefix)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False


def main(argv=None):
    """Entry point; returns the exit status."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        check_args(args)
        client = FplClient()
        if args.mode == 'evaluate':
            report.print_evaluate(evaluate(client, args.gw))
        else:
            opts = Options(team_id=args.team, ft=args.ft, chip=args.chip, min_chance=args.min_chance,
                           lookahead=args.lookahead, override_budget=args.override_budget,
                           max_hits=args.max_hits, explain=args.explain, time_limit=args.time_limit,
                           solver=args.solver, chip_report=not args.no_chip_report)
            report.print_optimize(optimize(client, opts))
    except FplError as e:
        log.error(f"[ERROR] {e}")
        return 1
    return 0
