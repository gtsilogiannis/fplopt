"""Command line: python -m fplopt (or the 'fplopt' command after pip install).

Streams:  results on stdout (text, or one JSON document with --json), progress and warnings
          on stderr, both UTF-8 on every platform.
Status:   0 success, otherwise the exit code of the error class (errors.py): 1 internal error,
          2 invalid argument, 3 network/API, 4 not found (404), 5 no plan, 6 user action needed,
          7 missing data file, 130 interrupted.
"""
import argparse
import logging
import os
import sys
import traceback

from . import jsonout, report
from .api import FPL_API, FplClient
from .engine import Options, optimize
from .errors import EXIT_INTERNAL, EXIT_INTERRUPTED, FplError, InputError, UserActionError
from .messages import collect
from .milp import SOLVER_TIME_LIMIT, highspy
from .predictions import evaluate
from .rules import MAX_FT

log = logging.getLogger('fplopt')

# Base URL of the FPL API. Overridden only by the environment variable FPLOPT_API_URL, which the
# tests use to point the program at a local fake server; normal runs never set it.
API_URL_ENV = "FPLOPT_API_URL"
# Folder for the downloaded data, relative to the working directory (shared with scripts/calibrate_xp.py,
# which keeps the vaastav files there; the FPL live data go to its subfolder fpl_live/).
CACHE_DIR = "data_cache"


class _Parser(argparse.ArgumentParser):
    """argparse prints its errors and exits with status 2 by itself; raising InputError instead
    lets main() report them like every other error (status 2, and JSON with --json)."""

    def error(self, message):
        raise InputError(message)


def build_parser():
    parser = _Parser(prog="fplopt", description="FPL squad optimizer (multi-period MILP)")
    parser.add_argument("--mode", type=str, choices=['optimize', 'evaluate'], default='optimize',
                        help="Execution mode (Default: optimize)")
    parser.add_argument("--team", type=int, help="Official FPL Team ID (Required for optimize)")
    parser.add_argument("--ft", type=int, default=None,
                        help="Free transfers available now (1-5). Default: rebuilt from your GW history")  # M8
    # C6: the chip to play NOW (override). The suggestion comes from the "Chip suggestions" section;
    # the model does not choose chips inside the main plan.
    parser.add_argument("--chip", type=str, choices=['WC', 'FH', 'BB', 'TC', 'NONE'], default='NONE',
                        help="Chip played in the current GW (Default: NONE; see the Chip suggestions section)")
    # Decision check: the user's own chip decision, compared with the plan without it and with the
    # best version of the same chip (decision.py)
    parser.add_argument("--check", type=str, choices=['TC', 'BB', 'TRANSFERS'], default=None,
                        help="Check your own decision: TC (with --captain), BB, or TRANSFERS (with --out, --in)")
    parser.add_argument("--check-gw", type=int, default=None,
                        help="Gameweek of the checked decision, inside the horizon (Default: the current GW)")
    parser.add_argument("--captain", type=int, default=None,
                        help="--check TC: FPL id of your captain (--explain NAME prints it)")
    parser.add_argument("--out", type=str, default=None, dest="out_ids",
                        help="--check TRANSFERS: ids of the players you sell now, comma-separated (e.g. 12,40)")
    parser.add_argument("--in", type=str, default=None, dest="in_ids",
                        help="--check TRANSFERS: ids of the players you buy now, comma-separated, as many")
    parser.add_argument("--no-chip-report", action="store_true",
                        help="Skip the Chip suggestions section (saves its solve time)")   # C10
    parser.add_argument("--min-chance", type=int, default=75,
                        help="Minimum availability probability percentage, 0-100 (Default: 75)")
    # M7: default 5. Timing on the offline harness (700 players, 2 cores): lookahead 3 ->
    # 10 s, 5 -> 88 s (proven optimal), 8 -> 244 s with a 0.45-point gap left.
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
    parser.add_argument("--solver", type=str, choices=['highs'], default=None,
                        help="MIP solver (Default and only choice: highs)")  # M10; CBC was removed after measurement
    # The log keeps one run per GW (the last one), so comparison runs
    # made after the decision run would replace its predictions
    parser.add_argument("--no-log", action="store_true",
                        help="Do not write this run's predictions to predictions_log.csv (comparison runs)")
    parser.add_argument("--refresh-cache", action="store_true",
                        help=f"Download the cached live data of final GWs again ({CACHE_DIR}/fpl_live/)")
    parser.add_argument("--json", action="store_true",
                        help="Print the result as one JSON document (UTF-8) on stdout instead of text")
    return parser


def check_args(args):
    """Checks that argparse cannot express (ranges, required-by-mode). Raises InputError."""
    if args.mode == 'evaluate':
        if not args.gw:
            raise InputError("Evaluation mode requires target Gameweek. Example: --mode evaluate --gw 5")
        if not 1 <= args.gw <= 38:
            raise InputError("--gw must be between 1 and 38.")
        return
    if not args.team:
        raise InputError("Optimization mode requires --team ID: the number in the address of your team's Points page (fantasy.premierleague.com/entry/ID/event/GW).")
    if args.team < 1:
        raise InputError("--team must be a positive team ID.")
    if args.max_hits is not None and args.max_hits < 0:  # H: negative cap -> infeasible
        raise InputError("--max-hits must be 0 or greater.")
    if args.ft is not None and not 1 <= args.ft <= MAX_FT:  # M8: FPL stores 1 to 5
        raise InputError(f"--ft must be between 1 and {MAX_FT}.")
    if args.lookahead < 1:
        raise InputError("--lookahead must be 1 or greater.")
    if not 0 <= args.min_chance <= 100:
        raise InputError("--min-chance must be between 0 and 100.")
    if args.time_limit < 1:
        raise InputError("--time-limit must be 1 second or more.")
    if args.override_budget is not None and args.override_budget <= 0:
        raise InputError("--override-budget must be a positive amount in GBPm (e.g. 100.9).")
    if args.check is None and (args.check_gw is not None or args.captain is not None):
        raise InputError("--check-gw and --captain are used only with --check.")
    if args.check != 'TRANSFERS' and (args.out_ids is not None or args.in_ids is not None):
        raise InputError("--out and --in are used only with --check TRANSFERS.")
    for name in ('out_ids', 'in_ids'):                  # "12,40" -> [12, 40]
        text = getattr(args, name)
        if text is not None:
            try:
                setattr(args, name, [int(t) for t in text.split(',') if t.strip()])
            except ValueError:
                raise InputError(f"--{name[:-4]} needs player ids separated by commas, e.g. 12,40 (got '{text}').")
    if args.check_gw is not None and not 1 <= args.check_gw <= 38:
        raise InputError("--check-gw must be between 1 and 38.")
    if args.solver == 'highs' and highspy is None:  # M10
        raise UserActionError("--solver highs needs the highspy package: pip install highspy")


def use_utf8_streams():
    """Text and JSON are written as UTF-8 on every platform. Without this, on Windows the output
    to a pipe (which is what a frontend reads) uses the ANSI code page (e.g. cp1252): '£' is
    changed and a name such as 'Horníček' can stop the program with UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError, OSError):   # not a text file (e.g. replaced by a test)
            pass


def configure_logging():
    """Every message of the package goes to stderr, as it is (the text carries its own prefix)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)


def main(argv=None):
    """Entry point; returns the exit status."""
    use_utf8_streams()
    configure_logging()
    argv = sys.argv[1:] if argv is None else list(argv)
    want_json = '--json' in argv              # known before parsing, so parse errors are JSON too
    mode = 'evaluate' if 'evaluate' in argv else 'optimize'
    with collect() as warnings:
        try:
            args = build_parser().parse_args(argv)
            mode = args.mode
            check_args(args)
            client = FplClient(base_url=os.environ.get(API_URL_ENV) or FPL_API, cache_dir=CACHE_DIR,
                               refresh_cache=args.refresh_cache)
            if args.mode == 'evaluate':
                result = evaluate(client, args.gw)
                if want_json:
                    print(jsonout.dumps(jsonout.evaluate_to_dict(result, warnings)))
                else:
                    report.print_evaluate(result)
            else:
                opts = Options(team_id=args.team, ft=args.ft, chip=args.chip, min_chance=args.min_chance,
                               lookahead=args.lookahead, override_budget=args.override_budget,
                               max_hits=args.max_hits, explain=args.explain, time_limit=args.time_limit,
                               solver=args.solver, chip_report=not args.no_chip_report,
                               write_log=not args.no_log, check=args.check, check_gw=args.check_gw,
                               check_captain=args.captain, check_out=args.out_ids, check_in=args.in_ids)
                result = optimize(client, opts)
                if want_json:
                    print(jsonout.dumps(jsonout.optimize_to_dict(result)))
                else:
                    report.print_optimize(result)
            return 0
        except FplError as e:
            log.error(f"[ERROR] {e}")
            return _fail(e, mode, want_json, warnings, e.exit_code)
        except KeyboardInterrupt:
            log.error("[ERROR] Interrupted.")
            return _fail(KeyboardInterrupt("Interrupted."), mode, want_json, warnings, EXIT_INTERRUPTED)
        except Exception as e:          # a bug: full traceback on stderr, status 1
            traceback.print_exc(file=sys.stderr)
            return _fail(e, mode, want_json, warnings, EXIT_INTERNAL)


def _fail(exc, mode, want_json, warnings, status):
    if want_json:
        doc = jsonout.error_to_dict(exc, mode, warnings)
        doc['error']['exit_code'] = status
        print(jsonout.dumps(doc))
    return status
