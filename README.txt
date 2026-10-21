FPL optimizer (fplopt) - quick commands (chat 9a(i), 2026-10-21)
The full README.md is written in chat 11.

Layout:
  src/fplopt/      the package (python -m fplopt, or the 'fplopt' command after installing)
  scripts/         backtest and calibration of the xP model (calibrate_xp.py, check_*.py)
  tests/           pytest suite (offline: synthetic season, no network)

Setup (once, in the repository folder):
  python -m venv venv
  source venv/bin/activate
  pip install -e ".[highs,dev]"
(-e = editable: the installed package is this folder, so changes to the code apply at once.
 The quotes stop zsh from reading [ ] as a file pattern.)

Before each GW deadline, AFTER the previous GW is finalised
(FPL locks the points at 09:00 UK time the day after the GW's last match):
  python -m fplopt --team 10248663

Results are printed on stdout, progress and warnings on stderr (both appear in the terminal).
The output ends with a CHIP SUGGESTIONS section: for every chip you can play now, the exact gain
now (best plan with the chip minus best plan without, next 3 GWs, unweighted expected points) and
an estimate for every GW up to the end of the chip's window (GW19 / GW38). If it suggests a chip,
re-run with --chip X to get the plan that uses it.

Options:
  --ft N                 free transfers now, 1-5 (default: rebuilt from your GW history)
  --chip WC|FH|BB|TC     chip played this GW (default: none; see CHIP SUGGESTIONS)
  --no-chip-report       skip the CHIP SUGGESTIONS section (saves its solve time)
  --lookahead N          gameweeks planned, current one included (default: 5)
  --time-limit S         seconds for the whole run of the solver (plan + chip analysis); the optimality
                         gap is printed (default: 180)
  --solver highs|cbc     MIP solver (default: highs if highspy is installed, else cbc)
  --max-hits N           optional cap on -4 hits per GW (default: no cap)
  --min-chance P         players below P% chance of playing are treated as out (default: 75)
  --override-budget X    selling value of the squad + bank, as shown in the app (e.g. 100.1)
  --explain "A,B"        print the xP components of these players (web names); the minutes model is
                         shown shrunk, with the raw estimate in brackets
  --json                 print the whole result as one JSON document (UTF-8) on stdout instead of text
                         (for the C frontend and the website; schema in claude/notes_09a_refactor.md)

Exit status: 0 OK, 1 internal error, 2 invalid argument, 3 network / FPL API, 4 not found (team id or
GW), 5 no plan found, 6 action needed by you (the message says which), 7 missing data file,
130 interrupted. With --json an error is also a JSON document: {"ok": false, "error": {...}}.

Examples:
  python -m fplopt --team 10248663 --chip WC
  python -m fplopt --team 10248663 --no-chip-report --time-limit 60

After the GW is finalised (model accuracy vs FPL's ep_next):
  python -m fplopt --mode evaluate --gw N

Backtest of the xP model on past seasons (run from the repository folder; downloads vaastav data
into data_cache/):
  python scripts/calibrate_xp.py --season 2025-26
  python scripts/check_papp.py               (P(appear) only, chat 6b)
  python scripts/check_minutes.py measure    (calibration of P(appear), P(60+), E[minutes], raw)
  python scripts/check_minutes.py fit        (fits the shrinkage constants on 2024-25, checks on 2025-26)
  python scripts/check_minutes.py verify     (xp_model.shrink_minutes equals the fit formulas)
  python scripts/check_minutes.py recency    (refits the constants for several MINUTES_RECENCY values)

Acceptance check of a change to the xP model (lower RMSE AND top-11 + captain gap not worse
than 2 paired standard errors, in both seasons):
  cp src/fplopt/xp_model.py xp_model_base.py        (before the change)
  python scripts/check_acceptance.py xp_model_base.py

Tests (offline):
  pytest                 fast tests (~10 s)
  pytest -m slow         full runs on the synthetic season (~12 min)
