FPL optimizer - quick commands (fpl_v2.py, version of chat 7, 2026-10-20)
The full README.md is written in chat 11.

Setup (once, in the project folder):
  python -m venv venv
  source venv/bin/activate
  pip install requests pandas "pulp==3.3.2" highspy

Before each GW deadline, AFTER the previous GW is finalised
(FPL locks the points at 09:00 UK time the day after the GW's last match):
  python fpl_v2.py --team 10248663

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

Examples:
  python fpl_v2.py --team 10248663 --chip WC
  python fpl_v2.py --team 10248663 --chip BB
  python fpl_v2.py --team 10248663 --no-chip-report --time-limit 60

After the GW is finalised (model accuracy vs FPL's ep_next):
  python fpl_v2.py --mode evaluate --gw N

Backtest of the xP model on past seasons (downloads vaastav data into data_cache/):
  python calibrate_xp.py --season 2025-26
  python check_papp.py                 (P(appear) only, chat 6b)
  python check_minutes.py measure     (calibration of P(appear), P(60+), E[minutes], raw)
  python check_minutes.py fit         (fits the shrinkage constants on 2024-25, checks on 2025-26)
  python check_minutes.py verify      (fpl_v2.shrink_minutes equals the fit formulas; consistency checks)
  python check_minutes.py recency     (refits the constants for several MINUTES_RECENCY values)

Acceptance check of a change to the xP model (lower RMSE AND top-11 + captain gap not worse
than 2 paired standard errors, in both seasons):
  cp fpl_v2.py fpl_v2_base.py          (before the change)
  python check_acceptance.py fpl_v2_base

Offline checks (no network; next to fpl_v2.py with harness_offline.py and harness_6b.py):
  python harness_6b.py wait            (also: ft, save, chips, budget, vice, gw1, full)
  python harness_7.py tc               (chat 7, hand-computed: tc, bb, wc, fh, rules, estimates, equiv)
