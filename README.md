# fplopt

[![tests](https://github.com/gtsilogiannis/fplopt/actions/workflows/tests.yml/badge.svg)](https://github.com/gtsilogiannis/fplopt/actions/workflows/tests.yml)

A Fantasy Premier League (FPL) squad optimizer. It estimates the expected points (xP) of every
player in every remaining gameweek, then solves a multi-period mixed-integer linear program
(MILP) that plans transfers, starting XI, captain, vice-captain and bench order over the next
gameweeks, with free transfers, −4 hits, the budget and chips modelled explicitly.

The xP model is checked with a walk-forward backtest on two past seasons, and every change to it
must pass a fixed acceptance criterion on both (see [Validation](#validation)).

> Not affiliated with the Premier League or Fantasy Premier League. The program reads the public,
> undocumented FPL API, which can change without notice.

**Status:** version 0.9.0 (before 1.0 the command-line options and the JSON output may still
change). The test suite runs on Linux and Windows with Python 3.12, 3.13 and 3.14 (GitHub
Actions); weekly runs on real data are made on Linux.

---

## Contents

- [What it does](#what-it-does)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [How it works](#how-it-works)
- [Validation](#validation)
- [Limitations and assumptions](#limitations-and-assumptions)
- [Development](#development)
- [AI assistance](#ai-assistance)
- [License](#license)

---

## What it does

- **Expected points per player and gameweek.** A component model (appearance, goals, assists,
  clean sheet, goals conceded, saves, defensive contributions, bonus, cards) built from FPL's own
  per-match data (xG, xA, xGC, defensive contributions, minutes). Double and blank gameweeks are
  handled per match.
- **Multi-period plan** (default: 5 gameweeks). One MILP decides the squad of every gameweek,
  so it can bank free transfers, take a hit now for a later gain, or wait. It models selling
  prices (the 50% profit rule), the bank, the 3-per-club limit, formations, the vice-captain and
  the expected value of the bench through autosubs.
- **Chip suggestions** (Wildcard, Free Hit, Bench Boost, Triple Captain): the exact gain of
  playing each available chip now, an estimate for every later gameweek until the chip expires,
  and a rule that suggests a chip only when now looks like its best gameweek.
- **Model evaluation**: every run logs its predictions; after the gameweek,
  `--mode evaluate` compares them with the actual points and with FPL's own `ep_next`.
- **Machine-readable output** (`--json`) and documented exit codes, for use by other programs.

## Installation

Requirements: **Python 3.12 or newer** (PuLP 4 is not released for older versions) and git.

Linux / macOS:

```sh
git clone https://github.com/gtsilogiannis/fplopt.git
cd fplopt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Windows (PowerShell):

```powershell
git clone https://github.com/gtsilogiannis/fplopt.git
cd fplopt
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
```

- `requirements.txt` pins the exact versions the tests passed with (PuLP 4.2.0 with the HiGHS
  solver through `highspy`, pandas, numpy, requests). Use `requirements-dev.txt` instead to also
  get pytest.
- `pip install -e .` installs the package in editable mode (the installed package is this
  folder) and creates the `fplopt` command. `--no-deps` keeps the pinned versions.

## Usage

### Your team ID

Open your team's **Points** page on the FPL website. The address is
`https://fantasy.premierleague.com/entry/<TEAM_ID>/event/<GW>`; the number after `entry/` is
your team ID.

### Weekly run

Run before the deadline, **after the previous gameweek is final** (FPL can still change points
and bonus until then; the program says so with an `[INFO]` line if it is not):

```sh
python -m fplopt --team <TEAM_ID>
```

`fplopt --team <TEAM_ID>` is the same command. A run takes about 1–3 minutes, most of it in the
solver: with the default 5-gameweek horizon, about 40–60 s for the plan and 90 s for the chip
analysis on a recent laptop. `--time-limit` caps the solver time of the whole run. A number of
the chip section that the time limit would cut short is not shown (`not computed`), and a chip
without all its numbers gets no suggestion: the numbers that are shown are the same on a slower
machine, which only shows fewer of them.

If the chip section suggests a chip, run again with that chip to get the plan that uses it:

```sh
python -m fplopt --team <TEAM_ID> --chip WC --no-log
```

A Wildcard is the slowest plan to solve. On 8 squads of past seasons (`bench_chip_solve.py` below)
the 5-gameweek Wildcard plan took 50–740 s (median 190 s) to reach the gap target, against
30–70 s for a Free Hit (2-core machine, two solves at a time). At the time limit the plan is
printed with its gap; give the Wildcard run more time, e.g. `--time-limit 900`.

### Options

| Option | Meaning | Default |
|---|---|---|
| `--team ID` | FPL team ID (required for `optimize`) | — |
| `--mode optimize\|evaluate` | plan the next gameweeks, or evaluate a past gameweek | `optimize` |
| `--gw N` | gameweek to evaluate (required for `evaluate`) | — |
| `--lookahead N` | gameweeks planned, current one included | 5 |
| `--ft N` | free transfers now, 1–5 | rebuilt from your history |
| `--chip WC\|FH\|BB\|TC\|NONE` | chip played in the current gameweek | `NONE` |
| `--no-chip-report` | skip the chip suggestions (saves their solve time) | off |
| `--max-hits N` | cap on paid transfers per gameweek | no cap |
| `--min-chance P` | players below P% chance of playing count as unavailable | 75 |
| `--override-budget X` | selling value of the squad + bank in £m, as shown in the app | computed |
| `--time-limit S` | solver seconds for the whole run (plan + chip analysis) | 180 |
| `--solver highs` | MIP solver (HiGHS is the only one) | HiGHS |
| `--explain "A,B"` | print the xP components of these players (FPL web names) | — |
| `--no-log` | do not write the predictions of this run to `predictions_log.csv` | off |
| `--refresh-cache` | download the cached live data of final gameweeks again | off |
| `--json` | print the result as one JSON document instead of text | off |

`--no-log` is for comparison runs: the log keeps one run per gameweek (the last one), so it
should hold the run you acted on.

### Evaluating the model

After a gameweek is final:

```sh
python -m fplopt --mode evaluate --gw 8
```

It compares the logged predictions of that gameweek with the actual points (mean absolute error
and bias) for all logged players, the relevant ones and your starting XI, side by side with FPL's
`ep_next`. One gameweek is a small sample; judge the model over many.

### Files written

Both are written in the folder you run the command from:

- `predictions_log.csv`: one row per player and gameweek (model xP, FPL `ep_next`, role).
- `data_cache/fpl_live/`: the live data of gameweeks that FPL marks as `finished` and
  `data_checked`, so later runs do not download them again.

### Environment variable

`FPLOPT_API_URL` replaces the base URL of the FPL API (default
`https://fantasy.premierleague.com/api`). The tests use it to point the program at a local fake
server; normal runs do not need it.

## Output

**Text (default).** Results on stdout: the squad of the current gameweek with roles (`>>` marks a
player to buy), the players to sell with the price each one returns, the plan of every gameweek
of the horizon (transfers, hits, captain), the expected points split into their parts, the
solver's optimality gap, and the chip suggestions. Progress lines (`[*]`, `[INFO]`, `[WARNING]`) go to stderr. Both streams are
UTF-8 on every platform.

**JSON (`--json`).** Exactly one JSON document on stdout per run, also when the run fails:

```json
{"schema_version": "1.0", "engine_version": "0.9.0", "ok": true, "mode": "optimize", ...}
{"schema_version": "1.0", "engine_version": "0.9.0", "ok": false, "mode": "optimize",
 "error": {"exit_code": 4, "type": "not_found", "message": "..."}, "warnings": []}
```

Per-gameweek values are arrays of `{"gw": ..., ...}` objects, numbers are not rounded and
`null` means "not applicable" or "unknown". The full schema is in
[docs/json-output.md](docs/json-output.md).

**Exit status.**

| Code | `error.type` | Meaning |
|---|---|---|
| 0 | — | success (also when the time limit stopped the solver; the gap says how far from proven) |
| 1 | `internal` | unexpected error (a bug) |
| 2 | `input` | invalid argument |
| 3 | `network` | FPL API unreachable, timeout, server error after retries, or a reply that is not JSON |
| 4 | `not_found` | HTTP 404: unknown team ID or gameweek |
| 5 | `solver` | no plan satisfies the constraints |
| 6 | `user_action` | the run needs you (the message says what, e.g. `--ft`) |
| 7 | `data` | a local file is missing (e.g. `evaluate` without `predictions_log.csv`) |
| 130 | `internal` | interrupted (Ctrl+C) |

## How it works

### 1. Expected points (`src/fplopt/xp_model.py`)

The xP of player *i* in a match is a sum of components, each one
(FPL points per event) × (expected number of events):

```math
\mathrm{xP}_i = \underbrace{P_{\text{app}} + P_{60}}_{\text{appearance}}
+ \text{goals} + \text{assists} + \text{clean sheet} + \text{conceded} + \text{saves}
+ \text{DefCon} + \text{bonus} + \text{cards}
```

- **Minutes model.** For each player, recency-weighted counts (weight 0.6 per gameweek of age)
  of appearances, 60-minute appearances and minutes in his team's matches, only at his current
  club. The raw frequencies are shrunk towards a prior that counts like a small pseudo-sample,
  e.g. $P_{\text{app}} = (S_{\text{app}} + a m)/(W + a)$. This keeps a player with one good
  week from being treated as a certain starter, and guarantees $P_{60} \le P_{\text{app}}$.
- **Rates per 90 minutes** (xG, xA, xGC, defensive contributions, saves, bonus, cards) are shrunk
  towards the mean of the player's position with 600 pseudo-minutes. For xG, xA and bonus the
  target also depends on price (∝ price², normalised so the position mean does not change).
- **Team strength.** xG for and against per match relative to the league average, with 10
  pseudo-matches of an average team; a home factor of 1.08.
- **Scoring rules** of the 2026/27 season. Non-linear rules use their exact expectation under the
  stated distribution: clean sheet $= e^{-1.08 \lambda}$ (Poisson goals against), "1 point per 3
  saves" and "−1 per 2 goals conceded" as $\mathbb{E}\lfloor X/k \rfloor$ with Poisson $X$,
  defensive contributions as a negative-binomial tail above the threshold.
- A **double gameweek** adds the xP of both matches; a **blank** gives 0. FPL's
  `chance_of_playing` scales the current gameweek. Players with no data at their current club yet
  (e.g. new signings) use FPL's `ep_next` instead of the model.

Every constant in the file is labelled `[measured]`, `[fitted]`, `[tuned]` or `[assumption]`,
with the data it came from.

### 2. The plan (`src/fplopt/milp.py`)

Binary variables per player *i* and gameweek *g*: in the squad $x_{ig}$, in the XI $l_{ig}$,
captain $c_{ig}$, vice-captain $v_{ig}$. The objective is the expected points of the horizon
minus the hits, with weight $w_g = 0.85^{g - g_0}$ for gameweeks further ahead:

```math
\max \sum_g w_g \Big[ \sum_i \mathrm{xP}_{ig}\,(l_{ig} + c_{ig})
+ \text{vice-captain} + \text{autosubs} - 4\,h_g \Big]
```

Main constraints:

- 15 players (2 GK, 5 DEF, 5 MID, 3 FWD), at most 3 per club; a valid XI (1 GK, at least
  3 DEF, 2 MID, 1 FWD); one captain from the XI.
- **Transfers and hits.** $T_g$ = players bought in gameweek *g*, $F_g$ = free transfers,
  $h_g$ = paid transfers:
  $h_g \ge T_g - F_g$, $F_{g+1} \le F_g - T_g + h_g + 1$ and $1 \le F_g \le 5$.
  Since $h_g = \max(0, T_g - F_g)$ at the optimum, the bound is $F_{g+1} \le \max(0, F_g - T_g) + 1$,
  and a larger $F_{g+1}$ never lowers the objective, so the model can always reach the FPL rule
  $F_{g+1} = \min(5, \max(0, F_g - T_g) + 1)$. The free transfers and hits of the plan are
  recomputed with that rule when the solution is read.
- **Budget** in every gameweek: an owned player counts at his selling price, a bought one at his
  current price (prices are taken as constant over the horizon).
- **Vice-captain**: $(1 - P_C)\cdot\mathrm{xP}_{VC}$, the points he adds when the captain does
  not play (independence assumed), linearised without big-M constants.
- **Bench**: the bench GK is valued like the vice-captain; bench slot *k* gets weight
  $\beta_k = P(\text{at least } k \text{ outfield starters do not play})$, computed exactly from
  the starters' appearance probabilities. $\beta$ depends on the chosen XI, so the MILP is
  re-solved with the new weights (up to 3 passes) until they stop changing.
- **Chips**: Wildcard and Free Hit switch off the hit constraints for one gameweek (a Free Hit
  squad reverts afterwards); Bench Boost scores the bench; Triple Captain adds the captain once
  more. Products of binary variables are written as continuous variables bounded by both.

HiGHS solves the model until the plan is proven within 0.05% of the best plan of this model, or
until the time limit; the remaining gap is always reported.

### 3. Chip suggestions (`src/fplopt/chips.py`)

For each chip that can be played now: the **exact gain now** = best 3-gameweek plan with the chip
minus best plan without it (unweighted points, with a bound on the error from both solves), and
an **estimate** for every later gameweek until the chip expires, made on the squad of the plan.
A chip is suggested only if its estimate now is the largest until expiry and its exact gain is
positive. The Wildcard is not estimated after the current gameweek (no one-gameweek proxy exists).

## Validation

### Backtest of the xP model

`scripts/calibrate_xp.py` predicts every gameweek of a completed season (GW2–38) using only the
data before it, from the per-gameweek files of
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League). The
baseline is the FPL projection stored in that dataset for the same gameweek: a **proxy** for the
`ep_next` that FPL shows before the deadline, and probably weaker than it.

The two most recent completed seasons are used; the current season (2026/27) is checked live
with `--mode evaluate`.

| | 2024/25 | 2025/26 |
|---|---|---|
| RMSE per player-gameweek (rows where FPL's projection > 0) | 2.5364 (12,693 rows) | 2.4996 (4,071 rows) |
| MAE, relevant players (model or FPL ≥ 2): model / FPL proxy | 2.15 / 2.57 | 2.10 / 2.54 |
| Bias, relevant players: model / FPL proxy | −0.10 / +0.38 | −0.07 / +0.41 |
| Best XI + captain per GW, predicted − actual (mean ± s.e., 37 GWs) | +0.23 ± 3.40 | +7.17 ± 2.52 |

How to read it:

- The 2025/26 file of the dataset has FPL's projection in only 11 gameweeks (GW1-6, 8, 9, 24, 29,
  38; it is 0 for every player in the others, checked 2026-10-23), so its RMSE and baseline rows
  come from those gameweeks; 2024/25 lacks it only in GW22, 32 and 34.
- The `[tuned]` constants were chosen on **2024/25 only**; 2025/26 is the check. Some
  `[measured]` and `[fitted]` constants pool both seasons, and the DefCon dispersion was fitted
  on 2025/26 (the only season with that data), so 2025/26 is not a fully independent test.
- The last row is a scale check: the 11 highest predictions of each gameweek plus the captain,
  with no budget. In 2025/26 the model **over-predicts** that group by about 7 points per
  gameweek; see [Limitations](#limitations-and-assumptions).
- A change to the xP model is accepted only if, in **both** seasons, its RMSE is lower and the
  best-XI gap does not get worse by more than 2 standard errors of the paired difference
  (`scripts/check_acceptance.py`).

Live comparisons with the real `ep_next` come from `--mode evaluate` during the season.

### Tests

`pytest` runs 101 fast tests (10–20 s) on a synthetic season, with no network: the MILP on
small pools checked by hand (free transfers, hits, chips, budget, vice-captain), selling prices,
the chip rules, the JSON output and exit codes, retries against a local HTTP server, the cache.
`pytest -m slow` runs 14 full optimizer runs (about 5–15 minutes) and compares the output with a
stored reference run.

## Limitations and assumptions

- **xP calibration.** The highest predictions are too high: players predicted above 6 points
  scored 0.6–0.9 less on average, and players over £10m are over-predicted by 0.7–1.1 points per
  gameweek from GW9 on. This affects the captain choice.
- **Not modelled:** own goals, penalty saves and misses; fixture congestion and minutes in other
  competitions; price changes within the horizon (prices are constant).
- **Assumptions:** the weight 0.85 per gameweek ahead; the matches of a double gameweek and the
  appearances of different players are independent; FPL's `chance_of_playing` is taken at face
  value; free transfers left at the end of the horizon have no value.
- **Chip suggestions** rest on estimates for future gameweeks made on the current plan's squad.
  Their exact gains can change by a few tenths between machines, because the chip solves share
  the remaining time limit and a slower machine can stop the bench-weight passes earlier.
- **Optimality** is relative to this model: a proven plan is the best plan *for these xP values*,
  not a guarantee of points.
- **The FPL API** is undocumented and can change; the program retries temporary failures (4
  attempts) and stops with exit code 3 otherwise.
- **Windows**: the test suite passes there (GitHub Actions), but the optimizer has not yet been
  run on real FPL data on a Windows machine.

## Development

```
src/fplopt/     the package
  cli.py          command line, exit codes          engine.py    one run -> result object
  xp_model.py     xP model (self-contained)         milp.py      MILP and solver (only PuLP user)
  chips.py        chip gains and suggestion rule    squad.py     squad, selling prices, free transfers
  data.py         gameweeks, fixtures, history      api.py       FPL API client, retries, cache
  report.py       text output                       jsonout.py   JSON output
  predictions.py  prediction log and evaluation     rules.py     transfer and chip rules
scripts/        backtest, calibration and acceptance check of the xP model; solver timing
tests/          pytest suite (offline)
docs/           JSON output reference
```

```sh
pip install -r requirements-dev.txt
pip install -e . --no-deps
pytest                    # fast tests
pytest -m slow            # full runs; use an otherwise idle machine
```

Backtest and acceptance check (run from the repository root; they download the vaastav data into
`data_cache/`):

```sh
python scripts/calibrate_xp.py --season 2025-26
cp src/fplopt/xp_model.py xp_model_base.py          # before changing the model
python scripts/check_acceptance.py xp_model_base.py # after: prints ACCEPTED or NOT ACCEPTED
```

Rolling-horizon backtest of the whole optimizer: every gameweek of a past season is planned
with only the data known before its deadline, the decisions of that gameweek are applied, and the
real points are counted (autosubs, vice-captain, chips, hits). Two settings are compared by
their paired difference per gameweek:

```sh
python scripts/backtest_season.py --season 2024-25 --lookahead 3 --out a.csv
python scripts/backtest_season.py --compare a.csv b.csv
```

Chip rules are compared on one path, the season played without chips: at every gameweek the
exact and estimated gains and the real gain of each chip are recorded, then the rules are replayed
on these records (one season and lookahead 3 take about an hour):

```sh
python scripts/chip_study.py collect --season 2024-25 --out chips_2425.jsonl
python scripts/chip_study.py evaluate chips_2425.jsonl chips_2526.jsonl
```

Other scripts: `check_minutes.py` (calibration of the minutes model), `check_papp.py`
(appearance probability only), `bench_solver.py` (solver timing on a 700-player synthetic
season), `bench_chip_solve.py` (time of Wildcard and Free Hit plans on squads saved by
`backtest_season.py --save-gws`).

### Tags in the comments

Comments in the code carry short tags such as `M5:` or `X8:`. Each tag names one change, so
that every line belonging to it can be found with a search (e.g. `git grep -n "M5"`). The
letter gives the area:

| Tags | Area |
|---|---|
| `A`, `B`, `C`, `D`, `E` | first fixes of the original version: **A** an unavailable player you own stays in the pool (his price counts in the budget and he can be sold); **B** the −4 cost of each hit; **C** the solver status is checked; **D** binary variables are read safely from floats (`> 0.5`, not `== 1`); **E** a bonus based on transfer trends was removed from the xP |
| `G1`–`G4` | xP per gameweek: horizon as a list of gameweeks, double and blank gameweeks, output |
| `H` | the optional cap on hits (`--max-hits`) |
| `X1`–`X10` | xP model: scoring rules, constants, history data, per-match model, `ep_next` fallback, `--explain`, price-dependent prior, club join dates, RMSE as the tuning score |
| `V1`, `V2` | prediction log and `--mode evaluate` |
| `S1`–`S5`, `R1`, `R2` | selling prices (50% rule, purchase prices rebuilt from the transfer history), budget, sales output; **R1** the squad after a Free Hit; **R2** a team created during the season starts like a GW1 team |
| `M1`–`M10` | multi-period MILP: squad per gameweek, free transfers and hits, Free Hit, budget, bench, vice-captain, time limit, free transfers from history, appearance probability of bench and vice-captain, HiGHS |
| `A1` | hits are weighted like the points of their gameweek |
| `C2`–`C12` | chips: Wildcard / Free Hit transfer rules, chip variables, Free Hit squad, Triple Captain / Bench Boost terms, available chips, xP up to GW38, chip gains and the suggestion rule; **C12** a chip number is shown only if computed in full (not cut by the time limit) |
| `K1`–`K3` | chip constraints: one chip per gameweek, each set at most once, no Free Hit in two consecutive gameweeks |
| `W1`–`W4` | rolling-horizon backtest of the decisions (`scripts/backtest_season.py`); **W2** value of the free transfers left after the horizon; **W3** counterfactual test of the chip rule (`scripts/chip_study.py`); **W4** solve times of Wildcard and Free Hit plans on saved backtest squads (`scripts/bench_chip_solve.py`) |

`C` alone (solver status) and `C2`–`C11` (chips) are different changes.

Comments also compare numbers between **development stages**, e.g. `RMSE 2.4996 (6b 2.5064)`:
the current value, then the value at stage 6b. The stages, in the order they were made:

| Stage | Change |
|---|---|
| 5 | first fixes (`A`–`E`) |
| 6 | xP per gameweek (`G`, `H`) |
| 8a | new xP model (`X`, `V`) |
| 8b | data: selling prices and squad history (`S`, `R`) |
| 6b | multi-period MILP (`M`, `A1`) |
| 8c | shrinkage of the minutes model |
| 7 | chips (`C2`–`C11`, `K`) |
| 9b | rolling-horizon backtest (`W`), teams created during the season (`R2`), chip numbers computed in full (`C12`) |

After stage 7 came the split into a package with tests, and the move to PuLP 4; those did not
change any result.

## AI assistance

I built this project around my own interests: football, FPL and programming. I decided what the
optimizer should do and how it is judged: the features (multi-period planning, chip suggestions,
model evaluation, the JSON interface for a future frontend), the rule that a change to the xP
model is kept only if it beats the previous version in the backtest of both seasons, and the
trade-offs at each step, such as removing the CBC solver after measuring it. I ran every version
on my own FPL team with real data and checked its output before accepting it.

The code was written with substantial help from an AI assistant (Claude, by Anthropic), which
wrote most of it under my direction.

## License

Copyright (C) 2026 Giannis Tsilogiannis

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU Affero General Public License, version 3 only (`AGPL-3.0-only`), as published by the Free
Software Foundation. It is distributed WITHOUT ANY WARRANTY. See [LICENSE](LICENSE).

In short: you may use, change and share it. If you distribute it, changed or not, or let other
people use a changed version over a network, you must offer them its source code under the same
license.
