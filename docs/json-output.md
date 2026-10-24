# JSON output (`--json`), schema 1.0

`python -m fplopt ... --json` prints **one JSON document** on stdout and nothing else. Progress
lines (`[*] ...`, `[WARNING] ...`) still go to stderr. The text is UTF-8; names are written as
they are (`Ødegaard`, not `Ødegaard`).

This document describes `schema_version` **1.0**, produced by `engine_version` 0.9.0. The code
that writes it is `src/fplopt/jsonout.py`.

## Contents

- [Rules](#rules)
- [Reading it from another program](#reading-it-from-another-program)
- [Common header](#common-header)
- [Mode `optimize`](#mode-optimize)
- [Mode `evaluate`](#mode-evaluate)
- [Errors](#errors)
- [Warning codes](#warning-codes)
- [Example](#example)

## Rules

1. Exactly one document per run, also when the run fails (then `"ok": false`, see
   [Errors](#errors)).
2. No object has numbers as keys: everything per gameweek is an array of objects with a `"gw"`
   field, e.g. `[{"gw": 8, "weight": 1.0}, {"gw": 9, "weight": 0.85}]`.
3. Field names are fixed. Enumerations are short strings. `null` means "does not apply" or
   "unknown"; each field below says which.
4. Numbers are not rounded (the text output rounds them). `NaN` and infinity never appear (they
   are not valid JSON); they are written as `null`.
5. Versioning: `schema_version` changes when a field is removed or changes meaning. New fields
   can be added without a change, so a reader must ignore fields it does not know. A reader
   should check the major version (the part before the dot).
6. Units: money in £m (e.g. `5.5`), points in FPL points, times in seconds. "Weighted" points
   are multiplied by the gameweek weight `0.85^(gw - current_gw)`; "unweighted" are not.

## Reading it from another program

Run the command as a child process, read all of stdout, wait for it to end, then:

- exit status `0` → parse the document; `ok` is `true`;
- any other status → the document has `"ok": false` and `error.message` is the text to show.

The working directory of the process is where `predictions_log.csv` and `data_cache/` are
written. Exit statuses are listed in the README and in [Errors](#errors).

## Common header

Every document starts with these fields:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `engine_version` | string | version of the `fplopt` package, e.g. `"0.9.0"` |
| `ok` | bool | `true` on success, `false` on error |
| `mode` | string | `"optimize"` or `"evaluate"` |

## Mode `optimize`

Top-level fields after the header:

| Field | Type | Section |
|---|---|---|
| `input` | object | [input](#input) |
| `context` | object | [context](#context) |
| `solver` | object | [solver](#solver) |
| `squad` | array (15) | [squad](#squad) |
| `plan` | array (one per GW of the horizon) | [plan](#plan) |
| `sales_now` | array | [sales_now](#sales_now) |
| `expected_points_now` | object | [expected_points_now](#expected_points_now) |
| `chips` | object | [chips](#chips) |
| `explain` | array | [explain](#explain) |
| `explain_not_found` | array of string | names given to `--explain` that matched no player (lower case) |
| `fallback_players` | int | players whose xP is FPL's `ep_next` (no data at their current club yet) |
| `predictions_log` | string or null | path of the log that was written; `null` with `--no-log` |
| `warnings` | array | [Warning codes](#warning-codes) |
| `timings` | object | [timings and api](#timings-and-api) |
| `api` | object | [timings and api](#timings-and-api) |

### input

The options of the run, as given (defaults filled in by the command line).

| Field | Type | Meaning |
|---|---|---|
| `team_id` | int | `--team` |
| `ft` | int or null | `--ft`; `null` = not given (rebuilt from the history) |
| `chip` | string | `--chip`: `"WC"`, `"FH"`, `"BB"`, `"TC"` or `"NONE"` |
| `min_chance` | int | `--min-chance` (percent) |
| `lookahead` | int | `--lookahead` |
| `override_budget` | number or null | `--override-budget`; `null` = not given |
| `max_hits` | int or null | `--max-hits`; `null` = no cap |
| `time_limit` | int | `--time-limit` (seconds) |
| `solver` | string or null | `--solver`; `null` = default (HiGHS) |
| `chip_report` | bool | `false` with `--no-chip-report` |
| `write_log` | bool | `false` with `--no-log` |

### context

| Field | Type | Meaning |
|---|---|---|
| `current_gw` | int | the gameweek being planned now (the next deadline) |
| `horizon` | array of int | gameweeks of the plan, `current_gw` first |
| `weights` | array of `{gw, weight}` | weight of each gameweek in the objective, `0.85^(gw - current_gw)` |
| `history_gws` | array of int | finished gameweeks the xP model learned from |
| `history_rows` | int | (player, gameweek) rows in that history |
| `blank_double` | array of `{gw, blank, double}` | gameweeks of the horizon where some club has no match (`blank`) or two or more (`double`); both are arrays of club short names. Gameweeks with neither are left out |
| `free_transfers` | object | `value`: int or `null` (`null` = unlimited, before the team's first deadline); `source`: `"history"` (rebuilt), `"override"` (`--ft`) or `"unlimited"`; `rebuilt`: int or `null`, the value rebuilt from the history, also when `--ft` was given |
| `budget` | object | `total`: number, the budget of the plan (£m); `squad_value`: number or `null`, selling value of the current squad (`null` in GW1); `bank`: number; `source`: `"computed"` (squad value + bank), `"override"` (`--override-budget`) or `"default"` (100.0 in GW1) |

### solver

| Field | Type | Meaning |
|---|---|---|
| `name` | string | solver used, e.g. `"HiGHS"` (may include its version) |
| `seconds` | number | solve time of the main plan, all passes |
| `passes` | int | solves of the bench-weight fixed point (1–3) |
| `proven` | bool | `true`: the last pass stopped at the gap target (0.05%); `false`: it stopped at the time limit |
| `gap` | number or null | best proven bound minus the plan's objective, in weighted points: the plan is at most this far below the best plan of the model. `null` = bound unknown |
| `objective` | number | objective of the plan (weighted points over the horizon, hits subtracted, tie-breaks included) |
| `bench_weights_change` | number | largest change of a bench weight in the last pass; above 0.02 the fixed point did not settle (warning `bench_not_converged`) |

### squad

The 15 players of the current gameweek: the XI ordered by position, then the bench in its order.

| Field | Type | Meaning |
|---|---|---|
| `id` | int | FPL player id (`element`) |
| `name` | string | FPL `web_name` |
| `team` | int | FPL club id |
| `position` | string | `"GK"`, `"DEF"`, `"MID"`, `"FWD"` |
| `cost` | number | £m: the selling price for a player you keep, the current price for one you buy |
| `xp_now` | number | xP in the current gameweek (all his matches; 0 in a blank) |
| `papp_now` | number | probability that he plays at least once in the current gameweek |
| `xp_horizon` | number | weighted xP over the horizon |
| `n_matches_now` | int | his club's matches in the current gameweek (0 = blank, 2 = double) |
| `role` | string | `"Captain"`, `"Vice"`, `"Start"` or `"Bench"` |
| `bench_order` | int or null | `0` = bench goalkeeper, `1`–`3` = outfield bench slots in order; `null` for the XI |
| `action` | string | `"KEEP"` (in your current squad) or `"BUY"` |
| `available` | bool | `false` = his chance of playing is below `--min-chance`; he cannot start this gameweek |
| `source` | string | `"model"` or `"FPL ep"` (his xP is FPL's `ep_next`) |

### plan

One object per gameweek of the horizon, in order. Only the current gameweek is a decision; the
later ones are re-planned every week with new data.

| Field | Type | Meaning |
|---|---|---|
| `gw` | int | gameweek |
| `chip` | string or null | chip played in that gameweek (`"WC"`, `"FH"`, `"BB"`, `"TC"`), `null` = none |
| `ft` | int or null | free transfers before that gameweek's transfers; `null` when they are unlimited |
| `ft_unlimited` | bool | `true` before the team's first deadline or with a Wildcard / Free Hit |
| `hits` | int | paid transfers (each costs 4 points) |
| `transfers` | array of `{out_id, out_name, in_id, in_name}` | sales and purchases, paired by position |
| `captain` | `{id, name}` | captain |
| `vice` | `{id, name}` or null | vice-captain; `null` if the model has no vice-captain term |
| `squad` | array of int | the 15 player ids |
| `xi` | array of int | the 11 starters |
| `bench` | array of int | bench in order: goalkeeper first, then outfield slots 1–3 |

### sales_now

Players sold in the current gameweek (empty in GW1).

| Field | Type | Meaning |
|---|---|---|
| `id` | int | player id |
| `name` | string | `web_name` |
| `price` | number | £m the sale returns (his selling price) |
| `available` | bool | as in `squad` |

### expected_points_now

Expected points of the current gameweek, unweighted, split into the terms of the objective.

| Field | Type | Meaning |
|---|---|---|
| `xi` | number | sum of the XI's xP (all 15 with Bench Boost); the captain is counted once here |
| `captain` | number | the captain's extra points: his xP × 1, or × 2 with Triple Captain |
| `vice` | number | expected extra points of the vice-captain, (1 − P(captain plays)) × his xP (× 2 with Triple Captain) |
| `autosubs` | number | expected points of bench players coming on for starters who do not play (0 with Bench Boost) |
| `hits` | int | points deducted (4 per paid transfer); subtract it |
| `total` | number | `xi + captain + vice + autosubs - hits` |
| `bench_weights` | array of 3 numbers, or null | P(bench slot k is used), k = 1–3; `null` with Bench Boost |

### chips

| Field | Type | Meaning |
|---|---|---|
| `source` | string | where the chip windows came from: the API, or the built-in rule if the API field is missing |
| `sets` | array of `{chip, start, stop, state}` | every chip set of the season: `start`/`stop` gameweeks; `state`: `"available"`, `"played GWn"` or `"expired"` |
| `status` | string | `"computed"`; `"no_chip_now"` (no chip can be played now); `"skipped_flag"` (`--no-chip-report`); `"skipped_chip"` (`--chip` was given) |
| `exact` | array of `{chip, window_end, delta, error_bound}` | one per chip that can be played now. `delta`: number or `null` (not computed in time), the exact gain now = best plan with the chip minus best plan without it, over `min(3, lookahead)` gameweeks, unweighted points; `error_bound`: number or `null`, bound on the error of `delta` from the two solver gaps; `window_end`: last gameweek of the chip's current window |
| `estimates` | array of `{chip, gw, delta}` | estimated one-gameweek gain of the chip in each gameweek up to `window_end`, on the squad of the plan (none for `WC`) |
| `suggestion` | `{chip, delta}` or null | the suggested chip and its exact gain; `null` = no chip this gameweek |
| `also_best_now` | array of string | other chips that also pass the rule now (one chip per gameweek) |
| `holds` | array of `{chip, now, best_gw, best, window_end}` | chips not suggested now: estimate now (`now`, may be `null`) and the gameweek with the best estimate |
| `incomplete` | array of string | chips with no suggestion at all because the time limit left out their exact gain or an estimate up to `window_end` (the rule is not applied to part of the numbers, so that the suggestion does not depend on the speed of the machine) |
| `wc_note` | `{wc_delta, suggested_delta}` or null | set when the Wildcard gains more now than the suggested chip |
| `not_computed` | array of string | parts skipped by the time limit, e.g. `"exact WC"`, `"estimate TC GW12"`, `"estimates from GW16"`. Every number that is given was computed in full (all its solves reached the gap target), so the numbers do not depend on the speed of the machine; a slower machine gives fewer of them |
| `seconds` | number or null | time of the chip analysis; `null` when it did not run (`skipped_flag`, `skipped_chip`) |

When `status` is not `"computed"`, `exact`, `estimates`, `holds`, `incomplete` and `also_best_now` are empty
and `suggestion` and `wc_note` are `null`.

### explain

One object per player named in `--explain` (empty without it).

| Field | Type | Meaning |
|---|---|---|
| `id`, `name`, `team` | int, string, string | player id, `web_name`, club short name |
| `pos_id` | int | 1 GK, 2 DEF, 3 MID, 4 FWD |
| `price` | number | current price (£m) |
| `source` | string | `"model"` or `"FPL ep"` |
| `chance` | int or null | FPL's chance of playing next round (percent); `null` = no news |
| `minutes_model` | object or null | `Papp`, `P60`, `Emin` (shrunk; per match), `raw_Papp`, `raw_P60`, `raw_Emin` (before shrinkage), `Wapp` (recency weight of the evidence), `minutes_total` (season minutes), `price_factor`; `null` for `"FPL ep"` players |
| `per90` | object | shrunk rates per 90 minutes: `xg`, `xa`, `xgc`, `dc`, `saves`, `bonus`, `yc`, `rc` (empty for `"FPL ep"` players) |
| `components` | object | xP of the current gameweek by component: `appearance`, `goals`, `assists`, `clean_sheet`, `conceded`, `saves`, `defcon`, `bonus`, `cards` (empty for `"FPL ep"` players) |
| `ep_next` | number | FPL's own projection for the next gameweek |
| `xp_by_gw` | array of `{gw, xp}` | xP in each gameweek of the horizon |

### timings and api

| Field | Type | Meaning |
|---|---|---|
| `timings.data_s` | number | downloading and preparing the data, xP included |
| `timings.solve_s` | number | main plan |
| `timings.chips_s` | number | chip analysis |
| `timings.total_s` | number | whole run |
| `api.requests` | int | HTTP requests sent (retries included) |
| `api.retries` | int | requests repeated after a temporary failure |
| `api.cache_hits` | int | live gameweek data read from `data_cache/fpl_live/` |
| `api.cache_writes` | int | live gameweek data written to the cache |

## Mode `evaluate`

`python -m fplopt --mode evaluate --gw N --json`

| Field | Type | Meaning |
|---|---|---|
| `gw` | int | the evaluated gameweek |
| `log_path` | string | the predictions log that was read |
| `rows` | array of `{label, n, mae, mae_fpl, bias, bias_fpl}` | one per group of players. `label`: `"All logged"`, `"Relevant (model or FPL >= 2)"`, `"Your starting XI"`; `n`: players; `mae`, `bias`: mean \|xP − actual\| and mean (xP − actual) of the model; `mae_fpl`, `bias_fpl`: the same for FPL's `ep_next`. A group with no players is left out. Empty when the log has no predictions for that gameweek (warning `no_predictions`) |
| `xi_captain` | `{predicted, actual}` or null | points of the logged starting XI with the captain counted twice; `null` when no XI was logged |
| `warnings` | array | [Warning codes](#warning-codes) |

## Errors

On any error the document is:

```json
{
  "schema_version": "1.0",
  "engine_version": "0.9.0",
  "ok": false,
  "mode": "optimize",
  "error": {"exit_code": 4, "type": "not_found", "message": "..."},
  "warnings": []
}
```

| `exit_code` | `type` | Meaning |
|---|---|---|
| 1 | `internal` | unexpected error (a bug); the traceback is on stderr |
| 2 | `input` | invalid argument |
| 3 | `network` | FPL API unreachable, timeout, HTTP 5xx / 429 after 4 attempts, other HTTP error, or a reply that is not JSON |
| 4 | `not_found` | HTTP 404: unknown team id or gameweek |
| 5 | `solver` | the solver found no plan |
| 6 | `user_action` | the run needs the user; `message` says what (e.g. pass `--ft`, install a package) |
| 7 | `data` | a local file is missing (e.g. `predictions_log.csv` for `evaluate`) |
| 130 | `internal` | interrupted (Ctrl+C) |

`exit_code` always equals the exit status of the process. `warnings` holds what was logged
before the error.

## Warning codes

`warnings` is an array of `{code, message}`. `message` is the text printed on stderr without its
`[WARNING]` / `[INFO]` / `[*]` prefix. Some entries are informational, not problems. A reader
must accept codes it does not know; an entry logged without a code has `"code": "warning"`.

| Code | Meaning |
|---|---|
| `previous_gw_not_final` | the previous gameweek is not final yet; the minutes model uses data up to the one before |
| `no_history` | no finished gameweek yet; every player uses FPL's `ep_next` |
| `join_dates_missing` | the API did not give the date each player joined his club; the minutes model uses all gameweeks |
| `ft_rebuild` | the free transfers rebuilt from the history disagree with the hit costs in the API; without `--ft` the run then stops with exit code 6 |
| `bank_mismatch` | the bank rebuilt from the transfer history differs from the one the API reports; selling prices may be off (`--override-budget` sets the budget) |
| `first_squad_prices` | the price bounds of the team's first squad do not contain the amount actually paid; selling prices may be off |
| `replay_squad_mismatch` | replaying the transfer history does not give the current squad; players without a known purchase price are valued at their current price |
| `chips` | the API has no chip windows (the built-in 2026/27 windows are used), or a chip is already active for the current gameweek (pass it with `--chip`) |
| `time_limit` | the solver stopped at the time limit; the plan is valid and `solver.gap` bounds its distance from the best plan |
| `bench_not_converged` | the bench weights still changed after the last pass; the last plan is used |
| `chips_not_computed` | the time limit skipped parts of the chip analysis (`chips.not_computed`) |
| `api_retry` | an API request failed temporarily and was repeated |
| `cache_unreadable`, `cache_write_failed` | a cache file could not be read (it is downloaded again) or written |
| `live_fields_missing` | the live gameweek data lack some statistics the model reads; they are taken as 0 |
| `no_predictions` | `evaluate`: the log has no predictions for that gameweek |

## Example

Abridged output of a run on the synthetic season of the tests (players are named `P<id>`;
lookahead 3; arrays shortened):

```json
{
  "schema_version": "1.0",
  "engine_version": "0.9.0",
  "ok": true,
  "mode": "optimize",
  "input": {"team_id": 1, "ft": null, "chip": "NONE", "min_chance": 75, "lookahead": 3,
            "override_budget": null, "max_hits": null, "time_limit": 180, "solver": null,
            "chip_report": true, "write_log": false},
  "context": {
    "current_gw": 8,
    "horizon": [8, 9, 10],
    "weights": [{"gw": 8, "weight": 1.0}, {"gw": 9, "weight": 0.85}, {"gw": 10, "weight": 0.7224999999999999}],
    "history_gws": [1, 2, 3, 4, 5, 6, 7],
    "history_rows": 2520,
    "blank_double": [],
    "free_transfers": {"value": 3, "source": "history", "rebuilt": 3},
    "budget": {"total": 99.1, "squad_value": 96.19999999999999, "bank": 2.9, "source": "computed"}
  },
  "solver": {"name": "HiGHS", "seconds": 22.6, "passes": 3, "proven": true,
             "gap": 0.0686, "objective": 141.832, "bench_weights_change": 0.0},
  "squad": [
    {"id": 214, "name": "P214", "team": 12, "position": "FWD", "cost": 8.5, "xp_now": 6.52,
     "papp_now": 0.878, "xp_horizon": 16.609, "n_matches_now": 1, "role": "Captain",
     "bench_order": null, "action": "BUY", "available": true, "source": "model"},
    {"id": 1, "name": "P1", "team": 1, "position": "GK", "cost": 5.0, "xp_now": 0.88,
     "papp_now": 0.465, "xp_horizon": 2.498, "n_matches_now": 1, "role": "Bench",
     "bench_order": 0, "action": "KEEP", "available": true, "source": "model"}
  ],
  "plan": [
    {"gw": 8, "chip": null, "ft": 3, "ft_unlimited": false, "hits": 2,
     "transfers": [{"out_id": 321, "out_name": "P321", "in_id": 214, "in_name": "P214"}],
     "captain": {"id": 214, "name": "P214"}, "vice": {"id": 108, "name": "P108"},
     "squad": [1, 19, 30, 39, 47, 57, 75, 108, 111, 171, 207, 214, 297, 327, 360],
     "xi": [19, 30, 39, 47, 57, 108, 171, 214, 297, 327, 360],
     "bench": [1, 207, 75, 111]}
  ],
  "sales_now": [{"id": 321, "name": "P321", "price": 6.5, "available": true}],
  "expected_points_now": {"xi": 45.95, "captain": 6.52, "vice": 0.795, "autosubs": 4.160,
                          "hits": 8, "total": 49.425, "bench_weights": [0.794, 0.436, 0.162]},
  "chips": {
    "source": "API (bootstrap-static chips)",
    "sets": [{"chip": "BB", "start": 1, "stop": 19, "state": "available"},
             {"chip": "WC", "start": 2, "stop": 19, "state": "played GW5"}],
    "status": "computed",
    "exact": [{"chip": "BB", "window_end": 19, "delta": 2.850, "error_bound": 0.141},
              {"chip": "TC", "window_end": 19, "delta": 7.315, "error_bound": 0.135}],
    "estimates": [{"chip": "TC", "gw": 8, "delta": 7.315}, {"chip": "TC", "gw": 17, "delta": 7.533}],
    "suggestion": null,
    "also_best_now": [],
    "holds": [{"chip": "TC", "now": 7.315, "best_gw": 17, "best": 7.533, "window_end": 19}],
    "incomplete": [],
    "wc_note": null,
    "not_computed": [],
    "seconds": 82.0
  },
  "explain": [],
  "explain_not_found": [],
  "fallback_players": 0,
  "predictions_log": null,
  "warnings": [],
  "timings": {"data_s": 0.30, "solve_s": 22.6, "chips_s": 82.0, "total_s": 105.0},
  "api": {"requests": 0, "retries": 0, "cache_hits": 0, "cache_writes": 0}
}
```
