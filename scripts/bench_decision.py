"""Timing and consistency of the decision-check solves for a Wildcard or Free Hit, on real past squads.

For a chip c (WC or FH) played in GW g, the decision check needs three values of the multi-period MILP:
    Z0    the main plan, no chip,
    Z+    the best plan with c in GW g (chip_fix),
    Z(D)  the best plan with c in GW g and the user's squad S in GW g (chip_fix + force_squad),
    G = Z(D) - Z0,  R = Z+ - Z(D) (>= 0 if both solves use the same bench weights).
The script calls milp.plan_squad directly with the options of the engine, so it measures the same solves
as the check. The squads S are made by rule, with no choice by hand:
    a  S = S+, the squad of Z+ in GW g (then R must be 0 within the gaps),
    b  S+ with one change: the outfield player of S+ with the lowest xP (FH: xP in GW g; WC: weighted xP
       from GW g to the end of the horizon) replaced by the cheapest player of the same position outside
       S+ that keeps the club limit and the budget (ties: the lowest id),
    c  S = the current squad (the chip played with no change).
The squads come from a saved backtest path (scripts/backtest_season.py --save-gws ...), as in
scripts/bench_chip_solve.py:

    python scripts/bench_decision.py --states st --cases 2025-26:8 2025-26:15 --chips WC FH --out dec.csv
    python scripts/bench_decision.py --states st --cases 2024-25:15 --chips WC --offset 2 --decisions a

Each row is one solve: case, chip, GW, solve name, seconds, passes, complete, gap, objective, J (the
objective without tie-breaks, decision.plan_value), value_sc (J with the bench weights implied by the plan's
own XI) and, for a decision, the number of players it changes and G, R with their bounds from the gaps.
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

from fplopt import milp                                             # noqa: E402
from fplopt.chips import gw_points                                  # noqa: E402
from fplopt.decision import diff_bounds, plan_value                 # noqa: E402
from fplopt.engine import Options, optimize                         # noqa: E402

FIELDS = ['case', 'chip', 'gw', 'solve', 'seconds', 'passes', 'complete', 'gap', 'objective', 'J', 'value_sc',
          'beta_change', 'changed_vs_best', 'changed_vs_current', 'G', 'G_low', 'G_high', 'R', 'R_low', 'R_high',
          'k2_ok', 'r_below_gap']


def tenths(x):
    """GBPm to integer tenths (prices in the API are tenths; float sums are not exact)."""
    return int(round(x * 10))


def squad_cost(pool, squad, start_squad):
    """Cost of a squad for the static budget check, in integer tenths.

    Owned players count at their selling price, the others at now_cost. With constant prices this is
    exactly the cheapest cost of the squad in the MILP."""
    owned = set(start_squad)
    return sum(tenths(pool['sell'][i]) if i in owned else tenths(pool['price'][i]) for i in squad)


def violations(pool, squad, start_squad, budget):
    """Rule checks of a 15-man squad: positions 2/5/5/3, at most 3 per club, budget. [] = valid."""
    out = []
    count = {p: sum(1 for i in squad if pool['pos'][i] == p) for p in (1, 2, 3, 4)}
    if (count[1], count[2], count[3], count[4]) != (2, 5, 5, 3) or len(set(squad)) != 15:
        out.append('positions')
    clubs = {}
    for i in squad:
        clubs[pool['team'][i]] = clubs.get(pool['team'][i], 0) + 1
    if max(clubs.values()) > 3:
        out.append('club_limit')
    if squad_cost(pool, squad, start_squad) > tenths(budget):
        out.append('over_budget')
    return out


def one_change(pool, best, start_squad, budget, chip, g, horizon, weights):
    """Decision b: S+ with its weakest outfield player replaced by the cheapest valid one of the same position.

    Weakest = lowest xP in GW g (FH lasts one GW) or lowest weighted xP from g to the end of the horizon
    (WC). Returns the new squad, or None if no valid change exists."""
    xp = pool['xp']
    gws = [g] if chip == 'FH' else [t for t in horizon if t >= g]
    score = {i: sum(weights[t] * xp[(i, t)] for t in gws) for i in best}
    owned = set(start_squad)
    cost = {i: tenths(pool['sell'][i]) if i in owned else tenths(pool['price'][i]) for i in pool['ids']}
    for out_id in sorted((i for i in best if pool['pos'][i] != 1), key=lambda i: (score[i], i)):
        # same position, not in S+; cheapest first, then the lowest id
        for in_id in sorted((i for i in pool['ids'] if pool['pos'][i] == pool['pos'][out_id] and i not in best),
                            key=lambda i: (cost[i], i)):
            squad = [i for i in best if i != out_id] + [in_id]
            if not violations(pool, squad, start_squad, budget):
                return squad
    return None


def self_consistent_value(res, pool, horizon, weights):
    """J of a plan with the bench weights implied by its own XI in every GW.

    This is the fixed point the passes aim for, so two plans solved with different bench weights can be
    compared. The proof of R >= 0 needs the same objective, and the bench weights of the last pass depend
    on each plan's XI: this value measures that risk."""
    total = 0.0
    for g in horizon:
        pg = res['plan'][g]
        beta = milp.bench_weights(pg['xi'], pool['pos'], {i: pool['papp'][(i, g)] for i in pool['ids']})
        total += weights[g] * gw_points(pg, g, pool, beta)['total']
    return total


def solve_row(name, res, seconds, pool, horizon, weights):
    """One CSV row of a solve (the decision fields are filled by the caller)."""
    row = {k: None for k in FIELDS}
    row.update(solve=name, seconds=round(seconds, 1), passes=res.get('passes'), complete=res.get('complete'))
    if 'plan' in res:
        row.update(gap=None if res.get('gap') is None else round(res['gap'], 3), objective=round(res['objective'], 3),
                   J=round(plan_value(res, pool, horizon, weights), 3),
                   value_sc=round(self_consistent_value(res, pool, horizon, weights), 3),
                   beta_change=None if res.get('beta_change') is None else round(res['beta_change'], 3))
    return row


def run_decisions(state, chip, g, opts, time_limit, decisions=('a', 'b', 'c'), z0_seconds=None):
    """Solve Z+ and Z(D) for each decision of 'decisions' on the state of an engine run (main plan = Z0).

    Returns the rows: z0, z_best, then z_a, z_b, z_c (those that exist)."""
    pool, horizon, weights = state.pool, state.horizon, state.weights
    start, budget, ft0 = state.start_squad, state.budget, state.ft0
    base = state.plan_result

    def solve(force_squad=None):
        t = time.time()
        res = milp.plan_squad(pool, horizon, weights, start, budget, ft0, 'NONE', opts.max_hits, opts.use_vice,
                              opts.use_bench, time_limit=time_limit, solver=opts.solver,
                              chip_fix={(chip, g): 1}, force_squad=force_squad)
        return res, time.time() - t

    rows = [solve_row('z0', base, base.get('seconds', 0.0) if z0_seconds is None else z0_seconds, pool, horizon,
                      weights)]
    best, sec = solve()
    rows.append(solve_row('z_best', best, sec, pool, horizon, weights))
    if 'plan' not in best:
        return rows
    s_best = best['plan'][g]['squad']
    squads = {'a': list(s_best), 'b': one_change(pool, s_best, start, budget, chip, g, horizon, weights),
              'c': list(start) if start else None}
    z0, zb = plan_value(base, pool, horizon, weights), plan_value(best, pool, horizon, weights)
    for d in decisions:
        sq = squads[d]
        if sq is None:
            continue
        bad = violations(pool, sq, start, budget)
        if bad:                                   # the squads are valid by construction; stop if not
            raise RuntimeError(f"decision {d}: invalid squad ({', '.join(bad)})")
        res, sec = solve({g: sq})
        row = solve_row(f"z_{d}", res, sec, pool, horizon, weights)
        row['changed_vs_best'] = len(set(sq) - set(s_best))
        row['changed_vs_current'] = len(set(sq) - set(start))
        if 'plan' in res and res.get('complete') and best.get('complete') and base.get('complete'):
            zd = plan_value(res, pool, horizon, weights)
            gap_d = res.get('gap')
            G, R = zd - z0, zb - zd
            g_lo, g_hi = diff_bounds(G, gap_d, base.get('gap'))
            r_lo, r_hi = diff_bounds(R, best.get('gap'), gap_d)
            row.update(G=round(G, 3), G_low=None if g_lo is None else round(g_lo, 3),
                       G_high=None if g_hi is None else round(g_hi, 3), R=round(R, 3),
                       R_low=None if r_lo is None else round(r_lo, 3), R_high=None if r_hi is None else round(r_hi, 3))
            if gap_d is not None and best.get('gap') is not None:
                # bool() because comparing numpy floats gives numpy.bool_, which is not a Python bool
                row['r_below_gap'] = bool(R < -gap_d)             # R below 0 by more than the gap
                if d == 'a':
                    row['k2_ok'] = bool(-gap_d <= R <= best['gap'])   # same squad as Z+: R = 0 within the gaps
        rows.append(row)
    return rows


def run_case(states, case, chip, lookahead, offset, time_limit, decisions):
    season, gw = case.split(':')
    _, client = B.load_state(season, int(gw), states)
    opts = Options(team_id=1, lookahead=lookahead, write_log=False, time_limit=time_limit, chip_report=False)
    r = optimize(client, opts)
    g = r.state.horizon[offset]
    rows = run_decisions(r.state, chip, g, opts, time_limit, decisions)
    for row in rows:
        row.update(case=case, chip=chip, gw=g)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--states', required=True, help="folder of the saved teams")
    ap.add_argument('--cases', nargs='+', required=True, help="season:gw, e.g. 2025-26:8")
    ap.add_argument('--chips', nargs='+', default=['WC', 'FH'], choices=['WC', 'FH'])
    ap.add_argument('--lookahead', type=int, default=5)
    ap.add_argument('--offset', type=int, default=0, help="GW of the chip = current GW + offset (inside the horizon)")
    ap.add_argument('--decisions', nargs='+', default=['a', 'b', 'c'], choices=['a', 'b', 'c'])
    ap.add_argument('--time-limit', type=int, default=B.NO_LIMIT)
    ap.add_argument('--out')
    args = ap.parse_args()
    if not 0 <= args.offset < args.lookahead:
        ap.error("--offset must be at least 0 and below --lookahead")
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    rows = []
    for case in args.cases:
        for chip in args.chips:
            for row in run_case(args.states, case, chip, args.lookahead, args.offset, args.time_limit,
                                args.decisions):
                rows.append(row)
                print(' | '.join(f"{k} {row[k]}" for k in FIELDS), flush=True)
                if args.out:
                    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
                        w = csv.DictWriter(fh, fieldnames=FIELDS)
                        w.writeheader()
                        w.writerows(rows)


if __name__ == '__main__':
    main()
