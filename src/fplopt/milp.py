"""The multi-period MILP and the solver interface.

Moved unchanged from fpl_v2.py in the refactor; later moved to the PuLP 4 API. This is
the only module that uses PuLP.
"""
import time  # M7: solve time is reported (it guides the choice of lookahead)

import pulp

from .errors import UserActionError
from .rules import CHIPS, HIT_COST, MAX_FT

try:
    import highspy  # M10: the MIP solver; PuLP 4 ships none (required, see requirements.txt)
except ImportError:
    highspy = None


def require_pulp_4():
    """This module uses the PuLP 4 API (solve() returns LpSolveStats; LpStatus, sol_status
    and PULP_CBC_CMD no longer exist). PuLP 3.x would fail later with an AttributeError, so stop
    early with a clear message."""
    if int(pulp.__version__.split('.')[0]) < 4:
        raise UserActionError(f"PuLP {pulp.__version__} found; this program needs PuLP 4.x. "
                              f"Install the pinned versions with: pip install -r requirements.txt")


# ---------------------------------------------------------------------------
# Solver and fixed-point constants (M5, M7, M10) and tie-breaks (M2, C3)
# ---------------------------------------------------------------------------
# [assumption] stop re-solving when the bench weights implied by the chosen XI
# differ from the ones used by at most this much (probability units).
BENCH_BETA_TOL = 0.02
BENCH_MAX_PASSES = 3  # [assumption] cap on solves for the bench-weight fixed point
# Tie-break, not a model of anything: among plans with the same expected points,
# prefer fewer transfers (e.g. a swap of two equal players in the last GW of the
# horizon, where a free transfer has no value for the solver). 0.001 points per
# transfer is far below any xP difference the model can resolve.
TRANSFER_TIEBREAK = 0.001
# M9 originally shrank P(appear) only for the vice-captain / autosub terms with
# PAPP_PRIOR_WEIGHT / PAPP_PRIOR_MEAN. Since 8c the whole minutes model is shrunk in
# build_features (xp_model.APP_PRIOR_*), so those terms use the same Papp as the xP.
# M7/M10: wall-clock limit (seconds) for the WHOLE solve, all bench-weight passes
# together. When it is reached, the solver returns the best plan found so far and its
# proven bound; the gap (bound - plan) is printed, so the result is a guarantee "at
# most GAP weighted points below the best possible plan of this model".
# [decision] set from measured solve times (see the --lookahead comment in cli.py).
SOLVER_TIME_LIMIT = 180
# M10: [decision] stop when the plan is proven within 0.05% of the best possible plan of
# this model (~0.1 weighted points at lookahead 5). The xP error is far larger (RMSE ~3
# points per player-GW), and proving the last 0.05% took most of the solve time
# (CBC, lookahead 5: 33 s exact -> 17 s). The gap is always printed.
MIP_REL_GAP = 0.0005
MIN_PASS_SECONDS = 10  # [assumption] no extra bench-weight pass with less time than this

# C2: largest possible value of T - F in one GW: at most 15 transfers in, at least 1 FT.
# The smallest constant that switches the hit and FT constraints off in a WC/FH GW
# so it is the tightest big-M for them; it is a rule of the game, not a tuning value.
CHIP_TRANSFER_M = 15 - 1
# Tie-break, not a model of anything (same role as TRANSFER_TIEBREAK): a chip whose use gains
# nothing is not played. Without it the solver played a Free Hit with no transfers in the last
# GW of a test horizon (harness_7 rules), because an unused chip has no value inside the MILP.
CHIP_TIEBREAK = 0.001


def is_chosen(var):
    """True if a binary LpVariable equals 1 in the solution (float-safe)."""
    # D: solvers return floats, e.g. 0.9999999 or 1e-10 for a binary variable.
    # 'varValue == 1' can miss a selected player; 0.5 is the safe midpoint.
    # varValue is None if the variable was never solved.
    return var.varValue is not None and var.varValue > 0.5


def poisson_binomial_tail(qs, kmax=3):
    """[P(N>=1), ..., P(N>=kmax)] where N = number of independent events with
    probabilities qs (here: starters who do not appear). Exact, by building the
    distribution of N one player at a time (dynamic programming)."""
    dist = [1.0]                                   # dist[n] = P(N = n) so far
    for q in qs:
        new = [0.0] * (len(dist) + 1)
        for n, pn in enumerate(dist):
            new[n] += pn * (1.0 - q)               # this starter plays
            new[n + 1] += pn * q                   # this starter does not
        dist = new
    return [sum(dist[k:]) for k in range(1, kmax + 1)]


def reference_xi(squad, pos, xp_g):
    """Best valid XI of a 15-man squad for one GW, by xP (greedy; exact for the
    FPL formation rules because only minimums bind inside a 2-5-5-3 squad)."""
    by_pos = {p: sorted((i for i in squad if pos[i] == p), key=lambda i: -xp_g[i]) for p in (1, 2, 3, 4)}
    xi = by_pos[1][:1] + by_pos[2][:3] + by_pos[3][:2] + by_pos[4][:1]   # formation minimums
    rest = sorted((i for p in (2, 3, 4) for i in by_pos[p] if i not in xi), key=lambda i: -xp_g[i])
    return xi + rest[:11 - len(xi)]


def bench_weights(xi, pos, papp_g):
    """M5: beta_k = P(at least k outfield starters do not appear), k = 1, 2, 3.
    [assumption] independence between players; formation rules and a bench
    player who does not appear himself are ignored."""
    return poisson_binomial_tail([1.0 - papp_g[i] for i in xi if pos[i] != 1], 3)


def fill_start(prob):
    """Complete the starting solution. PuLP 4 passes HiGHS only the variables that have a
    value (a PARTIAL start, which HiGHS then tries to complete). The PuLP 3 version of this module
    passed every column, with 0 for the variables without a value (HighsSolver.callSolver), and
    that start is feasible: the "keep the squad" start below leaves only transfer, vice-captain,
    autosub and chip-product variables without a value, and 0 is feasible for all of them.
    So every variable without a value gets 0 (clipped to its bounds), as before."""
    for var in prob.variables():
        if var.varValue is None:
            val = 0.0
            if var.lowBound is not None:            # None = no bound (minus infinity)
                val = max(val, var.lowBound)
            if var.upBound is not None:             # None = no bound (plus infinity)
                val = min(val, var.upBound)
            var.setInitialValue(val)


# Solver states that stop the search before the gap target (LpSolveStatus of PuLP 4).
# Only TimeLimit can occur with the options used here; the others are listed for safety.
STOPPED_EARLY = ('TimeLimit', 'IterationLimit', 'NodeLimit', 'SolutionLimit', 'MemoryLimit',
                 'Interrupted', 'Stopped')


def run_solver(prob, solver, time_limit, warm_start):
    """M10: solve with HiGHS; it stops at the relative gap MIP_REL_GAP or at the time limit.
    PuLP 4 returns the solve statistics (LpSolveStats) instead of setting prob.status and
    prob.sol_status, and reports the bound HiGHS has proven (best_bound, in the problem's own
    sense: an upper bound for this maximisation), so the HighsSolver subclass of the PuLP 3
    version is no longer needed. CBC was removed after measurement:
    HiGHS was as good or better in every measured case. 'solver' is kept as an argument for the
    --solver option. Returns {'solver', 'proven' (stopped by the gap, not by time),
    'has_solution', 'stopped_early', 'status', 'bound'}."""
    if solver not in (None, 'highs'):             # e.g. 'cbc' from code written for PuLP 3
        raise UserActionError(f"Unknown solver '{solver}': only 'highs' is supported.")
    if highspy is None:
        raise UserActionError("The HiGHS solver is missing: pip install -r requirements.txt")
    if warm_start:
        fill_start(prob)
    cmd = pulp.HiGHS(msg=False, warmStart=warm_start, timeLimit=time_limit, gapRel=MIP_REL_GAP)
    stats = prob.solve(cmd)
    status = stats.status.name                          # e.g. 'Optimal', 'TimeLimit', 'Infeasible'
    return {'solver': 'HiGHS ' + highspy.__version__ if hasattr(highspy, '__version__') else 'HiGHS',
            'proven': stats.status == pulp.LpSolveStatus.Optimal,
            'has_solution': bool(stats.has_solution), 'stopped_early': status in STOPPED_EARLY,
            'status': status, 'bound': stats.best_bound}


def is_var(e):
    """C3: True for a PuLP variable or expression (the chip switches below are either a
    variable or a fixed 0/1 number; a constraint with no variable is not added)."""
    return isinstance(e, (pulp.LpVariable, pulp.LpAffineExpression))


def solve_plan(pool, horizon, weight, start_squad, budget, ft0, chip='NONE', max_hits=None,
               beta=None, use_vice=True, use_bench=True, warm=None, time_limit=None, solver=None,
               chips_avail=None, chip_fix=None):
    """
    M1-M7: one MILP over the whole horizon.
    pool: dict of dicts keyed by player id: 'pos', 'team', 'price' (now_cost, GBPm),
          'sell' (selling price, owned players only), 'xp' {(i, g): xP}, 'papp' {(i, g):
          P(appears in GW g)}, 'avail' (False -> may not start in the first GW).
    start_squad: ids owned before the first GW of the horizon ([] at GW1).
    ft0: free transfers in the first GW; None = unlimited (before the first deadline).
    chip: chip of the first GW: 'WC'/'FH'/'BB'/'TC' = played, 'NONE' = none, 'AUTO' = the
          model decides (C3, only with chips_avail).
    chips_avail: C3 {chip: [(start GW, stop GW), ...]} windows of the chip sets still
          unused; the model may play each at most once per window. None = no chip planning.
    chip_fix: C3 {(chip, g): 0 or 1} forces chip decisions (used for the opportunity costs).
    beta: {g: [b1, b2, b3]} outfield bench weights (None -> no outfield bench value).
    warm: {variable name: value} of a previous solve, given to the solver as a starting
          solution (the bench fixed point re-solves an almost identical problem).
    Returns a dict with the solver status and the plan (see the end of the function).
    """
    ids = pool['ids']
    pos, team, price, sell, xp, papp = (pool[k] for k in ('pos', 'team', 'price', 'sell', 'xp', 'papp'))
    t0 = horizon[0]
    S0 = set(start_squad)
    ig = [(i, g) for i in ids for g in horizon]
    out_ids = [i for i in ids if pos[i] != 1]
    gk_ids = [i for i in ids if pos[i] == 1]

    prob = pulp.LpProblem("FPL_Multi_Period", pulp.LpMaximize)
    # --- C3: chip switches u[(c, g)]: a binary variable where the chip may be played,
    # otherwise a fixed number (1 = played, 0 = not) ---
    windows = chips_avail or {}
    fix = dict(chip_fix or {})
    u = {}
    for cp in CHIPS:
        for g in horizon:
            if (cp, g) in fix:
                u[(cp, g)] = fix[(cp, g)]
            elif g == t0 and chip != 'AUTO':
                u[(cp, g)] = 1 if cp == chip else 0         # --chip: forced in the current GW
            elif any(s <= g <= e for s, e in windows.get(cp, [])):
                # A variable is created by the problem it belongs to (PuLP 4 API, also in 3.3.x)
                u[(cp, g)] = prob.add_variable(f"chip_{cp}_{g}", cat='Binary')
            else:
                u[(cp, g)] = 0
    for g in horizon:                                       # (K1) at most one chip per GW
        if any(is_var(u[(cp, g)]) for cp in CHIPS):
            prob += pulp.lpSum(u[(cp, g)] for cp in CHIPS) <= 1
    for cp in CHIPS:                                        # (K2) each unused chip at most once per set
        for s, e in windows.get(cp, []):
            terms = [u[(cp, g)] for g in horizon if s <= g <= e]
            if any(is_var(t_) for t_ in terms):
                prob += pulp.lpSum(terms) <= 1
    for g in horizon:                                       # (K3) no Free Hit in two consecutive GWs
        if g + 1 in horizon and (is_var(u[('FH', g)]) or is_var(u[('FH', g + 1)])):
            prob += u[('FH', g)] + u[('FH', g + 1)] <= 1
    chip_vars = [u_ for u_ in u.values() if is_var(u_)]
    # phi_g = 1 if a Wildcard or Free Hit is played in GW g (one chip per GW, so 0 or 1)
    phi = {g: u[('WC', g)] + u[('FH', g)] for g in horizon}
    # first GW: transfers cost nothing (before the first deadline, or a WC/FH fixed in t0)
    free_now = ft0 is None or (not is_var(phi[t0]) and phi[t0] == 1)
    # add_variable_dicts: one variable per key of ig, named "x_(i,_g)" (same names as the
    # LpVariable.dicts of PuLP 3, so warm starts by name keep working)
    x = prob.add_variable_dicts("x", ig, cat='Binary')        # M1: squad in GW g
    l = prob.add_variable_dicts("l", ig, cat='Binary')        # XI in GW g
    c = prob.add_variable_dicts("c", ig, cat='Binary')        # captain in GW g
    # M1: transfers in. Continuous is enough: at the optimum in = max(0, x - x_prev),
    # because every extra unit can only cost FTs or hits. 'out' is not needed:
    # the squad has 15 players in every GW, so #out = #in.
    t_in = prob.add_variable_dicts("in", ig, lowBound=0, upBound=1)

    # C4: permanent squad y[(i, g)] after GW g = what the transfers of GW g+1 start from.
    # A Free Hit squad is temporary: after an FH in g the permanent squad is still y[g-1]
    # (M3 did this only for an FH fixed in the first GW).
    #   no FH in g:       y = x[g]
    #   FH fixed in g:    y = y[g-1]
    #   FH variable:      continuous y in [0, 1] with  y <= x[g] + u_FH,  y <= y[g-1] + 1 - u_FH.
    # The solver wants y as high as possible (in >= x - y), so y = x[g] for u_FH = 0 and
    # y = y[g-1] for u_FH = 1. Not needed after the last GW of the horizon.
    y = {}
    for g in horizon:
        uf = u[('FH', g)]
        for i in ids:
            y_prev = y[(i, g - 1)] if g > t0 else (1 if i in S0 else 0)
            if is_var(uf) and g != horizon[-1]:
                yv = prob.add_variable(f"y_{i}_{g}", lowBound=0, upBound=1)
                prob += yv <= x[(i, g)] + uf
                prob += yv <= y_prev + 1 - uf
                y[(i, g)] = yv
            elif not is_var(uf) and uf == 1:
                y[(i, g)] = y_prev                       # Free Hit fixed in g: squad reverts
            else:
                y[(i, g)] = x[(i, g)]

    def prev_x(i, g):
        # squad that the transfers of GW g start from (C4: after a Free Hit, the one before it)
        return y[(i, g - 1)] if g > t0 else (1 if i in S0 else 0)

    obj = []
    for g in horizon:
        w = weight[g]
        u_bb, u_tc = u[('BB', g)], u[('TC', g)]
        bb = not is_var(u_bb) and u_bb == 1                   # Bench Boost fixed in g (M3)
        mult = 2.0 if (not is_var(u_tc) and u_tc == 1) else 1.0   # extra captain multiplier (TC fixed)
        # --- squad rules (every GW) ---
        prob += pulp.lpSum(x[(i, g)] for i in ids) == 15
        for p_, n_ in ((1, 2), (2, 5), (3, 5), (4, 3)):
            prob += pulp.lpSum(x[(i, g)] for i in ids if pos[i] == p_) == n_
        for t_id in set(team.values()):
            prob += pulp.lpSum(x[(i, g)] for i in ids if team[i] == t_id) <= 3
        # --- M1: continuity  x[g] - x_prev <= in[g] ---
        for i in ids:
            prob += t_in[(i, g)] >= x[(i, g)] - prev_x(i, g)
        # --- XI, captain ---
        prob += pulp.lpSum(l[(i, g)] for i in ids) == 11
        prob += pulp.lpSum(l[(i, g)] for i in gk_ids) == 1
        prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == 2) >= 3
        prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == 3) >= 2
        prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == 4) >= 1
        prob += pulp.lpSum(c[(i, g)] for i in ids) == 1
        for i in ids:
            prob += l[(i, g)] <= x[(i, g)]
            prob += c[(i, g)] <= l[(i, g)]
        scorer = x if bb else l                               # BB: all 15 score
        obj.append(w * pulp.lpSum(xp[(i, g)] * scorer[(i, g)] for i in ids))
        obj.append(w * mult * pulp.lpSum(xp[(i, g)] * c[(i, g)] for i in ids))
        # C5: chip played by the model in g. Each extra term is a product of two binaries,
        # written with a continuous variable bounded by both:
        #   BB: rho_i = (x_i - l_i) * u_BB  (bench player i scores),
        #   TC: tau_i = c_i * u_TC          (captain scores once more).
        # The solver raises a variable with a positive coefficient to min(a, b) = a*b; for a
        # negative xP the lower bound rho >= a + b - 1 forces the product. A captain with
        # xP <= 0 is never chosen while a starter has xP > 0, so tau covers xP > 0 only.
        if is_var(u_bb):
            rho = {i: prob.add_variable(f"bb_{i}_{g}", lowBound=0, upBound=1) for i in ids if xp[(i, g)] != 0}
            for i, r_ in rho.items():
                if xp[(i, g)] > 0:
                    prob += r_ <= x[(i, g)] - l[(i, g)]
                    prob += r_ <= u_bb
                else:
                    prob += r_ >= x[(i, g)] - l[(i, g)] + u_bb - 1
            obj.append(w * pulp.lpSum(xp[(i, g)] * r_ for i, r_ in rho.items()))
        if is_var(u_tc):
            tau = {i: prob.add_variable(f"tc_{i}_{g}", lowBound=0, upBound=1) for i in ids if xp[(i, g)] > 0}
            for i, t_ in tau.items():
                prob += t_ <= c[(i, g)]
                prob += t_ <= u_tc
            obj.append(w * pulp.lpSum(xp[(i, g)] * t_ for i, t_ in tau.items()))
        m_big = max([xp[(i, g)] for i in ids] + [0.0])        # M >= any xP in this GW

        # --- M6: vice-captain. Bonus = xP_C + (1 - P_C) * xP_VC (independence). ---
        # q = 1 - P_C = 1 - sum_i P_i * c_i is LINEAR (exactly one captain), so the only
        # product left is q * xP_VC = sum_j xP_j * (v_j * q). It is split over the
        # vice-captain candidates: s_j = v_j * q with
        #     s_j <= v_j,   sum_j s_j <= q,   s_j >= 0,   bonus part = sum_j xP_j * s_j.
        # Exact for binary v: only the vice-captain can have s > 0, and s <= q.
        # No big-M constant appears, which keeps the LP relaxation tight.
        if use_vice and m_big > 0:
            v = {i: prob.add_variable(f"v_{i}_{g}", cat='Binary') for i in ids}
            s_vc = {i: prob.add_variable(f"svc_{i}_{g}", lowBound=0) for i in ids}
            prob += pulp.lpSum(v.values()) == 1
            prob += pulp.lpSum(s_vc.values()) <= 1 - pulp.lpSum(papp[(i, g)] * c[(i, g)] for i in ids)
            for i in ids:
                prob += v[i] <= l[(i, g)]
                prob += v[i] + c[(i, g)] <= 1
                prob += s_vc[i] <= v[i]
            obj.append(w * mult * pulp.lpSum(xp[(i, g)] * s_vc[i] for i in ids))
            if is_var(u_tc):
                # C5: with Triple Captain the vice-captain term is also tripled: sigma = s_vc * u_TC
                # (s_vc in [0, 1], u_TC binary -> min(s_vc, u_TC) = s_vc * u_TC)
                sig = {i: prob.add_variable(f"tcv_{i}_{g}", lowBound=0, upBound=1) for i in ids if xp[(i, g)] > 0}
                for i, s_ in sig.items():
                    prob += s_ <= s_vc[i]
                    prob += s_ <= u_tc
                obj.append(w * pulp.lpSum(xp[(i, g)] * s_ for i, s_ in sig.items()))
            pool.setdefault('_vice', {})[g] = v

        # --- M5: bench value through autosubs (not with Bench Boost: the bench scores anyway) ---
        if use_bench and not bb and m_big > 0:
            # Goalkeeper: exact under independence, same construction as the vice-captain,
            # with "captain" = starting GK and "vice" = bench GK (bench indicator x - l):
            #   value = (1 - P_startGK) * xP_benchGK,  s_i = (x_i - l_i) * (1 - P_startGK).
            s_gk = {i: prob.add_variable(f"sgk_{i}_{g}", lowBound=0) for i in gk_ids}
            prob += pulp.lpSum(s_gk.values()) <= 1 - pulp.lpSum(papp[(i, g)] * l[(i, g)] for i in gk_ids)
            if is_var(u_bb):
                prob += pulp.lpSum(s_gk.values()) <= 1 - u_bb    # C5: no autosub value in a BB GW
            for i in gk_ids:
                prob += s_gk[i] <= x[(i, g)] - l[(i, g)]
            obj.append(w * pulp.lpSum(xp[(i, g)] * s_gk[i] for i in gk_ids))
            # Outfield: bench slots 1-3 with weights beta (assignment variables; continuous,
            # the assignment constraints are totally unimodular -> integral at the optimum)
            if beta and g in beta:
                b = prob.add_variable_dicts(f"b_{g}", [(i, k) for i in out_ids for k in range(3)],
                                            lowBound=0, upBound=1)
                for i in out_ids:
                    prob += pulp.lpSum(b[(i, k)] for k in range(3)) <= x[(i, g)] - l[(i, g)]
                for k in range(3):
                    prob += pulp.lpSum(b[(i, k)] for i in out_ids) <= 1 - u_bb   # C5: 0 in a BB GW
                obj.append(w * pulp.lpSum(beta[g][k] * xp[(i, g)] * b[(i, k)]
                                          for i in out_ids for k in range(3)))
                pool.setdefault('_bench', {})[g] = b

    # A: an unavailable player may be kept, but not started in the first GW
    for i in ids:
        if not pool['avail'][i]:
            prob += l[(i, t0)] == 0

    # --- M4: budget in every GW, with constant prices [assumption] ---
    # k[i,g] = "still holds the ORIGINAL copy of owned player i" (sells for sell[i]);
    # a re-bought copy costs price[i]. Only needed when sell < price.
    gain_ids = [i for i in S0 if i in pool['price'] and sell[i] < price[i] - 1e-9]
    k_var = prob.add_variable_dicts("k", [(i, g) for i in gain_ids for g in horizon], lowBound=0, upBound=1)
    # C4: kap[(i, g)] = still owns the original copy PERMANENTLY after GW g. A Free Hit squad
    # may leave i out for one GW without selling the original (same rule as y above):
    #   no FH in g: kap = k[g];  FH fixed: kap = kap[g-1];  FH variable: kap <= k + u_FH, kap <= kap[g-1].
    kap = {}
    for g in horizon:
        uf = u[('FH', g)]
        for i in gain_ids:
            prob += k_var[(i, g)] <= x[(i, g)]
            kap_prev = kap[(i, g - 1)] if g > t0 else 1
            if is_var(kap_prev):
                prob += k_var[(i, g)] <= kap_prev              # once sold, the original is gone
            if is_var(uf) and g != horizon[-1]:
                kv = prob.add_variable(f"kap_{i}_{g}", lowBound=0, upBound=1)
                prob += kv <= k_var[(i, g)] + uf
                if is_var(kap_prev):
                    prob += kv <= kap_prev
                kap[(i, g)] = kv
            elif not is_var(uf) and uf == 1:
                kap[(i, g)] = kap_prev                         # Free Hit fixed in g
            else:
                kap[(i, g)] = k_var[(i, g)]
        cost = []
        for i in ids:
            if i in gain_ids:
                cost.append(sell[i] * k_var[(i, g)] + price[i] * (x[(i, g)] - k_var[(i, g)]))
            else:
                cost.append((sell[i] if i in S0 else price[i]) * x[(i, g)])
        prob += pulp.lpSum(cost) <= budget

    # --- M2: free transfers and hits (linearisation) ---
    T = {g: pulp.lpSum(t_in[(i, g)] for i in ids) for g in horizon}
    F = {t0: ft0}
    h = {}
    obj.append(-TRANSFER_TIEBREAK * pulp.lpSum(T.values()))
    if chip_vars:
        obj.append(-CHIP_TIEBREAK * pulp.lpSum(chip_vars))   # C3: no chip without a gain
    for g in horizon:
        if g > t0:
            F[g] = prob.add_variable(f"ft_{g}", lowBound=1, upBound=MAX_FT, cat='Integer')
            ph = phi[g - 1]
            if g - 1 == t0 and ft0 is None:
                prob += F[g] <= 1                       # first deadline passed -> 1 FT
            elif not is_var(ph) and ph == 1:
                prob += F[g] <= F[g - 1]                # chip fixed: FTs kept, none added
            elif is_var(ph):
                # C2: WC/FH decided by the model. phi = 1 switches (2) off and
                # (3) gives F[g] <= F[g-1]; phi = 0 leaves (2) as in M2 and (3) is implied by it.
                prob += F[g] <= F[g - 1] - T[g - 1] + h[g - 1] + 1 + CHIP_TRANSFER_M * ph   # (2)
                prob += F[g] <= F[g - 1] + 1 - ph                                          # (3)
            else:
                prob += F[g] <= F[g - 1] - T[g - 1] + h[g - 1] + 1
        if (g == t0 and free_now) or (not is_var(phi[g]) and phi[g] == 1):
            h[g] = 0                                    # WC / FH fixed / before the first deadline
            continue
        h[g] = prob.add_variable(f"hits_{g}", lowBound=0, upBound=max_hits, cat='Integer')
        prob += h[g] >= T[g] - F[g] - CHIP_TRANSFER_M * phi[g]   # (1); C2: phi = 0 -> M2 as before
        obj.append(-weight[g] * HIT_COST * h[g])        # M2/A1: weighted like that GW's points

    prob += pulp.lpSum(obj)
    if warm:
        for var in prob.variables():
            if var.name in warm and warm[var.name] is not None:
                val = warm[var.name]
                # solvers return values like -1e-12 for a variable with lower bound 0;
                # PuLP rejects an initial value outside the bounds, so clip it first
                if var.lowBound is not None:
                    val = max(val, var.lowBound)
                if var.upBound is not None:
                    val = min(val, var.upBound)
                var.setInitialValue(round(val) if var.cat == 'Integer' else val)
    elif S0:
        # M7: starting solution for the solver = keep the current squad in every GW (always
        # feasible: no transfers, and it costs exactly the budget minus the bank), best
        # XI by xP, captain and vice-captain = the two highest xP of that XI. Without it,
        # most of the solve time was spent looking for a FIRST feasible plan.
        ft_run = ft0 if ft0 is not None else 1
        for g in horizon:
            xp_g = {i: (xp[(i, g)] if (g > t0 or pool['avail'][i]) else -1.0) for i in start_squad}
            xi = reference_xi(start_squad, pos, xp_g)
            ranked = sorted(xi, key=lambda i: -xp_g[i])
            for i in ids:
                x[(i, g)].setInitialValue(1 if i in S0 else 0)
                l[(i, g)].setInitialValue(1 if i in xi else 0)
                c[(i, g)].setInitialValue(1 if i == ranked[0] else 0)
                if '_vice' in pool and g in pool['_vice']:
                    pool['_vice'][g][i].setInitialValue(1 if i == ranked[1] else 0)
            for i in gain_ids:
                k_var[(i, g)].setInitialValue(1)     # original copies kept -> budget holds
                if isinstance(kap[(i, g)], pulp.LpVariable) and kap[(i, g)] is not k_var[(i, g)]:
                    kap[(i, g)].setInitialValue(1)   # C4
            for i in ids:
                if isinstance(y[(i, g)], pulp.LpVariable) and y[(i, g)] is not x[(i, g)]:
                    y[(i, g)].setInitialValue(1 if i in S0 else 0)   # C4: permanent squad = kept squad
            for cp in CHIPS:
                if is_var(u[(cp, g)]):
                    u[(cp, g)].setInitialValue(0)    # C3: start from "no chip played"
            if g > t0:
                F[g].setInitialValue(ft_run)
            if isinstance(h[g], pulp.LpVariable):
                h[g].setInitialValue(0)
            chip_fixed_free = not is_var(phi[g]) and phi[g] == 1   # WC/FH fixed in g: FTs kept
            if not ((g == t0 and ft0 is None) or chip_fixed_free):
                ft_run = min(MAX_FT, ft_run + 1)     # no transfer -> one more FT (cap 5)
            elif ft0 is None and g == t0:
                ft_run = 1
    out = run_solver(prob, solver, time_limit or SOLVER_TIME_LIMIT, bool(warm) or bool(S0))
    # 'status' keeps the texts of the PuLP 3 version: 'Optimal' whenever a plan exists (also when
    # the time limit stopped the proof; 'proven' tells the two apart)
    res = {'status': 'Optimal' if out['has_solution'] else out['status'],
           'objective': pulp.value(prob.objective),
           'values': {var.name: var.varValue for var in prob.variables()},
           'proven': out['proven'], 'solver': out['solver']}
    # has_solution: optimal within MIP_REL_GAP, or a feasible plan found before the time limit
    # stopped the proof. Otherwise (no plan found, infeasible) there are no usable values.
    if not out['has_solution']:
        if out['stopped_early']:
            res['status'] = 'No plan found within the time limit'
        return res
    res['gap'] = None if out['bound'] is None else max(0.0, out['bound'] - res['objective'])

    # --- read the plan; FTs and hits are recomputed with the exact FPL rule ---
    plan, prev, ft = {}, list(start_squad), ft0
    for g in horizon:
        sq = [i for i in ids if is_chosen(x[(i, g)])]
        # C3: chip played in g (fixed or chosen by the model); None = no chip
        played = next((cp for cp in CHIPS if (is_chosen(u[(cp, g)]) if is_var(u[(cp, g)]) else u[(cp, g)] == 1)),
                      None)
        base = prev                                     # C4: permanent squad before g
        ins = [i for i in sq if i not in base]
        outs = [i for i in base if i not in sq]
        if (g == t0 and ft0 is None) or played in ('WC', 'FH'):
            hits, ft_next = 0, (1 if (g == t0 and ft0 is None) else ft)   # FTs kept, none added
        else:
            hits, ft_next = max(0, len(ins) - ft), min(MAX_FT, max(0, ft - len(ins)) + 1)
        xi = [i for i in sq if is_chosen(l[(i, g)])]
        cap = next(i for i in xi if is_chosen(c[(i, g)]))
        vice = None
        if '_vice' in pool and g in pool['_vice']:
            vice = next((i for i in xi if is_chosen(pool['_vice'][g][i])), None)
        bench_out = sorted((i for i in sq if i not in xi and pos[i] != 1), key=lambda i: -xp[(i, g)])
        if '_bench' in pool and g in pool['_bench']:
            b = pool['_bench'][g]                       # order by the slot the solver used
            bench_out = sorted(bench_out, key=lambda i: -sum((3 - k) * (b[(i, k)].varValue or 0) for k in range(3)))
        bench_gk = [i for i in sq if i not in xi and pos[i] == 1]
        plan[g] = {'squad': sq, 'xi': xi, 'captain': cap, 'vice': vice, 'bench': bench_gk + bench_out,
                   'in': ins, 'out': outs, 'ft': ft, 'hits': hits, 'chip': played}
        prev = base if played == 'FH' else sq           # C4: a Free Hit squad reverts
        ft = ft_next
    res['plan'] = plan
    pool.pop('_vice', None)
    pool.pop('_bench', None)
    return res


def plan_squad(pool, horizon, weight, start_squad, budget, ft0, chip='NONE', max_hits=None,
               use_vice=True, use_bench=True, time_limit=None, solver=None, chips_avail=None, chip_fix=None):
    """M5: fixed point for the outfield bench weights. Pass 1 uses the best XI of
    the current squad (or of the pool at GW1) as reference; each later pass uses the
    XI the solver chose, until the weights change by <= BENCH_BETA_TOL. Each pass
    starts the solver from the previous plan (warm start).
    Returns the result of the last pass that produced a plan (or the failed first one)."""
    pos, xp, papp = pool['pos'], pool['xp'], pool['papp']
    ref = start_squad if start_squad else pool['ids']
    beta = {g: bench_weights(reference_xi(ref, pos, {i: xp[(i, g)] for i in ref}), pos,
                             {i: papp[(i, g)] for i in ref}) for g in horizon} if use_bench else None
    best, seconds, history = None, 0.0, []
    for n_pass in range(1, BENCH_MAX_PASSES + 1):
        t_start = time.time()
        # M10: the time limit is for the whole run; each pass gets what is left
        left = (time_limit or SOLVER_TIME_LIMIT) - seconds
        if best is not None and left < MIN_PASS_SECONDS:
            break                                   # no time for another pass: keep the last plan
        res = solve_plan(pool, horizon, weight, start_squad, budget, ft0, chip, max_hits,
                         beta, use_vice, use_bench, warm=best['values'] if best else None,
                         time_limit=max(left, MIN_PASS_SECONDS), solver=solver,
                         chips_avail=chips_avail, chip_fix=chip_fix)   # C3
        seconds += time.time() - t_start
        history.append(beta)
        if 'plan' not in res:
            if best is None:                        # nothing found at all
                res['seconds'], res['passes'] = seconds, n_pass
                return res
            break                                   # keep the previous pass
        res['beta'], res['passes'] = beta, n_pass
        best = res
        if not use_bench:
            break
        implied = {g: bench_weights(res['plan'][g]['xi'], pos, {i: papp[(i, g)] for i in pool['ids']})
                   for g in horizon}
        best['beta_implied'] = implied
        best['beta_change'] = max(abs(implied[g][k] - beta[g][k]) for g in horizon for k in range(3))
        if best['beta_change'] <= BENCH_BETA_TOL:
            break
        beta = implied
    best['seconds'], best['beta_history'] = seconds, history
    return best
