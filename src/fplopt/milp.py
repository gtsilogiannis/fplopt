"""Multi-period MILP of the FPL squad and its solver interface.

This is the only module that uses PuLP.
"""
import time

import pulp

from .errors import UserActionError
from .rules import CHIPS, HIT_COST, MAX_FT

try:
    import highspy  # PuLP 4 ships no MIP solver
except ImportError:
    highspy = None


def require_pulp_4():
    """Stop with a clear message on PuLP 3.x, which lacks the PuLP 4 API used here and would
    otherwise fail later with an AttributeError."""
    if int(pulp.__version__.split('.')[0]) < 4:
        raise UserActionError(f"PuLP {pulp.__version__} found; this program needs PuLP 4.x. "
                              f"Install the pinned versions with: pip install -r requirements.txt")


# [assumption] Stop re-solving when the bench weights implied by the chosen XI
# differ from the ones used by at most this much (probability units).
BENCH_BETA_TOL = 0.02
BENCH_MAX_PASSES = 3  # [assumption] cap on solves for the bench-weight fixed point
# Tie-break, not a model of anything: among plans with the same expected points,
# prefer fewer transfers (e.g. a swap of two equal players in the last GW of the
# horizon, where a free transfer has no value for the solver). 0.001 points per
# transfer is far below any xP difference the model can resolve.
TRANSFER_TIEBREAK = 0.001
# Value (weighted points) of each free transfer left after the last GW of the horizon.
# 0 disables the term. [tuned] with scripts/backtest_season.py --ft-value (pending).
FT_END_VALUE = 0.0
# Wall-clock limit (seconds) for the whole solve, all bench-weight passes together. When it
# is reached, the solver returns the best plan found so far and its proven bound; the gap
# (bound minus plan) is printed, so the plan is at most that many weighted points below the
# best plan of this model.
# [decision] Set from measured solve times (see the --lookahead comment in cli.py).
SOLVER_TIME_LIMIT = 180
# [decision] Default time limit with --chip WC when --time-limit is not given. Measured on a recent
# laptop, lookahead 5, 8 squads of past seasons: one Wildcard solve took 24-460 s (median 112 s;
# 3 of 8 over 180 s). 900 s is about twice the slowest case; a solve that reaches the
# gap target earlier stops earlier.
WILDCARD_TIME_LIMIT = 900
# [decision] Stop when the plan is proven within 0.05% of the best plan of this model (about
# 0.1 weighted points at lookahead 5). The xP error is far larger (RMSE about 2.5 points per
# player-GW), and proving the last 0.05% took about half of the solve time at lookahead 5.
# The gap is always printed.
MIP_REL_GAP = 0.0005
MIN_PASS_SECONDS = 10  # [assumption] no extra bench-weight pass with less time than this

# Largest possible T - F in one GW (at most 15 transfers in, at least 1 FT): the tightest
# big-M that switches the hit and FT constraints off in a WC or FH GW. A rule of the game,
# not a tuning value.
CHIP_TRANSFER_M = 15 - 1
# Tie-break, not a model of anything: a chip that gains nothing is not played. An unused chip
# has no value inside the MILP, so without it the solver may play a Free Hit with no transfers
# in the last GW of the horizon.
CHIP_TIEBREAK = 0.001


def is_chosen(var):
    """True if a binary LpVariable equals 1 in the solution (float-safe)."""
    # Solvers return floats such as 0.9999999 or 1e-10 for a binary variable, so
    # 'varValue == 1' can miss a selected player. varValue is None for an unsolved variable.
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
    """beta_k = P(at least k outfield starters do not appear), k = 1, 2, 3.

    [assumption] Players are independent; formation rules and a bench player who does not
    appear himself are ignored."""
    return poisson_binomial_tail([1.0 - papp_g[i] for i in xi if pos[i] != 1], 3)


def fill_start(prob):
    """Give every variable without a starting value the value 0, clipped to its bounds.

    PuLP passes HiGHS only the variables that have a value, as a partial start that HiGHS must
    complete. The "keep the squad" start leaves only transfer, vice-captain, autosub and
    chip-product variables unset, and 0 is feasible for all of them, so the completed start is
    feasible."""
    for var in prob.variables():
        if var.varValue is None:
            val = 0.0
            if var.lowBound is not None:
                val = max(val, var.lowBound)
            if var.upBound is not None:
                val = min(val, var.upBound)
            var.setInitialValue(val)


# Solver states that stop the search before the gap target. Only TimeLimit can occur with the
# options used here.
STOPPED_EARLY = ('TimeLimit', 'IterationLimit', 'NodeLimit', 'SolutionLimit', 'MemoryLimit',
                 'Interrupted', 'Stopped')


def run_solver(prob, solver, time_limit, warm_start):
    """Solve with HiGHS until the relative gap MIP_REL_GAP or the time limit.

    'bound' is the bound HiGHS has proven, an upper bound for this maximization. 'solver' is
    the value of the --solver option. Returns {'solver', 'proven' (stopped by the gap, not by
    time), 'has_solution', 'stopped_early', 'status', 'bound'}."""
    if solver not in (None, 'highs'):
        raise UserActionError(f"Unknown solver '{solver}': only 'highs' is supported.")
    if highspy is None:
        raise UserActionError("The HiGHS solver is missing: pip install -r requirements.txt")
    if warm_start:
        fill_start(prob)
    cmd = pulp.HiGHS(msg=False, warmStart=warm_start, timeLimit=time_limit, gapRel=MIP_REL_GAP)
    stats = prob.solve(cmd)
    status = stats.status.name
    return {'solver': 'HiGHS ' + highspy.__version__ if hasattr(highspy, '__version__') else 'HiGHS',
            'proven': stats.status == pulp.LpSolveStatus.Optimal,
            'has_solution': bool(stats.has_solution), 'stopped_early': status in STOPPED_EARLY,
            'status': status, 'bound': stats.best_bound}


def is_var(e):
    """True for a PuLP variable or expression.

    A chip switch is either a variable or a fixed 0/1 number, and a constraint with no
    variable is not added."""
    return isinstance(e, (pulp.LpVariable, pulp.LpAffineExpression))


def solve_plan(pool, horizon, weight, start_squad, budget, ft0, chip='NONE', max_hits=None,
               beta=None, use_vice=True, use_bench=True, warm=None, time_limit=None, solver=None,
               chips_avail=None, chip_fix=None, force_captain=None, force_squad=None, force_formation=None):
    """Solve one MILP over the whole horizon.

    pool: dict of dicts keyed by player id: 'pos', 'team', 'price' (now_cost, GBPm),
          'sell' (selling price, owned players only), 'xp' {(i, g): xP}, 'papp' {(i, g):
          P(appears in GW g)}, 'avail' (False: may not start in the first GW).
    start_squad: ids owned before the first GW of the horizon ([] at GW1).
    ft0: free transfers in the first GW; None = unlimited (before the first deadline).
    chip: chip of the first GW: 'WC'/'FH'/'BB'/'TC' = played, 'NONE' = none, 'AUTO' = the
          model decides (only with chips_avail).
    chips_avail: {chip: [(start GW, stop GW), ...]} windows of the chip sets still unused;
          the model may play each at most once per window. None = no chip planning.
    chip_fix: {(chip, g): 0 or 1} forces chip decisions (used for the opportunity costs).
    force_captain: {g: player id}: a decision of the user held fixed, that player is the
          captain in GW g (e.g. "Triple Captain on X in GW g"). Everything else stays free; since
          captain <= XI <= squad, the model keeps him in the squad and the XI, and buys him if
          he is not owned (paying for that transfer as usual).
    force_squad: {g: 15 player ids}: the squad of GW g held fixed (the user's transfers, or
          'no transfer' with the current squad). The transfers into it are counted and paid for
          (free transfers, hits) as usual; the XI, captain and the other GWs stay free.
    force_formation: {g: (DEF, MID, FWD)}: the formation of the XI in GW g (with 1 GK), e.g.
          (3, 4, 3), for the comparison of formations of a fixed squad (chips.formation_values).
    beta: {g: [b1, b2, b3]} outfield bench weights (None: no outfield bench value).
    warm: {variable name: value} of a previous solve, given to the solver as a starting
          solution (the bench fixed point re-solves an almost identical problem).

    Returns a dict with the solver status and the plan.
    """
    ids = pool['ids']
    pos, team, price, sell, xp, papp = (pool[k] for k in ('pos', 'team', 'price', 'sell', 'xp', 'papp'))
    t0 = horizon[0]
    S0 = set(start_squad)
    ig = [(i, g) for i in ids for g in horizon]
    out_ids = [i for i in ids if pos[i] != 1]
    gk_ids = [i for i in ids if pos[i] == 1]
    # Only GWs of the horizon can be forced; an id outside the pool is a caller error.
    fcap = {g: i for g, i in (force_captain or {}).items() if g in horizon}
    for g, i in fcap.items():
        if i not in pos:
            raise ValueError(f"force_captain: player {i} (GW{g}) is not in the pool")
    fsq = {g: set(sq) for g, sq in (force_squad or {}).items() if g in horizon}
    # A forced formation must be legal: 10 outfield players, DEF >= 3, MID >= 2, FWD >= 1.
    ffm = {g: tuple(f) for g, f in (force_formation or {}).items() if g in horizon}
    for g, (d, m, f) in ffm.items():
        if d + m + f != 10 or d < 3 or m < 2 or f < 1:
            raise ValueError(f"force_formation: {d}-{m}-{f} (GW{g}) is not a valid formation")
    for g, sq in fsq.items():
        if len(sq) != 15 or not sq <= set(pos):
            raise ValueError(f"force_squad: GW{g} needs 15 distinct players of the pool")

    prob = pulp.LpProblem("FPL_Multi_Period", pulp.LpMaximize)
    # Chip switches u[(c, g)]: a binary variable where the chip may be played, otherwise a
    # fixed number (1 = played, 0 = not).
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
                u[(cp, g)] = prob.add_variable(f"chip_{cp}_{g}", cat='Binary')
            else:
                u[(cp, g)] = 0
    for g in horizon:                                       # at most one chip per GW
        if any(is_var(u[(cp, g)]) for cp in CHIPS):
            prob += pulp.lpSum(u[(cp, g)] for cp in CHIPS) <= 1
    for cp in CHIPS:                                        # each unused set at most once
        for s, e in windows.get(cp, []):
            terms = [u[(cp, g)] for g in horizon if s <= g <= e]
            if any(is_var(t_) for t_ in terms):
                prob += pulp.lpSum(terms) <= 1
    for g in horizon:                                       # no Free Hit in two consecutive GWs
        if g + 1 in horizon and (is_var(u[('FH', g)]) or is_var(u[('FH', g + 1)])):
            prob += u[('FH', g)] + u[('FH', g + 1)] <= 1
    chip_vars = [u_ for u_ in u.values() if is_var(u_)]
    # phi_g = 1 if a Wildcard or Free Hit is played in GW g (one chip per GW, so 0 or 1).
    phi = {g: u[('WC', g)] + u[('FH', g)] for g in horizon}
    # Transfers of the first GW are free before the first deadline or with a WC/FH fixed in it.
    free_now = ft0 is None or (not is_var(phi[t0]) and phi[t0] == 1)
    # Variables are named "x_(i,_g)"; warm starts match variables by name.
    x = prob.add_variable_dicts("x", ig, cat='Binary')        # squad in GW g
    l = prob.add_variable_dicts("l", ig, cat='Binary')        # XI in GW g
    c = prob.add_variable_dicts("c", ig, cat='Binary')        # captain in GW g
    # Transfers in. Continuous is enough: at the optimum in = max(0, x - x_prev), because every
    # extra unit can only cost FTs or hits. No 'out' variable is needed: the squad has 15
    # players in every GW, so #out = #in.
    t_in = prob.add_variable_dicts("in", ig, lowBound=0, upBound=1)

    # Permanent squad y[(i, g)] after GW g, the squad the transfers of GW g+1 start from.
    # A Free Hit squad is temporary: after an FH in g the permanent squad is still y[g-1].
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
        # Squad the transfers of GW g start from: after a Free Hit, the one before it.
        return y[(i, g - 1)] if g > t0 else (1 if i in S0 else 0)

    obj = []
    for g in horizon:
        w = weight[g]
        u_bb, u_tc = u[('BB', g)], u[('TC', g)]
        bb = not is_var(u_bb) and u_bb == 1                   # Bench Boost fixed in g
        mult = 2.0 if (not is_var(u_tc) and u_tc == 1) else 1.0   # extra captain multiplier (TC fixed)
        prob += pulp.lpSum(x[(i, g)] for i in ids) == 15
        for p_, n_ in ((1, 2), (2, 5), (3, 5), (4, 3)):
            prob += pulp.lpSum(x[(i, g)] for i in ids if pos[i] == p_) == n_
        for t_id in set(team.values()):
            prob += pulp.lpSum(x[(i, g)] for i in ids if team[i] == t_id) <= 3
        for i in ids:
            prob += t_in[(i, g)] >= x[(i, g)] - prev_x(i, g)
        prob += pulp.lpSum(l[(i, g)] for i in ids) == 11
        prob += pulp.lpSum(l[(i, g)] for i in gk_ids) == 1
        prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == 2) >= 3
        prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == 3) >= 2
        prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == 4) >= 1
        if g in ffm:
            for p_, n_ in zip((2, 3, 4), ffm[g]):
                prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == p_) == n_
        prob += pulp.lpSum(c[(i, g)] for i in ids) == 1
        if g in fcap:
            prob += c[(fcap[g], g)] == 1
        for i in fsq.get(g, ()):
            prob += x[(i, g)] == 1                            # 15 forced players fix the squad
        for i in ids:
            prob += l[(i, g)] <= x[(i, g)]
            prob += c[(i, g)] <= l[(i, g)]
        scorer = x if bb else l                               # BB: all 15 score
        obj.append(w * pulp.lpSum(xp[(i, g)] * scorer[(i, g)] for i in ids))
        obj.append(w * mult * pulp.lpSum(xp[(i, g)] * c[(i, g)] for i in ids))
        # Chip chosen by the model in g. Each extra term is a product of two binaries, written
        # with a continuous variable bounded by both:
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
        m_big = max([xp[(i, g)] for i in ids] + [0.0])        # >= any xP in this GW

        # Vice-captain: bonus = xP_C + (1 - P_C) * xP_VC, under independence.
        # q = 1 - P_C = 1 - sum_i P_i * c_i is linear (exactly one captain), so the only
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
                # With Triple Captain the vice-captain term is also tripled: sigma = s_vc * u_TC.
                # With s_vc in [0, 1] and u_TC binary, min(s_vc, u_TC) = s_vc * u_TC.
                sig = {i: prob.add_variable(f"tcv_{i}_{g}", lowBound=0, upBound=1) for i in ids if xp[(i, g)] > 0}
                for i, s_ in sig.items():
                    prob += s_ <= s_vc[i]
                    prob += s_ <= u_tc
                obj.append(w * pulp.lpSum(xp[(i, g)] * s_ for i, s_ in sig.items()))
            pool.setdefault('_vice', {})[g] = v

        # Bench value through autosubs (none with Bench Boost, where the bench scores anyway).
        if use_bench and not bb and m_big > 0:
            # Goalkeeper: exact under independence, same construction as the vice-captain,
            # with "captain" = starting GK and "vice" = bench GK (bench indicator x - l):
            #   value = (1 - P_startGK) * xP_benchGK,  s_i = (x_i - l_i) * (1 - P_startGK).
            s_gk = {i: prob.add_variable(f"sgk_{i}_{g}", lowBound=0) for i in gk_ids}
            prob += pulp.lpSum(s_gk.values()) <= 1 - pulp.lpSum(papp[(i, g)] * l[(i, g)] for i in gk_ids)
            if is_var(u_bb):
                prob += pulp.lpSum(s_gk.values()) <= 1 - u_bb    # no autosub value in a BB GW
            for i in gk_ids:
                prob += s_gk[i] <= x[(i, g)] - l[(i, g)]
            obj.append(w * pulp.lpSum(xp[(i, g)] * s_gk[i] for i in gk_ids))
            # Outfield: bench slots 1-3 with weights beta. The assignment variables can be
            # continuous: the assignment constraints are totally unimodular, so the optimum is
            # integral.
            if beta and g in beta:
                b = prob.add_variable_dicts(f"b_{g}", [(i, k) for i in out_ids for k in range(3)],
                                            lowBound=0, upBound=1)
                for i in out_ids:
                    prob += pulp.lpSum(b[(i, k)] for k in range(3)) <= x[(i, g)] - l[(i, g)]
                for k in range(3):
                    prob += pulp.lpSum(b[(i, k)] for i in out_ids) <= 1 - u_bb   # 0 in a BB GW
                obj.append(w * pulp.lpSum(beta[g][k] * xp[(i, g)] * b[(i, k)]
                                          for i in out_ids for k in range(3)))
                pool.setdefault('_bench', {})[g] = b

    # An unavailable player may be kept but not started in the first GW. The user's forced
    # captain and forced squad are exempt: FPL lets them start (with xP 0), and the constraint
    # would leave the decision with no plan at all.
    for i in ids:
        if not pool['avail'][i] and fcap.get(t0) != i and i not in fsq.get(t0, ()):
            prob += l[(i, t0)] == 0

    # Budget in every GW. [assumption] Prices are constant over the horizon.
    # k[i,g] = still holds the original copy of owned player i (sells for sell[i]); a re-bought
    # copy costs price[i]. Only needed when sell < price.
    gain_ids = [i for i in S0 if i in pool['price'] and sell[i] < price[i] - 1e-9]
    k_var = prob.add_variable_dicts("k", [(i, g) for i in gain_ids for g in horizon], lowBound=0, upBound=1)
    # kap[(i, g)] = still owns the original copy permanently after GW g. A Free Hit squad may
    # leave i out for one GW without selling the original (same rule as y above):
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

    # Free transfers and hits.
    T = {g: pulp.lpSum(t_in[(i, g)] for i in ids) for g in horizon}
    F = {t0: ft0}
    h = {}
    obj.append(-TRANSFER_TIEBREAK * pulp.lpSum(T.values()))
    if chip_vars:
        obj.append(-CHIP_TIEBREAK * pulp.lpSum(chip_vars))
    for g in horizon:
        if g > t0:
            F[g] = prob.add_variable(f"ft_{g}", lowBound=1, upBound=MAX_FT, cat='Integer')
            ph = phi[g - 1]
            if g - 1 == t0 and ft0 is None:
                prob += F[g] <= 1                       # 1 FT after the first deadline
            elif not is_var(ph) and ph == 1:
                prob += F[g] <= F[g - 1]                # chip fixed: FTs kept, none added
            elif is_var(ph):
                # WC/FH decided by the model. phi = 1 switches (2) off and (3) gives
                # F[g] <= F[g-1]; with phi = 0, (2) is the plain rule and implies (3).
                prob += F[g] <= F[g - 1] - T[g - 1] + h[g - 1] + 1 + CHIP_TRANSFER_M * ph   # (2)
                prob += F[g] <= F[g - 1] + 1 - ph                                          # (3)
            else:
                prob += F[g] <= F[g - 1] - T[g - 1] + h[g - 1] + 1
        if (g == t0 and free_now) or (not is_var(phi[g]) and phi[g] == 1):
            h[g] = 0                                    # WC / FH fixed / before the first deadline
            continue
        h[g] = prob.add_variable(f"hits_{g}", lowBound=0, upBound=max_hits, cat='Integer')
        prob += h[g] >= T[g] - F[g] - CHIP_TRANSFER_M * phi[g]   # (1)
        obj.append(-weight[g] * HIT_COST * h[g])        # weighted like that GW's points

    # Free transfers left for the GW after the horizon, worth FT_END_VALUE weighted points
    # each. Without it a banked transfer is worth 0 at the end of the horizon, so the plan uses
    # every free transfer even for a tiny gain. Same rule as F[g] above, one GW further.
    if FT_END_VALUE:
        g = horizon[-1]
        fe = prob.add_variable("ft_end", lowBound=0, upBound=MAX_FT)   # integer at the optimum
        ph = phi[g]
        if g == t0 and ft0 is None:
            prob += fe <= 1                             # 1 FT after the first deadline
        elif not is_var(ph) and ph == 1:
            prob += fe <= F[g]                          # chip fixed: FTs kept, none added
        elif is_var(ph):
            prob += fe <= F[g] - T[g] + h[g] + 1 + CHIP_TRANSFER_M * ph   # (2), as above
            prob += fe <= F[g] + 1 - ph                                  # (3)
        else:
            prob += fe <= F[g] - T[g] + h[g] + 1
        obj.append(FT_END_VALUE * fe)

    prob += pulp.lpSum(obj)
    # The "keep the squad" start is feasible only if every forced captain is owned, every
    # forced squad equals the kept one, and no formation is forced (the start XI follows the
    # formation minimums). Otherwise the solver searches without a start.
    use_start = bool(S0) and all(i in S0 for i in fcap.values())
    use_start = use_start and all(sq == S0 for sq in fsq.values())
    use_start = use_start and not ffm
    if warm:
        for var in prob.variables():
            if var.name in warm and warm[var.name] is not None:
                val = warm[var.name]
                # Solvers return values such as -1e-12 for a variable with lower bound 0, and
                # PuLP rejects an initial value outside the bounds.
                if var.lowBound is not None:
                    val = max(val, var.lowBound)
                if var.upBound is not None:
                    val = min(val, var.upBound)
                var.setInitialValue(round(val) if var.cat == 'Integer' else val)
    elif use_start:
        # Starting solution: keep the current squad in every GW (always feasible: no
        # transfers, and it costs exactly the budget minus the bank), the best XI by xP, and
        # the two highest xP of that XI as captain and vice-captain. Without it, most of the
        # solve time goes to finding a first feasible plan.
        ft_run = ft0 if ft0 is not None else 1
        for g in horizon:
            xp_g = {i: (xp[(i, g)] if (g > t0 or pool['avail'][i]) else -1.0) for i in start_squad}
            if g in fcap:
                # Ranked first, so he is in the start XI and its captain; use_start guarantees
                # that he is owned.
                xp_g[fcap[g]] = float('inf')
            xi = reference_xi(start_squad, pos, xp_g)
            ranked = sorted(xi, key=lambda i: -xp_g[i])
            for i in ids:
                x[(i, g)].setInitialValue(1 if i in S0 else 0)
                l[(i, g)].setInitialValue(1 if i in xi else 0)
                c[(i, g)].setInitialValue(1 if i == ranked[0] else 0)
                if '_vice' in pool and g in pool['_vice']:
                    pool['_vice'][g][i].setInitialValue(1 if i == ranked[1] else 0)
            for i in gain_ids:
                k_var[(i, g)].setInitialValue(1)     # original copies kept, so the budget holds
                if isinstance(kap[(i, g)], pulp.LpVariable) and kap[(i, g)] is not k_var[(i, g)]:
                    kap[(i, g)].setInitialValue(1)
            for i in ids:
                if isinstance(y[(i, g)], pulp.LpVariable) and y[(i, g)] is not x[(i, g)]:
                    y[(i, g)].setInitialValue(1 if i in S0 else 0)   # permanent squad = kept squad
            for cp in CHIPS:
                if is_var(u[(cp, g)]):
                    u[(cp, g)].setInitialValue(0)    # start from "no chip played"
            if g > t0:
                F[g].setInitialValue(ft_run)
            if isinstance(h[g], pulp.LpVariable):
                h[g].setInitialValue(0)
            chip_fixed_free = not is_var(phi[g]) and phi[g] == 1   # WC/FH fixed in g: FTs kept
            if not ((g == t0 and ft0 is None) or chip_fixed_free):
                ft_run = min(MAX_FT, ft_run + 1)     # no transfer: one more FT, up to the cap
            elif ft0 is None and g == t0:
                ft_run = 1
    out = run_solver(prob, solver, time_limit or SOLVER_TIME_LIMIT, bool(warm) or use_start)
    # 'status' is 'Optimal' whenever a plan exists, also when the time limit stopped the proof;
    # 'proven' tells the two apart.
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

    # Read the plan. FTs and hits are recomputed with the exact FPL rule.
    plan, prev, ft = {}, list(start_squad), ft0
    for g in horizon:
        sq = [i for i in ids if is_chosen(x[(i, g)])]
        # Chip played in g, fixed or chosen by the model; None = no chip.
        played = next((cp for cp in CHIPS if (is_chosen(u[(cp, g)]) if is_var(u[(cp, g)]) else u[(cp, g)] == 1)),
                      None)
        base = prev                                     # permanent squad before g
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
        prev = base if played == 'FH' else sq           # a Free Hit squad reverts
        ft = ft_next
    res['plan'] = plan
    pool.pop('_vice', None)
    pool.pop('_bench', None)
    return res


def plan_squad(pool, horizon, weight, start_squad, budget, ft0, chip='NONE', max_hits=None,
               use_vice=True, use_bench=True, time_limit=None, solver=None, chips_avail=None, chip_fix=None,
               force_captain=None, force_squad=None, force_formation=None):
    """Solve to a fixed point of the outfield bench weights.

    Pass 1 uses the best XI of the current squad (or of the pool at GW1) as reference; each
    later pass uses the XI the solver chose, until the weights change by at most
    BENCH_BETA_TOL. Each pass is warm-started from the previous plan.

    Returns the result of the last pass that produced a plan (or the failed first one), with
    'complete' = every pass reached the gap target and the passes stopped by the rule above, not
    by the time limit. A complete result does not depend on the speed of the machine: a longer
    time limit gives the same plan, up to last-digit floating-point differences."""
    pos, xp, papp = pool['pos'], pool['xp'], pool['papp']
    ref = start_squad if start_squad else pool['ids']
    beta = {g: bench_weights(reference_xi(ref, pos, {i: xp[(i, g)] for i in ref}), pos,
                             {i: papp[(i, g)] for i in ref}) for g in horizon} if use_bench else None
    best, seconds, history = None, 0.0, []
    complete = True                                 # False as soon as time cuts anything
    for n_pass in range(1, BENCH_MAX_PASSES + 1):
        t_start = time.time()
        # The time limit is for the whole run; each pass gets what is left.
        left = (time_limit or SOLVER_TIME_LIMIT) - seconds
        if best is not None and left < MIN_PASS_SECONDS:
            complete = False                        # a pass the rule wanted was not done
            break                                   # no time for another pass: keep the last plan
        res = solve_plan(pool, horizon, weight, start_squad, budget, ft0, chip, max_hits,
                         beta, use_vice, use_bench, warm=best['values'] if best else None,
                         time_limit=max(left, MIN_PASS_SECONDS), solver=solver,
                         chips_avail=chips_avail, chip_fix=chip_fix,
                         force_captain=force_captain, force_squad=force_squad,
                         force_formation=force_formation)
        seconds += time.time() - t_start
        history.append(beta)
        if 'plan' not in res:
            if best is None:                        # nothing found at all
                res['seconds'], res['passes'], res['complete'] = seconds, n_pass, False
                return res
            complete = False
            break                                   # keep the previous pass
        complete = complete and res['proven']       # a pass cut by time also changes later warm starts
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
    best['seconds'], best['beta_history'], best['complete'] = seconds, history, complete
    return best
