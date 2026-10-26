"""Decision check: a decision of the user held fixed in the multi-period MILP and compared with
    Z0    the best plan without that action (the main plan of the engine),
    Z+    the best plan with the same action, everything else chosen by the model,
    Z(D)  the best plan with the action exactly as the user chose it,
    G = Z(D) - Z0   gain of the decision over not taking the action (may be negative),
    R = Z+ - Z(D)   points left against the best version of the action (>= 0: forcing a choice
                    only removes plans, so it can never find a better one),
    G + R = Z+ - Z0 = gain of the action itself.
Every Z is the objective J of the plan: weighted expected points over the horizon, hits
subtracted, tie-breaks left out (plan_value). The transfers after the decision stay free: the
horizon measures the decision, it is not a plan the user has to follow.
Supported so far: Triple Captain on a player of the user's choice, and Bench Boost, in any
gameweek of the horizon. A captain who is not owned is bought by the decision plan (O1); the
transfer and its cost are part of Z(D).
Timing (step 3): is GW g the best GW of the chip's window? The one-GW estimates of the chip
analysis (chips.estimate_gw) for every GW from now to the end of the window, on the squads of the
main plan; GW g is best if its estimate is the maximum (the rule C10 of the chip suggestions).
They measure the chip itself (best captain), not the user's captain: R does that.
Wildcard, Free Hit and transfers come in later steps."""
import time

from .chips import estimate_gw, gw_points
from .milp import MIN_PASS_SECONDS, SOLVER_TIME_LIMIT, plan_squad

CHECK_CHIPS = ('TC', 'BB')   # decisions this module can check so far


def plan_value(res, pool, horizon, weight, use_bench=True):
    """J of a solved plan: sum over the horizon of weight[g] x the expected points of GW g
    (chips.gw_points: XI, captain, vice-captain, autosubs, minus hits), with the bench weights
    the solve used. Equals the solver's objective plus the tie-breaks it subtracts."""
    beta = res.get('beta') or {}
    return sum(weight[g] * gw_points(res['plan'][g], g, pool, beta.get(g), use_bench)['total'] for g in horizon)


def diff_bounds(est, gap_first, gap_second):
    """Interval of a difference A - B of two solved values. Each solve proves its value within its
    gap (true value in [solved, solved + gap]), so A - B lies in [est - gap_B, est + gap_A].
    None when a gap is unknown. [assumption] the tie-breaks (at most 0.001 per transfer or chip)
    are left out of these bounds: the gaps are on the solver's objective, which includes them."""
    if gap_first is None or gap_second is None:
        return None, None
    return est - gap_second, est + gap_first


def captain_bought(plan, horizon, captain, gw):
    """O1: last GW up to gw in which the plan buys the user's captain; None if he was owned."""
    bought = None
    for g in horizon:
        if g > gw:
            break
        if captain in plan[g]['in']:
            bought = g
    return bought


def chip_timing(pool, plan, horizon, chip, gw, window_end, budget, max_hits, use_vice, use_bench, solver,
                deadline):
    """Step 3: estimates of the chip for every GW from now (horizon[0]) to window_end, on the squad
    the main plan holds in that GW (after the horizon: its last squad, [assumption] as in
    chips.chip_report). Returns (timing, missing): timing = {'window_end', 'estimates' {gw: Delta},
    'at_gw' (estimate in the decision's GW), 'best_gw', 'best', 'is_best'}; C12: best_gw, best and
    is_best are None unless every GW of the window was estimated (missing lists what was not)."""
    t0, last = horizon[0], horizon[-1]
    est, missing = {}, []
    for t in range(t0, window_end + 1):
        if time.time() > deadline:
            missing.append(f"estimates from GW{t}")
            break
        v = estimate_gw(pool, plan[min(t, last)]['squad'], t, t0, [chip], budget, max_hits, use_vice, use_bench,
                        solver)[chip]
        if v is None:
            missing.append(f"estimate {chip} GW{t}")
        else:
            est[t] = v
    timing = {'window_end': window_end, 'estimates': est, 'at_gw': est.get(gw),
              'best_gw': None, 'best': None, 'is_best': None}
    if not missing:
        best_gw = max(est, key=est.get)                       # ties: the earliest GW
        timing['best_gw'], timing['best'] = best_gw, est[best_gw]
        # the 1e-9 lets an estimate equal to the maximum count as the maximum (as in chips.advise)
        # bool(): the engine's xP are numpy floats, and comparing them gives numpy.bool_, which is
        # not a Python bool (isinstance(numpy.False_, bool) is False)
        timing['is_best'] = bool(est[gw] >= est[best_gw] - 1e-9)
    return timing, missing


def check_decision(pool, horizon, weight, start_squad, budget, ft0, base, chip, gw, captain=None,
                   max_hits=None, use_vice=True, use_bench=True, time_limit=None, solver=None,
                   window_end=None):
    """Checks 'chip in GW gw' (with the user's captain for TC) against the main plan 'base'.
    base: result of milp.plan_squad with chip 'NONE' on the same pool, horizon, weights and options
          (Z0; it is not solved again here).
    Solves: Z+ (chip fixed in gw) and, for TC, Z(D) (chip and captain fixed). For BB there is no
    choice left to the user, so Z(D) = Z+ and R = 0. The time limit is shared by these solves.
    window_end: last GW of the chip's window that contains gw; given -> timing (step 3) with the
          time left after the solves. None -> no timing.
    C12: a number is given only if every solve behind it was complete; otherwise it is None and
    the missing solve is listed in 'not_computed'."""
    if chip not in CHECK_CHIPS:
        raise ValueError(f"decision check: chip {chip} is not supported yet ({', '.join(CHECK_CHIPS)})")
    if gw not in horizon:
        raise ValueError(f"decision check: GW{gw} is outside the horizon GW{horizon[0]}-GW{horizon[-1]}")
    if (chip == 'TC') != (captain is not None):
        raise ValueError("decision check: a captain is given with TC, and only with TC")
    if window_end is not None and window_end < gw:
        raise ValueError(f"decision check: GW{gw} is after the end of the chip's window (GW{window_end})")

    t_start = time.time()
    deadline = t_start + (time_limit or SOLVER_TIME_LIMIT)
    out = {'chip': chip, 'gw': gw, 'captain': captain, 'status': 'computed',
           'z0': None, 'z_best': None, 'z_decision': None,
           'gaps': {'z0': base.get('gap'), 'z_best': None, 'z_decision': None},
           'g': None, 'g_low': None, 'g_high': None, 'r': None, 'r_low': None, 'r_high': None,
           'gain_action': None, 'points_gw': {}, 'plan_best': None, 'plan_decision': None,
           'captain_bought': None, 'timing': None, 'not_computed': [], 'seconds': None}
    solved = {}
    if 'plan' in base and base.get('complete'):
        solved['z0'] = base
    else:
        out['not_computed'].append('Z0 (main plan)')

    # Z+ first, then the user's version (TC only); each solve starts from the kept squad as usual
    runs =[('z_best', None)] + ([('z_decision', {gw: captain})] if chip == 'TC' else [])
    for k, (name, fcap) in enumerate(runs):
        left = deadline - time.time()
        if left < MIN_PASS_SECONDS:
            out['not_computed'].append(name)
            continue
        res = plan_squad(pool, horizon, weight, start_squad, budget, ft0, 'NONE', max_hits, use_vice, use_bench,
                         time_limit=left / (len(runs) - k),          # equal share of what is left
                         solver=solver, chip_fix={(chip, gw): 1}, force_captain=fcap)
        if 'plan' not in res:
            out['not_computed'].append(f"{name} (no plan)")
        elif not res['complete']:                                    # C12: cut by the time limit
            out['not_computed'].append(name)
        else:
            solved[name] = res
    if chip == 'BB' and 'z_best' in solved:
        solved['z_decision'] = solved['z_best']                      # nothing left to choose

    for name, res in solved.items():
        out[name] = plan_value(res, pool, horizon, weight, use_bench)
        out['gaps'][name] = res.get('gap')
        out['points_gw'][name] = gw_points(res['plan'][gw], gw, pool, (res.get('beta') or {}).get(gw),
                                           use_bench)['total']       # D2: unweighted, GW of the decision
    gaps = out['gaps']
    if 'z0' in solved and 'z_decision' in solved:
        out['g'] = out['z_decision'] - out['z0']
        out['g_low'], out['g_high'] = diff_bounds(out['g'], gaps['z_decision'], gaps['z0'])
    if 'z_best' in solved and 'z_decision' in solved:
        out['r'] = out['z_best'] - out['z_decision']
        out['r_low'], out['r_high'] = diff_bounds(out['r'], gaps['z_best'], gaps['z_decision'])
    if 'z0' in solved and 'z_best' in solved:
        out['gain_action'] = out['z_best'] - out['z0']
    if 'z_best' in solved:
        out['plan_best'] = solved['z_best']['plan'][gw]
    if 'z_decision' in solved:
        out['plan_decision'] = solved['z_decision']['plan'][gw]
        if captain is not None:
            out['captain_bought'] = captain_bought(solved['z_decision']['plan'], horizon, captain, gw)
    if window_end is not None and 'plan' in base:
        out['timing'], missing = chip_timing(pool, base['plan'], horizon, chip, gw, window_end, budget, max_hits,
                                             use_vice, use_bench, solver, deadline)
        out['not_computed'] += missing
    if out['not_computed']:
        out['status'] = 'not_computed'
    out['seconds'] = time.time() - t_start
    return out


VERDICTS = {   # code -> meaning (the text output and docs/json-output.md use the same wording)
    'worse_than_none': "worse than not playing the chip: G is below 0 by more than the solver gaps",
    'no_clear_gain': "no gain beyond the solver gaps: G is within them around 0",
    'better_gw': "a later or earlier GW of the window has a higher estimate for the chip",
    'better_version': "the same chip with another choice (the model's captain) gains more: R is above 0 "
                      "by more than the solver gaps",
    'agrees': "the decision gains, the GW is the best of the window and no better version exists",
}


def verdict(res):
    """Section 0.4 of the spec: the verdict follows from the numbers, with no new constant (only the
    gap bounds and the timing rule C10). Returns a list of codes of VERDICTS, or [] when a number it
    needs was not computed (C12: no verdict from part of the numbers)."""
    if res.get('g_low') is None or res.get('g_high') is None:
        return []
    if res['g_high'] < 0:
        return ['worse_than_none']
    out = []
    if res['g_low'] <= 0:
        out.append('no_clear_gain')
    timing = res.get('timing')
    if timing is not None:
        if timing['is_best'] is None:
            return []                                  # timing incomplete -> no verdict
        if not timing['is_best']:
            out.append('better_gw')
    if res.get('r_low') is None:
        return []
    if res['r_low'] > 0:
        out.append('better_version')
    return out or ['agrees']
