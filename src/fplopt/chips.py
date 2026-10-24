"""Chip suggestions: the opportunity cost of each chip that can be
played now, and the rule that turns the numbers into a suggestion."""
import time

from .milp import MIN_PASS_SECONDS, plan_squad
from .rules import CHIPS, HIT_COST


def gw_points(pg, g, pool, beta_g, use_bench=True):
    """C8: expected points of GW g of a plan, UNWEIGHTED (no 0.85^k), split into parts:
    XI (all 15 with Bench Boost) + captain x m + vice-captain x m x (1 - P_C) + autosubs
    (not with Bench Boost) - hits, with m = 2 for Triple Captain, else 1. Same terms as the
    objective of solve_plan for that GW, without the weight and the tie-breaks."""
    xp, papp, pos = pool['xp'], pool['papp'], pool['pos']
    bb = pg['chip'] == 'BB'
    mult = 2.0 if pg['chip'] == 'TC' else 1.0
    out = {'xi': sum(xp[(i, g)] for i in (pg['squad'] if bb else pg['xi'])),
           'captain': mult * xp[(pg['captain'], g)], 'vice': 0.0, 'autosubs': 0.0,
           'hits': -HIT_COST * pg['hits']}
    if pg['vice'] is not None:
        out['vice'] = mult * (1.0 - papp[(pg['captain'], g)]) * xp[(pg['vice'], g)]
    if use_bench and not bb:
        gk_start = next(i for i in pg['xi'] if pos[i] == 1)
        out['autosubs'] = sum((1.0 - papp[(gk_start, g)]) * xp[(i, g)] for i in pg['bench'] if pos[i] == 1)
        outs = [i for i in pg['bench'] if pos[i] != 1]
        if beta_g:
            out['autosubs'] += sum(beta_g[k] * xp[(i, g)] for k, i in enumerate(outs))
    out['total'] = sum(out.values())
    return out


def plan_points(res, pool, horizon, use_bench=True):
    """C8: unweighted expected points of a whole plan (sum of gw_points over the horizon)."""
    beta = res.get('beta') or {}
    return sum(gw_points(res['plan'][g], g, pool, beta.get(g), use_bench)['total'] for g in horizon)


def squad_value(pool, squad, g, chip='NONE', use_vice=True, use_bench=True, solver=None):
    """C9: best one-GW value (unweighted) of a FIXED 15-man squad: plan_squad on a pool that
    holds only these 15 players, so the XI, captain, vice-captain and autosubs come from the
    same model as the plan. Every player may start: an unavailable one has xP 0 and P(app) 0,
    so the model benches him or counts him as autosubbed (as the game would)."""
    sub = {k: pool[k] for k in ('pos', 'team', 'price', 'xp', 'papp')}
    sub['ids'] = list(squad)
    sub['avail'] = {i: True for i in squad}
    sub['sell'] = {i: pool['price'][i] for i in squad}           # no transfers: prices do not matter
    budget = sum(sub['sell'].values()) + 1.0
    res = plan_squad(sub, [g], {g: 1.0}, list(squad), budget, 1, chip, 0, use_vice, use_bench,
                     time_limit=30, solver=solver)
    # C12: a value cut by the time limit would differ between machines: None (not computed) instead
    return plan_points(res, sub, [g], use_bench) if res.get('complete') else None


def chip_report(pool, horizon, weight, plan, start_squad, budget, ft0, chip_windows, max_hits,
                use_vice, use_bench, time_left, solver=None):
    """C10: opportunity cost of each chip that can be played now (design decision of the
    user: unweighted points).
      exact[c]    = Delta_c(t0): points of the best plan with chip c in t0 minus the best plan
                    without a chip, both solved with the full model over min(3, L) GWs and
                    scored UNWEIGHTED (plan_points).
      est[c][t]   = estimate of Delta_c(t) for t0 <= t <= end of the chip's window, on the
                    squad Q_t of the main plan (after the horizon: the squad of its last GW,
                    [assumption] no transfers after the horizon):
                    TC, BB: squad_value(Q_t, chip) - squad_value(Q_t);
                    FH: best one-GW Free Hit squad (whole pool) - squad_value(Q_t);
                    WC: not estimated after t0 ([assumption], no one-GW proxy exists).
    C12: a number is kept only if every solve behind it was complete (milp.plan_squad: gap target
    reached, bench-weight passes not cut by the time limit), so the numbers that are shown do not
    depend on the speed of the machine; a slower machine shows fewer of them ('skipped').
    Returns a dict with the numbers; the caller prints them."""
    t0 = horizon[0]
    deadline = time.time() + time_left
    out = {'exact': {}, 'gap': {}, 'est': {}, 'window': {}, 'skipped': [], 'plans': {}}
    now_chips = [cp for cp in CHIPS for s, e in chip_windows.get(cp, []) if s <= t0 <= e]
    for cp in now_chips:
        out['window'][cp] = next(e for s, e in chip_windows[cp] if s <= t0 <= e)
    if not now_chips:
        return out
    # --- exact Delta_c(t0): 1 + len(now_chips) solves with the full model ---
    h3 = horizon[:3]
    w3 = {g: weight[g] for g in h3}
    runs = [('NONE', 4.0)] + [(cp, 4.0 if cp == 'WC' else 2.0) for cp in now_chips]   # time shares
    base = None
    for k, (cp, share) in enumerate(runs):
        left = deadline - time.time()
        # this solve's share of the remaining time, in proportion to the measured solve times
        # (WC ~2x the others, measured), and at least MIN_PASS_SECONDS
        tl = max(MIN_PASS_SECONDS, left * share / sum(s for _, s in runs[k:]) * 0.75)
        if left < MIN_PASS_SECONDS or (cp != 'NONE' and base is None):
            out['skipped'].append(f"exact {cp}")
            continue
        res = plan_squad(pool, h3, w3, start_squad, budget, ft0, cp, max_hits, use_vice, use_bench,
                         time_limit=tl, solver=solver)
        if 'plan' not in res:
            out['skipped'].append(f"exact {cp} (no plan)")
            continue
        if not res['complete']:                     # C12: cut by the time limit
            out['skipped'].append(f"exact {cp}")
            continue
        pts = plan_points(res, pool, h3, use_bench)
        out['plans'][cp] = res['plan'][t0]          # W3: the GW-t0 decisions of this solve (backtest)
        if cp == 'NONE':
            base = (pts, res.get('gap'))
        else:
            out['exact'][cp] = pts - base[0]
            out['gap'][cp] = (res.get('gap') or 0.0) + (base[1] or 0.0)   # bound on the error of Delta
    # --- estimates Delta_c(t) up to the end of each window (one-GW solves) ---
    last = horizon[-1]
    for t in range(t0, max(out['window'].values()) + 1):
        if time.time() > deadline:
            out['skipped'].append(f"estimates from GW{t}")
            break
        q = plan[min(t, last)]['squad']
        chips_t = [cp for cp in now_chips if t <= out['window'][cp] and cp != 'WC']
        if not chips_t:
            continue
        v0 = squad_value(pool, q, t, 'NONE', use_vice, use_bench, solver)
        for cp in chips_t:
            if cp in ('TC', 'BB'):
                v = squad_value(pool, q, t, cp, use_vice, use_bench, solver)
            else:   # FH: whole pool, start from Q_t, sell = price ([assumption] constant prices)
                sub = dict(pool)
                sub['sell'] = {i: pool['sell'].get(i, pool['price'][i]) for i in q}
                sub['avail'] = {i: True for i in pool['ids']} if t > t0 else pool['avail']
                res = plan_squad(sub, [t], {t: 1.0}, list(q), budget, 1, 'FH', max_hits, use_vice, use_bench,
                                 time_limit=30, solver=solver)
                v = plan_points(res, sub, [t], use_bench) if res.get('complete') else None   # C12
            if v is not None and v0 is not None:
                out['est'].setdefault(cp, {})[t] = v - v0
            else:
                out['skipped'].append(f"estimate {cp} GW{t}")     # C12: not computed in time
    return out


def advise(rep, current_gw):
    """C10/C11: the suggestion rule applied to the numbers of chip_report.
    C12: the rule needs the exact Delta now and an estimate for EVERY GW up to the end of the
    window; a chip that misses any of them (time limit) gets no suggestion at all, neither 'play'
    nor 'hold', and is listed in 'incomplete' (a rule applied to part of the window could say
    'play' on one machine and 'hold' on another).
    A chip c is suggested now only if its estimate now is the maximum up to the end of its window
    AND its exact Delta now is positive; with several, the largest exact Delta (one chip per GW).
    WC has no estimates after now, so it is never suggested by the rule; C11 adds a note when
    the WC gains more now than the suggested chip.
    Returns {'now_ok': [...], 'pick': chip or None, 'also': [...],
             'holds': [(chip, estimate now or None, best GW, best estimate, window end), ...],
             'wc_note': (WC exact, pick exact) or None, 'incomplete': [chip, ...]}."""
    incomplete = [cp for cp in rep['window'] if cp != 'WC' and (
        cp not in rep['exact'] or
        any(t not in rep['est'].get(cp, {}) for t in range(current_gw, rep['window'][cp] + 1)))]
    est_all = {cp: est for cp, est in rep['est'].items() if cp not in incomplete}
    # the 1e-9 lets an estimate equal to the maximum count as the maximum (float rounding)
    now_ok = [cp for cp, est in est_all.items()
              if est.get(current_gw) is not None and est[current_gw] >= max(est.values()) - 1e-9
              and rep['exact'].get(cp, 0.0) > 0.0]
    holds = []
    for cp, est in est_all.items():
        if cp not in now_ok:
            t_best = max(est, key=est.get)
            holds.append((cp, est.get(current_gw), t_best, est[t_best], rep['window'][cp]))
    pick = max(now_ok, key=lambda cp: rep['exact'][cp]) if now_ok else None
    wc_note = None
    if pick is not None and pick != 'WC' and rep['exact'].get('WC', float('-inf')) > rep['exact'][pick]:
        wc_note = (rep['exact']['WC'], rep['exact'][pick])
    return {'now_ok': now_ok, 'pick': pick, 'also': [cp for cp in now_ok if cp != pick],
            'holds': holds, 'wc_note': wc_note, 'incomplete': incomplete}
