"""Offline checks for chat 6b (multi-period model). Uses the fake API of harness_offline.py.
Usage: python harness_6b.py <check>   (see CHECKS at the bottom). Not part of the optimizer."""
import sys, io, contextlib, time, os
import harness_offline as H
F = H.F

CAPTURE = {}
_orig_plan = F.plan_squad
_orig_solve = F.solve_plan
F.solve_plan = lambda *a, **kw: _orig_solve(*a, **{**kw, 'solver': kw.get('solver') or os.environ.get('SOLVER')})
def _spy(pool, horizon, weight, *a, **kw):           # keep the pool and result of the last run
    res = _orig_plan(pool, horizon, weight, *a, **kw)
    CAPTURE.update(pool=pool, res=res, horizon=horizon, weight=weight, args=a)
    return res
F.plan_squad = _spy


def run(ft=None, chip='NONE', L=3, max_hits=None, vice=True, bench=True, quiet=True, budget=None, tl=None,
        solver=None):
    """Full optimizer on the synthetic season (GW8, Free Hit in GW7 -> squad of GW6)."""
    H.MANAGER = H.build_manager()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        F.optimize_team(1, ft, chip, 75, L, budget, max_hits, None, vice, bench, time_limit=tl,
                        solver=solver or os.environ.get('SOLVER'))
    if not quiet:
        print(buf.getvalue())
    return buf.getvalue(), CAPTURE['res'], CAPTURE['pool']


# ---------- small synthetic pools for exact checks (no API) -------------------
def toy_pool(n_team=20, xp_base=None, extra=None):
    """3 GKs, 7 DEF, 7 MID, 5 FWD on separate teams, all price 5.0 and xP 2.0 per GW
    unless changed. extra: list of (id, pos, team, price, {g: xp}, {g: papp})."""
    pool = {'ids': [], 'pos': {}, 'team': {}, 'price': {}, 'sell': {}, 'xp': {}, 'papp': {}, 'avail': {}}
    pid = 0
    for p_, n_ in ((1, 3), (2, 7), (3, 7), (4, 5)):
        for k in range(n_):
            pid += 1
            pool['ids'].append(pid); pool['pos'][pid] = p_; pool['team'][pid] = pid
            pool['price'][pid] = 5.0; pool['avail'][pid] = True
    return pool


def fill(pool, horizon, xp=2.0, papp=1.0):
    for i in pool['ids']:
        for g in horizon:
            pool['xp'].setdefault((i, g), xp)
            pool['papp'].setdefault((i, g), papp)


def add(pool, i, pos, price, xp_by_g, papp_by_g=None, team=None):
    pool['ids'].append(i); pool['pos'][i] = pos; pool['team'][i] = team or i
    pool['price'][i] = price; pool['avail'][i] = True
    for g, v in xp_by_g.items():
        pool['xp'][(i, g)] = v
        pool['papp'][(i, g)] = (papp_by_g or {}).get(g, 1.0)


def base_squad(pool):
    by = {p: [i for i in pool['ids'] if pool['pos'][i] == p] for p in (1, 2, 3, 4)}
    sq = by[1][:2] + by[2][:5] + by[3][:5] + by[4][:3]
    for i in sq:
        pool['sell'][i] = pool['price'][i]
    for (i, g) in list(pool['xp']):              # fixed captain (FWD, 20 xP) so that the
        if i == sq[14]:                          # targets' gains are not doubled by captaincy
            pool['xp'][(i, g)] = 20.0
    return sq


def show(res, names=None):
    nm = (lambda i: names.get(i, i)) if names else (lambda i: i)
    for g, pg in res['plan'].items():
        print(f"   GW{g}: FT {pg['ft']} in {[nm(i) for i in pg['in']]} out {[nm(i) for i in pg['out']]} "
              f"hits {pg['hits']} C {nm(pg['captain'])} VC {nm(pg['vice']) if pg['vice'] else None}")


def check_wait():
    """notes_06 P3 + 'wait one GW for a free FT': two targets each +2 xP/GW over a
    3-GW horizon, 1 FT. Old model (lookahead fixed squad): 2 transfers now with a hit.
    New: 1 now, 1 next GW for free."""
    H_ = [8, 9, 10]
    w = {g: 0.85 ** (g - 8) for g in H_}
    pool = toy_pool(); fill(pool, H_)
    sq = base_squad(pool)
    add(pool, 101, 3, 5.0, {g: 4.0 for g in H_}); add(pool, 102, 3, 5.0, {g: 4.0 for g in H_})
    res = F.solve_plan(pool, H_, w, sq, 75.0, 1, use_vice=False, use_bench=False)
    print("[wait] 2 targets +2/GW, 1 FT, 3 GWs, no cap:"); show(res)
    ok = res['plan'][8]['hits'] == 0 and len(res['plan'][8]['in']) == 1 and len(res['plan'][9]['in']) == 1
    base = F.solve_plan(pool, H_, w, sq, 75.0, 0 or 1, use_vice=False, use_bench=False, max_hits=0)
    print(f"   -> waits for the GW9 free transfer: {ok} | gain over keeping the squad: "
          f"{res['objective'] - (sum(w[g] * (2.0 * 10 + 20 * 2) for g in H_)):.3f} "
          f"(hand: 2*2.5725 + 2*1.5725 = {2 * 2.5725 + 2 * 1.5725:.3f}, minus 0.002 tie-break)")
    # lookahead 1: the same situation can only see GW8 -> one transfer, no hit (gain 2 < 4)
    res1 = F.solve_plan(pool, [8], {8: 1.0}, sq, 75.0, 1, use_vice=False, use_bench=False)
    print(f"   lookahead 1: transfers {len(res1['plan'][8]['in'])}, hits {res1['plan'][8]['hits']}")
    # a hit is still taken when it pays: +4 per GW for one target, and a 2nd FT is useless later
    pool2 = toy_pool(); fill(pool2, H_); sq2 = base_squad(pool2)
    add(pool2, 101, 3, 5.0, {g: 7.0 for g in H_}); add(pool2, 102, 3, 5.0, {g: 7.0 for g in H_})
    res2 = F.solve_plan(pool2, H_, w, sq2, 75.0, 1, use_vice=False, use_bench=False)
    print("[hit] two targets +5/GW: hit now 5*2.5725 - 4 = 8.86 > wait 5*1.5725 = 7.86:"); show(res2)
    print(f"   -> takes the hit in GW8: {res2['plan'][8]['hits'] == 1}")


def check_save():
    """1 FT and no target that gains anything -> FT is kept (FT 2 in GW9)."""
    H_ = [8, 9, 10]; w = {g: 0.85 ** (g - 8) for g in H_}
    pool = toy_pool(); fill(pool, H_); sq = base_squad(pool)
    res = F.solve_plan(pool, H_, w, sq, 75.0, 1, use_vice=False, use_bench=False)
    print("[save] no improving target:"); show(res)
    print(f"   -> no transfer in any GW: {all(not pg['in'] for pg in res['plan'].values())}, "
          f"FT GW9 = {res['plan'][9]['ft']}, GW10 = {res['plan'][10]['ft']}")
    # FT cap: start with 5 FT and nothing to do -> stays 5 (no 6th)
    res5 = F.solve_plan(pool, H_, w, sq, 75.0, 5, use_vice=False, use_bench=False)
    print(f"   start 5 FT -> GW9 {res5['plan'][9]['ft']}, GW10 {res5['plan'][10]['ft']} (cap 5)")
    # target gains only in GW10: with 1 FT the model waits and uses 2+ FT later? (one transfer at GW10)
    pool2 = toy_pool(); fill(pool2, H_); sq2 = base_squad(pool2)
    add(pool2, 101, 3, 5.0, {8: 2.0, 9: 2.0, 10: 5.0})
    res2 = F.solve_plan(pool2, H_, w, sq2, 75.0, 1, use_vice=False, use_bench=False)
    print("[save] target good only in GW10:"); show(res2)


def check_chips():
    H_ = [8, 9, 10]; w = {g: 0.85 ** (g - 8) for g in H_}
    pool = toy_pool(); fill(pool, H_); sq = base_squad(pool)
    # three targets good in every GW, one target good only in GW8 (a 'free-hit' type pick)
    for k in (101, 102, 103):
        add(pool, k, 3, 5.0, {g: 4.0 for g in H_})
    add(pool, 104, 4, 5.0, {8: 9.0, 9: 0.0, 10: 0.0})
    res = F.solve_plan(pool, H_, w, sq, 75.0, 2, chip='WC', use_vice=False, use_bench=False)
    print("[WC] 2 FT:"); show(res)
    print(f"   -> GW8 hits 0: {res['plan'][8]['hits'] == 0}; FT GW9 = {res['plan'][9]['ft']} (kept 2, none added)")
    res = F.solve_plan(pool, H_, w, sq, 75.0, 2, chip='FH', use_vice=False, use_bench=False)
    print("[FH] 2 FT:"); show(res)
    gw9_base = set(sq)
    print(f"   -> GW9 transfers counted from the original squad: out of GW9 all in original squad "
          f"{set(res['plan'][9]['out']) <= gw9_base}; FT GW9 = {res['plan'][9]['ft']}")


def check_gw1():
    """Before the first deadline: no squad, unlimited free transfers, then 1 FT."""
    H_ = [1, 2, 3]; w = {g: 0.85 ** (g - 1) for g in H_}
    pool = toy_pool(); fill(pool, H_)
    add(pool, 101, 3, 5.0, {1: 2.0, 2: 6.0, 3: 6.0})
    add(pool, 102, 3, 5.0, {1: 2.0, 2: 6.0, 3: 6.0})
    res = F.solve_plan(pool, H_, w, [], 75.0, None, use_vice=True, use_bench=True,
                       beta={g: [0.5, 0.15, 0.03] for g in H_})
    print("[GW1] empty squad, unlimited now:", res['status']); show(res)
    print(f"   -> GW1 hits 0 with 15 'transfers': {res['plan'][1]['hits'] == 0}, FT GW2 = {res['plan'][2]['ft']}")


def check_budget():
    """Owned player A: bought 7.0, now 7.3 -> sells 7.1. Selling and re-buying costs 7.3."""
    H_ = [8, 9, 10]; w = {g: 0.85 ** (g - 8) for g in H_}
    pool = toy_pool(); fill(pool, H_); sq = base_squad(pool)
    a = sq[5]                                    # first MID of the squad
    pool['price'][a] = 7.3; pool['sell'][a] = 7.1
    for g in H_:
        pool['xp'][(a, g)] = 6.0 if g != 9 else 0.0   # blanks in GW9
    budget = sum(pool['sell'][i] for i in sq) + 0.1   # bank 0.1
    add(pool, 201, 3, 7.2, {8: 0.0, 9: 6.0, 10: 0.0})  # strong only in GW9
    res = F.solve_plan(pool, H_, w, sq, budget, 2, use_vice=False, use_bench=False)
    print("[budget] owned A (sell 7.1, buy back 7.3), bank 0.1:"); show(res)
    for g, pg in res['plan'].items():
        spend = sum(pool['sell'][i] if (i in sq and all(i in res['plan'][h]['squad'] for h in H_ if h <= g))
                    else pool['price'][i] for i in pg['squad'])
        print(f"   GW{g}: spend {spend:.1f} <= budget {budget:.1f}: {spend <= budget + 1e-9}")
    print(f"   A sold in GW9 and NOT bought back in GW10 (needs 7.3, has 7.2): "
          f"{a in res['plan'][9]['out'] and a not in res['plan'][10]['squad']}")
    pool['price'][a] = 7.1                       # what a model WITHOUT k would assume
    res = F.solve_plan(pool, H_, w, sq, budget, 2, use_vice=False, use_bench=False)
    print("   same case if re-buying cost the selling price (the error k prevents):"); show(res)


def check_vice():
    """Captain choice with a 75% player (Haaland GW8 case, notes_08b §11.7)."""
    H_ = [8]
    pool = toy_pool(); fill(pool, H_); sq = base_squad(pool)
    pool['xp'][(sq[14], 8)] = 2.0                # no fixed 20-xP captain in this check
    hal, bru = sq[12], sq[8]                     # a FWD and a MID of the squad
    pool['xp'][(hal, 8)], pool['papp'][(hal, 8)] = 5.62, 0.75
    pool['xp'][(bru, 8)], pool['papp'][(bru, 8)] = 5.79, 0.97
    names = {hal: 'Haaland', bru: 'Bruno'}
    for vice in (False, True):
        res = F.solve_plan(pool, H_, {8: 1.0}, sq, 75.0, 1, use_vice=vice, use_bench=False)
        pg = res['plan'][8]
        print(f"[vice={vice}] C {names.get(pg['captain'])}, VC {names.get(pg['vice'], pg['vice'])}, "
              f"objective {res['objective']:.3f}")
    print(f"   expected with VC: 11 starters*... captain bonus Haaland C = 5.62 + 0.25*5.79 = {5.62 + .25*5.79:.3f}; "
          f"Bruno C = 5.79 + 0.03*5.62 = {5.79 + .03*5.62:.3f}")


def check_equiv():
    """Lookahead 1, no vice / bench, --max-hits 1: same plan as the old optimizer."""
    import importlib, fpl_v2_old as OLD
    for ft, chip in ((1, 'NONE'), (2, 'NONE'), (1, 'WC'), (1, 'FH'), (1, 'BB'), (1, 'TC')):
        H.MANAGER = H.build_manager()
        cap = {}
        orig_log_old = OLD.log_predictions
        OLD.fetch_fpl_api = H.fake
        OLD.log_predictions = lambda gw, pl, fs: cap.update(old={p['id']: p['role'] for p in fs})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            OLD.optimize_team(1, ft, chip, 75, 1, None, 1, None)
        old_obj = float(buf.getvalue().split("Objective (weighted xP over horizon - hit cost): ")[1].split()[0])
        out, res, pool = run(ft=ft, chip=chip, L=1, max_hits=1, vice=False, bench=False)
        new = res['plan'][8]
        new_roles = {i: ('Captain' if i == new['captain'] else 'Start' if i in new['xi'] else 'Bench')
                     for i in new['squad']}
        xp_obj = res['objective'] + F.TRANSFER_TIEBREAK * sum(len(pg['in']) for pg in res['plan'].values())
        print(f"[equiv ft={ft} {chip}] old objective {old_obj:.2f} | new without tie-break {xp_obj:.2f} | "
              f"same squad {set(cap['old']) == set(new['squad'])} | same roles {cap['old'] == new_roles}")


def check_timing():
    """python harness_6b.py timing <time limit per solve> <lookahead> ..."""
    tl = int(sys.argv[2])   # SOLVER=cbc|highs in the environment selects the solver
    for L in (int(a) for a in sys.argv[3:]):
        t = time.time()
        out, res, pool = run(L=L, tl=tl)
        print(f"[timing] {res['solver']}, players {len(pool['ids'])}, lookahead {L}, limit {tl}s: {res['seconds']:.1f} s over "
              f"{res['passes']} pass(es), total run {time.time() - t:.1f} s, proven {res['proven']}, "
              f"gap {res['gap']}, objective {res['objective']:.3f}", flush=True)
        print("   " + [ln for ln in out.split('\n') if ln.startswith('Expected points')][0], flush=True)


def check_full():
    out, res, pool = run(quiet=False)
    print("beta history:", [{g: [round(b, 3) for b in v] for g, v in bh.items()} for bh in res['beta_history']])


def check_ft():
    """M8: free transfers rebuilt from entry/{id}/history/."""
    f = F.free_transfers_from_history
    def hist(first, rows, chips=()):
        return {'current': [{'event': g, 'event_transfers': rows.get(g, (0, 0))[0],
                             'event_transfers_cost': rows.get(g, (0, 0))[1]} for g in range(first, 8)],
                'chips': [{'name': n, 'event': g} for n, g in chips]}
    # harness manager: start GW1; 1 transfer GW3; WC GW5 (2); 1 transfer GW6; FH GW7 (3)
    h = hist(1, {3: (1, 0), 5: (2, 0), 6: (1, 0), 7: (3, 0)}, [('wildcard', 5), ('freehit', 7)])
    print("[ft] harness manager (hand: F2=1 F3=2 F4=2 F5=3 F6=3(WC) F7=3 F8=3(FH)):", f(h, 8))
    # user's own team (notes_08b §11.3): first GW3, no chips, 3 transfers between GW4 and GW7,
    # each allocation allowed by the bank history
    for rows in ({4: (1, 0), 5: (1, 0), 7: (1, 0)}, {5: (2, 0), 7: (1, 0)}, {5: (2, 0), 6: (1, 0)}):
        print(f"[ft] first GW3, transfers {rows}: {f(hist(3, rows), 8)}")
    print("[ft] cap: no transfers GW2-7 from GW1:", f(hist(1, {}), 8), "(5 max)")
    print("[ft] hit taken: GW3 2 transfers with 2 FT cost 0, GW4 2 with 1 FT cost 4:",
          f(hist(1, {3: (2, 0), 4: (2, 4)}), 8))
    print("[ft] inconsistent cost (extra FTs the rule does not know):", f(hist(1, {3: (3, 0)}), 8))
    print("[ft] team that has not passed its first deadline:", f(hist(8, {}), 8) if False else f({'current': [], 'chips': []}, 8))


CHECKS = {'ft': check_ft, 'wait': check_wait, 'save': check_save, 'chips': check_chips, 'budget': check_budget,
          'vice': check_vice, 'gw1': check_gw1, 'equiv': check_equiv, 'timing': check_timing, 'full': check_full}
if __name__ == '__main__':
    CHECKS[sys.argv[1]]()
