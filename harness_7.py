"""Offline checks for chat 7 (chip variables in the MILP). Uses the toy pools of harness_6b.py
and the fake API of harness_offline.py. Usage: python harness_7.py <check> [args]
Not part of the optimizer."""
import sys, time, os, io, contextlib
import harness_6b as H6
F = H6.F
H = H6.H

ALL = {c: [(1, 38)] for c in F.CHIPS}       # every chip available in every GW (one use)


def show(res):
    for g, pg in res['plan'].items():
        print(f"   GW{g}: chip {pg['chip']} | FT {pg['ft']} in {pg['in']} out {pg['out']} hits {pg['hits']} "
              f"C {pg['captain']} VC {pg['vice']}")


def W(hz):
    return {g: 0.85 ** (g - hz[0]) for g in hz}


def check_equiv():
    """The VARIABLE formulation (C2-C5) must give max(Z(chip fixed in g), Z(no chip)) when the
    chip is allowed only in GW g: fixed chips use the old (M3) code paths, so this compares
    the two formulations. Toy pool (exact) and harness pool (vice + bench terms on)."""
    hz = [8, 9, 10]
    pool = H6.toy_pool(); H6.fill(pool, hz); sq = H6.base_squad(pool)
    for k in (101, 102, 103):
        H6.add(pool, k, 3, 5.0, {g: 4.0 for g in hz})
    H6.add(pool, 104, 4, 5.0, {8: 9.0, 9: 0.0, 10: 0.0})
    H6.add(pool, 105, 1, 4.0, {8: 1.0, 9: 6.0, 10: 1.0})          # bench GK strong in GW9 (BB)
    ok = True
    for c in F.CHIPS:
        for g in hz:
            for vb in (False, True):
                kw = dict(use_vice=vb, use_bench=vb, beta={h: [0.5, 0.15, 0.03] for h in hz} if vb else None)
                z_none = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, **kw)['objective']
                z_fix = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip_fix={(c, g): 1}, **kw)['objective']
                res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO', chips_avail={c: [(g, g)]}, **kw)
                exp = max(z_none, z_fix - F.CHIP_TIEBREAK)   # a played chip pays the 0.001 tie-break
                good = abs(res['objective'] - exp) < 1e-6
                ok &= good
                print(f"[equiv toy {c} GW{g} vice/bench={vb}] variable {res['objective']:.4f} | "
                      f"max(fixed {z_fix:.4f}, none {z_none:.4f}) = {exp:.4f} | {'OK' if good else 'DIFF'}")
    print("ALL OK" if ok else "SOME DIFFER")


def check_equiv_big():
    """Same comparison on the synthetic season (GW8, lookahead 3, vice + bench, real bench weights)."""
    orig = F.chip_report                       # skip the chip analysis: CAPTURE must hold the main plan
    F.chip_report = lambda *a, **k: {'exact': {}, 'gap': {}, 'est': {}, 'window': {}, 'skipped': []}
    out, res0, pool = H6.run(L=3)
    F.chip_report = orig
    hz, w, args = H6.CAPTURE['horizon'], H6.CAPTURE['weight'], H6.CAPTURE['args']
    start, budget, ft0 = args[0], args[1], args[2]
    beta = res0['beta']
    ok = True
    for c in F.CHIPS:
        for g in hz:
            if c in ('WC', 'FH') and g == 1:
                continue
            z_none = F.solve_plan(pool, hz, w, start, budget, ft0, beta=beta)['objective']
            z_fix = F.solve_plan(pool, hz, w, start, budget, ft0, beta=beta, chip_fix={(c, g): 1})['objective']
            res = F.solve_plan(pool, hz, w, start, budget, ft0, beta=beta, chip='AUTO', chips_avail={c: [(g, g)]})
            exp = max(z_none, z_fix - F.CHIP_TIEBREAK)   # a played chip pays the 0.001 tie-break
            good = abs(res['objective'] - exp) <= 2 * F.MIP_REL_GAP * abs(exp) + 1e-6   # all within the gap
            ok &= good
            print(f"[equiv big {c} GW{g}] variable {res['objective']:.3f} (chip {res['plan'][g]['chip']}) | "
                  f"fixed {z_fix:.3f}, none {z_none:.3f} | {'OK' if good else 'DIFF'}", flush=True)
    print("ALL OK" if ok else "SOME DIFFER")


def check_tc():
    """Hand check TC: captain (FWD, fixed 20 xP) has 30 xP in GW10. weights 1, 0.85, 0.7225.
    Delta_TC(8) = 20, Delta_TC(9) = 0.85*20 = 17, Delta_TC(10) = 0.7225*30 = 21.675 -> GW10."""
    hz = [8, 9, 10]
    pool = H6.toy_pool(); H6.fill(pool, hz); sq = H6.base_squad(pool)
    pool['xp'][(sq[14], 10)] = 30.0
    z0 = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, use_vice=False, use_bench=False)['objective']
    res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO', chips_avail={'TC': [(1, 19)]},
                       use_vice=False, use_bench=False)
    show(res)
    print(f"   chip GW: {[g for g, p in res['plan'].items() if p['chip']]} (hand: [10]) | "
          f"Delta {res['objective'] - z0:.4f} (hand: 0.7225*30 - chip tie-break = {0.7225 * 30 - F.CHIP_TIEBREAK:.4f})")
    # vice-captain term tripled too: captain P(app) 0.5 in GW10, VC 10 xP
    pool['papp'][(sq[14], 10)] = 0.5
    pool['xp'][(sq[13], 10)] = 10.0
    z0 = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, use_vice=True, use_bench=False)['objective']
    res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO', chips_avail={'TC': [(1, 19)]},
                       use_vice=True, use_bench=False)
    hand = 0.7225 * (30 + 0.5 * 10)
    print(f"   with VC (captain P 0.5, VC 10 xP in GW10): chip GW {[g for g, p in res['plan'].items() if p['chip']]}, "
          f"Delta {res['objective'] - z0:.4f} (hand: 0.7225*(30 + 0.5*10) - 0.001 = {hand - F.CHIP_TIEBREAK:.4f})")


def check_bb():
    """Hand check BB (no autosub value, so the bench is worth 0 without the chip): bench of the
    kept squad = 1 GK + 3 outfield players with 2 xP = 8 in every GW (in GW9 the 2nd GK starts,
    the 1st, 2 xP, is benched). Delta_BB = 8, 6.8, 5.78 -> GW8. Then variants (see prints)."""
    hz = [8, 9, 10]
    pool = H6.toy_pool(); H6.fill(pool, hz); sq = H6.base_squad(pool)
    gk2 = sq[1]
    pool['xp'][(gk2, 9)] = 10.0                     # starts in GW9 -> GK1 (2 xP) is the bench GK
    z0 = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, use_vice=False, use_bench=False, max_hits=0)['objective']
    res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO', chips_avail={'BB': [(1, 19)]},
                       use_vice=False, use_bench=False, max_hits=0)
    show(res)
    print(f"   chip GW: {[g for g, p in res['plan'].items() if p['chip']]} | Delta {res['objective'] - z0:.4f} "
          f"(hand: GW8 8.0, GW9 0.85*8 = 6.8, GW10 0.7225*8 = 5.78 -> GW8, 8.0 - 0.001 = 7.999)")
    # make the bench strong in GW9: three outfield bench players 2 -> 7 xP are NOT starters only if
    # the XI is still better; so instead raise ALL 15 players' xP in GW9 by 1 -> bench 4 * 3 = 12
    for i in sq:
        pool['xp'][(i, 9)] += 1.0
    z0 = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, use_vice=False, use_bench=False, max_hits=0)['objective']
    res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO', chips_avail={'BB': [(1, 19)]},
                       use_vice=False, use_bench=False, max_hits=0)
    bench9 = sorted(pool['xp'][(i, 9)] for i in sq)[:4]
    print(f"   all +1 in GW9 (bench xP {bench9}): chip GW {[g for g, p in res['plan'].items() if p['chip']]}, "
          f"Delta {res['objective'] - z0:.4f} (hand: GW8 8, GW9 0.85*12 = 10.2 -> GW9, 10.2 - 0.001 = 10.199)")
    # with autosub value on: the BB gain is the bench xP minus the autosub value it replaces
    beta = {g: [0.5, 0.15, 0.03] for g in hz}
    z0 = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, use_vice=False, use_bench=True, beta=beta, max_hits=0)['objective']
    res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO', chips_avail={'BB': [(1, 19)]},
                       use_vice=False, use_bench=True, beta=beta, max_hits=0)
    # GW9 bench: GK 3 xP (P(start GK)=1 -> autosub 0), outfield 3,3,3 with beta 0.5, 0.15, 0.03
    hand9 = 0.85 * (12 - 3 * (0.5 + 0.15 + 0.03))
    hand8 = 8 - 2 * (0.5 + 0.15 + 0.03)
    print(f"   with autosub value (beta 0.5/0.15/0.03): chip GW {[g for g, p in res['plan'].items() if p['chip']]}, "
          f"Delta {res['objective'] - z0:.4f} (hand: GW8 {hand8:.4f}, GW9 0.85*(12 - 3*0.68) = {hand9:.4f}, minus 0.001)")


def check_wc():
    """Hand check WC: 3 target MIDs with 4 xP (vs 2) in every GW, 1 FT, no hits allowed.
    No WC: one transfer per GW: 2*(1+.85+.7225) + 2*(.85+.7225) + 2*.7225 = 9.735
    WC GW8: all three in GW8: 3 * 2*(2.5725) = 15.435 -> Delta 5.70
    WC GW9: one in GW8 + two in GW9: 5.145 + 2*3.145 = 11.435 -> Delta 1.70 -> GW8."""
    hz = [8, 9, 10]
    pool = H6.toy_pool(); H6.fill(pool, hz); sq = H6.base_squad(pool)
    for k in (101, 102, 103):
        H6.add(pool, k, 3, 5.0, {g: 4.0 for g in hz})
    kw = dict(use_vice=False, use_bench=False, max_hits=0)
    z0 = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, **kw)
    res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO', chips_avail={'WC': [(2, 19)]}, **kw)
    print("[WC] none:"); show(z0)
    print("[WC] auto:"); show(res)
    tb = F.TRANSFER_TIEBREAK
    print(f"   chip GW {[g for g, p in res['plan'].items() if p['chip']]} (hand [8]) | Delta "
          f"{res['objective'] - z0['objective']:.4f} (hand: 15.435 - 9.735 - 0.001 = 5.699; transfer tie-breaks equal) | "
          f"FT GW9 = {res['plan'][9]['ft']} (hand: F9 = F8 = 1, none added)")
    for g in hz:
        zf = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip_fix={('WC', g): 1}, **kw)['objective']
        print(f"   fixed WC GW{g}: Delta {zf - z0['objective']:.4f}")


def check_fh():
    """Hand check FH: four MIDs with 10 xP in GW9 only (0 otherwise), 1 FT, no hits.
    No FH: bank the GW8 FT, buy 2 in GW9 (F9 = 2): +2*8*0.85 = 13.6; in GW10 they stay
    (the XI still has 3 normal MIDs: 2 + 2 + ... no loss).
    FH GW9: 4 in GW9 for one GW: +4*8*0.85 = 27.2, squad reverts in GW10 -> Delta 13.6.
    Both: tie-break 0.001 per transfer: none 2 transfers, FH 4 -> Delta 13.598."""
    hz = [8, 9, 10]
    pool = H6.toy_pool(); H6.fill(pool, hz); sq = H6.base_squad(pool)
    for k in (101, 102, 103, 104):
        H6.add(pool, k, 3, 5.0, {8: 0.0, 9: 10.0, 10: 0.0})
    kw = dict(use_vice=False, use_bench=False, max_hits=0)
    z0 = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, **kw)
    res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO', chips_avail={'FH': [(2, 19)]}, **kw)
    print("[FH] none:"); show(z0)
    print("[FH] auto:"); show(res)
    print(f"   chip GW {[g for g, p in res['plan'].items() if p['chip']]} (hand [9]) | Delta "
          f"{res['objective'] - z0['objective']:.4f} (hand 27.2 - 13.6 - 0.002 - 0.001 = 13.597) | "
          f"GW10 squad = permanent squad of GW8: {set(res['plan'][10]['squad']) == set(res['plan'][8]['squad'])} | GW10 transfers {res['plan'][10]['in']} | "
          f"FT GW10 = {res['plan'][10]['ft']} (hand: F10 = F9)")


def check_rules():
    """K1 one chip per GW, K2 once per set, K3 no FH in consecutive GWs."""
    hz = [18, 19, 20, 21]
    pool = H6.toy_pool(); H6.fill(pool, hz); sq = H6.base_squad(pool)
    for i in sq:                                   # TC and BB both best in GW19
        pool['xp'][(i, 19)] = 3.0
    pool['xp'][(sq[14], 19)] = 40.0
    res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO',
                       chips_avail={'TC': [(1, 19), (20, 38)], 'BB': [(1, 19)]}, use_vice=False, use_bench=False,
                       max_hits=0)
    print("[K1] TC and BB both best in GW19:"); show(res)
    chips = {g: p['chip'] for g, p in res['plan'].items()}
    per_gw = {g: sum(1 for c in F.CHIPS if (res['values'].get(f"chip_{c}_{g}") or 0) > 0.5) for g in hz}
    print(f"   chips per GW {per_gw} (each <= 1: {all(v <= 1 for v in per_gw.values())}); "
          f"TC twice (one per set): {sum(1 for c in chips.values() if c == 'TC')}")
    # K3: Free Hit both sets, a spike in GW19 and GW20 for four MIDs -> only one of them
    pool = H6.toy_pool(); H6.fill(pool, hz); sq = H6.base_squad(pool)
    for k in (101, 102, 103, 104):
        H6.add(pool, k, 3, 5.0, {18: 0.0, 19: 10.0, 20: 10.0, 21: 0.0})
    res = F.solve_plan(pool, hz, W(hz), sq, 75.0, 1, chip='AUTO', chips_avail={'FH': [(2, 19), (20, 38)]},
                       use_vice=False, use_bench=False, max_hits=0)
    print("[K3] FH available in both sets, spike in GW19 and GW20:"); show(res)
    fh = [g for g, p in res['plan'].items() if p['chip'] == 'FH']
    print(f"   FH GWs {fh}: not consecutive {all(b - a > 1 for a, b in zip(fh, fh[1:]))}")


def check_estimates():
    """Hand check of the chip estimates (C9/C10) on the toy squad: 10 starters with 2 xP + the
    fixed FWD captain with 20 xP, bench = GK + 3 outfield with 2 xP.
    No vice/bench terms: value = 10*2 + 20 + captain 20 = 60; TC +20; BB +8 (bench 4*2).
    With P(app) = 0.9 for all: beta = 1-0.9^10 = 0.6513, 0.2639, 0.0702; bench GK value
    (1-0.9)*2 = 0.2 -> autosubs = 0.2 + 2*(0.6513+0.2639+0.0702) = 2.1708; vice 0.1*2 = 0.2;
    BB gain = 8 - 2.1708 = 5.8292; TC gain = 20 + 0.1*2 = 20.2.
    FH: four MIDs with 10 xP in GW9 -> one-GW gain 4*(10-2) = 32."""
    hz = [8, 9, 10]
    pool = H6.toy_pool(); H6.fill(pool, hz); sq = H6.base_squad(pool)
    v0 = F.squad_value(pool, sq, 8, 'NONE', use_vice=False, use_bench=False)
    vt = F.squad_value(pool, sq, 8, 'TC', use_vice=False, use_bench=False)
    vb = F.squad_value(pool, sq, 8, 'BB', use_vice=False, use_bench=False)
    print(f"[est] no vice/bench: value {v0:.4f} (hand 60) | TC {vt - v0:+.4f} (hand +20) | BB {vb - v0:+.4f} (hand +8)")
    for i in pool['ids']:
        for g in hz:
            pool['papp'][(i, g)] = 0.9
    v0 = F.squad_value(pool, sq, 8, 'NONE')
    vt = F.squad_value(pool, sq, 8, 'TC')
    vb = F.squad_value(pool, sq, 8, 'BB')
    b = F.poisson_binomial_tail([0.1] * 10)
    auto = 0.1 * 2 + 2 * sum(b)
    print(f"   beta {[round(x, 4) for x in b]} | value {v0:.4f} (hand 60 + 0.2 + {auto:.4f} = {60.2 + auto:.4f}) | "
          f"TC {vt - v0:+.4f} (hand +20.2) | BB {vb - v0:+.4f} (hand 8 - {auto:.4f} = {8 - auto:+.4f})")
    for k in (101, 102, 103, 104):
        H6.add(pool, k, 3, 5.0, {8: 0.0, 9: 10.0, 10: 0.0})
    v0 = F.squad_value(pool, sq, 9, 'NONE', use_vice=False, use_bench=False)
    sub = dict(pool); sub['sell'] = {i: pool['price'][i] for i in sq}
    r = F.plan_squad(sub, [9], {9: 1.0}, sq, 75.0, 1, 'FH', None, False, False)
    print(f"   FH GW9: {F.plan_points(r, sub, [9], False) - v0:+.4f} (hand +32)")
    # gw_points must equal the 'Expected points' line of the main output
    orig = F.chip_report                       # skip the chip analysis: CAPTURE must hold the main plan
    F.chip_report = lambda *a, **k: {'exact': {}, 'gap': {}, 'est': {}, 'window': {}, 'skipped': []}
    out, res, pool2 = H6.run(L=3)
    F.chip_report = orig
    line =[ln for ln in out.split('\n') if ln.startswith('Expected points')][0]
    print(f"   gw_points of the main plan GW8: {F.gw_points(res['plan'][8], 8, pool2, res['beta'][8])['total']:.2f} | {line}")


def check_timing():
    """python harness_7.py timing <limit> <L> [mode]. mode: none | auto | fixed:<chip>:<g>"""
    tl, L = int(sys.argv[2]), int(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else 'auto'
    vb = True   # vice/bench terms on (the optimizer's output needs them)
    H.MANAGER = H.build_manager()
    orig = F.plan_squad
    def plan(pool, horizon, weight, start, budget, ft0, chip='NONE', max_hits=None, use_vice=True,
             use_bench=True, time_limit=None, solver=None, **kw):
        if mode == 'auto':
            kw.update(chip='AUTO', chips_avail={c: [(2 if c in ('WC', 'FH') else 1, 19), (20, 38)] for c in F.CHIPS})
        elif mode.startswith('fixed'):
            _, c, g = mode.split(':')
            kw['chip_fix'] = {(c, int(g)): 1}
        kw.setdefault('chip', chip)
        return orig(pool, horizon, weight, start, budget, ft0, max_hits=max_hits, use_vice=vb, use_bench=vb,
                    time_limit=time_limit, solver=solver, **kw)
    F.plan_squad = plan
    t = time.time()
    out, res, pool = H6.run(L=L, tl=tl)
    chips = {g: p['chip'] for g, p in res['plan'].items() if p['chip']}
    print(f"[timing {mode}] players {len(pool['ids'])}, L {L}, vice/bench {vb}: solve {res['seconds']:.1f} s, "
          f"{res['passes']} pass(es), proven {res['proven']}, gap {res['gap']}, objective {res['objective']:.3f}, "
          f"chips {chips}, total {time.time() - t:.1f} s", flush=True)


CHECKS = {'equiv': check_equiv, 'equiv_big': check_equiv_big, 'tc': check_tc, 'bb': check_bb, 'wc': check_wc,
          'fh': check_fh, 'rules': check_rules, 'timing': check_timing, 'estimates': check_estimates}
if __name__ == '__main__':
    CHECKS[sys.argv[1]]()
