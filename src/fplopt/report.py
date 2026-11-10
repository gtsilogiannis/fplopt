"""Text output of a run (stdout). Progress lines and warnings ('[*] ...', '[WARNING] ...')
are not printed here: the engine logs them while it runs and the command line sends them to
stderr. The text below is the same as the output of the original fpl_v2.py."""
import sys

from .summary import build_summary


def print_summary(r, out=None):
    """15c, the default output: the plain-English summary (summary.py); --explain blocks first, as in
    the full output, because the user asked for them."""
    out = out or sys.stdout
    print_explain(r, out)
    for line in build_summary(r)['text']:
        print(line, file=out)


def print_explain(r, out):
    """X7: the --explain blocks (nothing without --explain)."""
    def p(line=""):
        print(line, file=out)

    for e in r.explain:                                   # X7: --explain
        p(f"\n[explain] {e.name} (id {e.id}, {e.team_name}, pos {e.pos_id}, £{e.price:.1f}m) "
          f"source={e.source} chance={e.chance}")
        mm = e.minutes_model
        if mm is not None:
            p(f"   minutes model (shrunk; raw in brackets): P(appear)={mm['Papp']:.2f} [{mm['raw_Papp']:.2f}] "
              f"P(60+)={mm['P60']:.2f} [{mm['raw_P60']:.2f}] E[min]={mm['Emin']:.0f} [{mm['raw_Emin']:.0f}] "
              f"| evidence W={mm['Wapp']:.2f} | season minutes={mm['minutes_total']:.0f} "
              f"| price factor={mm['price_factor']:.2f}")
            p("   per 90: " + ", ".join(f"{k}={v:.2f}" for k, v in e.per90))
            p(f"   GW{r.current_gw}: " + ", ".join(f"{k}={v:+.2f}" for k, v in e.components.items()))
        else:
            p(f"   no GW at current club yet -> ep_next={e.ep_next:.2f}")
        p("   xP by GW: " + ", ".join(f"GW{g}={x:.2f}" for g, x in e.xp_by_gw))
    for n in r.explain_not_found:
        p(f"[explain] '{n}' not found (check web_name; unavailable players outside your squad are skipped)")


def print_optimize(r, out=None):
    """Prints an engine.OptimizeResult in full (--details)."""
    out = out or sys.stdout

    def p(line=""):
        print(line, file=out)

    print_explain(r, out)
    g0 = r.current_gw
    p("\n" + "=" * 84)
    p(f" SQUAD PLAN | GW{g0} (horizon: GW{r.horizon[0]}-GW{r.horizon[-1]})")
    p("=" * 84)
    _print_squad(p, r.squad, g0)
    if g0 > 1:
        p("   Cost = selling price for players you keep, current price for players you buy.")  # S3

    p("-" * 84)
    _print_plan(p, r.plan)

    if g0 > 1:
        _print_sales(p, r.sales_now)
    _print_expected(p, r.expected_now, g0)
    print_formations(r, out)                              # 15c: the formations of the same squad
    p(f"Objective (weighted over horizon, hits included): {r.solver.objective:.2f}")

    if r.decision is not None:                            # --check only: the text without it is unchanged
        print_decision(r, out)
    print_chips(r, out)
    print_chip_plan(r, out)


def _print_squad(p, squad, g0):
    """The squad table of the current GW (main plan, or the plan with the suggested chip)."""
    p(f"   {'Player':<18} | Pos | Cost  | xP GW{g0:<3}| P(app) | xP horizon | Role")
    for s in squad:
        indicator = ">>" if s.action == 'BUY' else "  "
        flag = "" if s.available else "  (unavailable)"                                    # A
        gw_tag = "  (blank)" if s.n_matches_now == 0 else ("  (DGW)" if s.n_matches_now >= 2 else "")   # G4
        src_tag = "  (FPL ep)" if s.source != 'model' else ""                               # X5
        role = s.role
        if role == 'Bench':
            role = 'Bench GK' if s.pos_id == 1 else f"Bench {s.bench_slot}"                # M5: bench order
        p(f"{indicator} {s.name:<18} | {s.position:<3} | £{s.cost:<4.1f} | {s.xp_now:<8.2f} | "
          f"{s.papp_now:<6.2f} | {s.xp_horizon:<10.2f} | {role}{flag}{gw_tag}{src_tag}")


def _print_plan(p, plan):
    """The transfer plan over the horizon."""
    p(" Transfer plan (FT = free transfers available before that GW's transfers):")
    gws = {pg.gw for pg in plan}
    for pg in plan:
        moves = ", ".join(f"{t.out_name} -> {t.in_name}" for t in pg.transfers) or "none"
        chip_tag = f" [{pg.chip}]" if pg.chip else ""
        ft_txt = "unlimited" if pg.ft_unlimited else pg.ft
        vice_txt = f", VC {pg.vice_name}" if pg.vice_id is not None else ""
        p(f"   GW{pg.gw}{chip_tag}: FT {ft_txt} | {moves} | hits {pg.hits} | C {pg.captain_name}{vice_txt}")
        if pg.chip == 'FH' and pg.gw + 1 in gws:
            p(f"   Free Hit: GW{pg.gw + 1} starts again from the squad before GW{pg.gw}.")    # M3/C4


def _print_sales(p, sales_now):
    sold = [f"{name} £{price:.1f}m" + ("" if avail else " (unavailable)") for _, name, price, avail in sales_now]
    if sold:
        p(f"Sales now (SELL): {', '.join(sold)}")
    else:
        p("Action now: roll the transfer (no change this GW).")


def _print_expected(p, ep, g0):
    if ep.bench_weights is not None:
        p(f"   Bench weights GW{g0} (P(slot used)): " + ", ".join(f"{b:.2f}" for b in ep.bench_weights)
          + "  [approximation]")
    p(f"Expected points GW{g0}: XI {ep.xi:.2f} + captain {ep.captain:.2f} + vice {ep.vice:.2f} "
      f"+ autosubs {ep.autosubs:.2f} - hits {ep.hits} = {ep.total:.2f}")


def print_formations(r, out):
    """15c: the plan's formation in the current GW and the next two best ones for the same squad
    (unweighted expected points of the GW, captain and bench solved again with each formation)."""
    if not r.formations:
        return
    pg = r.plan[0]
    pos = {s.id: s.pos_id for s in r.squad}
    mine = "-".join(str(sum(1 for i in pg.xi if pos[i] == p_)) for p_ in (2, 3, 4))
    done = [f for f in r.formations if f['points'] is not None]
    chosen = next((f for f in done if f['formation'] == mine), None)
    if chosen is None:
        print(f"   Formation GW{r.current_gw}: {mine} (formations not computed in the time limit)", file=out)
        return
    others = sorted((f for f in done if f is not chosen), key=lambda f: -f['points'])[:2]
    cells = [f"{f['formation']} {f['points']:.2f} ({f['points'] - chosen['points']:+.2f})" for f in others]
    if cells:
        cells[0] = "next best " + cells[0]
    # points before hits: the hits are the same with every formation of the same squad
    print(f"   Formation GW{r.current_gw} (points before hits): {mine} {chosen['points']:.2f}"
          + "".join(f" | {c}" for c in cells), file=out)


def _signed(v, lo=None, hi=None):
    """+1.23 (between +1.10 and +1.30); 'not computed' for None (C12)."""
    if v is None:
        return "not computed"
    return f"{v:+.2f}" + (f" (between {lo:+.2f} and {hi:+.2f})" if lo is not None and hi is not None else "")


def print_decision(r, out):
    """Decision check: the user's chip decision against the plan without it and the best version."""
    def p(line=""):
        print(line, file=out)

    d = r.decision
    p("\n" + "=" * 84)
    if d.chip == 'TRANSFERS':
        moves = ", ".join(f"{d.names[o]} -> {d.names[i]}" for o, i in zip(d.transfers_out, d.transfers_in))
        p(f" DECISION CHECK | transfers in GW{d.gw}: {moves}")
    else:
        who = f" on {d.captain_name}" if d.captain_id is not None else ""
        p(f" DECISION CHECK | {d.chip}{who} in GW{d.gw}")
    p("=" * 84)
    if d.status == 'invalid':
        for v in d.violations:
            p(f"   Invalid: {v['message']}.")
        p(">> Verdict: this decision is not allowed by the rules of the game.")
        return
    x = d.result
    if d.chip == 'TRANSFERS':
        _print_transfers(r, d, x, p)
        return
    hz = f"GW{r.horizon[0]}-GW{r.horizon[-1]}"
    p(f"   Expected points over {hz}, weighted as in the plan; the transfers after GW{d.gw} stay free:")
    p(f"      G = your decision - no {d.chip}:      {_signed(x['g'], x['g_low'], x['g_high'])}")
    if d.chip == 'BB':
        p("      R = best BB - your decision:      0 (Bench Boost leaves nothing to choose)")
    else:
        p(f"      R = best {d.chip} - your decision:    {_signed(x['r'], x['r_low'], x['r_high'])}")
    p(f"      Gain of {d.chip} itself (G + R):      {_signed(x['gain_action'])}")
    pts = x['points_gw']
    cells = [f"{label} {pts[k]:.2f}" for k, label in (('z0', 'no chip'), ('z_decision', 'your decision'),
                                                       ('z_best', f'best {d.chip}')) if k in pts]
    if cells:
        p(f"   GW{d.gw} alone, unweighted: " + " | ".join(cells))
    best = x['plan_best']
    if d.chip == 'TC' and best is not None and best['captain'] != d.captain_id:
        p(f"   Best {d.chip} in GW{d.gw}: captain {d.names.get(best['captain'], best['captain'])}.")
    t = x['timing']
    if t is not None:
        est = t['estimates']
        top = max(est.values()) if est else None
        cells = [f"GW{g} {v:+.1f}{'*' if v == top else ''}" for g, v in sorted(est.items())]
        how = "squad of the plan, best captain" if d.chip == 'TC' else "squad of the plan"
        p(f"   {d.chip} estimates by GW ({how}; * = highest to GW{t['window_end']}):")
        for k in range(0, len(cells), 8):                 # 8 GWs per line, as in the chip section
            p("      " + "  ".join(cells[k:k + 8]))
    _print_tail(d, x, p)


def _print_tail(d, x, p):
    """Notes, what was not computed, the verdict and the reminder of the model's error."""
    for n in d.notes:
        p(f"   Note: {n['message']}.")
    if x['not_computed']:
        p(f"   Not computed in the time limit: {', '.join(x['not_computed'])} (raise --time-limit).")
    for code in d.verdict:
        p(">> Verdict: " + _verdict_text(code, d, x))
    if not d.verdict:
        p(">> Verdict: none: a number it needs was not computed (raise --time-limit).")
    p("   Expected points of the model (xP error ~2.5 per player-GW): differences of a few tenths are noise.")


def _model_moves(d, x):
    pg = x['plan_best']
    if not pg['in']:
        return "no transfer"
    # out and in listed apart: which sale pays for which purchase is not a decision of the plan
    return f"out {', '.join(d.names[i] for i in pg['out'])} | in {', '.join(d.names[i] for i in pg['in'])}"


def _print_transfers(r, d, x, p):
    """D3: the user's transfers against rolling the transfer and against the model's transfers."""
    hz = f"GW{r.horizon[0]}-GW{r.horizon[-1]}"
    p(f"   Expected points over {hz}, weighted as in the plan; the transfers after GW{d.gw} stay free:")
    p(f"      G = your transfers - no transfer:      {_signed(x['g'], x['g_low'], x['g_high'])}")
    p(f"      R = model's transfers - yours:         {_signed(x['r'], x['r_low'], x['r_high'])}")
    p(f"      Gain of the model's transfers (G + R): {_signed(x['gain_action'])}")
    pts = x['points_gw']
    cells = [f"{label} {pts[k]:.2f}" for k, label in (('z0', 'no transfer'), ('z_decision', 'yours'),
                                                       ('z_best', "model's")) if k in pts]
    if cells:
        p(f"   GW{d.gw} alone, unweighted, hits included: " + " | ".join(cells))
    if x['plan_decision'] is not None:
        p(f"   Your transfers: {x['plan_decision']['hits']} hit(s); captain "
          f"{d.names.get(x['plan_decision']['captain'], x['plan_decision']['captain'])}.")
    if x['plan_best'] is not None:
        p(f"   Model's transfers in GW{d.gw}: {_model_moves(d, x)} ({x['plan_best']['hits']} hit(s)).")
    _print_tail(d, x, p)


def _verdict_text(code, d, x):
    t = x['timing']
    if d.chip == 'TRANSFERS':
        return {'worse_than_none': f"worse than no transfer ({x['g']:+.2f}): roll the transfer.",
                'no_clear_gain': "no clear gain over no transfer: G is within the solver tolerance.",
                'better_version': f"the model's transfers ({_model_moves(d, x)}) gain {x['r']:+.2f} more.",
                'agrees': f"make them: they gain {x['g']:+.2f} over no transfer and no better set exists.",
                }[code]
    if code == 'worse_than_none':
        return f"worse than not playing {d.chip} in GW{d.gw} ({x['g']:+.2f}): keep the chip."
    if code == 'no_clear_gain':
        return f"no clear gain from {d.chip} in GW{d.gw}: G is within the solver tolerance."
    if code == 'better_gw':
        return (f"{d.chip} is estimated higher in GW{t['best_gw']} ({t['best']:+.1f}) than in GW{d.gw} "
                f"({t['at_gw']:+.1f}).")
    if code == 'better_version':
        cap = d.names.get(x['plan_best']['captain'], x['plan_best']['captain'])
        return f"captain {cap} gains {x['r']:+.2f} more with {d.chip} than your choice."
    return f"play it: it gains {x['g']:+.2f}, GW{d.gw} is the best GW of the window and no better version exists."


def print_chips(r, out):
    """C10/C11: the CHIP SUGGESTIONS section."""
    def p(line=""):
        print(line, file=out)

    c, g0 = r.chips, r.current_gw
    p("\n" + "=" * 84)
    p(f" CHIP SUGGESTIONS | GW{g0} (expected points, unweighted)")
    p("=" * 84)
    p(f"   Chips ({c.source}): " + " | ".join(f"{cp} GW{s}-{e} {st}" for cp, s, e, st in c.sets))
    if c.status == 'skipped_flag':
        p("   Skipped (--no-chip-report).")
        return
    if c.status == 'skipped_chip':
        p(f"   Skipped: --chip {r.options.chip} is played this GW (one chip per GW).")
        return
    if c.status == 'no_chip_now':
        p(f"   No chip can be played in GW{g0}.")
        return
    h3 = r.horizon[:3]
    p(f"   Exact Delta now = best plan with the chip in GW{g0} minus best plan without a chip, "
      f"full model, GW{h3[0]}-GW{h3[-1]}:")
    p("      " + " | ".join(f"{cp} {c.exact[cp]:+.2f} (error <= {c.error[cp]:.2f})"
                           if cp in c.exact else f"{cp} not computed" for cp in c.window))
    p("   Estimates by GW (squad of the plan; after "
      f"GW{r.horizon[-1]} its last squad, no transfers [assumption]); * = best GW to the end of the window:")
    for cp, est in c.estimates.items():
        best = max(est.values())
        cells = [f"GW{t} {d:+.1f}{'*' if d == best else ''}" for t, d in sorted(est.items())]
        for k in range(0, len(cells), 8):                 # 8 GWs per line
            p(f"      {cp if k == 0 else '':<3} " + "  ".join(cells[k:k + 8]))
    adv = c.advice
    for cp, now, t_best, best, window_end in adv['holds']:
        now_txt = f"now {now:+.1f}, " if now is not None else ""
        # the margin is printed because estimates of different GWs that differ by a few
        # tenths are within the error of the xP model (RMSE ~2.5 per player-GW)
        p(f"   {cp}: hold ({now_txt}best estimate GW{t_best} {best:+.1f}; window ends GW{window_end})")
    for cp in adv.get('incomplete', []):
        # C12: no suggestion from part of the numbers (it could differ between machines)
        p(f"   {cp}: no suggestion: not every number up to GW{c.window[cp]} was computed in the time limit "
          f"(raise --time-limit).")
    if 'WC' in c.window:
        txt = f"{c.exact['WC']:+.2f}" if 'WC' in c.exact else "not computed"
        p(f"   WC: exact Delta now {txt}; later GWs are not estimated [assumption] -> compare yourself "
          f"(window ends GW{c.window['WC']}).")
    pick = adv['pick']
    if pick is not None:
        # C13: the plan that uses it follows (PLAN WITH ...), unless it was switched off
        then = ("Its plan follows." if r.chip_plan is not None
                else f"Re-run with --chip {pick} for the plan that uses it.")
        p(f">> Suggestion: play {pick} in GW{g0} (exact Delta {c.exact[pick]:+.2f}). {then}")
        if adv['also']:
            p(f"   Also best now: {', '.join(adv['also'])} (one chip per GW).")
        if adv['wc_note'] is not None:
            wc, pk = adv['wc_note']
            p(f"   Note: WC gains more now ({wc:+.2f} vs {pk:+.2f}); "
              f"playing {pick} rules out WC this GW, and the WC timing is not estimated [assumption]. "
              f"Compare with --chip WC.")
    else:
        p(">> Suggestion: no chip this GW.")


def print_chip_plan(r, out):
    """C13: the plan with the suggested chip (the same as a run with --chip X)."""
    cp = r.chip_plan
    if cp is None:
        return

    def p(line=""):
        print(line, file=out)

    g0 = r.current_gw
    p("\n" + "=" * 84)
    p(f" PLAN WITH {cp.chip} | GW{g0} (the same plan as --chip {cp.chip})")
    p("=" * 84)
    if cp.status != 'computed':
        p(f"   Not computed in the time limit: run again with --chip {cp.chip} (or a longer --time-limit).")
        return
    _print_squad(p, cp.squad, g0)
    p("-" * 84)
    _print_plan(p, cp.plan)
    if g0 > 1:
        _print_sales(p, cp.sales_now)
    _print_expected(p, cp.expected_now, g0)
    gap = f"{cp.solver.gap:.2f}" if cp.solver.gap is not None else "unknown"
    p(f"Solved in {cp.seconds:.1f} s, within {gap} weighted points of the best plan with {cp.chip}; "
      f"objective {cp.solver.objective:.2f} (main plan {r.solver.objective:.2f}).")


def print_evaluate(r, out=None):
    """Prints a predictions.EvaluateResult (nothing if the log had no row for the GW)."""
    out = out or sys.stdout
    if not r.rows:
        return

    def p(line=""):
        print(line, file=out)

    p("\n" + "=" * 72)
    p(f" MODEL EVALUATION | GW{r.gw}")
    p("=" * 72)
    p(f"{'Players':<34} {'n':>4} | {'MAE':>6} {'FPL':>6} | {'bias':>6} {'FPL':>6}")
    p("-" * 72)
    for row in r.rows:
        p(f"{row.label:<34} {row.n:>4} | {row.mae:>6.2f} {row.mae_fpl:>6.2f} | "
          f"{row.bias:>+6.2f} {row.bias_fpl:>+6.2f}")
    p("-" * 72)
    p("MAE = mean |predicted - actual|, bias = mean (predicted - actual). FPL = ep_next baseline.")
    if r.xi_predicted is not None:
        p(f"XI + captain: predicted {r.xi_predicted:.1f}, actual {r.xi_actual:.0f} (no chips, no autosubs)")
    p("One GW is a small, noisy sample; judge the model over several GWs.")
