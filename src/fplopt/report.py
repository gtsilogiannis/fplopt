"""Text output of a run (stdout). Progress lines and warnings ('[*] ...', '[WARNING] ...')
are not printed here: the engine logs them while it runs and the command line sends them to
stderr. The text below is the same as the output of the original fpl_v2.py."""
import sys


def print_optimize(r, out=None):
    """Prints an engine.OptimizeResult."""
    out = out or sys.stdout

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

    g0 = r.current_gw
    p("\n" + "=" * 84)
    p(f" SQUAD PLAN | GW{g0} (horizon: GW{r.horizon[0]}-GW{r.horizon[-1]})")
    p("=" * 84)
    p(f"   {'Player':<18} | Pos | Cost  | xP GW{g0:<3}| P(app) | xP horizon | Role")
    for s in r.squad:
        indicator = ">>" if s.action == 'BUY' else "  "
        flag = "" if s.available else "  (unavailable)"                                    # A
        gw_tag = "  (blank)" if s.n_matches_now == 0 else ("  (DGW)" if s.n_matches_now >= 2 else "")   # G4
        src_tag = "  (FPL ep)" if s.source != 'model' else ""                               # X5
        role = s.role
        if role == 'Bench':
            role = 'Bench GK' if s.pos_id == 1 else f"Bench {s.bench_slot}"                # M5: bench order
        p(f"{indicator} {s.name:<18} | {s.position:<3} | £{s.cost:<4.1f} | {s.xp_now:<8.2f} | "
          f"{s.papp_now:<6.2f} | {s.xp_horizon:<10.2f} | {role}{flag}{gw_tag}{src_tag}")
    if g0 > 1:
        p("   Cost = selling price for players you keep, current price for players you buy.")  # S3

    p("-" * 84)
    p(" Transfer plan (FT = free transfers available before that GW's transfers):")
    gws = {pg.gw for pg in r.plan}
    for pg in r.plan:
        moves = ", ".join(f"{t.out_name} -> {t.in_name}" for t in pg.transfers) or "none"
        chip_tag = f" [{pg.chip}]" if pg.chip else ""
        ft_txt = "unlimited" if pg.ft_unlimited else pg.ft
        vice_txt = f", VC {pg.vice_name}" if pg.vice_id is not None else ""
        p(f"   GW{pg.gw}{chip_tag}: FT {ft_txt} | {moves} | hits {pg.hits} | C {pg.captain_name}{vice_txt}")
        if pg.chip == 'FH' and pg.gw + 1 in gws:
            p(f"   Free Hit: GW{pg.gw + 1} starts again from the squad before GW{pg.gw}.")    # M3/C4

    if g0 > 1:
        sold = [f"{name} £{price:.1f}m" + ("" if avail else " (unavailable)") for _, name, price, avail in r.sales_now]
        if sold:
            p(f"Sales now (SELL): {', '.join(sold)}")
        else:
            p("Action now: roll the transfer (no change this GW).")

    ep = r.expected_now
    if ep.bench_weights is not None:
        p(f"   Bench weights GW{g0} (P(slot used)): " + ", ".join(f"{b:.2f}" for b in ep.bench_weights)
          + "  [approximation]")
    p(f"Expected points GW{g0}: XI {ep.xi:.2f} + captain {ep.captain:.2f} + vice {ep.vice:.2f} "
      f"+ autosubs {ep.autosubs:.2f} - hits {ep.hits} = {ep.total:.2f}")
    p(f"Objective (weighted over horizon, hits included): {r.solver.objective:.2f}")

    print_chips(r, out)


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
        p(f">> Suggestion: play {pick} in GW{g0} (exact Delta {c.exact[pick]:+.2f}). "
          f"Re-run with --chip {pick} for the plan that uses it.")
        if adv['also']:
            p(f"   Also best now: {', '.join(adv['also'])} (one chip per GW).")
        if adv['wc_note'] is not None:
            wc, pk = adv['wc_note']
            p(f"   Note: WC gains more now ({wc:+.2f} vs {pk:+.2f}); "
              f"playing {pick} rules out WC this GW, and the WC timing is not estimated [assumption]. "
              f"Compare with --chip WC.")
    else:
        p(">> Suggestion: no chip this GW.")


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
