"""Plain-English summary of a run (chat 15c): "what do I do this gameweek", for readers with no
background in maths or programming.

build_summary() turns an engine.OptimizeResult into (1) a dict for the JSON output and (2) the
lines of text the command line prints. It computes NO new decision: every fact comes from the
result (S1), so the summary can never disagree with the full output (--details). Only rounding
and wording are added here.
"""
import math
import re
import textwrap
from datetime import datetime

from .rules import HIT_COST, MAX_FT

# [measured] 80% range of (actual - predicted) points of the best XI + captain in one gameweek:
# 10th and 90th percentile over 44 GWs of walk-forward backtest (scripts/calibrate_xp.py,
# top11), 2024/25 (34 GWs) and 2025/26 (10 GWs), only GWs where the dataset has FPL's own
# projection (the only GWs with an availability proxy; a live run leaves out injured players
# too). Pooled mean -1.0, SD 19.6. Limits: top11 picks the 11 highest predictions with no
# budget or formation and no vice / autosubs, so it is close to, not equal to, the plan's XI;
# with 44 GWs the ends are uncertain by a few points.
RANGE_LOW = -25.4
RANGE_HIGH = 24.0

CHIP_NAME = {'WC': 'Wildcard', 'FH': 'Free Hit', 'BB': 'Bench Boost', 'TC': 'Triple Captain'}
CHIP_ORDER = ('BB', 'TC', 'WC', 'FH')      # display order of the CHIPS section (the order of the FPL app)
# English names, NOT strftime('%a'): strftime follows the system locale (e.g. 'Σαβ' in Greek)
DAYS = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
WIDTH = 72          # text width of the summary (fits an 80-column terminal)
NOISE = "Chip gains are estimates: differences of a few tenths are within the model's error."


# --------------------------------------------------------------------------- rounding
def half_up(x):
    """Rounds to the nearest integer, .5 up (Python's round() sends 62.5 to 62)."""
    return int(math.floor(x + 0.5))


def about(x):
    """Totals: 'about 64'."""
    return f"about {half_up(x)}"


def signed(x):
    """Differences: '+1.8', '-4.0'; a value that rounds to 0.0 is 'no difference'."""
    return "no difference" if abs(x) < 0.05 else f"{x:+.1f}"


def money(x):
    return f"£{abs(x):.1f}m"


def tenths(x):
    """GBPm -> whole tenths (prices are whole tenths; the game counts money this way)."""
    return round(10 * x)


def format_deadline(deadline_utc, tz=None):
    """'2026-10-31T11:00:00Z' -> 'Sat 31 Oct, 13:00' in the time zone tz (None = the time zone of
    the operating system). astimezone() applies summer/winter time of that date."""
    t = datetime.fromisoformat(deadline_utc.replace('Z', '+00:00')).astimezone(tz)
    return f"{DAYS[t.weekday()]} {t.day} {MONTHS[t.month - 1]}, {t.hour:02d}:{t.minute:02d}"


# --------------------------------------------------------------------------- transfers
def pays_for(bank, deltas):
    """Which sales pay for the other purchases. All amounts in whole tenths of GBPm.
    bank:   money in the bank before the transfers;
    deltas: per move, selling price of the player out minus price of the player in.
    A move k with delta_k > 0 pays for the others iff without it the purchases do not fit and
    with it they do:   bank + sum_{j != k} delta_j < 0 <= bank + sum_j delta_j."""
    total = bank + sum(deltas)
    return [d > 0 and total - d < 0 <= total for d in deltas]


def _transfers(r, pg, squad=None, sales_now=None, vs_no_transfer=True):
    """Moves of the current GW with their money; None for the bank fields at GW1 (no squad yet).
    squad, sales_now: those of another plan of the same GW (C13: the plan with the suggested chip);
    None = the main plan's. vs_no_transfer=False: no gain over 'no transfer' (only the main plan has it)."""
    squad = r.squad if squad is None else squad
    sales_now = r.sales_now if sales_now is None else sales_now
    sell = {i: price for i, _, price, _ in sales_now}               # selling price of each sale
    cost = {s.id: s.cost for s in squad}                            # current price of each buy
    bank = None if r.squad_value is None else tenths(r.budget) - tenths(r.squad_value)
    deltas = [tenths(sell[t.out_id]) - tenths(cost[t.in_id]) if t.out_id in sell else None
              for t in pg.transfers]
    known = bank is not None and None not in deltas
    flags = pays_for(bank, deltas) if known else [False] * len(deltas)
    moves = [{'out': {'id': int(t.out_id), 'name': t.out_name}, 'in': {'id': int(t.in_id), 'name': t.in_name},
              'money': d / 10 if d is not None else None, 'pays_for_others': f}
             for t, d, f in zip(pg.transfers, deltas, flags)]
    return {
        'free_transfers': pg.ft, 'unlimited': pg.ft_unlimited, 'hits': pg.hits,
        'hit_points': HIT_COST * pg.hits, 'moves': moves,
        'bank_before': bank / 10 if bank is not None else None,
        'bank_after': (bank + sum(deltas)) / 10 if known else None,
        # unweighted gain over making no transfer (engine, decision.transfer_gain); None = not computed
        'vs_no_transfer': r.no_transfer if vs_no_transfer else None,
    }


def _transfer_rows(r, t, chip):
    """Rows of the TRANSFERS section for _layout(): (label, value), (None, sentence) or ('', raw line)."""
    g0 = r.current_gw
    if chip in ('WC', 'FH'):
        rows = [('Free transfers', f"unlimited ({CHIP_NAME[chip]})")]
        if t['moves']:
            rows += [('Sell', ", ".join(m['out']['name'] for m in t['moves'])),
                     ('Buy', ", ".join(m['in']['name'] for m in t['moves']))]
        if chip == 'FH':
            rows.append((None, f"After GW{g0} your squad goes back to the one you have now."))
        return rows
    if t['unlimited']:
        return [('Free transfers', "unlimited (your first deadline): build the squad below")]
    ft = t['free_transfers']
    if not t['moves']:
        if ft is not None and ft >= MAX_FT:
            return [('Free transfers', str(ft)),
                    (None, f"No transfer this gameweek: {MAX_FT} is the most you can keep, so one is lost.")]
        if ft is not None:
            return [('Free transfers', str(ft)),
                    (None, f"No transfer this gameweek: you will have {ft + 1} next gameweek.")]
        return [(None, "No transfer this gameweek.")]
    hits = t['hits']
    # the points cost is shown only when there is one
    cost = "" if hits == 0 else f" ({hits} extra transfer{'s' if hits != 1 else ''}: -{t['hit_points']} points)"
    rows = [('Free transfers', f"{ft}{cost}")]
    w = max(len(m['out']['name']) for m in t['moves'])        # dynamic columns: the longest name sets them
    v = max(len(m['in']['name']) for m in t['moves'])
    for m in t['moves']:
        d = m['money']
        # money of the move: '-' = you pay more than the sale returns, '+' = money left over
        price = "" if d is None else ("(same price)" if d == 0 else f"({'+' if d > 0 else '-'}{money(d)})")
        rows.append(('', f"    {m['out']['name']:<{w}} -> {m['in']['name']:<{v}}   {price}".rstrip()))
    if t['bank_before'] is not None and t['bank_after'] is not None:
        rows.append(('Bank', f"{money(t['bank_before'])} now -> {money(t['bank_after'])} after"))
    payers = [m['out']['name'] for m in t['moves'] if m['pays_for_others']]
    if payers:
        rows.append((None, f"Selling {' and '.join(payers)} is what makes the other purchases fit your budget."))
    vs = t['vs_no_transfer']
    if vs is not None:
        h = r.horizon
        now, all_ = signed(vs['this_gw']), signed(vs['next_gws'])
        if vs['n_gws'] == 1:
            gain = f"{now} points vs no transfer this gameweek"
        elif now == all_:                       # the same after rounding: the gain is all in this GW
            gain = f"{now} points vs no transfer, all of it this gameweek"
        else:
            gain = f"{now} points vs no transfer this gameweek, {all_} over GW{h[0]}-GW{h[-1]}"
        rows.append(('Gain', gain))
    return rows


# --------------------------------------------------------------------------- chips
def _chips(r):
    c = r.chips
    res = {'status': c.status, 'playing': None, 'play': None, 'also': [], 'hold': [], 'no_advice': [],
           'wildcard_now': None}
    if c.status == 'skipped_chip':
        res['playing'] = r.options.chip                      # --chip: the plan already uses it
    if c.status != 'computed' or c.advice is None:
        return res
    adv = c.advice
    if adv['pick'] is not None:
        res['play'] = {'chip': adv['pick'], 'gain': float(c.exact[adv['pick']])}
        res['also'] = list(adv['also'])
    res['hold'] = [{'chip': cp, 'now': None if now is None else float(now), 'best_gw': int(tb), 'best': float(best)}
                   for cp, now, tb, best, _ in adv['holds']]
    res['no_advice'] = list(adv['incomplete'])
    if 'WC' in c.window and 'WC' in c.exact and adv['pick'] != 'WC':
        res['wildcard_now'] = float(c.exact['WC'])           # WC timing is not estimated
    return res


def _with_states(r, res):
    """Adds 'by_chip': one {chip, status, text} per chip, in the display order (for the text and the JSON)."""
    if res['status'] != 'skipped_flag':
        res['by_chip'] = [{'chip': cp, 'status': st, 'text': cap(tx)}
                          for cp, (st, tx) in ((cp, _chip_state(r, res, cp)) for cp in CHIP_ORDER)]
    else:
        res['by_chip'] = []
    return res


def _chip_state(r, ch, cp):
    """(status code, text) of one chip this gameweek, for the CHIPS section and the JSON."""
    g0, c = r.current_gw, r.chips
    if ch['playing'] is not None:
        if cp == ch['playing']:
            return 'playing', "playing it this gameweek: the plan above uses it"
        if cp in c.window or any(st == 'available' and s0 <= g0 <= s1 for x, s0, s1, st in c.sets if x == cp):
            return 'other_chip', "not this gameweek (one chip per gameweek)"
    if ch['play'] is not None and cp == ch['play']['chip']:
        return 'play', f"play it this gameweek ({signed(ch['play']['gain'])} points)"
    if cp in ch['also']:
        return 'also', f"also good now ({signed(c.exact[cp])} points; one chip per gameweek)"
    for h in ch['hold']:
        if h['chip'] == cp:
            now = f"now {signed(h['now'])}, " if h['now'] is not None else ""
            return 'hold', f"keep it ({now}best GW{h['best_gw']}: {signed(h['best'])})"
    if cp == 'WC' and ch['wildcard_now'] is not None:
        return 'wildcard_now', (f"now {signed(ch['wildcard_now'])}; later gameweeks are not estimated, "
                                f"so the timing is your call")
    if cp in ch['no_advice']:
        return 'no_advice', "no advice (not enough time; see below)"
    # not playable now: say why, from the chip sets of the season (state: available / played GWn / expired)
    sets = sorted((s0, s1, st) for x, s0, s1, st in c.sets if x == cp)
    now_set = next(((s0, s1, st) for s0, s1, st in sets if s0 <= g0 <= s1), None)
    later = next((s0 for s0, s1, st in sets if s0 > g0 and st == 'available'), None)
    nxt = f"; the next one from GW{later}" if later is not None else ""
    if now_set is not None and now_set[2].startswith('played'):
        return 'used', f"used in {now_set[2].split()[-1]}{nxt}"
    if now_set is not None and now_set[2] == 'expired':
        return 'expired', f"expired{nxt}"
    if now_set is None and later is not None:
        return 'not_yet', f"can be played from GW{later}"
    if now_set is None:
        return 'unavailable', "none left this season"
    return 'not_estimated', "not estimated this gameweek"     # e.g. WC/FH before a team's first deadline


def _chip_rows(r, ch):
    """Rows of the CHIPS section for _layout(); [] = no section (--no-chip-report)."""
    if ch['status'] == 'skipped_flag':
        return []
    rows = [(CHIP_NAME[st['chip']], st['text']) for st in ch['by_chip']]
    rows.append(('', ''))
    if ch['playing'] is not None:
        pass                                                   # the plan already uses it: nothing to suggest
    elif ch['play'] is not None:
        cp = ch['play']['chip']
        if r.chip_plan is not None and r.chip_plan.chip == cp:
            rows.append((None, f"Suggested: {CHIP_NAME[cp]} this gameweek. Its plan is below."))   # C13
        else:
            rows.append((None, f"Suggested: {CHIP_NAME[cp]}. The plan above does not use it: run again with "
                               f"--chip {cp} to see the {CHIP_NAME[cp]} squad."))
    elif ch['status'] == 'no_chip_now':
        rows.append((None, "Suggested: No chip can be played this gameweek."))
    else:
        rows.append((None, "Suggested: No chip this gameweek."))
    if ch['play'] is not None or ch['hold'] or ch['wildcard_now'] is not None:
        rows.append((None, NOISE))
    while rows and rows[-1] == ('', ''):
        rows.pop()
    return rows


# --------------------------------------------------------------------------- lineup
def _tag(row):
    """Plain words for a player's flags in the lineup."""
    t = []
    if not row.available:
        t.append("injured or doubtful")
    if row.n_matches_now == 0:
        t.append("no match this gameweek")
    elif row.n_matches_now >= 2:
        t.append("plays twice this gameweek")
    return f"{row.name} ({', '.join(t)})" if t else row.name


def _xi_rows(squad, chip):
    """(title, rows) of a STARTING XI section from the squad rows of a plan (XI by position, then the bench)."""
    xi = [s for s in squad if s.role != 'Bench']
    bench = [s for s in squad if s.role == 'Bench']          # already in bench order (GK first)
    n = {p: sum(1 for x in xi if x.position == p) for p in ('DEF', 'MID', 'FWD')}
    rows = [(p, ", ".join(_tag(x) for x in xi if x.position == p)) for p in ('GK', 'DEF', 'MID', 'FWD')]
    # bench in its order: the GK first (his own slot), then outfield slots 1-3
    names = [_tag(b) + " (GK)" if b.pos_id == 1 else f"{b.bench_slot}. {_tag(b)}" for b in bench]
    rows.append(('Bench', ", ".join(names)))
    if chip == 'BB':
        rows.append((None, "Bench Boost: the bench players score too."))
    return f"STARTING XI ({n['DEF']}-{n['MID']}-{n['FWD']})", rows


def _captain_text(row, chip):
    """The captain's own points and what they count for with the armband (x2, or x3 with Triple Captain)."""
    as_cap = f"about {half_up(3 * row.xp_now)} as Triple Captain" if chip == 'TC' else \
        f"about {half_up(2 * row.xp_now)} as captain"
    return f"{row.name} ({about(row.xp_now)} expected points, {as_cap})"


# --------------------------------------------------------------------------- C13: plan with the suggested chip
def _chip_plan(r):
    """(dict for the JSON, title, rows) of the plan with the suggested chip; (None, None, []) without one.
    No new number: everything comes from engine.ChipPlan (the same plan as a run with --chip X)."""
    cp = r.chip_plan
    if cp is None:
        return None, None, []
    title = f"WITH {CHIP_NAME[cp.chip].upper()}"
    if cp.status != 'computed':
        return ({'chip': cp.chip, 'status': cp.status}, title,
                [(None, f"Not computed in the time limit: run again with --chip {cp.chip} "
                        f"(or a longer --time-limit).")])
    pg, main = cp.plan[0], r.plan[0]
    t = _transfers(r, pg, cp.squad, cp.sales_now, vs_no_transfer=False)
    same_moves = [(m.out_id, m.in_id) for m in pg.transfers] == [(m.out_id, m.in_id) for m in main.transfers]
    same_xi = [(s.id, s.role, s.bench_slot) for s in cp.squad] == [(s.id, s.role, s.bench_slot) for s in r.squad]
    rows_by = {s.id: s for s in cp.squad}
    cap = rows_by[pg.captain_id]
    total, without = float(cp.expected_now.total), float(r.expected_now.total)
    rows = []
    if same_moves:
        rows.append(('Transfers', "same as above"))
    else:
        rows += _transfer_rows(r, t, pg.chip)
    xi_title, xi_rows = _xi_rows(cp.squad, pg.chip)
    if same_xi:
        rows.append(('Starting XI', "same as above"))
    else:
        rows.append(('Starting XI', xi_title.split('(')[1].rstrip(')')))     # the formation, e.g. 3-4-3
        rows += [(f"  {lb}" if lb else lb, v) for lb, v in xi_rows]
    rows.append(('Captain', _captain_text(cap, pg.chip)))
    if pg.vice_id is not None and pg.vice_id != main.vice_id:
        rows.append(('Vice-captain', pg.vice_name))
    rows.append(('Expected', f"about {half_up(total)} this gameweek (about {half_up(without)} without the chip)"))
    d = {'chip': cp.chip, 'status': cp.status, 'same_transfers': same_moves, 'same_lineup': same_xi,
         'transfers': t,
         'captain': {'id': int(cap.id), 'name': cap.name, 'expected_points': float(cap.xp_now)},
         'vice': {'id': int(pg.vice_id), 'name': pg.vice_name} if pg.vice_id is not None else None,
         'expected_points': total, 'expected_points_without': without}
    return d, title, rows


# --------------------------------------------------------------------------- decision check
def _decision(r):
    """15c step 4: the user's own decision (--check) in plain words: (dict for the JSON, rows for _layout).
    No new number: the gains come from decision.check_decision, the verdict from its codes.
    Gains in the decision's GW are unweighted (decision B); the gain over the horizon is the G of the
    check (later gameweeks count less) and is shown only when it differs after rounding."""
    d = r.decision
    if d is None:
        return None, []
    gw, x, h = d.gw, d.result or {}, r.horizon
    when = "this gameweek" if gw == r.current_gw else f"in GW{gw}"     # the decision's GW, in words
    name = lambda i: d.names.get(i, str(i))
    moves = []
    if d.chip == 'TRANSFERS':
        what, none_ = f"Transfers in GW{gw}", "no transfer"
        w = max(len(name(o)) for o in d.transfers_out)           # one move per line, OUT column aligned
        moves = [('', f"    {name(o):<{w}} -> {name(i)}") for o, i in zip(d.transfers_out, d.transfers_in)]
    else:
        on = f" on {d.captain_name}" if d.captain_id is not None else ""
        what, none_ = f"{CHIP_NAME[d.chip]}{on} in GW{gw}", "not playing it"
    out = {'chip': d.chip, 'gw': gw, 'status': d.status, 'verdict': list(d.verdict), 'gain_this_gw': None,
           'gain_horizon': x.get('g'), 'better_captain': None, 'better_gw': None, 'model_transfers': None,
           'violations': [v['code'] for v in d.violations]}
    rows = [('Decision', what)] + moves
    if d.status == 'invalid':
        for v in d.violations:
            msg = v['message'].replace(f"{d.chip} cannot", f"{CHIP_NAME.get(d.chip, d.chip)} cannot")
            rows.append(('Not allowed', msg[0].upper() + msg[1:] + "."))
        out['text'] = rows
        return out, rows
    for n in d.notes:                                      # facts behind the numbers, in plain words
        if n['code'] == 'captain_unavailable':
            rows.append(('Note', f"{d.captain_name} is injured or doubtful: he is expected to score 0, and the "
                                 f"armband may pass to your vice-captain."))
        elif n['code'] == 'captain_bought':
            rows.append(('Note', f"{d.captain_name} is not in your squad: the decision includes buying him in "
                                 f"GW{x.get('captain_bought')}."))
        elif n['code'] == 'player_unavailable':
            rows.append(('Note', n['message'].split(' is below')[0] + " is injured or doubtful: he is expected "
                                                                    "to score 0."))
        elif n['code'] == 'hits':
            k = len(d.transfers_in) - (r.ft or 0)
            rows.append(('Note', f"{k} extra transfer{'s' if k != 1 else ''}: -{HIT_COST * k} points, counted "
                                 f"in the gain."))
    pts = x.get('points_gw') or {}
    if 'z_decision' in pts and 'z0' in pts:
        now = pts['z_decision'] - pts['z0']
        out['gain_this_gw'] = float(now)
        gain = f"{signed(now)} points vs {none_} {when}"
        if x.get('g') is not None and signed(x['g']) != signed(now) and len(h) > 1:
            gain += f"; {signed(x['g'])} over GW{h[0]}-GW{h[-1]} (later gameweeks count less)"
        rows.append(('Gain', gain))
    best = x.get('plan_best')
    if d.chip == 'TC' and best is not None:
        if best['captain'] != d.captain_id and 'z_best' in pts and 'z_decision' in pts:
            more = pts['z_best'] - pts['z_decision']
            out['better_captain'] = {'id': int(best['captain']), 'name': name(best['captain']), 'more': float(more)}
            rows.append(('Better captain', f"{name(best['captain'])} ({signed(more)} points {when})"))
        else:
            best_when = "now" if gw == r.current_gw else f"in GW{gw}"
            rows.append(('Better captain', f"none ({d.captain_name} is the best Triple Captain {best_when})"))
    if d.chip == 'TRANSFERS' and best is not None and x.get('r') is not None:
        sell = ", ".join(name(i) for i in best['out']) or "nobody"
        buy = ", ".join(name(i) for i in best['in']) or "nobody"
        out['model_transfers'] = {'out': [int(i) for i in best['out']], 'in': [int(i) for i in best['in']],
                                  'more': float(x['r'])}
        if 'better_version' in d.verdict:
            rows.append(('Model\'s transfers', f"sell {sell}; buy {buy} ({signed(x['r'])} more over "
                                               f"GW{h[0]}-GW{h[-1]}, later gameweeks count less)"))
        else:
            rows.append(('Model\'s transfers', "the same, or no better set"))
    t = x.get('timing')
    if t is not None and t.get('best_gw') is not None:
        if t['is_best']:
            rows.append(('Better gameweek', f"none up to GW{t['window_end']} ({when}: {signed(t['at_gw'])})"))
        else:
            out['better_gw'] = {'gw': int(t['best_gw']), 'estimate': float(t['best']), 'at_gw': float(t['at_gw'])}
            rows.append(('Better gameweek', f"GW{t['best_gw']} is estimated higher ({signed(t['best'])} vs "
                                            f"{signed(t['at_gw'])} {when})"))
    rows.append(('Verdict', _verdict_words(d, x, out)))
    out['text'] = rows
    return out, rows


def _verdict_words(d, x, out):
    """One sentence per verdict code (decision.verdict), joined."""
    if not d.verdict:
        return "none: not every number was computed in the time limit. Run again with --time-limit 900."
    tr = d.chip == 'TRANSFERS'
    words = []
    for code in d.verdict:
        if code == 'agrees':
            words.append("make them: no better set of transfers was found." if tr else
                         "play it: no better captain or gameweek in the model." if d.chip == 'TC' else
                         "play it: no better gameweek in the model.")
        elif code == 'worse_than_none':
            words.append("don't: it is worse than making no transfer." if tr else
                         "don't: it is worse than not playing it. Keep the chip.")
        elif code == 'no_clear_gain':
            words.append("no clear gain: the difference is within the model's error.")
        elif code == 'better_gw':
            b = out['better_gw'] or {}
            words.append(f"a good chip, but GW{b.get('gw')} looks better. {NOISE}")
        elif code == 'better_version':
            if tr:
                words.append("the model's transfers gain more (see above).")
            elif out['better_captain']:
                words.append(f"captain {out['better_captain']['name']} gains more with the chip.")
    return " ".join(w[0].upper() + w[1:] for w in words)


# --------------------------------------------------------------------------- alerts
def _alerts(r):
    """Only what changes what the user should do (spec §0.5); the rest stays in --details."""
    codes = {w['code'] for w in r.warnings}
    out = []
    if 'previous_gw_not_final' in codes:
        out.append({'code': 'previous_gw_not_final', 'message':
                    f"GW{r.current_gw - 1} points are not final yet. Run again after they are (usually "
                    f"the morning after its last match) for better advice."})
    if not r.solver.proven:
        out.append({'code': 'time_limit', 'message':
                    "The search stopped at its time limit: the plan is valid but may not be the best. "
                    "Run again with --time-limit 900."})
    adv = r.chips.advice or {}
    if adv.get('incomplete'):
        out.append({'code': 'chips_not_computed', 'message':
                    "No advice for " + ", ".join(CHIP_NAME[c] for c in adv['incomplete'])
                    + ": not enough time. Run again with --time-limit 900."})
    for s in r.squad:
        if not s.available and s.action == 'KEEP':
            where = "on your bench" if s.role == 'Bench' else "in your starting XI"
            out.append({'code': 'player_unavailable', 'message': f"{s.name} is injured or doubtful and is {where}."})
    hi = r.squad_value_high                                  # None: selling value exact (or --override-budget)
    if hi is not None and r.squad_value is not None and tenths(hi) > tenths(r.squad_value):
        extra = (tenths(hi) - tenths(r.squad_value)) / 10
        out.append({'code': 'selling_value_range', 'message':
                    f"Some selling prices are known only to within {money(extra)}. Your budget may be "
                    f"{money(extra)} higher than shown. Check it in the app."})
    return out


# --------------------------------------------------------------------------- whole summary
def cap(text):
    """First letter upper case ('keep it' -> 'Keep it'); names and numbers stay as they are."""
    return text[:1].upper() + text[1:] if text[:1].islower() else text


def _layout(title, rows, align=True):
    """One section: '== TITLE ==', then its rows. (label, value) -> 'label: value'; align=True puts
    every value of the section in one column (the longest label sets it; a long value wraps under
    that column), align=False leaves one space after each label. (None, sentence) -> the sentence
    wrapped from the left margin; ('', line) -> the line as it is ('' = an empty line)."""
    w = max((len(lb) for lb, _ in rows if lb), default=0) + 2          # 'label:' plus one space
    out = [f"== {title} =="]
    for label, value in rows:
        if label == '':
            out.append(value)                                          # pre-aligned (transfer moves)
        elif label is None:
            out += textwrap.wrap(value, WIDTH)
        else:
            head = f"{label}:".ljust(w) if align else f"{label}: "
            out += textwrap.wrap(cap(value), WIDTH, initial_indent=head, subsequent_indent=" " * len(head))
    return out


def build_summary(r, tz=None):
    """engine.OptimizeResult -> dict (JSON field 'summary'); its 'text' holds the printed lines.
    tz: time zone of the deadline (None = the operating system's)."""
    g0, pg = r.current_gw, r.plan[0]
    chip = pg.chip
    rows = {s.id: s for s in r.squad}
    xi = [s for s in r.squad if s.role != 'Bench']           # r.squad: the XI by position, then the bench
    bench = [s for s in r.squad if s.role == 'Bench']        # already in bench order (GK first)
    lineup = {p: [s.name for s in xi if s.position == p] for p in ('GK', 'DEF', 'MID', 'FWD')}
    lineup['bench'] = [s.name for s in bench]
    cap = rows[pg.captain_id]
    deadline = r.deadline                                    # UTC, as the API gives it; None if missing
    total = float(r.expected_now.total)                # plain float: numpy types are not JSON
    s = {
        'gw': g0, 'deadline_utc': deadline,
        'transfers': _transfers(r, pg),
        'captain': {'id': int(cap.id), 'name': cap.name, 'expected_points': float(cap.xp_now)},
        'vice': {'id': int(pg.vice_id), 'name': pg.vice_name} if pg.vice_id is not None else None,
        'lineup': lineup,
        'chip': None,                                        # filled below (the chip states need it)
        'expected_points': total,
        'expected_range': [max(0.0, total + RANGE_LOW), total + RANGE_HIGH],
        'alerts': _alerts(r),
        'decision': None,                                    # --check: filled below
        'chip_plan': None,                                   # C13: filled below
    }
    decision, decision_rows = _decision(r)
    if decision is not None:
        s['decision'] = {k: v for k, v in decision.items() if k != 'text'}

    s['chip'] = _with_states(r, _chips(r))

    lines = [f"Fantasy Premier League plan for GW{g0}"]
    if deadline:
        lines.append(f"Deadline: {format_deadline(deadline, tz)} (your time)")
    if decision_rows:                                        # what the user asked for comes first
        lines += [""] + _layout("YOUR DECISION", decision_rows)
    lines += [""] + _layout("TRANSFERS", _transfer_rows(r, s['transfers'], chip), align=False)
    xi_title, xi_rows = _xi_rows(r.squad, chip)
    lines += [""] + _layout(xi_title, xi_rows)
    caps = [('C', _captain_text(cap, chip))]
    if s['vice'] is not None:
        caps.append(('VC', s['vice']['name']))
    lines += [""] + _layout("CAPTAINS", caps)
    chip_rows = _chip_rows(r, s['chip'])
    if chip_rows:
        lines += [""] + _layout("CHIPS", chip_rows)
    s['chip_plan'], cp_title, cp_rows = _chip_plan(r)             # C13
    if cp_rows:
        lines += [""] + _layout(cp_title, cp_rows)
    lo, hi = s['expected_range']
    lines += [""] + _layout("EXPECTED POINTS", [(None, f"About {half_up(total)} this gameweek"),
                                                ('In 8 of 10 gameweeks', f"between {half_up(lo)} and {half_up(hi)}")])
    if s['alerts']:
        lines += ["", "== BEFORE YOU DECIDE =="]
        for a in s['alerts']:
            for sentence in re.split(r"(?<=\.)\s+", a['message']):     # one sentence per line
                lines += textwrap.wrap(sentence, WIDTH, subsequent_indent="  ")
    lines += ["", "Full numbers: Run again with --details"]
    s['text'] = lines
    return s
