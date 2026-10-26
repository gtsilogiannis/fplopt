"""Plain-English summary of a run (chat 15c): "what do I do this gameweek", for readers with no
background in maths or programming.

build_summary() turns an engine.OptimizeResult into (1) a dict for the JSON output and (2) the
lines of text the command line prints. It computes NO new decision: every fact comes from the
result (S1), so the summary can never disagree with the full output (--details). Only rounding
and wording are added here.
"""
import math
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


def _transfers(r, pg):
    """Moves of the current GW with their money; None for the bank fields at GW1 (no squad yet)."""
    sell = {i: price for i, _, price, _ in r.sales_now}            # selling price of each sale
    cost = {s.id: s.cost for s in r.squad}                          # current price of each buy
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
        # unweighted gain over making no transfer (step 2 of 15c fills it in the engine)
        'vs_no_transfer': getattr(r, 'no_transfer', None),
    }


def _transfer_lines(r, t, chip):
    g0 = r.current_gw
    out = []
    if chip in ('WC', 'FH'):
        out.append(f"TRANSFERS  {CHIP_NAME[chip]}: unlimited free transfers this gameweek")
        if t['moves']:
            out.append("  Sell: " + ", ".join(m['out']['name'] for m in t['moves']))
            out.append("  Buy:  " + ", ".join(m['in']['name'] for m in t['moves']))
        if chip == 'FH':
            out.append(f"  After GW{g0} your squad goes back to the one you have now.")
        return out
    if t['unlimited']:
        out.append("TRANSFERS  unlimited this gameweek (your first deadline): build the squad below")
        return out
    if not t['moves']:
        ft = t['free_transfers']
        if ft is not None and ft >= MAX_FT:
            out.append(f"TRANSFERS  none. You already hold {MAX_FT} free transfers, the most you can keep:")
            out.append("  one is lost if you make no transfer.")
        elif ft is not None:
            out.append(f"TRANSFERS  none. Save your free transfer: you will have {ft + 1} next gameweek.")
        else:
            out.append("TRANSFERS  none this gameweek.")
        return out
    n_ft, hits = t['free_transfers'], t['hits']
    cost_txt = "no points cost" if hits == 0 else f"{hits} extra transfer(s): -{t['hit_points']} points"
    out.append(f"TRANSFERS  ({n_ft} free transfer{'s' if n_ft != 1 else ''} - {cost_txt})")
    w = max(len(m['out']['name']) for m in t['moves'])
    v = max(len(m['in']['name']) for m in t['moves'])
    for m in t['moves']:
        d = m['money']
        price = "" if d is None else ("same price" if d == 0 else
                                      (f"frees {money(d)}" if d > 0 else f"costs {money(d)} more"))
        out.append(f"  Sell {m['out']['name']:<{w}} -> buy {m['in']['name']:<{v}}   {price}".rstrip())
    if t['bank_before'] is not None and t['bank_after'] is not None:
        out.append(f"  Bank: {money(t['bank_before'])} now, {money(t['bank_after'])} after.")
    payers = [m['out']['name'] for m in t['moves'] if m['pays_for_others']]
    if payers:
        out += _wrap(f"Selling {' and '.join(payers)} is what makes the other purchases fit your budget.")
    vs = t['vs_no_transfer']
    if vs is not None:
        h = r.horizon
        out += _wrap(f"Compared with making no transfer: {signed(vs['this_gw'])} expected points this "
                     f"gameweek, {signed(vs['next_gws'])} over GW{h[0]}-GW{h[-1]} ({vs['n_gws']} gameweeks).")
    return out


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


def _chip_lines(r, ch):
    g0 = r.current_gw
    if ch['status'] == 'skipped_flag':
        return []
    if ch['playing'] is not None:
        return [f"CHIPS  You are playing {CHIP_NAME[ch['playing']]} in GW{g0}: the plan above uses it."]
    if ch['status'] == 'no_chip_now':
        return [f"CHIPS  No chip can be played in GW{g0}."]
    out = []
    if ch['play'] is not None:
        name = CHIP_NAME[ch['play']['chip']]
        out.append(f"CHIPS  Suggested: {name} this gameweek (gain about {signed(ch['play']['gain'])}).")
        out += _wrap(f"The plan above does not use it. To see the plan with the {name}, run again "
                     f"with --chip {ch['play']['chip']}.", indent=4)
        if ch['also']:
            out += _wrap("Also good now: " + ", ".join(CHIP_NAME[c] for c in ch['also'])
                         + " (only one chip per gameweek).", indent=4)
    else:
        out.append("CHIPS  No chip this gameweek.")
    for h in ch['hold']:
        now = f"Now about {signed(h['now'])}, best" if h['now'] is not None else "Best"
        out += _wrap(f"{CHIP_NAME[h['chip']]}: keep it. {now} in GW{h['best_gw']} ({signed(h['best'])}).")
    if ch['wildcard_now'] is not None:
        out += _wrap(f"Wildcard: gain now about {signed(ch['wildcard_now'])}; later gameweeks are not "
                     f"estimated, so the timing is your call.")
    for cp in ch['no_advice']:
        out += _wrap(f"{CHIP_NAME[cp]}: no advice (not enough time; see below).")
    if ch['play'] is not None or ch['hold'] or ch['wildcard_now'] is not None:
        out += _wrap(NOISE)
    return out


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
    hi = getattr(r, 'squad_value_high', None)                # step 2 of 15c fills it in the engine
    if hi is not None and r.squad_value is not None and tenths(hi) > tenths(r.squad_value):
        extra = (tenths(hi) - tenths(r.squad_value)) / 10
        out.append({'code': 'selling_value_range', 'message':
                    f"Some selling prices are known only to within {money(extra)}: your budget may be "
                    f"{money(extra)} higher than shown. Check it in the app."})
    return out


# --------------------------------------------------------------------------- whole summary
def _wrap(text, indent=2):
    return textwrap.wrap(text, WIDTH, initial_indent=" " * indent, subsequent_indent=" " * indent)


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
    deadline = getattr(r, 'deadline', None)                  # step 2 of 15c fills it in the engine
    total = float(r.expected_now.total)                # plain float: numpy types are not JSON
    s = {
        'gw': g0, 'deadline_utc': deadline,
        'transfers': _transfers(r, pg),
        'captain': {'id': int(cap.id), 'name': cap.name, 'expected_points': float(cap.xp_now)},
        'vice': {'id': int(pg.vice_id), 'name': pg.vice_name} if pg.vice_id is not None else None,
        'lineup': lineup,
        'chip': _chips(r),
        'expected_points': total,
        'expected_range': [max(0.0, total + RANGE_LOW), total + RANGE_HIGH],
        'alerts': _alerts(r),
        'decision': None,                                    # decision check: step 4 of 15c
    }

    def tag(row):
        """Plain words for a player's flags in the lineup."""
        t = []
        if not row.available:
            t.append("injured or doubtful")
        if row.n_matches_now == 0:
            t.append("no match this gameweek")
        elif row.n_matches_now >= 2:
            t.append("plays twice this gameweek")
        return f"{row.name} ({', '.join(t)})" if t else row.name

    head = f"FPL plan for GW{g0}"
    if deadline:
        head += f" - deadline {format_deadline(deadline, tz)} (your time)"
    lines = [head, ""]
    lines += _transfer_lines(r, s['transfers'], chip)
    lines.append("")
    mult = "tripled (Triple Captain)" if chip == 'TC' else "doubled"
    lines.append(f"CAPTAIN       {cap.name} (expected {about(cap.xp_now)} points, {mult})")
    if s['vice'] is not None:
        lines.append(f"VICE-CAPTAIN  {s['vice']['name']}")
    lines.append("")
    n = {p: len(lineup[p]) for p in ('DEF', 'MID', 'FWD')}
    lines.append(f"STARTING XI ({n['DEF']}-{n['MID']}-{n['FWD']})")
    for p in ('GK', 'DEF', 'MID', 'FWD'):
        names = ", ".join(tag(x) for x in xi if x.position == p)
        lines += textwrap.wrap(f"{p:<4} {names}", WIDTH, initial_indent="  ", subsequent_indent=" " * 7)
    lines.append("BENCH (in this order)" + (" - Bench Boost: they all score" if chip == 'BB' else ""))
    lines += _wrap(", ".join(tag(b) + (" (GK)" if b.pos_id == 1 else "") for b in bench))
    chip_lines = _chip_lines(r, s['chip'])
    if chip_lines:
        lines += [""] + chip_lines
    lo, hi = s['expected_range']
    lines += ["", f"EXPECTED POINTS  {about(total)} this gameweek",
              f"  (in 8 of 10 gameweeks: between {half_up(lo)} and {half_up(hi)})"]
    if s['alerts']:
        lines += ["", "BEFORE YOU DECIDE"]
        for a in s['alerts']:
            lines += _wrap(a['message'])
    lines += ["", "Full numbers: run again with --details"]
    s['text'] = lines
    return s
