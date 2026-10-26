"""The manager's current situation, rebuilt from the public API: squad, bank and selling
prices (S1-S5, R1), free transfers (M8) and the chip sets still available (C6)."""
import logging

import pandas as pd

from .rules import CHIP_API_NAME, CHIP_RULE_WINDOWS, CHIPS, HIT_COST, MAX_FT

log = logging.getLogger(__name__)


def selling_price(purchase, now):
    """S1: FPL sell-on rule (2026/27): a price rise is shared 50/50 with the game,
    rounded DOWN to GBP 0.1m; a price fall is passed on in full.
    Arguments and result in tenths of GBPm (integers, as in the API): 70 = GBP 7.0m.
    Example: bought 70, now 73 -> profit 3 -> 3 // 2 = 1 -> sells for 71."""
    if now > purchase:
        return purchase + (now - purchase) // 2   # '//' = integer (floor) division
    return now


def load_current_squad(client, team_id, current_gw, bootstrap, history=None):
    """
    Squad, bank and selling prices that the next transfers start from.
    R1: a Free Hit squad reverts at the next deadline. If the Free Hit was played in
        GW-1, the picks of GW-1 are the TEMPORARY squad; the real one is GW-2's.
    S1: the public API has no selling prices (my-team/ needs a login), so purchase
        prices are rebuilt: the picks of the manager's first GW, then every later
        transfer in time order (element_in_cost = purchase price), skipping
        Free Hit transfers because they are reverted.
    Returns (list of player ids, bank in GBPm, {player id: selling price in GBPm},
             {player id: selling price from the upper bound of an unknown purchase price}).
    """
    if history is None:   # M8: the engine passes it (also used for the free transfers)
        history = client.entry_history(team_id)
    # chips: [{'name': 'freehit', 'event': 5, ...}, ...]
    fh_gws = {c['event'] for c in history.get('chips', []) if c.get('name') == 'freehit'}
    first_gw = history['current'][0]['event'] if history.get('current') else 1  # GW the team started

    squad_gw = current_gw - 1
    if squad_gw in fh_gws and squad_gw - 1 >= first_gw:
        squad_gw -= 1   # R1: GW-1 was a Free Hit -> the squad reverts to the one of GW-2
        log.info(f"[*] Free Hit played in GW{current_gw - 1}: squad and bank taken from GW{squad_gw}.")
    picks = client.picks(team_id, squad_gw)
    squad = [p['element'] for p in picks['picks']]
    bank = picks['entry_history']['bank']          # tenths of GBPm

    now_cost = {p['id']: p['now_cost'] for p in bootstrap['elements']}
    start_price = {p['id']: p['now_cost'] - p['cost_change_start'] for p in bootstrap['elements']}
    first_picks = picks if first_gw == squad_gw else \
        client.picks(team_id, first_gw)   # no second request if same GW
    first_squad = [p['element'] for p in first_picks['picks']]
    first_bank = first_picks['entry_history']['bank']
    # S5: the EXACT total paid for the first squad is known: every team starts with
    # GBP 100.0m, so total = 1000 - bank of the first GW. Each player's own price is
    # bracketed: lo <= purchase price <= hi.
    total = 1000 - first_bank
    if first_gw == 1:
        # Prices do not change before the season starts (premierleague.com), so a
        # GW1 squad was bought at the start price = now_cost - cost_change_start.
        lo = {pid: start_price.get(pid, now_cost.get(pid, 0)) for pid in first_squad}
        hi = dict(lo)
    else:
        # S4/S5: team created mid-season ('joined_time' of entry/{id}/). Price history
        # exists only per match (element-summary 'value'), so the price at creation is
        # between his last match BEFORE it (lo) and his first match AFTER it (hi).
        # Real run (team created 2026-09-01, first GW3): sum(lo) = 996, sum(hi) = 1000,
        # exact total 998. [assumption] the price moved one way between the two matches.
        joined = pd.Timestamp(client.entry(team_id)['joined_time'])
        lo, hi = {}, {}
        for pid in first_squad:
            rows = sorted((h for h in client.element_summary(pid)['history']
                           if h.get('kickoff_time')), key=lambda h: pd.Timestamp(h['kickoff_time']))
            before = [h['value'] for h in rows if pd.Timestamp(h['kickoff_time']) <= joined]
            after = [h['value'] for h in rows if pd.Timestamp(h['kickoff_time']) > joined]
            p0 = before[-1] if before else start_price.get(pid, now_cost.get(pid, 0))   # no match yet -> start price
            p1 = after[0] if after else p0
            lo[pid], hi[pid] = min(p0, p1), max(p0, p1)   # a falling price swaps the bounds
    # If the exact total equals the sum of one bound, every player is AT that bound:
    # sum(price - lo) = 0 with every term >= 0 forces price = lo (same for hi).
    if sum(lo.values()) == total:
        hi = dict(lo)
    elif sum(hi.values()) == total:
        lo = dict(hi)
    elif not sum(lo.values()) <= total <= sum(hi.values()):
        log.warning(f"[WARNING] First-squad prices (GBP {sum(lo.values()) / 10:.1f}-{sum(hi.values()) / 10:.1f}m) "
                    f"do not contain the amount actually paid (GBP {total / 10:.1f}m); selling prices may be off.",
                    extra={'code': 'first_squad_prices'})

    # Replay the transfers on both bounds (a transfer sets an EXACT purchase price).
    pur_lo, pur_hi = dict(lo), dict(hi)
    replay_bank = first_bank
    transfers = client.transfers(team_id)
    for t in sorted(transfers, key=lambda t: t['time']):   # oldest first (ISO time strings sort correctly)
        # event <= first_gw: already in the first picks; > squad_gw: after the squad we use;
        # Free Hit GW: reverted. Wildcard transfers are permanent and are kept.
        if t['event'] <= first_gw or t['event'] > squad_gw or t['event'] in fh_gws:
            continue
        for pur in (pur_lo, pur_hi):
            pur.pop(t['element_out'], None)        # sold: forget his purchase price
            pur[t['element_in']] = t['element_in_cost']
        replay_bank += t['element_out_cost'] - t['element_in_cost']   # out_cost = what the sale returned

    # Checks: the rebuilt squad and bank must match the API (S5: this now checks the
    # transfers alone, because it starts from the API's bank of the first GW).
    if set(pur_lo) != set(squad):
        log.warning(f"[WARNING] Transfer replay does not match the GW{squad_gw} squad; players without a "
                    f"known purchase price are valued at their current price.", extra={'code': 'replay_squad_mismatch'})
    if replay_bank != bank:
        log.warning(f"[WARNING] Rebuilt bank £{replay_bank / 10:.1f}m != API bank £{bank / 10:.1f}m: "
                    f"selling prices may be off. Check them in the app; use --override-budget if needed.",
                    extra={'code': 'bank_mismatch'})

    def sell_of(pur):   # {id: selling price in GBPm}; unknown purchase price -> current price
        return {pid: selling_price(pur.get(pid, now_cost.get(pid, 0)), now_cost.get(pid, 0)) / 10.0
                for pid in squad}
    sell, sell_hi = sell_of(pur_lo), sell_of(pur_hi)
    if sum(sell_hi.values()) > sum(sell.values()) + 1e-9:
        # S5: the selling price never decreases when the purchase price increases, so
        # the lower bound gives the lower budget. Using it never suggests a transfer
        # the game would refuse; at worst it misses one that needs the extra money.
        name = {p['id']: p['web_name'] for p in bootstrap['elements']}
        unsure = ", ".join(name.get(pid, str(pid)) for pid in squad if sell_hi[pid] != sell[pid])
        log.info(f"[*] Selling value of the squad is between £{sum(sell.values()):.1f}m and "
                 f"£{sum(sell_hi.values()):.1f}m (purchase price not exactly known: {unsure}). "
                 f"Using the lower value; --override-budget sets the exact one from the app.",
                 extra={'code': 'selling_value_range'})    # 15c: shown in the summary too
    # sell_hi: selling prices from the upper bound of the purchase prices (= sell when all are known)
    return squad, bank / 10.0, sell, sell_hi


def free_transfers_from_history(history, current_gw):
    """M8: free transfers available for current_gw, rebuilt from entry/{id}/history/.
    F(first GW + 1) = 1; then, for each later GW g before current_gw:
      Wildcard / Free Hit in g:  F(g+1) = F(g)
      otherwise:                 F(g+1) = min(5, max(0, F(g) - T(g)) + 1)
    where T(g) = event_transfers. Check: event_transfers_cost must equal
    HIT_COST * max(0, T(g) - F(g)); a mismatch means the rebuilt count is wrong.
    Returns (ft or None, list of mismatch messages). None = transfers are still
    unlimited (the team's first deadline has not passed)."""
    rows = {r['event']: r for r in history.get('current', [])}
    if not rows:
        return None, []
    first_gw = min(rows)
    if current_gw <= first_gw:
        return None, []
    chip_at = {c['event']: c.get('name') for c in history.get('chips', [])}
    ft, issues = 1, []
    for g in range(first_gw + 1, current_gw):
        r = rows.get(g, {})
        t = int(r.get('event_transfers') or 0)
        if chip_at.get(g) in ('wildcard', 'freehit'):
            continue                                   # FTs kept, no new one (ft unchanged)
        cost = int(r.get('event_transfers_cost') or 0)
        if cost != HIT_COST * max(0, t - ft):
            issues.append(f"GW{g}: {t} transfers with {ft} FT should cost {HIT_COST * max(0, t - ft)}, "
                          f"API says {cost}")
        ft = min(MAX_FT, max(0, ft - t) + 1)
    return ft, issues


def available_chips(bootstrap, history, current_gw):
    """C6: chip sets not played yet, from two endpoints:
      bootstrap-static 'chips': one entry per chip set: name ('wildcard', 'freehit', 'bboost',
          '3xc'), number (set 1 or 2), start_event, stop_event;
      entry/{id}/history/ 'chips': one entry per chip played: name, event.
    A set is used if a chip of that name was played inside its window.
    Returns ({chip: [(first GW still playable, stop GW), ...]}, rows for the output, warnings).
    If bootstrap-static has no usable 'chips' list, the rule windows of CHIP_RULE_WINDOWS are used."""
    played = [(ch.get('name'), ch.get('event')) for ch in (history or {}).get('chips', [])]
    code_of = {v: k for k, v in CHIP_API_NAME.items()}          # 'wildcard' -> 'WC', ...
    sets, warnings = [], []
    for ch in bootstrap.get('chips') or []:
        if not isinstance(ch, dict) or ch.get('name') not in code_of:
            continue                                            # other chips (e.g. an old 'manager')
        if ch.get('start_event') is None or ch.get('stop_event') is None:
            continue
        sets.append((code_of[ch['name']], int(ch['start_event']), int(ch['stop_event'])))
    source = 'API (bootstrap-static chips)'
    if not sets:
        warnings.append("bootstrap-static has no 'chips' list with start_event/stop_event; "
                        "using the 2026/27 rule windows (GW1/2-19, GW20-38)")
        sets = [(cp, s, e) for cp, ws in CHIP_RULE_WINDOWS.items() for s, e in ws]
        source = 'rule windows (CHIP_RULE_WINDOWS)'
    windows, rows = {}, []
    for cp, s, e in sorted(sets, key=lambda r: (CHIPS.index(r[0]), r[1])):
        used = [ev for n, ev in played if n == CHIP_API_NAME[cp] and ev is not None and s <= ev <= e]
        if used and used[0] >= current_gw:
            # played for the current GW already (before its deadline): the optimizer cannot see
            # the transfers made with it, so the user should pass it with --chip
            warnings.append(f"{cp} is already active for GW{used[0]}; pass --chip {cp}")
        if used:
            state = f"played GW{used[0]}"
        elif e < current_gw:
            state = "expired"
        else:
            state = "available"
            windows.setdefault(cp, []).append((max(s, current_gw), e))
        rows.append((cp, s, e, state))
    return windows, rows, warnings, source
