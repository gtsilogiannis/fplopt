"""Offline harness (chat 8b): fake FPL API with a synthetic season, no network.
Not part of the optimizer; basis for the pytest suite of chat 9.
Put it next to fpl_v2.py. Usage: python harness_offline.py <scenario>
Scenarios: unit, fh, nofh, midseason, midmix, corrupt, full, full2, wc
FPLMOD=<module> runs another version of the optimizer on the same data (e.g. an old copy)."""
import sys, os, random, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import importlib, os as _o; F = importlib.import_module(_o.environ.get("FPLMOD", "fpl_v2"))
import fpl_v2 as NEW
if not hasattr(F, "selling_price"): F.selling_price = NEW.selling_price

random.seed(7)
CUR = 8                                  # next GW
TEAMS = list(range(1, 21))
POS_N = {1: 2, 2: 6, 3: 6, 4: 4}
if os.environ.get("HARNESS_BIG"):        # 6b timing: 35 players per team = 700 (real pool ~700)
    POS_N = {1: 4, 2: 11, 3: 13, 4: 7}

# ---- players and price paths -------------------------------------------------
elements, path = [], {}
pid = 0
for t in TEAMS:
    for pos, n in POS_N.items():
        for k in range(n):
            pid += 1
            base = {1: 45, 2: 45, 3: 55, 4: 60}[pos] + random.choice([0, 0, 5, 10, 25])
            slope = random.choice([-1, 0, 0, 1, 2])          # tenths per GW
            path[pid] = [base + slope * (g - 1) // 2 for g in range(1, 40)]   # price at GW g = path[g-1]
            elements.append({'id': pid, 'web_name': f"P{pid}", 'team': t, 'element_type': pos,
                             'chance_of_playing_next_round': None, 'ep_next': '2.0',
                             'points_per_game': '2.0', 'team_join_date': '2026-07-01'})
for e in elements:
    e['now_cost'] = path[e['id']][CUR - 1]
    e['cost_change_start'] = e['now_cost'] - path[e['id']][0]
price = lambda p, g: path[p][g - 1]
by = lambda pos, team: [e['id'] for e in elements if e['element_type'] == pos and e['team'] == team]

events = [{'id': g, 'is_next': g == CUR, 'finished': g < CUR,
           'deadline_time': (pd.Timestamp('2026-08-15T10:00:00Z') + pd.Timedelta(days=7 * (g - 1))).isoformat()} for g in range(1, 39)]
teams = [{'id': t, 'short_name': f"T{t}"} for t in TEAMS]
fixtures = []
for g in range(1, 39):
    order = TEAMS[:]; random.Random(g).shuffle(order)
    for a, b in zip(order[::2], order[1::2]):
        fixtures.append({'event': g, 'team_h': a, 'team_a': b, 'finished': g < CUR})

BIG = bool(os.environ.get("HARNESS_BIG"))
def minutes_choices(e):
    # 6b (big harness only): realistic roles. One nailed GK per team; outfield players
    # are nailed (55%), rotation (25%) or bench (20%). The small harness keeps the
    # original i.i.d. minutes so that the 8b checks stay comparable.
    if not BIG:
        return [0, 0, 25, 90, 90, 90]
    if e['element_type'] == 1:
        first_gk = min(i['id'] for i in elements if i['team'] == e['team'] and i['element_type'] == 1)
        return [90] * 19 + [0] if e['id'] == first_gk else [0] * 19 + [90]
    r = random.Random(e['id'] * 7919).random()
    if r < 0.55:
        return [90] * 8 + [70, 80, 0]
    if r < 0.80:
        return [0, 25, 60, 90, 90]
    return [0, 0, 0, 0, 10]

def live(g):
    out = []
    for e in elements:
        m = random.Random(e['id'] * 100 + g).choice(minutes_choices(e))
        s = {'minutes': m, 'expected_goals': f"{0.003 * m * e['element_type']:.2f}",
             'expected_assists': f"{0.002 * m:.2f}", 'expected_goals_conceded': f"{0.012 * m:.2f}",
             'defensive_contribution': m // 9, 'saves': m // 30 if e['element_type'] == 1 else 0,
             'bonus': 0, 'yellow_cards': 0, 'red_cards': 0, 'total_points': 2 if m >= 60 else (1 if m else 0)}
        out.append({'id': e['id'], 'stats': s})
    return {'elements': out}

# ---- manager: GW1 squad, transfers in GW3 (normal), GW5 (wildcard), GW6, GW7 (free hit) --
def build_manager(first_gw=1, fh_last=True, corrupt=False, mix=False):
    sq = by(1, 1)[:1] + by(1, 2)[:1] + [by(2, t)[0] for t in (3, 4, 5, 6, 7)] + \
         [by(3, t)[0] for t in (8, 9, 10, 11, 12)] + [by(4, t)[0] for t in (13, 14, 15)]
    purchase = {p: price(p, max(first_gw - 1, 1)) for p in sq}
    if mix:   # some players bought after a price change that happened between two matches
        moved = [p for p in sq if price(p, first_gw) != price(p, first_gw - 1)][:2]
        for p in moved:
            purchase[p] = price(p, first_gw - 1) + (1 if price(p, first_gw) > price(p, first_gw - 1) else -1)
        print("[harness] mixed purchase prices for", moved, [(price(p, first_gw-1), purchase[p], price(p, first_gw)) for p in moved])
    bank = 1000 - sum(purchase.values())
    picks, banks, transfers = {}, {}, []
    plan = {3: [(sq[5], by(2, 16)[0])],                                     # normal transfer
            5: [(sq[7], by(3, 17)[0]), (sq[13], by(4, 18)[0])],             # wildcard
            6: [(by(2, 16)[0], by(2, 19)[0])],                              # normal, sells the GW3 buy
            7: [(sq[2], by(2, 20)[1]), (sq[8], by(3, 20)[0]), (sq[9], by(3, 16)[0])]}  # free hit
    chips = [{'name': 'wildcard', 'event': 5}] + ([{'name': 'freehit', 'event': 7}] if fh_last else [])
    saved = None
    for g in range(first_gw, CUR):
        if g == 7 and fh_last:
            saved = (list(sq), dict(purchase), bank)                        # FH: remember to revert
        for out, inn in plan.get(g, []) if g > first_gw else []:
            sp = F.selling_price(purchase[out], price(out, g))
            bank += sp - price(inn, g)
            transfers.append({'element_in': inn, 'element_in_cost': price(inn, g), 'element_out': out,
                              'element_out_cost': sp, 'event': g, 'entry': 1,
                              'time': f"2026-09-{g:02d}T08:00:00Z"})
            sq[sq.index(out)] = inn
            del purchase[out]; purchase[inn] = price(inn, g)
        picks[g] = list(sq); banks[g] = bank
    if saved:
        sq, purchase, bank = saved                                         # reverted squad for GW8
    if corrupt:
        transfers[0]['element_out_cost'] += 3                              # breaks the bank check
    random.Random(1).shuffle(transfers)                                    # API order not guaranteed
    # 6b: per-GW transfer counts as in entry/{id}/history/ (no hits in this manager's history)
    n_tr = {g: (len(plan.get(g, [])) if g > first_gw else 0) for g in range(first_gw, CUR)}
    return dict(picks=picks, banks=banks, transfers=transfers, chips=chips,
                current=[{'event': g, 'event_transfers': n_tr[g], 'event_transfers_cost': 0}
                         for g in range(first_gw, CUR)], truth=(sq, purchase, bank))

MANAGER = None
CALLS = []
def fake(url):
    CALLS.append(url)
    u = url.replace(F.FPL_API, '')
    if u == '/bootstrap-static/':
        # chat 7: chip sets as in bootstrap-static 2026/27 (NOCHIPS=1 -> field missing, rule fallback)
        chips = [] if os.environ.get("NOCHIPS") else \
            [{'name': n, 'number': k, 'start_event': (s if k == 1 else 20), 'stop_event': (19 if k == 1 else 38)}
             for n, s in (('wildcard', 2), ('freehit', 2), ('bboost', 1), ('3xc', 1)) for k in (1, 2)]
        return {'elements': elements, 'events': events, 'teams': teams, 'chips': chips}
    if u == '/fixtures/':
        return fixtures
    if u.startswith('/event/') and u.endswith('/live/'):
        return live(int(u.split('/')[2]))
    if u.endswith('/history/') and u.startswith('/entry/'):
        return {'current': MANAGER['current'], 'chips': MANAGER['chips']}
    if u.endswith('/transfers/'):
        return MANAGER['transfers']
    if '/picks/' in u:
        g = int(u.split('/')[4])
        return {'picks': [{'element': p} for p in MANAGER['picks'][g]],
                'entry_history': {'bank': MANAGER['banks'][g]}, 'active_chip': None}
    if u.startswith('/element-summary/'):
        p = int(u.split('/')[2])
        dl = {e['id']: pd.Timestamp(e['deadline_time']) for e in events}
        return {'history': [{'round': g, 'value': price(p, g),
                             'kickoff_time': (dl[g] + pd.Timedelta(days=1)).isoformat()} for g in range(1, CUR)]}
    if u.startswith('/entry/') and u.count('/') == 3:
        g0 = MANAGER['current'][0]['event']
        dl = pd.Timestamp(events[g0 - 1]['deadline_time'])
        return {'joined_time': (dl - pd.Timedelta(days=2)).isoformat(), 'started_event': g0}
    raise KeyError(url)
F.fetch_fpl_api = fake

def check_squad(first_gw=1, fh_last=True, corrupt=False, mix=False):
    global MANAGER
    MANAGER = build_manager(first_gw, fh_last, corrupt, mix)
    CALLS.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sq, bank, sell = F.load_current_squad(1, CUR, {'elements': elements})
    t_sq, t_pur, t_bank = MANAGER['truth']
    exp = {p: F.selling_price(t_pur[p], price(p, CUR)) / 10 for p in t_sq}
    print(buf.getvalue().strip() or "(no messages)")
    print(f"squad matches truth: {sorted(sq) == sorted(t_sq)} | bank {bank} vs truth {t_bank / 10} | "
          f"selling prices match: {sell == exp} | sum sell {sum(sell.values()):.1f} vs truth {sum(exp.values()):.1f} | "
          f"never above truth: {all(sell[p] <= exp[p] + 1e-9 for p in t_sq)} | requests: {len(CALLS)}")
    gain = [(p, t_pur[p], price(p, CUR), sell[p]) for p in t_sq if price(p, CUR) != t_pur[p]][:4]
    print("examples (id, purchase, now, sell):", gain)

if __name__ == '__main__':
    sc = sys.argv[1]
    if sc == 'unit':
        for a, b in [(70, 73), (70, 74), (70, 70), (70, 68), (70, 71), (55, 56), (55, 57)]:
            print(f"selling_price({a}, {b}) = {F.selling_price(a, b)}")
    elif sc == 'fh':
        check_squad()
    elif sc == 'nofh':
        check_squad(fh_last=False)
    elif sc == 'midseason':
        check_squad(first_gw=2)
    elif sc == 'midmix':
        check_squad(first_gw=2, mix=True)
    elif sc == 'corrupt':
        check_squad(corrupt=True)
    elif sc in ('full', 'full2', 'wc', 'auto'):
        MANAGER = build_manager()
        orig_log = F.log_predictions
        def spy(gw, players, final_squad):
            spend = sum(p['cost'] for p in final_squad)
            print(f"[harness] spend of chosen squad (sell for KEEP, now for BUY) = {spend:.1f}")
            truth = set(MANAGER['truth'][0]); chosen = {p['id'] for p in final_squad}
            tp = MANAGER['truth'][1]
            real = sum(NEW.selling_price(tp[i], price(i, CUR)) / 10 if i in truth else price(i, CUR) / 10 for i in chosen)
            print(f"[harness] vs REAL squad: transfers needed = {len(chosen - truth)}, "
                  f"real spend = {real:.1f}, real budget = {sum(NEW.selling_price(tp[i], price(i, CUR)) for i in truth) / 10 + MANAGER['truth'][2] / 10:.1f}")
            orig_log(gw, players, final_squad)
        F.log_predictions = spy
        F.optimize_team(1, None if sc == 'auto' else (2 if sc == 'full2' else 1), 'WC' if sc == 'wc' else 'NONE', 75, 3, None, 1, None)
        print("requests:", len(CALLS), "| truth bank:", MANAGER['truth'][2] / 10)
