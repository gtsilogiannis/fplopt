"""Synthetic FPL season and a fake API for offline tests (no network).

Port of claude/harness_offline.py (chat 8b). The data are generated with the same random
sequence (seed 7), so every number the harness printed is reproduced exactly; the
reference values are in claude/baseline_09a.txt.

Season: 20 teams, GW8 is the next gameweek (GW1-7 finished). Small pool: 18 players per
team (360). big=True: 35 per team (700, the real pool size) with realistic minutes roles.

Manager (team id 1): GW1 squad, one transfer in GW3, Wildcard in GW5 (2 transfers), one
transfer in GW6, Free Hit in GW7 (3 transfers, reverted for GW8).
"""
import random

import pandas as pd

CUR = 8                                    # next gameweek
TEAMS = list(range(1, 21))
POS_N_SMALL = {1: 2, 2: 6, 3: 6, 4: 4}     # players per team and position
POS_N_BIG = {1: 4, 2: 11, 3: 13, 4: 7}


def sell_rule(purchase, now):
    """FPL sell-on rule, written independently of the code under test (it is the oracle
    the tests compare with): half of a rise, rounded down to 0.1m; a fall in full."""
    return purchase + (now - purchase) // 2 if now > purchase else now


class Season:
    """Players, prices, fixtures and per-GW live data of the synthetic season."""

    def __init__(self, big=False, chips=True, data_checked=None):
        """data_checked: None = field absent from the gameweeks (as in the harness), True/False = set
        for the finished ones (the cache stores only finished AND data_checked gameweeks)."""
        rng = random.Random(7)             # same sequence as random.seed(7) in the harness
        self.big = big
        self.with_chips = chips
        pos_n = POS_N_BIG if big else POS_N_SMALL
        self.elements, self.path = [], {}
        pid = 0
        for t in TEAMS:
            for pos, n in pos_n.items():
                for _ in range(n):
                    pid += 1
                    base = {1: 45, 2: 45, 3: 55, 4: 60}[pos] + rng.choice([0, 0, 5, 10, 25])
                    slope = rng.choice([-1, 0, 0, 1, 2])            # tenths per GW
                    self.path[pid] = [base + slope * (g - 1) // 2 for g in range(1, 40)]
                    self.elements.append({'id': pid, 'web_name': f"P{pid}", 'team': t, 'element_type': pos,
                                          'chance_of_playing_next_round': None, 'ep_next': '2.0',
                                          'points_per_game': '2.0', 'team_join_date': '2026-07-01'})
        for e in self.elements:
            e['now_cost'] = self.path[e['id']][CUR - 1]
            e['cost_change_start'] = e['now_cost'] - self.path[e['id']][0]
        self.events = [{'id': g, 'is_next': g == CUR, 'finished': g < CUR,
                        'deadline_time': (pd.Timestamp('2026-08-15T10:00:00Z')
                                          + pd.Timedelta(days=7 * (g - 1))).isoformat()}
                       for g in range(1, 39)]
        if data_checked is not None:
            for e in self.events:
                e['data_checked'] = bool(data_checked) and e['finished']
        self.teams = [{'id': t, 'short_name': f"T{t}"} for t in TEAMS]
        self.fixtures = []
        for g in range(1, 39):
            order = TEAMS[:]
            random.Random(g).shuffle(order)
            for a, b in zip(order[::2], order[1::2]):
                self.fixtures.append({'event': g, 'team_h': a, 'team_a': b, 'finished': g < CUR})

    def price(self, pid, g):
        return self.path[pid][g - 1]

    def by(self, pos, team):
        return [e['id'] for e in self.elements if e['element_type'] == pos and e['team'] == team]

    def minutes_choices(self, e):
        # small pool: i.i.d. minutes (as in chat 8b); big pool: one nailed GK per team and
        # outfield players nailed (55%), rotation (25%) or bench (20%) (chat 6b)
        if not self.big:
            return [0, 0, 25, 90, 90, 90]
        if e['element_type'] == 1:
            first_gk = min(i['id'] for i in self.elements if i['team'] == e['team'] and i['element_type'] == 1)
            return [90] * 19 + [0] if e['id'] == first_gk else [0] * 19 + [90]
        r = random.Random(e['id'] * 7919).random()
        if r < 0.55:
            return [90] * 8 + [70, 80, 0]
        if r < 0.80:
            return [0, 25, 60, 90, 90]
        return [0, 0, 0, 0, 10]

    def live(self, g):
        out = []
        for e in self.elements:
            m = random.Random(e['id'] * 100 + g).choice(self.minutes_choices(e))
            s = {'minutes': m, 'expected_goals': f"{0.003 * m * e['element_type']:.2f}",
                 'expected_assists': f"{0.002 * m:.2f}", 'expected_goals_conceded': f"{0.012 * m:.2f}",
                 'defensive_contribution': m // 9, 'saves': m // 30 if e['element_type'] == 1 else 0,
                 'bonus': 0, 'yellow_cards': 0, 'red_cards': 0,
                 'total_points': 2 if m >= 60 else (1 if m else 0)}
            out.append({'id': e['id'], 'stats': s})
        return {'elements': out}

    def bootstrap(self):
        # chip sets as in bootstrap-static 2026/27; chips=False -> field missing (rule fallback)
        chips = [] if not self.with_chips else \
            [{'name': n, 'number': k, 'start_event': (s if k == 1 else 20), 'stop_event': (19 if k == 1 else 38)}
             for n, s in (('wildcard', 2), ('freehit', 2), ('bboost', 1), ('3xc', 1)) for k in (1, 2)]
        return {'elements': self.elements, 'events': self.events, 'teams': self.teams, 'chips': chips}


def build_manager(season, first_gw=1, fh_last=True, corrupt=False, mix=False):
    """History of the test manager; returns the API views and the TRUE squad for GW8."""
    by, price = season.by, season.price
    sq = by(1, 1)[:1] + by(1, 2)[:1] + [by(2, t)[0] for t in (3, 4, 5, 6, 7)] + \
        [by(3, t)[0] for t in (8, 9, 10, 11, 12)] + [by(4, t)[0] for t in (13, 14, 15)]
    purchase = {p: price(p, max(first_gw - 1, 1)) for p in sq}
    mixed = None
    if mix:   # some players bought after a price change that happened between two matches
        moved = [p for p in sq if price(p, first_gw) != price(p, first_gw - 1)][:2]
        for p in moved:
            purchase[p] = price(p, first_gw - 1) + (1 if price(p, first_gw) > price(p, first_gw - 1) else -1)
        mixed = (moved, [(price(p, first_gw - 1), purchase[p], price(p, first_gw)) for p in moved])
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
            sp = sell_rule(purchase[out], price(out, g))
            bank += sp - price(inn, g)
            transfers.append({'element_in': inn, 'element_in_cost': price(inn, g), 'element_out': out,
                              'element_out_cost': sp, 'event': g, 'entry': 1,
                              'time': f"2026-09-{g:02d}T08:00:00Z"})
            sq[sq.index(out)] = inn
            del purchase[out]
            purchase[inn] = price(inn, g)
        picks[g] = list(sq)
        banks[g] = bank
    if saved:
        sq, purchase, bank = saved                                         # reverted squad for GW8
    if corrupt:
        transfers[0]['element_out_cost'] += 3                              # breaks the bank check
    random.Random(1).shuffle(transfers)                                    # API order not guaranteed
    n_tr = {g: (len(plan.get(g, [])) if g > first_gw else 0) for g in range(first_gw, CUR)}
    return dict(picks=picks, banks=banks, transfers=transfers, chips=chips, mixed=mixed,
                current=[{'event': g, 'event_transfers': n_tr[g], 'event_transfers_cost': 0}
                         for g in range(first_gw, CUR)], truth=(sq, purchase, bank))


class FakeApi:
    """Answers the FPL API paths the optimizer requests ('/bootstrap-static/', ...) from the
    synthetic season and manager. Records every path it was asked for."""

    def __init__(self, season, manager):
        self.season, self.manager = season, manager
        self.calls = []

    def respond(self, path):
        self.calls.append(path)
        s, m = self.season, self.manager
        if path == '/bootstrap-static/':
            return s.bootstrap()
        if path == '/fixtures/':
            return s.fixtures
        if path.startswith('/event/') and path.endswith('/live/'):
            return s.live(int(path.split('/')[2]))
        if path.endswith('/history/') and path.startswith('/entry/'):
            return {'current': m['current'], 'chips': m['chips']}
        if path.endswith('/transfers/'):
            return m['transfers']
        if '/picks/' in path:
            g = int(path.split('/')[4])
            return {'picks': [{'element': p} for p in m['picks'][g]],
                    'entry_history': {'bank': m['banks'][g]}, 'active_chip': None}
        if path.startswith('/element-summary/'):
            p = int(path.split('/')[2])
            dl = {e['id']: pd.Timestamp(e['deadline_time']) for e in s.events}
            return {'history': [{'round': g, 'value': s.price(p, g),
                                 'kickoff_time': (dl[g] + pd.Timedelta(days=1)).isoformat()} for g in range(1, CUR)]}
        if path.startswith('/entry/') and path.count('/') == 3:
            g0 = m['current'][0]['event']
            dl = pd.Timestamp(s.events[g0 - 1]['deadline_time'])
            return {'joined_time': (dl - pd.Timedelta(days=2)).isoformat(), 'started_event': g0}
        raise KeyError(path)
