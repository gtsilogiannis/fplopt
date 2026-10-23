"""W1: rolling-horizon backtest of the whole optimizer on a past season.

At every gameweek t the optimizer sees only what was known before the deadline of t, plans
gameweeks t .. t+L-1, and only the decisions of t are applied (squad, transfers, XI, bench order,
captain, vice-captain, chip). The REAL points of t are then counted with the game's rules
(autosubs, vice-captain, chips, hits) and the next gameweek starts from the new state.

    python scripts/backtest_season.py --season 2024-25 --lookahead 3 --out bt_2425_L3.csv
    python scripts/backtest_season.py --season 2025-26 --first-gw 2 --last-gw 6     (quick check)
    python scripts/backtest_season.py --compare bt_a.csv bt_b.csv                  (paired difference)

How the past is replayed (the optimizer itself is not changed: it is fed a fake FPL API that
answers from the vaastav data, github.com/vaastav/Fantasy-Premier-League):
  - rules: the 2026/27 rules of the code (two chip sets, GW1-19 and GW20-38) in BOTH seasons;
    the player points of the data do not depend on the chip rules.
  - start: a new team before the GW2 deadline (GBP 100.0m, unlimited transfers), the same for
    every setting. GW1 is not played: the xP model needs one finished GW.
  - price at the deadline of GW g: the player's 'value' in his first row of GW g (no row: the
    last known value) [assumption: prices rarely move between a deadline and the matches].
  - availability: FPL's own projection for GW t (the dataset's xP of row t-1) equal to 0 means
    flagged out (chance 0); the proxy of check_papp.py. A team without a match in GW t is not
    flagged (its projection is 0 because of the blank). GWs without the projection in the data
    (most of 2025/26) flag nobody.
  - club join date: the first GW of the player's current run of rows at his club, from the rows
    before t only (no end-of-season data).
  - fixtures: the final schedule of the season [assumption: postponements are known from the
    start; in reality some blanks/doubles were announced later].
  - the solver runs without a time limit that could cut it short (--time-limit default 10^6 s):
    each solve stops at the gap target, every bench-weight pass and chip solve is done, so the
    result does not depend on the speed or load of the machine (measured: four runs of the
    synthetic season gave identical plans, objectives, gaps and chip gains).
Output: one CSV row per GW (points, hits, chip, captain, expected points of the plan, ...).
"""
import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate_xp import download                                   # noqa: E402  (same cache)

from fplopt.api import FplClient                                    # noqa: E402
from fplopt.engine import Options, optimize                         # noqa: E402
from fplopt.rules import CHIP_API_NAME, CHIP_RULE_WINDOWS, HIT_COST  # noqa: E402
from fplopt.squad import selling_price                              # noqa: E402

POS = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}    # 2024/25 also has 'AM' rows: dropped
NO_LIMIT = 10 ** 6                               # seconds: in practice no time limit


# --------------------------------------------------------------------------- data
class SeasonData:
    """Everything the fake API needs, from vaastav's merged_gw.csv, fixtures.csv and teams.csv."""

    def __init__(self, season):
        d = pd.read_csv(download(season, 'gws/merged_gw.csv'))
        d = d[d['position'].isin(POS)].copy()
        teams = pd.read_csv(download(season, 'teams.csv'))
        fx = pd.read_csv(download(season, 'fixtures.csv'))
        team_id = dict(zip(teams['name'], teams['id']))
        d['team_id'] = d['team'].map(team_id)
        if d['team_id'].isna().any():
            raise ValueError(f"unknown team names: {sorted(d.loc[d['team_id'].isna(), 'team'].unique())}")
        if 'defensive_contribution' not in d:                # DefCon exists only from 2025/26
            d['defensive_contribution'] = 0
        d['kickoff'] = pd.to_datetime(d['kickoff_time'], utc=True)
        d = d.sort_values(['element', 'GW', 'kickoff'])
        self.season, self.d = season, d
        self.teams = [{'id': int(i), 'short_name': s, 'name': n}
                      for i, s, n in zip(teams['id'], teams['short_name'], teams['name'])]
        fx = fx[fx['event'].notna()].copy()
        fx['kickoff'] = pd.to_datetime(fx['kickoff_time'], utc=True)
        self.fixtures = [{'event': int(r.event), 'team_h': int(r.team_h), 'team_a': int(r.team_a),
                          'kickoff': r.kickoff} for r in fx.itertuples()]
        # deadline of GW g ~ 90 minutes before its first kickoff [assumption]
        self.deadline = {g: k - pd.Timedelta(minutes=90) for g, k in fx.groupby('event')['kickoff'].min().items()}
        self.n_matches = {}                                  # (team, GW) -> matches
        for f in self.fixtures:
            for t in (f['team_h'], f['team_a']):
                self.n_matches[(t, f['event'])] = self.n_matches.get((t, f['event']), 0) + 1
        g1 = d.groupby(['element', 'GW'])
        first = g1.first()                                   # first row of each (player, GW)
        # price per (player, GW) in tenths; GWs without a row take the last known value,
        # and GWs before the player's first row the first value (only used as a fallback)
        pr = first['value'].unstack().reindex(columns=range(1, 39))
        self.price = pr.ffill(axis=1).bfill(axis=1).astype(int)
        self.ep = first['xP']                                # FPL projection recorded with the row
        # GWs whose rows carry FPL's projection: in vaastav's 2025/26 file the column is 0 for every
        # player in most GWs (checked 2026-10-23: only GW1-6, 8, 9, 24, 29, 38 have it; 2024/25
        # lacks GW22, 32, 34). Same test as calibrate_xp.py: mean projection of the GW > 0.5.
        self.has_ep = {int(g) for g, v in d.groupby('GW')['xP'].mean().items() if v > 0.5}
        agg = g1.agg(minutes=('minutes', 'sum'), points=('total_points', 'sum'))
        self.minutes = agg['minutes'].to_dict()              # (player, GW) -> minutes (DGW: sum)
        self.points = agg['points'].to_dict()                # (player, GW) -> FPL points
        self.pos = d.groupby('element')['position'].first().map(POS).to_dict()
        self.name = d.groupby('element')['name'].first().to_dict()
        self.first_gw = d.groupby('element')['GW'].min().to_dict()
        # (player, GW) -> team of his last row in that GW
        self.team_at = g1['team_id'].last().astype(int).to_dict()
        self.gws_of = d.groupby('element')['GW'].apply(lambda s: sorted(set(s))).to_dict()
        # per-GW live stats, the fields fplopt.data.fetch_history reads
        stat_cols = ['minutes', 'total_points', 'expected_goals', 'expected_assists',
                     'expected_goals_conceded', 'defensive_contribution', 'saves', 'bonus',
                     'yellow_cards', 'red_cards']
        live = d.groupby(['GW', 'element'])[stat_cols].sum()
        self.live = {g: [{'id': int(pid), 'stats': {c: (float(v) if c.startswith('expected') else int(v))
                                                     for c, v in zip(stat_cols, row)}}
                         for pid, row in zip(df.index.get_level_values(1), df.itertuples(index=False))]
                     for g, df in live.groupby(level=0)}

    def team_of(self, pid, t):
        """Club of the player at the deadline of GW t: his last row before or in t."""
        past = [g for g in self.gws_of[pid] if g <= t]
        return self.team_at[(pid, past[-1] if past else self.gws_of[pid][0])]

    def join_gw(self, pid, t):
        """First GW of the player's current run at his club, from rows before t only."""
        club = self.team_of(pid, t)
        past = [g for g in self.gws_of[pid] if g < t]
        start = None
        for g in reversed(past):
            if self.team_at[(pid, g)] != club:
                break
            start = g
        return start

    def chance(self, pid, t):
        """Availability proxy for GW t: None (no news) or 0 (flagged out). Without FPL's projection
        for GW t in the data (see has_ep) nobody is flagged: the model then only learns of an
        injury from the minutes played [assumption: no better source without future data]."""
        if t - 1 not in self.has_ep or self.n_matches.get((self.team_of(pid, t), t), 0) == 0:
            return None                                      # no data, or blank GW (projection 0 is not news)
        ep = self.ep.get((pid, t - 1))
        return 0 if ep is not None and not pd.isna(ep) and ep == 0 else None

    def ep_next(self, pid, t, ppg):
        """FPL's projection for GW t (row t-1); where the data lacks it: points per game x matches
        in GW t [assumption]. The optimizer uses it only for players with no GW at their club."""
        ep = self.ep.get((pid, t - 1))
        if t - 1 in self.has_ep and ep is not None and not pd.isna(ep):
            return float(ep)
        return ppg * self.n_matches.get((self.team_of(pid, t), t), 0)


# --------------------------------------------------------------------------- manager state
class Manager:
    """The simulated team, in the form the FPL API reports it (picks, bank, transfers, history)."""

    def __init__(self):
        self.picks, self.banks, self.transfers, self.current, self.chips = {}, {}, [], [], []
        self.squad, self.purchase, self.bank = [], {}, 1000   # tenths of GBPm
        self.saved = None                                      # (squad, purchase, bank) during a FH

    def apply(self, data, g, plan_gw):
        """Applies the decisions of GW g (plan_gw = OptimizeResult.plan[0])."""
        if self.saved is not None:                             # last GW was a Free Hit: revert
            self.squad, self.purchase, self.bank = self.saved
            self.saved = None
        chip = plan_gw.chip
        if chip == 'FH':
            self.saved = (list(self.squad), dict(self.purchase), self.bank)
        new = list(plan_gw.squad)
        out = [p for p in self.squad if p not in new]
        inn = [p for p in new if p not in self.squad]
        if self.squad and len(out) != len(inn):
            raise RuntimeError(f"GW{g}: {len(out)} sales but {len(inn)} buys")
        stamp = data.deadline[g] - pd.Timedelta(hours=1)
        for k, p in enumerate(out):
            sp = selling_price(self.purchase[p], int(data.price.at[p, g]))
            self.bank += sp
            del self.purchase[p]
            q = inn[k]                                          # pairing does not matter for the API
            cost = int(data.price.at[q, g])
            self.bank -= cost
            self.purchase[q] = cost
            self.transfers.append({'element_in': q, 'element_in_cost': cost, 'element_out': p,
                                   'element_out_cost': sp, 'event': g, 'entry': 1,
                                   'time': (stamp + pd.Timedelta(seconds=k)).isoformat()})
        if not self.squad:                                      # first squad of the team
            for q in inn:
                self.purchase[q] = int(data.price.at[q, g])
                self.bank -= self.purchase[q]
        if self.bank < 0:
            raise RuntimeError(f"GW{g}: bank {self.bank / 10:.1f} < 0")
        self.squad = new
        first = not self.current
        self.picks[g], self.banks[g] = list(new), self.bank
        self.current.append({'event': g, 'event_transfers': 0 if first else len(out),
                             'event_transfers_cost': HIT_COST * plan_gw.hits})
        if chip in CHIP_API_NAME:
            self.chips.append({'name': CHIP_API_NAME[chip], 'event': g})


# --------------------------------------------------------------------------- fake API
class VaastavApi:
    """Answers the FPL API paths the optimizer requests, as they were before the deadline of GW t."""

    def __init__(self, data, manager, t):
        self.data, self.m, self.t = data, manager, t

    def bootstrap(self):
        d, t = self.data, self.t
        elements = []
        for pid, g0 in d.first_gw.items():
            if g0 > t:
                continue                                        # not in the game yet
            jg = d.join_gw(pid, t)
            join = None if jg is None else (
                '2000-01-01' if jg == d.gws_of[pid][0] and jg == 1 else
                (d.deadline[jg] - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
            pts = [d.points[(pid, g)] for g in d.gws_of[pid] if g < t]
            ppg = sum(pts) / len(pts) if pts else 0.0
            elements.append({'id': int(pid), 'web_name': d.name[pid], 'team': d.team_of(pid, t),
                             'element_type': d.pos[pid], 'now_cost': int(d.price.at[pid, t]),
                             'cost_change_start': int(d.price.at[pid, t] - d.price.at[pid, 1]),
                             'chance_of_playing_next_round': d.chance(pid, t),
                             'ep_next': f"{d.ep_next(pid, t, ppg):.1f}",
                             'points_per_game': f"{ppg:.1f}",
                             'team_join_date': join})
        events = [{'id': g, 'is_next': g == t, 'finished': g < t, 'data_checked': g < t,
                   'deadline_time': d.deadline[g].isoformat()} for g in range(1, 39)]
        chips = [{'name': CHIP_API_NAME[c], 'number': k + 1, 'start_event': s, 'stop_event': e}
                 for c, ws in CHIP_RULE_WINDOWS.items() for k, (s, e) in enumerate(ws)]
        return {'elements': elements, 'events': events, 'teams': d.teams, 'chips': chips}

    def respond(self, path):
        d, m, t = self.data, self.m, self.t
        if path == '/bootstrap-static/':
            return self.bootstrap()
        if path == '/fixtures/':
            return [{'event': f['event'], 'team_h': f['team_h'], 'team_a': f['team_a'],
                     'finished': f['event'] < t} for f in d.fixtures]
        if path.startswith('/event/') and path.endswith('/live/'):
            return {'elements': d.live[int(path.split('/')[2])]}
        if path.startswith('/entry/') and path.endswith('/history/'):
            return {'current': list(m.current), 'chips': list(m.chips)}
        if path.endswith('/transfers/'):
            return list(m.transfers)
        if '/picks/' in path:
            g = int(path.split('/')[4])
            return {'picks': [{'element': p} for p in m.picks[g]],
                    'entry_history': {'bank': m.banks[g]}, 'active_chip': None}
        if path.startswith('/element-summary/'):
            pid = int(path.split('/')[2])
            rows = d.d[(d.d['element'] == pid) & (d.d['GW'] < t)]
            return {'history': [{'round': int(r.GW), 'value': int(r.value), 'kickoff_time': r.kickoff.isoformat()}
                                for r in rows.itertuples()]}
        if path.startswith('/entry/') and path.count('/') == 3:
            g0 = m.current[0]['event'] if m.current else t
            # created just before its first deadline: the price at creation is the deadline price
            return {'joined_time': (d.deadline[g0] - pd.Timedelta(hours=2)).isoformat(), 'started_event': g0}
        raise KeyError(path)


class Client(FplClient):
    """FplClient whose requests are answered by a VaastavApi (no network, no cache)."""

    def __init__(self, api):
        super().__init__()
        self.api = api

    def get(self, path):
        return self.api.respond(path)


# --------------------------------------------------------------------------- scoring
def valid_xi(xi, pos):
    """Formation rule: 1 GK, at least 3 DEF, 2 MID (implied by the squad limits) and 1 FWD."""
    n = [sum(1 for i in xi if pos[i] == k) for k in (1, 2, 3, 4)]
    return n[0] == 1 and n[1] >= 3 and n[2] >= 2 and n[3] >= 1


def score_gw(pg, pos, minutes, points):
    """Real FPL points of one GW for the decisions pg (PlanGW): XI after autosubs (none with Bench
    Boost), captain x2 (x3 with Triple Captain), passed to the vice-captain if the captain did not
    play, minus 4 per hit. A player 'plays' with minutes > 0 in any of his matches of the GW.
    Autosubs: the bench in its order (GK slot first); a bench player who played replaces the first
    starter who did not, among those whose replacement keeps a valid formation (GK only by GK).
    Returns (total, {'xi': final XI, 'captain_used': id or None})."""
    played = {i: minutes.get(i, 0) > 0 for i in pg.squad}
    if pg.chip == 'BB':
        xi = list(pg.squad)
    else:
        xi = list(pg.xi)
        for b in pg.bench:                                  # GK slot first, then outfield 1-3
            if not played[b]:
                continue
            for s in xi:
                if played[s] or (pos[s] == 1) != (pos[b] == 1):
                    continue
                trial = [b if i == s else i for i in xi]
                if valid_xi(trial, pos):
                    xi = trial
                    break
    total = sum(points.get(i, 0) for i in xi)
    mult = 3 if pg.chip == 'TC' else 2
    cap = pg.captain_id if played[pg.captain_id] else (
        pg.vice_id if pg.vice_id is not None and played[pg.vice_id] else None)
    if cap is not None:
        total += (mult - 1) * points.get(cap, 0)
    return total - HIT_COST * pg.hits, {'xi': xi, 'captain_used': cap}


# --------------------------------------------------------------------------- simulation
def simulate(season, lookahead, first_gw, last_gw, chips, time_limit, out_path):
    data = SeasonData(season)
    m = Manager()
    rows = []
    for t in range(first_gw, last_gw + 1):
        t0 = time.time()
        opts = Options(team_id=1, lookahead=lookahead, write_log=False, time_limit=time_limit,
                       chip_report=(chips == 'advice'))
        r = optimize(Client(VaastavApi(data, m, t)), opts)
        played_chip = 'NONE'
        if chips == 'advice' and r.chips.advice and r.chips.advice.get('pick'):
            played_chip = r.chips.advice['pick']             # the suggestion, then the plan with it
            opts.chip, opts.chip_report = played_chip, False
            r = optimize(Client(VaastavApi(data, m, t)), opts)
        pg = r.plan[0]
        # check of the fake API: the budget the optimizer rebuilt (lower bound of the selling
        # values, S5) against the exact one the simulation knows
        sq, pur, bank = m.saved if m.saved is not None else (m.squad, m.purchase, m.bank)   # after a FH revert
        exact = (bank + sum(selling_price(pur[p], int(data.price.at[p, t])) for p in sq)) / 10 if sq else 100.0
        m.apply(data, t, pg)
        pos = {i: data.pos[i] for i in pg.squad}
        mins = {i: data.minutes.get((i, t), 0) for i in pg.squad}
        pts = {i: data.points.get((i, t), 0) for i in pg.squad}
        total, info = score_gw(pg, pos, mins, pts)
        rows.append({'season': season, 'gw': t, 'points': total, 'hits': pg.hits, 'chip': played_chip,
                     'transfers': len(pg.transfers), 'ft': r.ft, 'bank': m.bank / 10,
                     'budget_seen': round(r.budget, 1), 'budget_exact': exact,
                     'captain': data.name[pg.captain_id], 'captain_used': data.name.get(info['captain_used']),
                     'expected': round(r.expected_now.total, 2), 'objective': round(r.solver.objective, 3),
                     'gap': None if r.solver.gap is None else round(r.solver.gap, 3),
                     'proven': r.solver.proven, 'passes': r.solver.passes,
                     'seconds': round(time.time() - t0, 1)})
        print(f"GW{t:2d}  pts {total:3d}  hits {pg.hits}  chip {played_chip:4s}  tr {len(pg.transfers)}  "
              f"exp {r.expected_now.total:6.2f}  cap {data.name[pg.captain_id][:18]:18s}  "
              f"{rows[-1]['seconds']:6.1f} s", flush=True)
        if out_path:
            write_rows(out_path, rows)
    tot = sum(r['points'] for r in rows)
    print(f"{season} GW{first_gw}-{last_gw}: {tot} points, {sum(r['hits'] for r in rows)} hits")
    return rows


def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def compare(path_a, path_b):
    """Paired difference A - B of the points per GW (same season, same GWs): mean, standard error
    s_d / sqrt(n), and whether |mean| > 2 SE. The GWs are not independent (the squads drift
    apart), so the standard error is an approximation."""
    a = pd.read_csv(path_a).set_index(['season', 'gw'])['points']
    b = pd.read_csv(path_b).set_index(['season', 'gw'])['points']
    dd = (a - b).dropna()
    n = len(dd)
    mean, sd = dd.mean(), dd.std(ddof=1) if n > 1 else float('nan')
    se = sd / math.sqrt(n) if n > 1 else float('nan')
    print(f"A = {path_a}: {a.loc[dd.index].sum()} points | B = {path_b}: {b.loc[dd.index].sum()} points | {n} GWs")
    print(f"A - B per GW: {mean:+.2f} +- {se:.2f} (SE); season total {dd.sum():+.0f}; "
          f"{'significant' if abs(mean) > 2 * se else 'not significant'} at 2 SE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--season', default='2024-25')
    ap.add_argument('--lookahead', type=int, default=3)
    ap.add_argument('--first-gw', type=int, default=2)
    ap.add_argument('--last-gw', type=int, default=38)
    ap.add_argument('--chips', choices=['none', 'advice'], default='none',
                    help="none = never play a chip; advice = play the chip the optimizer suggests")
    ap.add_argument('--time-limit', type=int, default=NO_LIMIT)
    ap.add_argument('--out', help="CSV with one row per GW (rewritten after every GW)")
    ap.add_argument('--compare', nargs=2, metavar=('A.csv', 'B.csv'))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    logging.basicConfig(level=logging.WARNING, format='%(message)s')   # optimizer warnings only
    simulate(args.season, args.lookahead, args.first_gw, args.last_gw, args.chips, args.time_limit, args.out)


if __name__ == '__main__':
    main()
