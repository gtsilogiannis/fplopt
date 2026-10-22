"""
Walk-forward backtest of the xP model (fplopt.xp_model) on past seasons.

For every gameweek t, the model sees only data from GWs < t (as in a live run),
predicts each player's points for GW t, and the prediction is compared with the
actual points. Data: github.com/vaastav/Fantasy-Premier-League (per-match rows).

Usage (from the repository root; downloads into data_cache/):
  python scripts/calibrate_xp.py --season 2025-26
  python scripts/calibrate_xp.py --season 2024-25 --tune     # grid search of the [tuned] constants (slow, ~10 min)
"""
import argparse
import itertools
import os
import numpy as np
import pandas as pd
import requests
from fplopt import xp_model as F

DATA_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{path}"
CACHE_DIR = "data_cache"
POS = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}   # 2024/25 also has 'AM' (assistant manager): dropped


def download(season, path):
    """Download once into data_cache/ and reuse afterwards."""
    local = os.path.join(CACHE_DIR, f"{season}_{path.replace('/', '_')}")
    if not os.path.isfile(local):
        os.makedirs(CACHE_DIR, exist_ok=True)
        r = requests.get(DATA_URL.format(season=season, path=path), timeout=60)
        r.raise_for_status()
        with open(local, 'wb') as fh:
            fh.write(r.content)
    return local


def load(season):
    """Per-match rows -> the same per-(player, GW) table that fplopt.data.fetch_history() builds live."""
    d = pd.read_csv(download(season, 'gws/merged_gw.csv'))
    teams = pd.read_csv(download(season, 'teams.csv'))
    d = d[d['position'].isin(POS)].copy()
    d['team_id'] = d['team'].map(dict(zip(teams['name'], teams['id'])))
    d['pos'] = d['position'].map(POS)
    if 'defensive_contribution' not in d:          # DefCon exists only from 2025/26
        d['defensive_contribution'] = 0
    d = d.rename(columns={**{k: v for k, v in F.HIST_STATS.items()}, 'GW': 'gw'})
    hist = (d.groupby(['element', 'gw'])
             .agg(team=('team_id', 'last'), n_matches=('fixture', 'count'), minutes=('minutes', 'sum'),
                  pts=('total_points', 'sum'), **{c: (c, 'sum') for c in F.STAT_COLS})
             .reset_index())
    pos_of = d.groupby('element')['pos'].last().to_dict()
    # X9: first GW at the current club, from the END-of-season team_join_date.
    # predict() only uses it when it is not later than the GW being predicted.
    raw = pd.read_csv(download(season, 'players_raw.csv'))
    deadlines = pd.to_datetime(d.groupby('gw')['kickoff_time'].min(), utc=True).to_dict()  # first kickoff ~ deadline
    join = {pid: (jd if isinstance(jd, str) else None) for pid, jd in zip(raw['id'], raw['team_join_date'])}
    start_gw_of = F.first_gw_at_club(join, deadlines)
    # FPL's own projection, made right after GW t-1 (the dataset's xP for row t-1).
    # The xP of row t itself is NOT used: it was recorded after GW t and contains
    # its result (checked: high scorers in GW t already have high xP in row t).
    ep = d.groupby(['element', 'gw'])['xP'].first().reset_index()
    ep['gw'] += 1
    hist = hist.merge(ep.rename(columns={'xP': 'fpl_ep'}), on=['element', 'gw'], how='left')
    hist.attrs['start_gw_of'] = start_gw_of   # attrs: metadata dict attached to a DataFrame
    return d, hist, pos_of


def price_table(d):
    """{gw: {player_id: price in GBPm}}: the price at the GW deadline (latest known at or before it)."""
    pr = d.pivot_table(index='element', columns='gw', values='value', aggfunc='first').ffill(axis=1) / 10.0
    return {gw: pr[gw].dropna().to_dict() for gw in pr.columns}


def predict(d, hist, pos_of, gws, prices=None):
    prices = prices if prices is not None else price_table(d)
    out = []
    for gw in gws:
        # X9: a join date after this GW was not known yet at that time -> ignore it
        start = {k: v for k, v in hist.attrs.get('start_gw_of', {}).items() if v <= gw}
        feats, teams, L = F.build_features(hist, pos_of, gw, prices.get(gw), start)
        comp = {}
        for r in d[d['gw'] == gw].itertuples():
            f = feats.get(r.element)
            if f is None:
                continue
            c = F.match_xp(f, r.pos, teams.get(r.team_id, F.NEUTRAL_TEAM),
                           teams.get(r.opponent_team, F.NEUTRAL_TEAM), bool(r.was_home), L)
            acc = comp.setdefault(r.element, dict.fromkeys(c, 0.0))
            for k, v in c.items():
                acc[k] += v
        price_now = prices.get(gw, {})
        for r in hist[hist['gw'] == gw].itertuples():
            row = {'element': r.element, 'gw': gw, 'pos': pos_of[r.element], 'pts': r.pts,
                   'minutes': r.minutes, 'fpl_ep': r.fpl_ep, 'price': price_now.get(r.element, np.nan)}
            if r.element in comp:
                row['xp'] = sum(comp[r.element].values())
                row['source'] = 'model'
                row.update({'c_' + k: v for k, v in comp[r.element].items()})
            else:
                # X9: no minutes yet -> the live script uses FPL's ep_next; here the
                # FPL projection made after GW t-1 (0 if the dataset has none)
                row['xp'] = 0.0 if pd.isna(r.fpl_ep) else r.fpl_ep
                row['source'] = 'FPL ep'
            out.append(row)
    p = pd.DataFrame(out)
    # Availability: the live script drops players with chance_of_playing below
    # --min-chance. History has no such field; FPL's ep = 0 is used as the proxy.
    # Only in GWs where the dataset has FPL's xP at all (it is 0 for everyone in some GWs).
    has_ep = p.groupby('gw')['fpl_ep'].transform('mean') > 0.5
    p.loc[has_ep & (p['fpl_ep'] == 0), 'xp'] = 0.0
    p['has_ep'] = has_ep
    return p


def score(p):
    """Tuning objective: RMSE over GWs with an FPL projection and players FPL expects
    may play (fpl_ep > 0). The subset does not depend on the model, so every
    parameter set is scored on the same rows. RMSE, not MAE: squared error is
    minimised by the conditional MEAN (= expected points); absolute error by the
    MEDIAN, which is below the mean for right-skewed points and favours under-prediction."""
    s = p[p['has_ep'] & (p['fpl_ep'] > 0)]
    return float(np.sqrt(((s['xp'] - s['pts']) ** 2).mean())), len(s)


def price_bias(p):
    """Mean predicted - actual by price band and season phase (availability proxy applied)."""
    q = p[~p['has_ep'] | (p['fpl_ep'] != 0)].copy()
    q['phase'] = np.where(q['gw'] <= 8, 'GW2-8', 'GW9-38')
    q['band'] = pd.cut(q['price'], [0, 5, 6.5, 8, 10, 20])
    t = q.groupby(['phase', 'band'], observed=True).agg(n=('xp', 'size'), predicted=('xp', 'mean'),
                                                       actual=('pts', 'mean'))
    t['bias'] = t['predicted'] - t['actual']
    return t


def actual_components(d):
    """Actual FPL points split into the same components as match_xp()."""
    a = pd.DataFrame({'element': d['element'], 'gw': d['gw']})
    pos = d['pos']
    a['appearance'] = (d['minutes'] > 0).astype(int) + (d['minutes'] >= 60).astype(int)
    a['goals'] = pos.map(F.PTS_GOAL) * d['goals_scored']
    a['assists'] = F.PTS_ASSIST * d['assists']
    a['clean_sheet'] = pos.map(F.PTS_CLEAN_SHEET) * d['clean_sheets']
    a['conceded'] = np.where(pos <= 2, -(d['goals_conceded'] // F.CONCEDED_PER_MINUS_ONE), 0)
    a['saves'] = np.where(pos == 1, d['saves'] // F.SAVES_PER_POINT, 0)
    thr = pos.map({1: 10 ** 6, **F.DEFCON_THRESHOLD})
    a['defcon'] = F.PTS_DEFCON * (d['dc'] >= thr)
    a['bonus'] = d['bonus']
    a['cards'] = F.PTS_YELLOW * d['yc'] + F.PTS_RED * d['rc']
    return a.groupby(['element', 'gw']).sum().reset_index()


def top11(p):
    """Scale check (acceptance criterion of the project): for every GW, the 11
    highest predictions + the captain (the highest one, counted twice), with no budget or
    formation. Returns an array with one row per GW: (predicted, actual)."""
    tops = []
    for gw, g in p.groupby('gw'):
        top = g.nlargest(11, 'xp')
        cap = top['xp'].idxmax()                  # index label of the highest prediction
        tops.append((top['xp'].sum() + top.loc[cap, 'xp'], top['pts'].sum() + top.loc[cap, 'pts']))
    return np.array(tops)


def report(season, d, p):
    valid = p[p.groupby('gw')['fpl_ep'].transform('mean') > 0.5]   # GWs where the dataset has FPL's xP
    rel = valid[(valid['fpl_ep'] >= 2) | (valid['xp'] >= 2)]
    print(f"\n=== {season}: accuracy (GWs with FPL baseline: {valid['gw'].nunique()}) ===")
    print(f"{'subset':<10}{'n':>7} | {'MAE':>6}{'FPL':>7} | {'RMSE':>6}{'FPL':>7} | {'bias':>6}{'FPL':>7} | "
          f"{'corr':>6}{'FPL':>7}")
    for name, s in [('all', valid), ('relevant', rel)]:
        em, ef = s['xp'] - s['pts'], s['fpl_ep'] - s['pts']
        print(f"{name:<10}{len(s):>7} | {em.abs().mean():>6.2f}{ef.abs().mean():>7.2f} | "
              f"{np.sqrt((em ** 2).mean()):>6.2f}{np.sqrt((ef ** 2).mean()):>7.2f} | "
              f"{em.mean():>+6.2f}{ef.mean():>+7.2f} | {s['xp'].corr(s['pts']):>6.3f}{s['fpl_ep'].corr(s['pts']):>7.3f}")
    rmse, n = score(p)
    print(f"Tuning score (RMSE, fpl_ep > 0): {rmse:.4f} on {n} rows | "
          f"fallback (FPL ep) rows: {(p['source'] != 'model').sum()}")

    print(f"\n=== {season}: bias by price (predicted - actual) ===")
    print(price_bias(p).round(2).to_string())

    print(f"\n=== {season}: components, predicted / actual mean points per player-GW ===")
    a = actual_components(d)
    m = p.merge(a, on=['element', 'gw'])
    m = m[(m['source'] == 'model') & (~m['has_ep'] | (m['fpl_ep'] != 0))]   # model rows, available
    comps = ['appearance', 'goals', 'assists', 'clean_sheet', 'conceded', 'saves', 'defcon', 'bonus', 'cards']
    print("pos  " + " ".join(f"{c[:11]:>12}" for c in comps) + f"{'total':>13}")
    for pos, g in m.groupby('pos'):
        cells = [f"{g['c_' + c].mean():.2f}/{g[c].mean():.2f}" for c in comps]
        print(f"{pos:<5}" + " ".join(f"{c:>12}" for c in cells) + f"{g['xp'].mean():>7.2f}/{g['pts'].mean():.2f}")

    print(f"\n=== {season}: calibration by predicted xP ===")
    p2 = p[~p['has_ep'] | (p['fpl_ep'] != 0)]
    bins = pd.cut(p2['xp'], [-1, 0.5, 1, 2, 3, 4, 5, 6, 30])
    t = p2.groupby(bins, observed=True).agg(n=('xp', 'size'), predicted=('xp', 'mean'), actual=('pts', 'mean'))
    print(t.round(2).T.to_string())

    tops = top11(p)
    gap = tops[:, 0] - tops[:, 1]
    print(f"\nTop-11 + captain per GW (no budget): predicted {tops[:, 0].mean():.1f}, actual {tops[:, 1].mean():.1f} "
          f"| gap {gap.mean():+.2f} +/- {gap.std(ddof=1) / np.sqrt(len(gap)):.2f} (standard error, {len(gap)} GWs)")


def tune(d, hist, pos_of):
    """Grid search of the [tuned] constants by RMSE (see score()). TEAM_PRIOR_MATCHES
    stays at 10: in an earlier grid it changed MAE by < 0.01."""
    prices = price_table(d)
    rows = []
    # 8c: the recency grid is centred on the chosen 0.6 (the earlier grid 0.8/0.9/1.0 did not contain it)
    for rec, kp, b in itertools.product([0.5, 0.6, 0.7], [300, 600, 1200, 2400], [0.0, 0.5, 1.0, 1.5, 2.0]):
        F.MINUTES_RECENCY, F.PLAYER_PRIOR_MINUTES, F.PRICE_PRIOR_EXPONENT = rec, kp, b
        rmse, n = score(predict(d, hist, pos_of, range(2, 39), prices))
        rows.append((rec, kp, b, rmse))
        print(f"recency={rec} K_player={kp} price_exponent={b}: RMSE {rmse:.4f} (n={n})", flush=True)
    best = min(rows, key=lambda r: r[3])
    print(f"Best: recency={best[0]} K_player={best[1]} price_exponent={best[2]} (RMSE {best[3]:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Walk-forward backtest of the fplopt xP model")
    ap.add_argument("--season", default="2025-26", help="e.g. 2024-25, 2025-26")
    ap.add_argument("--tune", action="store_true", help="grid-search the [tuned] constants")
    args = ap.parse_args()
    d, hist, pos_of = load(args.season)
    d = d.rename(columns={'GW': 'gw'})
    if args.tune:
        tune(d, hist, pos_of)
    else:
        report(args.season, d, predict(d, hist, pos_of, range(2, 39)))
