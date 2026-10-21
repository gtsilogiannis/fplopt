"""Chat 6b (M9): calibration of P(appear) of the minutes model.
Since 8c the same shrinkage is applied in build_features (a = 0.4, m = 0.45 unchanged) and
check_minutes.py measures and fits P(appear), P(60+) and E[minutes] together; this script
stays as the original, simpler P(appear) check.
Usage (next to fpl_v2.py and calibrate_xp.py): python check_papp.py
Data: the same vaastav files and data_cache/ as calibrate_xp.py.
Walk-forward, same estimator as build_features: Papp = recency-weighted share
(MINUTES_RECENCY^(age)) of the player's team-match GWs before t in which he played.
Compares it with the actual appearance in GW t. Data: vaastav merged_gw.csv.
Availability proxy (notes_08a 5.1): FPL xP of row t-1 (= FPL estimate for GW t,
made after GW t-1) > 0; rows with xP(t-1) == 0 correspond to players flagged out."""
import numpy as np
import pandas as pd
import fpl_v2 as F
from calibrate_xp import download   # same download + cache as the xP backtest

R = F.MINUTES_RECENCY


def load(season):
    d = pd.read_csv(download(season, 'gws/merged_gw.csv'))
    return d.groupby(['element', 'GW']).agg(minutes=('minutes', 'sum'), n=('fixture', 'count'),
                                            xP=('xP', 'first'), pos=('position', 'first')).reset_index()


def table(season):
    g = load(season)
    rows = []
    for e, df in g.groupby('element'):
        df = df.sort_values('GW')
        gws, mins, ns, xps = df['GW'].values, df['minutes'].values, df['n'].values, df['xP'].values
        for k in range(1, len(gws)):
            if gws[k - 1] != gws[k] - 1:
                continue                          # need row t-1 for the availability proxy
            w = R ** (gws[k] - 1 - gws[:k])
            rows.append({'season': season, 't': gws[k], 'pos': df['pos'].values[k], 'S': float((w * (mins[:k] > 0)).sum()),
                         'W': float(w.sum()), 'n': int(ns[k]), 'played': bool(mins[k] > 0),
                         'avail': xps[k - 1] > 0})
    return pd.DataFrame(rows)


def p_of(d, a, m):
    p1 = (d['S'] + a * m) / (d['W'] + a)              # shrunk per-match P(appear)
    return 1.0 - (1.0 - p1) ** d['n']                  # DGW: at least one of n matches


def logloss(d, a, m):
    p = p_of(d, a, m).clip(1e-6, 1 - 1e-6)
    y = d['played']
    return float(-(y * np.log(p) + (~y) * np.log(1 - p)).mean())


def report(d, p, label):
    bins = [-0.001, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.999, 1.0]
    t = d.assign(p=p, bin=pd.cut(p, bins)).groupby('bin', observed=True).agg(
        n=('played', 'size'), predicted=('p', 'mean'), actual=('played', 'mean'))
    print(f"-- {label}\n{t.round(3).to_string()}")


if __name__ == '__main__':
    df = pd.concat([table('2024-25'), table('2025-26')], ignore_index=True)
    df = df[(df['t'] >= 2) & df['avail']]
    tr, te = df[df['season'] == '2024-25'], df[df['season'] == '2025-26']
    print(f"rows (available proxy): 2024-25 {len(tr)}, 2025-26 {len(te)}")
    report(te, p_of(te, 0.0, 0.0), "2025-26, raw estimator without shrinkage (a = 0)")
    best = None
    for a in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
        for m in np.arange(0.2, 0.81, 0.05):
            ll = logloss(tr, a, m)
            if best is None or ll < best[0]:
                best = (ll, a, round(float(m), 2))
    ll, a, m = best
    print(f"\nfit on 2024-25: a = {a}, m = {m}, logloss {ll:.4f} "
          f"(no shrinkage: {logloss(tr, 1e-9, 0.5):.4f})")
    print(f"check on 2025-26: logloss {logloss(te, a, m):.4f} (no shrinkage {logloss(te, 1e-9, 0.5):.4f})")
    best2 = min(((logloss(te, a2, m2), a2, round(float(m2), 2)) for a2 in (0.1, 0.15, 0.2, 0.3, 0.4, 0.6)
                 for m2 in np.arange(0.2, 0.81, 0.05)))
    print(f"best on 2025-26 alone: a = {best2[1]}, m = {best2[2]}, logloss {best2[0]:.4f}")
    report(te, p_of(te, a, m), f"2025-26 with shrinkage a={a}, m={m} (fitted on 2024-25)")
    for ps, dp in te.groupby('pos'):
        top = dp[dp['S'] >= dp['W'] - 1e-9]          # played in every previous GW
        print(f"   2025-26 {ps}: played every previous GW -> n {len(top)}, predicted "
              f"{p_of(top, a, m).mean():.3f}, actual {top['played'].mean():.3f}")
    Wmax = 1 / (1 - R)
    print(f"\nW max = {Wmax:.2f}; played every GW -> P = {(Wmax + a * m) / (Wmax + a):.3f}; "
          f"never played -> {(a * m) / (Wmax + a):.3f}")
