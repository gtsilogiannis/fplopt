"""Chat 8c: calibration of the minutes model (P(appear), P(60+), E[minutes]) and fit of its shrinkage.
Usage (from the repository root):  python scripts/check_minutes.py [measure|fit|verify|recency]
Data: the same vaastav files and data_cache/ as calibrate_xp.py.

Walk-forward: for every GW t = 2..38 the sums behind the model (xp_model.minutes_sums) are built
from GWs < t only, and compared with what the player did in GW t.
Sample = the rows calibrate_xp.score() uses: GWs where the dataset has FPL's projection and
the player is not flagged out (FPL ep of that GW > 0, the availability proxy of notes_08a 5.1),
and the player is in the model (has at least one GW at his current club).
P(60+) and minutes are judged on single-match GWs only (in a DGW the model works per match).
"""
import sys
import numpy as np
import pandas as pd
from fplopt import xp_model as F
from calibrate_xp import load

SEASONS = ('2024-25', '2025-26')
POS_NAME = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
_CACHE = {}


def season_data(season):
    if season not in _CACHE:
        _CACHE[season] = load(season)
    return _CACHE[season]


def table(season, recency=None, lag=0, age_from_last=False):
    """One row per (player, GW t) of the sample: sums S_app, S_60, S_min, W and the outcome of GW t.
    lag = 1 hides GW t-1 from the model (the live situation when the last GW is not final yet)."""
    d, hist, pos_of = season_data(season)
    start_all = hist.attrs['start_gw_of']
    has_ep = hist.groupby('gw')['fpl_ep'].transform('mean') > 0.5
    ok = hist[has_ep & (hist['fpl_ep'] > 0)]
    parts = []
    for t in range(2 + lag, 39):
        start = {k: v for k, v in start_all.items() if v <= t}
        # lag: the model of GW t only knows GWs <= t-1-lag, but is asked about GW t (calendar ages)
        # age_from_last emulates 'age counted from the last GW present': ask the function for GW t - lag
        s = F.minutes_sums(hist[hist['gw'] <= t - 1 - lag], t - lag if age_from_last else t, start, recency)
        tgt = ok[ok['gw'] == t][['element', 'minutes', 'n_matches']]
        x = tgt.join(s, on='element', how='inner')
        x = x[x['W'] > 0]
        x['t'] = t
        parts.append(x)
    x = pd.concat(parts, ignore_index=True)
    x['pos'] = x['element'].map(pos_of)
    x['season'] = season
    x['played'] = (x['minutes'] > 0).astype(float)
    x['m'] = x['minutes'] / x['n_matches']
    x['y60'] = (x['m'] >= 60).astype(float)
    return x


# ---------------------------------------------------------------- estimators
def p_app(df, a, m):
    """P(appear in a match) = (S_app + a*m) / (W + a). a, m scalars or per-row arrays."""
    return (df['S_app'] + a * m) / (df['W'] + a)


def cond(df, num, b, mq):
    """Shrunk conditional mean given an appearance: (S_x + b*mq) / (S_app + b)."""
    return (df[num] + b * mq) / (df['S_app'] + b)


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ll_app(df, a, m):
    """Log-loss of P(at least one appearance in the GW); DGW: 1 - (1 - p)^n."""
    p = 1.0 - (1.0 - p_app(df, a, m)) ** df['n_matches']
    return logloss(df['played'].values, p.values)


def prior_arrays(df, par, key):
    """par = scalar or {pos: scalar} -> per-row array."""
    if isinstance(par, dict):
        return df['pos'].map(par).values
    return par


# ---------------------------------------------------------------- report helpers
def bins_table(pred, act, edges):
    t = pd.DataFrame({'pred': pred, 'act': act})
    t['bin'] = pd.cut(t['pred'], edges, include_lowest=True)
    return t.groupby('bin', observed=True).agg(n=('pred', 'size'), predicted=('pred', 'mean'), actual=('act', 'mean')).round(3)


def measure(recency=None):
    x = pd.concat([table(s, recency) for s in SEASONS], ignore_index=True)
    x['Papp'], x['P60'], x['Emin'] = x['S_app'] / x['W'], x['S_60'] / x['W'], x['S_min'] / x['W']
    for s in SEASONS:
        xs = x[(x['season'] == s)]
        one = xs[xs['n_matches'] == 1]
        print(f"\n===== {s}: rows {len(xs)} (single-match GWs {len(one)}); raw estimates, no shrinkage =====")
        print("-- P(60+) raw vs actual share of GWs with 60+ minutes")
        print(bins_table(one['P60'], one['y60'], [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.999, 1.0]).to_string())
        print("-- Emin raw vs actual minutes")
        print(bins_table(one['Emin'], one['m'], [0, 5, 20, 40, 60, 75, 85, 89.9, 90.0]).to_string())
        print("-- played in every previous GW (S_app = W), by position: predicted / actual")
        rows = []
        for p, g in one[one['S_app'] >= one['W'] - 1e-9].groupby('pos'):
            rows.append({'pos': POS_NAME[p], 'n': len(g), 'Papp': g['Papp'].mean(), 'actual app': g['played'].mean(),
                         'P60': g['P60'].mean(), 'actual 60+': g['y60'].mean(),
                         'Emin': g['Emin'].mean(), 'actual min': g['m'].mean()})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        print("-- all rows, by position: predicted / actual")
        rows = []
        for p, g in one.groupby('pos'):
            rows.append({'pos': POS_NAME[p], 'n': len(g), 'Papp': g['Papp'].mean(), 'actual app': g['played'].mean(),
                         'P60': g['P60'].mean(), 'actual 60+': g['y60'].mean(),
                         'Emin': g['Emin'].mean(), 'actual min': g['m'].mean()})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        print("-- conditional on an appearance in the target GW, by position: raw prediction / actual "
              "(both over the rows where he played)")
        rows = []
        for p, g in one[(one['S_app'] > 0) & (one['played'] == 1)].groupby('pos'):
            rows.append({'pos': POS_NAME[p], 'n app': len(g), 'q60 raw (S60/Sapp)': (g['S_60'] / g['S_app']).mean(),
                         'q60 actual': g['y60'].mean(), 'mu raw (Smin/Sapp)': (g['S_min'] / g['S_app']).mean(),
                         'mu actual': g['m'].mean()})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    return x


# ---------------------------------------------------------------- fitting
GRID_A = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
GRID_M = tuple(np.round(np.arange(0.2, 0.91, 0.05), 2))
GRID_MU = tuple(float(v) for v in range(40, 91, 5))


def fit_papp(tr, per_pos=False, global_a=None):
    """Grid fit of (a, m) by log-loss. per_pos: m per position with a common a (a fitted)."""
    best = None
    if not per_pos:
        for a in GRID_A:
            for m in GRID_M:
                ll = ll_app(tr, a, m)
                if best is None or ll < best[0]:
                    best = (ll, a, m)
        return best
    # per-position prior mean, common weight: coordinate search over a, then m per position
    for a in GRID_A:
        ms = {}
        for p in POS_NAME:
            g = tr[tr['pos'] == p]
            ms[p] = min(GRID_M, key=lambda m: ll_app(g, a, m))
        ll = ll_app(tr, a, tr['pos'].map(ms).values)
        if best is None or ll < best[0]:
            best = (ll, a, ms)
    return best


def p60_cond(df, a, m, b, mq):
    return p_app(df, a, m) * cond(df, 'S_60', b, mq)


def emin_cond(df, a, m, c, mu):
    return p_app(df, a, m) * cond(df, 'S_min', c, mu)


def one_match(df):
    return df[df['n_matches'] == 1]


def fit_p60(tr, a, m):
    tr = one_match(tr)
    best = None
    for b in GRID_A:
        for mq in GRID_M:
            ll = logloss(tr['y60'].values, p60_cond(tr, a, m, b, mq).values)
            if best is None or ll < best[0]:
                best = (ll, b, mq)
    return best


def fit_emin(tr, a, m):
    tr = one_match(tr)
    best = None
    for c in GRID_A:
        for mu in GRID_MU:
            mse = float(((emin_cond(tr, a, m, c, mu) - tr['m']) ** 2).mean())
            if best is None or mse < best[0]:
                best = (mse, c, mu)
    return best


def fit(recency=None, verbose=True):
    tr, te = table(SEASONS[0], recency), table(SEASONS[1], recency)
    res = {}
    # ---- P(appear)
    base_tr, base_te = ll_app(tr, 1e-9, 0.5), ll_app(te, 1e-9, 0.5)
    ll, a, m = fit_papp(tr)
    ll_pp, a_pp, ms_pp = fit_papp(tr, per_pos=True)
    if verbose:
        print(f"\n[P(appear)] log-loss  raw: {base_tr:.4f} / {base_te:.4f}  (fit season / test season)")
        print(f"   global fit on {SEASONS[0]}: a={a}, m={m}: {ll:.4f} / {ll_app(te, a, m):.4f}")
        print(f"   per-position m, common a: a={a_pp}, m={ms_pp}: {ll_pp:.4f} / "
              f"{ll_app(te, a_pp, te['pos'].map(ms_pp).values):.4f}")
        print(f"   best on test season alone (global): {fit_papp(te)}")
    res['papp'] = (a, m)
    # ---- P(60+): raw / direct shrinkage / conditional
    t1, e1 = one_match(tr), one_match(te)
    raw_p60 = lambda d: (d['S_60'] / d['W']).values
    ll_raw = (logloss(t1['y60'].values, raw_p60(t1)), logloss(e1['y60'].values, raw_p60(e1)))
    best_d = min(((logloss(t1['y60'].values, ((t1['S_60'] + a2 * m2) / (t1['W'] + a2)).values), a2, m2)
                  for a2 in GRID_A for m2 in GRID_M))
    _, a2, m2 = best_d
    ll_dir = (best_d[0], logloss(e1['y60'].values, ((e1['S_60'] + a2 * m2) / (e1['W'] + a2)).values))
    ll_c, b, mq = fit_p60(tr, a, m)
    ll_cte = logloss(e1['y60'].values, p60_cond(e1, a, m, b, mq).values)
    if verbose:
        print(f"\n[P(60+)] log-loss (single-match GWs)  fit season / test season")
        print(f"   raw                      : {ll_raw[0]:.4f} / {ll_raw[1]:.4f}")
        print(f"   direct shrinkage a={a2}, m={m2}: {ll_dir[0]:.4f} / {ll_dir[1]:.4f}")
        print(f"   conditional  P60 = Papp* x q60*, b={b}, m_q={mq}: {ll_c:.4f} / {ll_cte:.4f}")
    res['p60'] = (b, mq)
    # ---- Emin
    mse_raw = (float((((t1['S_min'] / t1['W']) - t1['m']) ** 2).mean()), float((((e1['S_min'] / e1['W']) - e1['m']) ** 2).mean()))
    best_dm = min(((float(((((t1['S_min'] + a3 * m3) / (t1['W'] + a3)) - t1['m']) ** 2).mean()), a3, m3)
                   for a3 in GRID_A for m3 in GRID_MU))
    _, a3, m3 = best_dm
    mse_dir = (best_dm[0], float(((((e1['S_min'] + a3 * m3) / (e1['W'] + a3)) - e1['m']) ** 2).mean()))
    mse_c, c, mu = fit_emin(tr, a, m)
    mse_cte = float(((emin_cond(e1, a, m, c, mu) - e1['m']) ** 2).mean())
    if verbose:
        print(f"\n[Emin] MSE of minutes (single-match GWs)  fit season / test season")
        print(f"   raw                      : {mse_raw[0]:.2f} / {mse_raw[1]:.2f}")
        print(f"   direct shrinkage a={a3}, mean={m3}: {mse_dir[0]:.2f} / {mse_dir[1]:.2f}")
        print(f"   conditional  Emin = Papp* x mu*, c={c}, mu={mu}: {mse_c:.2f} / {mse_cte:.2f}")
    res['emin'] = (c, mu)
    return res, tr, te


def fit_per_position(recency=None):
    """Does a per-position prior mean for the CONDITIONAL layer help on the test season?"""
    res, tr, te = fit(recency, verbose=False)
    a, m = res['papp']
    b, mq = res['p60']
    c, mu = res['emin']
    t1, e1 = one_match(tr), one_match(te)
    print(f"\n[per-position conditional priors] global: b={b}, m_q={mq}; c={c}, mu={mu}")
    # P(60+ | app): per-position prior mean, same weight b
    mq_pos = {p: min(GRID_M, key=lambda v: logloss(t1[t1['pos'] == p]['y60'].values,
                                                    p60_cond(t1[t1['pos'] == p], a, m, b, v).values))
              for p in POS_NAME}
    mu_pos = {p: min(GRID_MU, key=lambda v: float(((emin_cond(t1[t1['pos'] == p], a, m, c, v) - t1[t1['pos'] == p]['m']) ** 2).mean()))
              for p in POS_NAME}
    print(f"   m_q per position (fit season): { {POS_NAME[p]: v for p, v in mq_pos.items()} }")
    print(f"   mu  per position (fit season): { {POS_NAME[p]: v for p, v in mu_pos.items()} }")
    for name, d in ((f'fit {SEASONS[0]}', t1), (f'test {SEASONS[1]}', e1)):
        g_p = logloss(d['y60'].values, p60_cond(d, a, m, b, mq).values)
        pp_p = logloss(d['y60'].values, p60_cond(d, a, m, b, d['pos'].map(mq_pos).values).values)
        g_m = float(((emin_cond(d, a, m, c, mu) - d['m']) ** 2).mean())
        pp_m = float(((emin_cond(d, a, m, c, d['pos'].map(mu_pos).values) - d['m']) ** 2).mean())
        print(f"   {name}: P60 log-loss global {g_p:.4f} -> per-position {pp_p:.4f} | Emin MSE global {g_m:.2f} -> per-position {pp_m:.2f}")
    # best per-position prior means fitted on the TEST season alone (stability across seasons)
    mq_te = {p: min(GRID_M, key=lambda v: logloss(e1[e1['pos'] == p]['y60'].values,
                                                   p60_cond(e1[e1['pos'] == p], a, m, b, v).values)) for p in POS_NAME}
    mu_te = {p: min(GRID_MU, key=lambda v: float(((emin_cond(e1[e1['pos'] == p], a, m, c, v) - e1[e1['pos'] == p]['m']) ** 2).mean()))
             for p in POS_NAME}
    print(f"   same fit on the test season alone: m_q { {POS_NAME[p]: v for p, v in mq_te.items()} } "
          f"mu { {POS_NAME[p]: v for p, v in mu_te.items()} }")
    return res


def verify():
    """The production code (xp_model.shrink_minutes) must equal the formulas used by the fit, and
    the consistency inequalities must hold on every row of the walk-forward sample."""
    a, m = F.APP_PRIOR_WEIGHT, F.APP_PRIOR_MEAN
    b, mq, c, mu = F.P60_PRIOR_WEIGHT, F.P60_PRIOR_MEAN, F.MIN_PRIOR_WEIGHT, F.MIN_PRIOR_MEAN
    for season in SEASONS:
        d = table(season)
        papp, p60, emin = F.shrink_minutes(d['W'], d['S_app'], d['S_60'], d['S_min'])
        err = max(float((papp - p_app(d, a, m)).abs().max()),
                  float((p60 - p60_cond(d, a, m, b, mq)).abs().max()),
                  float((emin - emin_cond(d, a, m, c, mu)).abs().max()))
        print(f"{season}: rows {len(d)} | max |production - fit formula| = {err:.2e}")
        print(f"   P60 <= Papp: {bool((p60 <= papp + 1e-12).all())} | Emin <= 90*Papp: {bool((emin <= 90 * papp + 1e-12).all())}"
              f" | 60*P60 <= Emin: {bool((60 * p60 <= emin + 1e-9).all())} (needs b = c: {b == c})")


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'measure'
    if what == 'measure':
        measure()
    elif what == 'fit':
        fit()
        fit_per_position()
    elif what == 'verify':
        verify()
    elif what == 'recency':
        # re-fit of (a, m), (b, m_q), (c, m_mu) for several MINUTES_RECENCY values (notes_08c §6)
        for R in (0.4, 0.5, 0.6, 0.7, 0.8):
            res, _, _ = fit(R, verbose=False)
            print(f"recency {R} (W_max {1 / (1 - R):.2f}): {res}")
