"""Expected-points (xP) model: FPL scoring rules, model constants and pure functions.

This module is deliberately SELF-CONTAINED (only math and pandas, no import from the rest of
the package):
  - scripts/calibrate_xp.py runs exactly this code on past seasons (walk-forward backtest);
  - scripts/check_acceptance.py loads two copies of this file (baseline and candidate) as
    independent modules, which only works for a file without package-relative imports;
  - the tuning scripts change the constants below at runtime (xp_model.MINUTES_RECENCY = ...),
    and every function reads them from this module's globals when it runs.
"""
import math

import pandas as pd

# ---------------------------------------------------------------------------
# X1: FPL scoring rules 2026/27. Verified 2026-10-17 against premierleague.com
# "FPL basics explained: Scoring points" (2026-07-20). Keys = element_type:
# 1 GK, 2 DEF, 3 MID, 4 FWD.
# ---------------------------------------------------------------------------
PTS_GOAL = {1: 10, 2: 6, 3: 5, 4: 4}
PTS_ASSIST = 3
PTS_CLEAN_SHEET = {1: 4, 2: 4, 3: 1, 4: 0}        # needs 60+ minutes
PTS_DEFCON = 2                                     # capped at 2 per match
DEFCON_THRESHOLD = {2: 10, 3: 12, 4: 12}           # DEF: CBIT, MID/FWD: CBIRT; GK not eligible
SAVES_PER_POINT = 3                                # GK: +1 per 3 saves
CONCEDED_PER_MINUS_ONE = 2                         # GK/DEF: -1 per 2 goals conceded
PTS_YELLOW = -1
PTS_RED = -3

# ---------------------------------------------------------------------------
# X2: model constants. Every value is one of:
#   [measured]   computed from vaastav/Fantasy-Premier-League per-match data
#   [fitted]     chosen so that predicted and actual points of one component match
#   [tuned]      chosen by walk-forward RMSE on 2024/25, checked on 2025/26
#                (calibrate_xp.py --tune). X10: RMSE, not MAE (chat-8a used MAE):
#                MAE is minimised by the median, which is below the mean for
#                right-skewed FPL points, so it favoured under-predicting attackers.
#   [assumption] not derived from data
# Seasons: 2024/25 and 2025/26 (2025/26 is the only season with DefCon data).
# ---------------------------------------------------------------------------
# [measured] FPL assists per Opta xA. FPL awards assists Opta does not count
# (rebounds, penalties/free kicks won, deflections). 2024/25 / 2025/26:
# DEF 1.11 / 1.29, MID 1.37 / 1.33, FWD 2.08 / 2.11 -> pooled values below.
# GK: [assumption] 1.0 (only ~3 xA per season, too few to measure).
ASSIST_PER_XA = {1: 1.0, 2: 1.20, 3: 1.35, 4: 2.09}
# [measured] goals per xG, pooled 2024/25 + 2025/26: DEF 0.80, MID 0.99, FWD 1.00.
# GK: [assumption] 1.0.
GOAL_PER_XG = {1: 1.0, 2: 0.80, 3: 0.99, 4: 1.00}
# [measured] home/away team xG ratio: 1.12 (2024/25), 1.23 (2025/26), pooled 1.17.
# Split symmetrically: home x1.08, away /1.08 (sqrt(1.17) = 1.08).
HOME_ADV = 1.08
# [measured] mean minutes of an appearance that reached 60 minutes: 85.1 / 85.3.
MIN_GIVEN_60 = 85.2
# [fitted] P(clean sheet) = exp(-1.08 * expected goals conceded). With 1.0 the
# model over-predicted CS points (goals are more dispersed than Poisson(xG)).
# Best value: 1.10 on 2024/25, 1.05 on 2025/26.
CS_LAMBDA_SCALE = 1.08
# [fitted on 2025/26, in-sample] negative-binomial size of the defensive-action
# count per match. Smaller = more dispersed. Poisson (size -> infinity)
# under-predicted DefCon points (DEF 0.138 vs 0.172 actual per player-GW).
DEFCON_NB_SIZE = {2: 4.0, 3: 10.0, 4: 10.0}
# [tuned] recency weight per gameweek for the minutes model (1.0 = no recency).
# RMSE 2024/25 / 2025/26: 0.4 -> 2.5511/2.5107, 0.5 -> 2.5457/2.5070,
# 0.6 -> 2.5429/2.5064, 0.7 -> 2.5431/2.5089, 0.8 -> 2.5472/2.5141 (K=600, b=1.5);
# 0.9 / 1.0 worse in the full grid.
# 8c re-check WITH the shrinkage (a, b, c kept): 0.4 -> 2.5470/2.5018, 0.5 -> 2.5442/2.5002,
# 0.6 -> 2.5431/2.5007, 0.7 -> 2.5442/2.5037, 0.8 -> 2.5485/2.5089. 0.5 wins only in 2025/26
# (by 0.0005), so by the acceptance rule (lower in BOTH seasons) 0.6 stays. Refitting a, b, c
# for each value (check_minutes.py fit <R>) did not change this.
MINUTES_RECENCY = 0.6
# 8c: SHRINKAGE of the minutes model. The raw estimates S/W of the recency-weighted
# sums (W = sum w, S_app = sum w*[appeared], S_60 = sum w*[60+ min], S_min = sum w*minutes,
# w = MINUTES_RECENCY^age) are frequencies over ~2.5 effective GWs. Measured walk-forward on
# 2024/25 and 2025/26: a raw 1.00 means 87-88% (P60) / 82-84 minutes (Emin),
# a raw 0.00 means 3-9%. Each quantity is pulled towards a prior that counts like a
# pseudo-sample of weight a, b or c:
#     Papp = (S_app + a*m)   / (W + a)          P(appears in a match)
#     q    = (S_60  + b*m_q) / (S_app + b)      P(60+ min | appears)
#     mu   = (S_min + c*m_mu) / (S_app + c)     E[minutes | appears]
#     P60  = Papp * q,   Emin = Papp * mu
# P60 <= Papp and Emin <= 90*Papp hold by construction (q <= 1, mu <= 90); with b = c also
# 60*P60 <= Emin (S_min >= 60*S_60 and m_mu >= 60*m_q). a, b, c are in units of the weights
# (a: recency weight, b and c: weighted appearances); they depend on MINUTES_RECENCY.
# One global prior for all positions: per-position priors improved the log-loss of P60 by
# <= 0.001 and the MSE of minutes by <= 2 (of ~730-810) and were not adopted.
# [fitted] on 2024/25 (check_minutes.py fit: log-loss for P(appear) and P(60+), MSE of minutes),
# checked on 2025/26 and in the xP backtest of both seasons (calibrate_xp.py).
# [tuned] b = c = a = 0.4: the fitted b was 0.8 on the grid, but the common weight b = c
# (needed for 60*P60 <= Emin) gave a lower RMSE in both seasons (2.5431/2.5007 vs 2.5441/2.5035,
# measured with K=1200 and price exponent 1.0, before their 8c retune).
APP_PRIOR_WEIGHT = 0.4      # a [fitted]
APP_PRIOR_MEAN = 0.45       # m [fitted]: P(appear) of a player with no information
P60_PRIOR_WEIGHT = 0.4      # b [tuned]
P60_PRIOR_MEAN = 0.8        # m_q [fitted]
MIN_PRIOR_WEIGHT = 0.4      # c [fitted]
MIN_PRIOR_MEAN = 80.0       # m_mu [fitted] minutes, given that he appears
# [tuned] pseudo-minutes of the prior added to each player's per-90 rates
# (shrinkage: a player with few minutes is pulled to his prior, see X8).
# Grid {300, 600, 1200, 2400}; final value by the selection rule below (8c: 1200 -> 600).
PLAYER_PRIOR_MINUTES = 600.0
# [tuned] pseudo-matches of the league average added to each team's xG for/against.
TEAM_PRIOR_MATCHES = 10.0
# X8: the shrinkage target ("prior") of these per-90 rates depends on the player's
# price: prior = position mean * price^b / (minute-weighted mean of price^b in the
# position). The division keeps the position average unchanged; b = 0 gives the
# plain position mean (chat-8a version). Reason: with the position mean, early in
# the season every premium player was pulled towards an average player
# (backtest GW2-8, players >= GBP10m: bias -3.91 in 2024/25, -1.36 in 2025/26).
PRICE_PRIOR_STATS = ['xg', 'xa', 'bonus']
# [tuned] grid {0, 0.5, 1, 1.5, 2}. Selection rule for PLAYER_PRIOR_MINUTES (K) and this
# exponent (b), fitted on 2024/25 ONLY (decision 2026-10-20): among settings within 0.005 RMSE
# of the best, the smallest |predicted - actual| of the best XI + captain; 2025/26 is the check.
# 8a used both seasons (K=1200, b=1.0). 8c (shrunk minutes model):
# 2024/25 best RMSE 2.5352 (K=300, b=2.0); K=600, b=2.0: RMSE 2.5364, gap +0.2 -> chosen.
# Check on 2025/26: RMSE 2.4996 (6b 2.5064), gap +7.2 (6b +5.9; paired difference +1.26 +/- 1.13,
# within the acceptance tolerance of the roadmap). The gap grows with b (K=600: -6.5, -3.1,
# +0.2 for b = 1, 1.5, 2), so b > 2 would not be chosen.
PRICE_PRIOR_EXPONENT = 2.0

# X3: per-(player, GW) statistics used by the model (live endpoint names -> short names)
HIST_STATS = {
    'expected_goals': 'xg', 'expected_assists': 'xa', 'expected_goals_conceded': 'xgc',
    'defensive_contribution': 'dc', 'saves': 'saves', 'bonus': 'bonus',
    'yellow_cards': 'yc', 'red_cards': 'rc',
}
STAT_COLS = list(HIST_STATS.values())
HIST_COLS = ['element', 'gw', 'team', 'n_matches', 'minutes'] + STAT_COLS
NEUTRAL_TEAM = {'att': 1.0, 'def': 1.0}  # league-average team (no data yet)


# ---------------------------------------------------------------------------
# X4: xP model. Pure functions: no API calls, so calibrate_xp.py runs exactly
# the same code on historical data.
# ---------------------------------------------------------------------------
def poisson_pmf(lam, kmax):
    """[P(X=0), ..., P(X=kmax)] for X ~ Poisson(lam), via P(k) = P(k-1) * lam / k."""
    p = [math.exp(-lam)]
    for k in range(1, kmax + 1):
        p.append(p[-1] * lam / k)
    return p


def expected_floor_div(lam, k):
    """E[floor(X / k)] for X ~ Poisson(lam). Used for '1 point per 3 saves' and
    '-1 per 2 goals conceded', which are not linear in X."""
    if lam <= 0:
        return 0.0
    kmax = int(lam + 10 * math.sqrt(lam) + 20)  # tail beyond this is negligible
    return sum((i // k) * p for i, p in enumerate(poisson_pmf(lam, kmax)))


def negbin_tail(mean, size, threshold):
    """P(X >= threshold) for X ~ NegativeBinomial(mean, size); Var = mean + mean^2/size."""
    if mean <= 0:
        return 0.0
    q = size / (size + mean)
    p = math.exp(size * math.log(q))   # P(X = 0)
    cdf = p
    for k in range(1, threshold):
        p *= (k - 1 + size) / k * (mean / (size + mean))   # P(k) from P(k-1)
        cdf += p
    return max(0.0, 1.0 - cdf)


def minutes_sums(hist, gw_now, start_gw_of=None, recency=None):
    """8c: recency-weighted SUMS behind the minutes model, one row per player:
        W     = sum w                      (evidence: weight of all his team-match GWs)
        S_app = sum w * [minutes > 0]      (weighted number of appearances)
        S_60  = sum w * [m >= 60]          (weighted number of 60+ minute appearances)
        S_min = sum w * m                  (weighted minutes), m = minutes per team match
    with w = recency^(gw_now - 1 - gw). Only GWs before gw_now in which his team played
    (blanks carry no information), and only GWs at his CURRENT club (X9). The raw
    estimates are S_app/W, S_60/W, S_min/W; shrink_minutes() turns the sums into the
    estimates the xP model uses. Kept separate so that check_minutes.py measures the
    same estimator as the model."""
    r = MINUTES_RECENCY if recency is None else recency
    # Age = gw_now - 1 - gw is CALENDAR age: if the GW before gw_now is not final yet and so
    # missing from hist, W is smaller and the shrinkage stronger. Measured better than
    # counting age from the last GW present in the data.
    h = hist[(hist['gw'] < gw_now) & (hist['n_matches'] > 0)]
    # X9: only GWs at his CURRENT club. A GW before the player joined would count
    # as "did not appear" (0 minutes) or, for a move between PL clubs, describe his
    # role at the old club. GWs on the bench before a debut at the same club stay:
    # removing them made the backtest worse (they are real "did not play" evidence).
    if start_gw_of:
        h = h[h['gw'] >= h['element'].map(start_gw_of).fillna(0)]   # unknown join GW -> keep all
    hh = h.assign(w=r ** (gw_now - 1 - h['gw']),
                  m=h['minutes'] / h['n_matches'])               # per match (DGW: average)
    hh = hh.assign(S_app=(hh['minutes'] > 0) * hh['w'],
                   S_60=(hh['m'] >= 60) * hh['w'],
                   S_min=hh['m'] * hh['w'])
    return hh.groupby('element')[['w', 'S_app', 'S_60', 'S_min']].sum().rename(columns={'w': 'W'})


def shrink_minutes(W, S_app, S_60, S_min):
    """8c: (Papp, P60, Emin) from the sums of minutes_sums(), see the constants above
    (APP_PRIOR_*, P60_PRIOR_*, MIN_PRIOR_*). Works on scalars and on pandas Series.
    Papp is a probability per team match, Emin minutes per match."""
    papp = (S_app + APP_PRIOR_WEIGHT * APP_PRIOR_MEAN) / (W + APP_PRIOR_WEIGHT)
    q60 = (S_60 + P60_PRIOR_WEIGHT * P60_PRIOR_MEAN) / (S_app + P60_PRIOR_WEIGHT)
    mu = (S_min + MIN_PRIOR_WEIGHT * MIN_PRIOR_MEAN) / (S_app + MIN_PRIOR_WEIGHT)
    return papp, papp * q60, papp * mu


def build_features(hist, pos_of, gw_now, price_of=None, start_gw_of=None):
    """
    From per-(player, GW) history before gw_now, compute
      teams[t] = {'att', 'def'}: xG for / against per match relative to the league average
      feats[i] = minutes model (Papp, P60, Emin; shrunk, see shrink_minutes) + per-90 rates
               (shrunk to a price-dependent prior)
      L        = league-average xG per team per match
    price_of:    {player_id: price in GBPm}; None -> plain position mean (exponent ignored).
    start_gw_of: {player_id: first GW at his current club} (see first_gw_at_club); None -> all GWs.
    Players with no GW in the minutes model are NOT in feats (caller uses FPL's ep_next).
    Returns ({}, {}, None) when there is no history (GW1).
    """
    h = hist[(hist['gw'] < gw_now) & (hist['n_matches'] > 0)]   # blanks carry no information
    if h.empty or h['minutes'].sum() == 0:
        return {}, {}, None

    # Team strength. 11 players are on the pitch, so 990 player-minutes = one match:
    #   xG for per match     = 990 * sum(player xG)  / sum(player minutes)
    #   xG against per match =  90 * sum(player xGC) / sum(player minutes)
    # (every player's xGC is the team's xGA while he is on the pitch).
    L = 990.0 * h['xg'].sum() / h['minutes'].sum()
    k = TEAM_PRIOR_MATCHES
    teams = {}
    for t, r in h.groupby('team')[['xg', 'xgc', 'minutes']].sum().iterrows():
        xgf = 990.0 * (r['xg'] + L * k) / (r['minutes'] + 990.0 * k)          # + k average matches
        xga = 90.0 * (r['xgc'] + 11.0 * L * k) / (r['minutes'] + 990.0 * k)
        teams[t] = {'att': xgf / L, 'def': xga / L}   # 1.0 = league average

    # Per-90 rates with shrinkage: rate = (stat + K*prior) / (minutes + K),
    # prior = minute-weighted mean of the player's position in the same data.
    psum = h.groupby('element')[STAT_COLS + ['minutes']].sum()
    psum['pos'] = psum.index.map(pos_of)
    prior = {pos: {c: g[c].sum() / max(g['minutes'].sum(), 1.0) for c in STAT_COLS}
             for pos, g in psum.groupby('pos')}

    # X8: price factor s_i = price_i^b / (minute-weighted mean of price^b in his position).
    # The minute-weighted mean of s over a position is 1, so the position's total
    # rate is unchanged; only its split between cheap and expensive players changes.
    b = PRICE_PRIOR_EXPONENT if price_of is not None else 0.0
    psum['price'] = psum.index.map(price_of or {}).astype(float)
    psum['pb'] = psum['price'] ** b                          # NaN where the price is unknown
    pb_mean = {pos: (g['pb'] * g['minutes']).sum() / g.loc[g['pb'].notna(), 'minutes'].sum()
               for pos, g in psum.groupby('pos') if g['pb'].notna().any()}

    # Minutes model: recency-weighted share of team matches in which the player
    # appeared / reached 60 minutes, and mean minutes per team match (see minutes_sums);
    # these sums are shrunk towards a prior below (8c, shrink_minutes).
    # The per-90 rates above still use all GWs (0-minute rows add nothing to them).
    msum = minutes_sums(hist, gw_now, start_gw_of)

    feats = {}
    kp = PLAYER_PRIOR_MINUTES
    for el, r in psum.iterrows():
        pos = r['pos']
        if pos not in prior or el not in msum.index or msum.loc[el, 'W'] <= 0:
            continue   # X9: no GW at his current club yet -> ep_next fallback
        mw = msum.loc[el]
        # 8c: shrunk minutes model (Papp, P60, Emin are what the xP, the vice-captain and the
        # bench terms all use). 'raw_*' = the unshrunk S/W, shown by --explain only.
        papp, p60, emin = shrink_minutes(mw['W'], mw['S_app'], mw['S_60'], mw['S_min'])
        f = {'Papp': papp, 'P60': p60, 'Emin': emin,
             'raw_Papp': mw['S_app'] / mw['W'], 'raw_P60': mw['S_60'] / mw['W'], 'raw_Emin': mw['S_min'] / mw['W'],
             'minutes_total': r['minutes'],
             'Wapp': mw['W']}   # total recency weight (evidence) behind the minutes model
        # X8: unknown price (or b = 0) -> factor 1 -> plain position mean
        s = r['pb'] / pb_mean[pos] if (pd.notna(r['pb']) and pos in pb_mean) else 1.0
        f['price_factor'] = s
        for c in STAT_COLS:
            target = prior[pos][c] * (s if c in PRICE_PRIOR_STATS else 1.0)
            f[c + '90'] = 90.0 * (r[c] + kp * target) / (r['minutes'] + kp)
        feats[el] = f
    return feats, teams, L


def first_gw_at_club(join_dates, deadlines):
    """X9: {player_id: first GW whose deadline is on/after the date he joined his
    current club}. join_dates: {id: 'YYYY-MM-DD' or None}; deadlines: {gw: pd.Timestamp (UTC)}.
    A player who joined after every listed deadline gets max(gw) + 1."""
    gws = sorted(deadlines)
    out = {}
    for pid, jd in join_dates.items():
        if not jd:
            continue                     # unknown -> build_features keeps all his GWs
        t = pd.Timestamp(jd, tz='UTC')
        out[pid] = next((g for g in gws if deadlines[g] >= t), gws[-1] + 1)
    return out


def match_xp(f, pos, own, opp, is_home, L):
    """
    Expected FPL points of one player in one match = sum of components, each
    computed as (FPL points per event) x (expected number of events).
    Same structure for every position; only the points table differs.
    """
    h = HOME_ADV if is_home else 1.0 / HOME_ADV
    att_mult = opp['def'] * h                      # our attack vs this defence, relative to average
    lam_against = L * opp['att'] * own['def'] / h  # expected goals conceded, full match
    papp, p60, emin = f['Papp'], f['P60'], f['Emin']
    min_if_app = emin / papp if papp > 0 else 0.0  # expected minutes given he plays
    c = {}
    c['appearance'] = papp + p60                   # 1 pt for playing, +1 for 60+ minutes
    c['goals'] = PTS_GOAL[pos] * GOAL_PER_XG[pos] * f['xg90'] * emin / 90.0 * att_mult
    c['assists'] = PTS_ASSIST * ASSIST_PER_XA[pos] * f['xa90'] * emin / 90.0 * att_mult
    c['clean_sheet'] = (PTS_CLEAN_SHEET[pos] * p60
                        * math.exp(-CS_LAMBDA_SCALE * lam_against * MIN_GIVEN_60 / 90.0))
    c['conceded'] = 0.0
    c['saves'] = 0.0
    c['defcon'] = 0.0
    if pos in (1, 2):
        # -1 per 2 goals conceded while on the pitch: E[floor(G/2)], G ~ Poisson
        c['conceded'] = -papp * expected_floor_div(lam_against * min_if_app / 90.0, CONCEDED_PER_MINUS_ONE)
    if pos == 1:
        lam_saves = f['saves90'] * min_if_app / 90.0 * opp['att'] / h   # opponent attacks at 1/h
        c['saves'] = papp * expected_floor_div(lam_saves, SAVES_PER_POINT)
    else:
        # DefCon: 2 pts if the action count reaches the threshold; only 60+ minute
        # appearances are counted (measured 2025/26: <60 min reached it in <=0.5%)
        mean_dc = f['dc90'] * MIN_GIVEN_60 / 90.0
        c['defcon'] = PTS_DEFCON * p60 * negbin_tail(mean_dc, DEFCON_NB_SIZE[pos], DEFCON_THRESHOLD[pos])
    c['bonus'] = f['bonus90'] * emin / 90.0
    c['cards'] = (PTS_YELLOW * f['yc90'] + PTS_RED * f['rc90']) * emin / 90.0
    # Not modelled (small): own goals, penalty saves, penalty misses.
    return c
