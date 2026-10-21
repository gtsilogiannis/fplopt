"""
Advanced FPL MILP Optimizer Engine (v2.0 - Native Stack)
Zero-dependency overhead: Only requires requests, pandas, pulp.
Usage:
  - Optimization: python fpl_v2.py --team <TEAM_ID> [OPTIONS]
  - Calibration:  python fpl_v2.py --mode evaluate --gw <GW_NUMBER>
"""

import requests
import pandas as pd
import pulp
import argparse
import sys
import os
import math
import time  # M7: solve time is reported (lookahead choice, notes_06b)
import tempfile  # M7: CBC log file, read for the optimality gap
try:
    import highspy  # M10: optional MIP solver (pip install highspy); CBC is used without it
except ImportError:
    highspy = None
# U1 (chat 8b): 're', 'json', 'unicodedata' were used only by the removed Understat
# scraper and clean_name(); 'csv' was unused (the log is written with pandas).

# PuLP 4.x removed LpVariable.dicts, LpStatus and the bundled CBC (PULP_CBC_CMD).
# This script is written against the 3.x API, so stop early with a clear message
# instead of crashing later with an AttributeError.
if int(pulp.__version__.split('.')[0]) >= 4:
    sys.exit(f"[ERROR] PuLP {pulp.__version__} found; this script needs PuLP 3.x. "
             f"Install it with: pip install 'pulp==3.3.2'")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
FPL_API = "https://fantasy.premierleague.com/api"
# X6: STRENGTH_BASELINE removed (FPL strength ratings are no longer used).
# Discount weight per gameweek AHEAD in the objective (decision parameter, not part of xP).
# Assumption (unvalidated): kept at 0.85 until the multi-period model (chat 6b).
TIME_DECAY_RATE = 0.85
# V1: new log file (one row per player per GW, see log_predictions). The old
# predictions_history.csv has a different schema and is no longer read.
PREDICTIONS_FILE = "predictions_log.csv"
XI_ROLES = ['Start', 'Captain', 'Vice']   # M6: logged roles of the starting XI ('Vice' from chat 6b)
HIT_COST = 4  # FPL rule: points deducted per transfer beyond free transfers (2026/27)

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
# 8c: SHRINKAGE of the minutes model (notes_08c). The raw estimates S/W of the recency-weighted
# sums (W = sum w, S_app = sum w*[appeared], S_60 = sum w*[60+ min], S_min = sum w*minutes,
# w = MINUTES_RECENCY^age) are frequencies over ~2.5 effective GWs. Measured walk-forward on
# 2024/25 and 2025/26 (notes_08c §2): a raw 1.00 means 87-88% (P60) / 82-84 minutes (Emin),
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
# <= 0.001 and the MSE of minutes by <= 2 (of ~730-810) and were not adopted (notes_08c §4).
# [fitted] on 2024/25 (check_minutes.py fit: log-loss for P(appear) and P(60+), MSE of minutes),
# checked on 2025/26 and in the xP backtest of both seasons (calibrate_xp.py, notes_08c §5).
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
# 8a used both seasons (K=1200, b=1.0). 8c (shrunk minutes model, notes_08c §6.2):
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


def fetch_fpl_api(url):
    """Centralized HTTP GET request handler with explicit error trapping."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=12)
        if response.status_code == 404:
            print(f"[ERROR] 404 Not Found: Verify Team ID or Target Gameweek.\nEndpoint: {url}")
            sys.exit(1)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network/FPL API Handshake Failure: {e}")
        sys.exit(1)


# U1 (chat 8b): fetch_understat_data() and clean_name() removed. Understat no longer
# embeds 'playersData' in the league page (data now comes from an undocumented XHR
# endpoint), the xP model uses the per-GW Opta xG/xA of the FPL live endpoint (X6),
# and no backtest showed a gain from a second xG source. Reasons: notes_08b.


def is_chosen(var):
    """True if a binary LpVariable equals 1 in the solution (float-safe)."""
    # D: solvers return floats, e.g. 0.9999999 or 1e-10 for a binary variable.
    # 'varValue == 1' can miss a selected player; 0.5 is the safe midpoint.
    # varValue is None if the variable was never solved.
    return var.varValue is not None and var.varValue > 0.5


def get_current_gw(events):
    for event in events:
        if event['is_next']:
            return event['id']
    return 1


def selling_price(purchase, now):
    """S1: FPL sell-on rule (2026/27): a price rise is shared 50/50 with the game,
    rounded DOWN to GBP 0.1m; a price fall is passed on in full.
    Arguments and result in tenths of GBPm (integers, as in the API): 70 = GBP 7.0m.
    Example: bought 70, now 73 -> profit 3 -> 3 // 2 = 1 -> sells for 71."""
    if now > purchase:
        return purchase + (now - purchase) // 2   # '//' = integer (floor) division
    return now


def load_current_squad(team_id, current_gw, bootstrap, history=None):
    """
    Squad, bank and selling prices that the next transfers start from.
    R1: a Free Hit squad reverts at the next deadline. If the Free Hit was played in
        GW-1, the picks of GW-1 are the TEMPORARY squad; the real one is GW-2's.
    S1: the public API has no selling prices (my-team/ needs a login), so purchase
        prices are rebuilt: the picks of the manager's first GW, then every later
        transfer in time order (element_in_cost = purchase price), skipping
        Free Hit transfers because they are reverted.
    Returns (list of player ids, bank in GBPm, {player id: selling price in GBPm}).
    """
    if history is None:   # M8: optimize_team passes it (also used for the free transfers)
        history = fetch_fpl_api(f"{FPL_API}/entry/{team_id}/history/")
    # chips: [{'name': 'freehit', 'event': 5, ...}, ...]
    fh_gws = {c['event'] for c in history.get('chips', []) if c.get('name') == 'freehit'}
    first_gw = history['current'][0]['event'] if history.get('current') else 1  # GW the team started

    squad_gw = current_gw - 1
    if squad_gw in fh_gws and squad_gw - 1 >= first_gw:
        squad_gw -= 1   # R1: GW-1 was a Free Hit -> the squad reverts to the one of GW-2
        print(f"[*] Free Hit played in GW{current_gw - 1}: squad and bank taken from GW{squad_gw}.")
    picks = fetch_fpl_api(f"{FPL_API}/entry/{team_id}/event/{squad_gw}/picks/")
    squad = [p['element'] for p in picks['picks']]
    bank = picks['entry_history']['bank']          # tenths of GBPm

    now_cost = {p['id']: p['now_cost'] for p in bootstrap['elements']}
    start_price = {p['id']: p['now_cost'] - p['cost_change_start'] for p in bootstrap['elements']}
    first_picks = picks if first_gw == squad_gw else \
        fetch_fpl_api(f"{FPL_API}/entry/{team_id}/event/{first_gw}/picks/")   # no second request if same GW
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
        joined = pd.Timestamp(fetch_fpl_api(f"{FPL_API}/entry/{team_id}/")['joined_time'])
        lo, hi = {}, {}
        for pid in first_squad:
            rows = sorted((h for h in fetch_fpl_api(f"{FPL_API}/element-summary/{pid}/")['history']
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
        print(f"[WARNING] First-squad prices (GBP {sum(lo.values()) / 10:.1f}-{sum(hi.values()) / 10:.1f}m) "
              f"do not contain the amount actually paid (GBP {total / 10:.1f}m); selling prices may be off.")

    # Replay the transfers on both bounds (a transfer sets an EXACT purchase price).
    pur_lo, pur_hi = dict(lo), dict(hi)
    replay_bank = first_bank
    transfers = fetch_fpl_api(f"{FPL_API}/entry/{team_id}/transfers/")
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
        print(f"[WARNING] Transfer replay does not match the GW{squad_gw} squad; players without a "
              f"known purchase price are valued at their current price.")
    if replay_bank != bank:
        print(f"[WARNING] Rebuilt bank £{replay_bank / 10:.1f}m != API bank £{bank / 10:.1f}m: "
              f"selling prices may be off. Check them in the app; use --override-budget if needed.")

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
        print(f"[*] Selling value of the squad is between £{sum(sell.values()):.1f}m and "
              f"£{sum(sell_hi.values()):.1f}m (purchase price not exactly known: {unsure}). "
              f"Using the lower value; --override-budget sets the exact one from the app.")
    return squad, bank / 10.0, sell


def build_fixture_map(current_gw, max_gw_lookahead, all_fixtures):
    # X3: uses the full fixture list (one request to /fixtures/) instead of one
    # request per gameweek; the same list also gives past match counts.
    team_fixtures = {i: [] for i in range(1, 21)}
    horizon_gws = set(range(current_gw, min(current_gw + max_gw_lookahead, 39)))
    for f in all_fixtures:
        gw = f['event']
        if gw not in horizon_gws or f['finished']:  # event is None for unscheduled fixtures
            continue
        team_fixtures[f['team_h']].append({'opp': f['team_a'], 'is_home': True, 'gw': gw})
        team_fixtures[f['team_a']].append({'opp': f['team_h'], 'is_home': False, 'gw': gw})
    return team_fixtures


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
    # counting age from the last GW present in the data (notes_08c §8).
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


def fetch_history(bootstrap, all_fixtures, current_gw):
    """X3: per-(player, GW) stats of every FINISHED gameweek before current_gw,
    from /event/{gw}/live/ (one request per gameweek)."""
    team_of = {p['id']: p['team'] for p in bootstrap['elements']}
    gws = [e['id'] for e in bootstrap['events'] if e['finished'] and e['id'] < current_gw]
    n_matches = {}
    for f in all_fixtures:
        if f['event'] in gws:
            for t in (f['team_h'], f['team_a']):
                n_matches[(t, f['event'])] = n_matches.get((t, f['event']), 0) + 1
    rows = []
    missing = set()
    for gw in gws:
        live = fetch_fpl_api(f"{FPL_API}/event/{gw}/live/")
        for el in live['elements']:
            pid = el['id']
            if pid not in team_of:
                continue
            s = el['stats']
            # Limitation: a player who moved between PL clubs this season is assigned
            # to his CURRENT club for all past GWs (the live endpoint has no team field).
            row = {'element': pid, 'gw': gw, 'team': team_of[pid],
                   'n_matches': n_matches.get((team_of[pid], gw), 0),
                   'minutes': int(s.get('minutes') or 0)}
            for api_name, short in HIST_STATS.items():
                if api_name not in s:
                    missing.add(api_name)
                row[short] = float(s.get(api_name) or 0.0)   # xG fields are strings in the API
            rows.append(row)
    if missing:
        print(f"[WARNING] Live endpoint lacks fields {sorted(missing)}; treated as 0.")
    print(f"[*] History: {len(gws)} finished GWs, {len(rows)} player-GW rows")
    return pd.DataFrame(rows, columns=HIST_COLS)


def log_predictions(gw, players, final_squad):
    """V1: one row per player in the pool for this GW (not only the XI), so the
    evaluation is not limited to the players the optimizer already liked.
    Re-running for the same GW replaces that GW's rows (no duplicates)."""
    role = {p['id']: p['role'] for p in final_squad}
    new = pd.DataFrame([{
        'Gameweek': gw, 'Player_ID': p['id'], 'Name': p['name'], 'Position': p['position'],
        'Model_xP': p['xp_gw'][gw], 'FPL_ep_next': p['ep_next'],
        'Role': role.get(p['id'], 'Pool'),  # 'Pool' = not in the chosen squad ('None' reads back as NaN)
        'Source': p['source'],
    } for p in players if gw in p['xp_gw']])
    if os.path.isfile(PREDICTIONS_FILE):
        old = pd.read_csv(PREDICTIONS_FILE)
        old = old[old['Gameweek'] != gw]           # drop the previous run for this GW
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(PREDICTIONS_FILE, index=False)
    print(f"[*] {len(players)} predictions for GW{gw} written to {PREDICTIONS_FILE}")


def evaluate_model(gw):
    """V2: compares logged predictions for one GW with the actual points.
    Per-player errors use unmultiplied points (the captain is not doubled), and
    FPL's own ep_next for the same players is the baseline."""
    if not os.path.isfile(PREDICTIONS_FILE):
        print(f"[ERROR] {PREDICTIONS_FILE} not found. Run an optimization before the GW deadline first.")
        sys.exit(1)

    df = pd.read_csv(PREDICTIONS_FILE)
    df = df[df['Gameweek'] == gw]
    if df.empty:
        print(f"[WARNING] {PREDICTIONS_FILE} has no predictions for GW{gw}.")
        return

    live_data = fetch_fpl_api(f"{FPL_API}/event/{gw}/live/")
    actual = {el['id']: el['stats']['total_points'] for el in live_data['elements']}
    df = df.assign(Actual=df['Player_ID'].map(actual).fillna(0.0))

    def report(label, d):
        if d.empty:
            return
        e_m = d['Model_xP'] - d['Actual']
        e_f = d['FPL_ep_next'] - d['Actual']
        print(f"{label:<34} {len(d):>4} | {e_m.abs().mean():>6.2f} {e_f.abs().mean():>6.2f} | "
              f"{e_m.mean():>+6.2f} {e_f.mean():>+6.2f}")

    print("\n" + "=" * 72)
    print(f" MODEL EVALUATION | GW{gw}")
    print("=" * 72)
    print(f"{'Players':<34} {'n':>4} | {'MAE':>6} {'FPL':>6} | {'bias':>6} {'FPL':>6}")
    print("-" * 72)
    report("All logged", df)
    report("Relevant (model or FPL >= 2)", df[(df['Model_xP'] >= 2) | (df['FPL_ep_next'] >= 2)])
    report("Your starting XI", df[df['Role'].isin(XI_ROLES)])
    print("-" * 72)
    print("MAE = mean |predicted - actual|, bias = mean (predicted - actual). FPL = ep_next baseline.")

    # Team total, captain counted twice on both sides. Autosubs are ignored.
    xi = df[df['Role'].isin(XI_ROLES)]
    if not xi.empty:
        cap = xi[xi['Role'] == 'Captain']
        pred = xi['Model_xP'].sum() + cap['Model_xP'].sum()
        act = xi['Actual'].sum() + cap['Actual'].sum()
        print(f"XI + captain: predicted {pred:.1f}, actual {act:.0f} (no chips, no autosubs)")
    print("One GW is a small, noisy sample; judge the model over several GWs.")


# ---------------------------------------------------------------------------
# M: multi-period model (chat 6b). Rules verified 2026-10-19 (notes_06b §2):
#   - 1 free transfer (FT) per GW, unused FTs roll over, at most 5 stored.
#   - Every transfer beyond the FTs costs HIT_COST points in that GW.
#   - Wildcard / Free Hit: unlimited free transfers in that GW; the FTs held
#     before the chip are kept for the next GW, but no new one is added
#     (F[g+1] = F[g]). Free Hit squad reverts at the next deadline.
#   - Captain and vice-captain come from the XI; if the captain plays 0 minutes
#     the multiplier (x2, or x3 with Triple Captain) moves to the vice-captain.
#   - Autosubs: a starter with 0 minutes is replaced by the first bench player
#     (in bench order) who played and keeps a valid formation; GK only by GK.
# ---------------------------------------------------------------------------
MAX_FT = 5  # FPL rule 2026/27: maximum number of stored free transfers
# [assumption] stop re-solving when the bench weights implied by the chosen XI
# differ from the ones used by at most this much (probability units).
BENCH_BETA_TOL = 0.02
BENCH_MAX_PASSES = 3  # [assumption] cap on solves for the bench-weight fixed point
# Tie-break, not a model of anything: among plans with the same expected points,
# prefer fewer transfers (e.g. a swap of two equal players in the last GW of the
# horizon, where a free transfer has no value for the solver). 0.001 points per
# transfer is far below any xP difference the model can resolve.
TRANSFER_TIEBREAK = 0.001
# M9 (chat 6b) shrank P(appear) only for the vice-captain / autosub terms with
# PAPP_PRIOR_WEIGHT / PAPP_PRIOR_MEAN. Since 8c the whole minutes model is shrunk in
# build_features (APP_PRIOR_* above), so those terms use the same Papp as the xP.
# M7/M10: wall-clock limit (seconds) for the WHOLE solve, all bench-weight passes
# together. When it is reached, the solver returns the best plan found so far and its
# proven bound; the gap (bound - plan) is printed, so the result is a guarantee "at
# most GAP weighted points below the best possible plan of this model".
# [decision] set from the timing runs in notes_06b §9 and §18.
SOLVER_TIME_LIMIT = 180
# M10: [decision] stop when the plan is proven within 0.05% of the best possible plan of
# this model (~0.1 weighted points at lookahead 5). The xP error is far larger (RMSE ~3
# points per player-GW), and proving the last 0.05% took most of the solve time
# (CBC, lookahead 5: 33 s exact -> 17 s; notes_06b §18). The gap is always printed.
MIP_REL_GAP = 0.0005
MIN_PASS_SECONDS = 10  # [assumption] no extra bench-weight pass with less time than this

# ---------------------------------------------------------------------------
# C1 (chat 7): chips 2026/27, verified 2026-10-20 (premierleague.com "What's happening
# with FPL chips in 2026/27"; onefpl.com; notes_07 §1):
#   - two sets of WC, FH, BB, TC; set 1 must be played by the GW19 deadline, set 2 GW20-38
#   - one chip per GW; WC and FH not in GW1; no FH in two consecutive GWs (GW19 -> GW20)
#   - WC/FH keep the banked free transfers (F[g+1] = F[g], as in M2/M3)
# The API (bootstrap-static 'chips') lists each chip with start_event / stop_event; the
# windows below are used only if that field is missing.
# ---------------------------------------------------------------------------
CHIPS = ('WC', 'FH', 'BB', 'TC')
CHIP_API_NAME = {'WC': 'wildcard', 'FH': 'freehit', 'BB': 'bboost', 'TC': '3xc'}   # names in the API
CHIP_RULE_WINDOWS = {'WC': [(2, 19), (20, 38)], 'FH': [(2, 19), (20, 38)],
                     'BB': [(1, 19), (20, 38)], 'TC': [(1, 19), (20, 38)]}
# C2: largest possible value of T - F in one GW: at most 15 transfers in, at least 1 FT.
# The smallest constant that switches the hit and FT constraints off in a WC/FH GW
# (notes_07 §2), so it is the tightest big-M for them; it is a rule of the game, not a tuning value.
CHIP_TRANSFER_M = 15 - 1
# Tie-break, not a model of anything (same role as TRANSFER_TIEBREAK): a chip whose use gains
# nothing is not played. Without it the solver played a Free Hit with no transfers in the last
# GW of a test horizon (harness_7 rules), because an unused chip has no value inside the MILP.
CHIP_TIEBREAK = 0.001


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


def poisson_binomial_tail(qs, kmax=3):
    """[P(N>=1), ..., P(N>=kmax)] where N = number of independent events with
    probabilities qs (here: starters who do not appear). Exact, by building the
    distribution of N one player at a time (dynamic programming)."""
    dist = [1.0]                                   # dist[n] = P(N = n) so far
    for q in qs:
        new = [0.0] * (len(dist) + 1)
        for n, pn in enumerate(dist):
            new[n] += pn * (1.0 - q)               # this starter plays
            new[n + 1] += pn * q                   # this starter does not
        dist = new
    return [sum(dist[k:]) for k in range(1, kmax + 1)]


def reference_xi(squad, pos, xp_g):
    """Best valid XI of a 15-man squad for one GW, by xP (greedy; exact for the
    FPL formation rules because only minimums bind inside a 2-5-5-3 squad)."""
    by_pos = {p: sorted((i for i in squad if pos[i] == p), key=lambda i: -xp_g[i]) for p in (1, 2, 3, 4)}
    xi = by_pos[1][:1] + by_pos[2][:3] + by_pos[3][:2] + by_pos[4][:1]   # formation minimums
    rest = sorted((i for p in (2, 3, 4) for i in by_pos[p] if i not in xi), key=lambda i: -xp_g[i])
    return xi + rest[:11 - len(xi)]


def bench_weights(xi, pos, papp_g):
    """M5: beta_k = P(at least k outfield starters do not appear), k = 1, 2, 3.
    [assumption] independence between players; formation rules and a bench
    player who does not appear himself are ignored (notes_06b §8)."""
    return poisson_binomial_tail([1.0 - papp_g[i] for i in xi if pos[i] != 1], 3)


def cbc_bound(log_path):
    """M7: best bound proven by CBC for this maximisation, read from its log. The final
    summary prints 'Upper bound: X' when CBC stops on the time limit; if it is missing,
    the last progress line 'best possible -X' is used (printed for the internal
    minimisation, i.e. with the sign flipped). None if neither is found."""
    try:
        with open(log_path) as fh:
            lines = fh.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            if line.startswith('Upper bound:'):
                return float(line.split(':')[1])
            if 'best possible' in line:
                return -float(line.split('best possible')[1].split()[0].rstrip(','))
        except (IndexError, ValueError):
            continue
    return None


if highspy is not None:
    class HighsSolver(pulp.HiGHS):
        """M10: PuLP's HiGHS interface plus (1) a starting solution from the variables'
        initial values (PuLP 3.3.2 passes none to HiGHS) and (2) the proven bound."""
        def __init__(self, warm_start=False, **kw):
            super().__init__(msg=False, **kw)
            self.warm_start, self.bound = warm_start, None

        def callSolver(self, lp):
            if self.warm_start:
                sol = highspy.HighsSolution()
                # column order = lp.variables(), the order used by buildSolverModel
                sol.col_value = [v.varValue if v.varValue is not None else 0.0 for v in lp.variables()]
                lp.solverModel.setSolution(sol)
            lp.solverModel.run()

        def findSolutionValues(self, lp):
            info = lp.solverModel.getInfo()
            # HiGHS minimises -objective, so its dual (lower) bound is minus our upper bound
            if info.mip_dual_bound not in (float('inf'), float('-inf')):
                self.bound = -info.mip_dual_bound
            return super().findSolutionValues(lp)


def run_solver(prob, solver, time_limit, warm_start):
    """M10: solve with CBC or HiGHS. Both stop at the relative gap MIP_REL_GAP or at the
    time limit. Returns {'solver', 'proven' (stopped by the gap, not by time), 'bound'}."""
    name = solver or default_solver()
    if name == 'highs':
        cmd = HighsSolver(warm_start=warm_start, timeLimit=time_limit, gapRel=MIP_REL_GAP)
        prob.solve(cmd)
        return {'solver': 'HiGHS ' + highspy.__version__ if hasattr(highspy, '__version__') else 'HiGHS',
                'proven': prob.sol_status == pulp.LpSolutionOptimal, 'bound': cmd.bound}
    log_path = os.path.join(tempfile.gettempdir(), f"fpl_cbc_{os.getpid()}.log")
    prob.solve(pulp.PULP_CBC_CMD(msg=0, warmStart=warm_start, logPath=log_path,
                                 timeLimit=time_limit, gapRel=MIP_REL_GAP))
    proven = prob.sol_status == pulp.LpSolutionOptimal
    bound = None if proven else cbc_bound(log_path)
    if proven and prob.objective is not None:
        # stopped by the gap: the bound is not printed in a parsable way; MIP_REL_GAP
        # is the guarantee (bound <= objective * (1 + MIP_REL_GAP))
        bound = pulp.value(prob.objective) * (1.0 + MIP_REL_GAP)
    try:
        os.remove(log_path)
    except OSError:
        pass
    return {'solver': 'CBC', 'proven': proven, 'bound': bound}


def default_solver():
    """M10: HiGHS when highspy is installed (faster and more predictable on the larger
    horizons, notes_06b §18), otherwise the CBC bundled with PuLP 3.x."""
    return 'highs' if highspy is not None else 'cbc'


def is_var(e):
    """C3: True for a PuLP variable or expression (the chip switches below are either a
    variable or a fixed 0/1 number; a constraint with no variable is not added)."""
    return isinstance(e, (pulp.LpVariable, pulp.LpAffineExpression))


def solve_plan(pool, horizon, weight, start_squad, budget, ft0, chip='NONE', max_hits=None,
               beta=None, use_vice=True, use_bench=True, warm=None, time_limit=None, solver=None,
               chips_avail=None, chip_fix=None):
    """
    M1-M7: one MILP over the whole horizon.
    pool: dict of dicts keyed by player id: 'pos', 'team', 'price' (now_cost, GBPm),
          'sell' (selling price, owned players only), 'xp' {(i, g): xP}, 'papp' {(i, g):
          P(appears in GW g)}, 'avail' (False -> may not start in the first GW).
    start_squad: ids owned before the first GW of the horizon ([] at GW1).
    ft0: free transfers in the first GW; None = unlimited (before the first deadline).
    chip: chip of the first GW: 'WC'/'FH'/'BB'/'TC' = played, 'NONE' = none, 'AUTO' = the
          model decides (C3, only with chips_avail).
    chips_avail: C3 {chip: [(start GW, stop GW), ...]} windows of the chip sets still
          unused; the model may play each at most once per window. None = no chip planning.
    chip_fix: C3 {(chip, g): 0 or 1} forces chip decisions (used for the opportunity costs).
    beta: {g: [b1, b2, b3]} outfield bench weights (None -> no outfield bench value).
    warm: {variable name: value} of a previous solve, given to CBC as a starting
          solution (the bench fixed point re-solves an almost identical problem).
    Returns a dict with the solver status and the plan (see the end of the function).
    """
    ids = pool['ids']
    pos, team, price, sell, xp, papp = (pool[k] for k in ('pos', 'team', 'price', 'sell', 'xp', 'papp'))
    t0 = horizon[0]
    S0 = set(start_squad)
    ig = [(i, g) for i in ids for g in horizon]
    out_ids = [i for i in ids if pos[i] != 1]
    gk_ids = [i for i in ids if pos[i] == 1]

    prob = pulp.LpProblem("FPL_Multi_Period", pulp.LpMaximize)
    # --- C3: chip switches u[(c, g)]: a binary variable where the chip may be played,
    # otherwise a fixed number (1 = played, 0 = not). notes_07 §2 ---
    windows = chips_avail or {}
    fix = dict(chip_fix or {})
    u = {}
    for cp in CHIPS:
        for g in horizon:
            if (cp, g) in fix:
                u[(cp, g)] = fix[(cp, g)]
            elif g == t0 and chip != 'AUTO':
                u[(cp, g)] = 1 if cp == chip else 0         # --chip: forced in the current GW
            elif any(s <= g <= e for s, e in windows.get(cp, [])):
                u[(cp, g)] = pulp.LpVariable(f"chip_{cp}_{g}", cat='Binary')
            else:
                u[(cp, g)] = 0
    for g in horizon:                                       # (K1) at most one chip per GW
        if any(is_var(u[(cp, g)]) for cp in CHIPS):
            prob += pulp.lpSum(u[(cp, g)] for cp in CHIPS) <= 1
    for cp in CHIPS:                                        # (K2) each unused chip at most once per set
        for s, e in windows.get(cp, []):
            terms = [u[(cp, g)] for g in horizon if s <= g <= e]
            if any(is_var(t_) for t_ in terms):
                prob += pulp.lpSum(terms) <= 1
    for g in horizon:                                       # (K3) no Free Hit in two consecutive GWs
        if g + 1 in horizon and (is_var(u[('FH', g)]) or is_var(u[('FH', g + 1)])):
            prob += u[('FH', g)] + u[('FH', g + 1)] <= 1
    chip_vars = [u_ for u_ in u.values() if is_var(u_)]
    # phi_g = 1 if a Wildcard or Free Hit is played in GW g (one chip per GW, so 0 or 1)
    phi = {g: u[('WC', g)] + u[('FH', g)] for g in horizon}
    # first GW: transfers cost nothing (before the first deadline, or a WC/FH fixed in t0)
    free_now = ft0 is None or (not is_var(phi[t0]) and phi[t0] == 1)
    x = pulp.LpVariable.dicts("x", ig, cat='Binary')          # M1: squad in GW g
    l = pulp.LpVariable.dicts("l", ig, cat='Binary')          # XI in GW g
    c = pulp.LpVariable.dicts("c", ig, cat='Binary')          # captain in GW g
    # M1: transfers in. Continuous is enough: at the optimum in = max(0, x - x_prev),
    # because every extra unit can only cost FTs or hits. 'out' is not needed:
    # the squad has 15 players in every GW, so #out = #in.
    t_in = pulp.LpVariable.dicts("in", ig, lowBound=0, upBound=1)

    # C4: permanent squad y[(i, g)] after GW g = what the transfers of GW g+1 start from.
    # A Free Hit squad is temporary: after an FH in g the permanent squad is still y[g-1]
    # (M3 did this only for an FH fixed in the first GW).
    #   no FH in g:       y = x[g]
    #   FH fixed in g:    y = y[g-1]
    #   FH variable:      continuous y in [0, 1] with  y <= x[g] + u_FH,  y <= y[g-1] + 1 - u_FH.
    # The solver wants y as high as possible (in >= x - y), so y = x[g] for u_FH = 0 and
    # y = y[g-1] for u_FH = 1 (notes_07 §2). Not needed after the last GW of the horizon.
    y = {}
    for g in horizon:
        uf = u[('FH', g)]
        for i in ids:
            y_prev = y[(i, g - 1)] if g > t0 else (1 if i in S0 else 0)
            if is_var(uf) and g != horizon[-1]:
                yv = pulp.LpVariable(f"y_{i}_{g}", lowBound=0, upBound=1)
                prob += yv <= x[(i, g)] + uf
                prob += yv <= y_prev + 1 - uf
                y[(i, g)] = yv
            elif not is_var(uf) and uf == 1:
                y[(i, g)] = y_prev                       # Free Hit fixed in g: squad reverts
            else:
                y[(i, g)] = x[(i, g)]

    def prev_x(i, g):
        # squad that the transfers of GW g start from (C4: after a Free Hit, the one before it)
        return y[(i, g - 1)] if g > t0 else (1 if i in S0 else 0)

    obj = []
    for g in horizon:
        w = weight[g]
        u_bb, u_tc = u[('BB', g)], u[('TC', g)]
        bb = not is_var(u_bb) and u_bb == 1                   # Bench Boost fixed in g (M3)
        mult = 2.0 if (not is_var(u_tc) and u_tc == 1) else 1.0   # extra captain multiplier (TC fixed)
        # --- squad rules (every GW) ---
        prob += pulp.lpSum(x[(i, g)] for i in ids) == 15
        for p_, n_ in ((1, 2), (2, 5), (3, 5), (4, 3)):
            prob += pulp.lpSum(x[(i, g)] for i in ids if pos[i] == p_) == n_
        for t_id in set(team.values()):
            prob += pulp.lpSum(x[(i, g)] for i in ids if team[i] == t_id) <= 3
        # --- M1: continuity  x[g] - x_prev <= in[g] ---
        for i in ids:
            prob += t_in[(i, g)] >= x[(i, g)] - prev_x(i, g)
        # --- XI, captain ---
        prob += pulp.lpSum(l[(i, g)] for i in ids) == 11
        prob += pulp.lpSum(l[(i, g)] for i in gk_ids) == 1
        prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == 2) >= 3
        prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == 3) >= 2
        prob += pulp.lpSum(l[(i, g)] for i in ids if pos[i] == 4) >= 1
        prob += pulp.lpSum(c[(i, g)] for i in ids) == 1
        for i in ids:
            prob += l[(i, g)] <= x[(i, g)]
            prob += c[(i, g)] <= l[(i, g)]
        scorer = x if bb else l                               # BB: all 15 score
        obj.append(w * pulp.lpSum(xp[(i, g)] * scorer[(i, g)] for i in ids))
        obj.append(w * mult * pulp.lpSum(xp[(i, g)] * c[(i, g)] for i in ids))
        # C5: chip played by the model in g. Each extra term is a product of two binaries,
        # written with a continuous variable bounded by both (notes_07 §2):
        #   BB: rho_i = (x_i - l_i) * u_BB  (bench player i scores),
        #   TC: tau_i = c_i * u_TC          (captain scores once more).
        # The solver raises a variable with a positive coefficient to min(a, b) = a*b; for a
        # negative xP the lower bound rho >= a + b - 1 forces the product. A captain with
        # xP <= 0 is never chosen while a starter has xP > 0, so tau covers xP > 0 only.
        if is_var(u_bb):
            rho = {i: pulp.LpVariable(f"bb_{i}_{g}", lowBound=0, upBound=1) for i in ids if xp[(i, g)] != 0}
            for i, r_ in rho.items():
                if xp[(i, g)] > 0:
                    prob += r_ <= x[(i, g)] - l[(i, g)]
                    prob += r_ <= u_bb
                else:
                    prob += r_ >= x[(i, g)] - l[(i, g)] + u_bb - 1
            obj.append(w * pulp.lpSum(xp[(i, g)] * r_ for i, r_ in rho.items()))
        if is_var(u_tc):
            tau = {i: pulp.LpVariable(f"tc_{i}_{g}", lowBound=0, upBound=1) for i in ids if xp[(i, g)] > 0}
            for i, t_ in tau.items():
                prob += t_ <= c[(i, g)]
                prob += t_ <= u_tc
            obj.append(w * pulp.lpSum(xp[(i, g)] * t_ for i, t_ in tau.items()))
        m_big = max([xp[(i, g)] for i in ids] + [0.0])        # M >= any xP in this GW

        # --- M6: vice-captain. Bonus = xP_C + (1 - P_C) * xP_VC (independence). ---
        # q = 1 - P_C = 1 - sum_i P_i * c_i is LINEAR (exactly one captain), so the only
        # product left is q * xP_VC = sum_j xP_j * (v_j * q). It is split over the
        # vice-captain candidates: s_j = v_j * q with
        #     s_j <= v_j,   sum_j s_j <= q,   s_j >= 0,   bonus part = sum_j xP_j * s_j.
        # Exact for binary v: only the vice-captain can have s > 0, and s <= q.
        # No big-M constant appears, which keeps the LP relaxation tight (notes_06b §9).
        if use_vice and m_big > 0:
            v = {i: pulp.LpVariable(f"v_{i}_{g}", cat='Binary') for i in ids}
            s_vc = {i: pulp.LpVariable(f"svc_{i}_{g}", lowBound=0) for i in ids}
            prob += pulp.lpSum(v.values()) == 1
            prob += pulp.lpSum(s_vc.values()) <= 1 - pulp.lpSum(papp[(i, g)] * c[(i, g)] for i in ids)
            for i in ids:
                prob += v[i] <= l[(i, g)]
                prob += v[i] + c[(i, g)] <= 1
                prob += s_vc[i] <= v[i]
            obj.append(w * mult * pulp.lpSum(xp[(i, g)] * s_vc[i] for i in ids))
            if is_var(u_tc):
                # C5: with Triple Captain the vice-captain term is also tripled: sigma = s_vc * u_TC
                # (s_vc in [0, 1], u_TC binary -> min(s_vc, u_TC) = s_vc * u_TC)
                sig = {i: pulp.LpVariable(f"tcv_{i}_{g}", lowBound=0, upBound=1) for i in ids if xp[(i, g)] > 0}
                for i, s_ in sig.items():
                    prob += s_ <= s_vc[i]
                    prob += s_ <= u_tc
                obj.append(w * pulp.lpSum(xp[(i, g)] * s_ for i, s_ in sig.items()))
            pool.setdefault('_vice', {})[g] = v

        # --- M5: bench value through autosubs (not with Bench Boost: the bench scores anyway) ---
        if use_bench and not bb and m_big > 0:
            # Goalkeeper: exact under independence, same construction as the vice-captain,
            # with "captain" = starting GK and "vice" = bench GK (bench indicator x - l):
            #   value = (1 - P_startGK) * xP_benchGK,  s_i = (x_i - l_i) * (1 - P_startGK).
            s_gk = {i: pulp.LpVariable(f"sgk_{i}_{g}", lowBound=0) for i in gk_ids}
            prob += pulp.lpSum(s_gk.values()) <= 1 - pulp.lpSum(papp[(i, g)] * l[(i, g)] for i in gk_ids)
            if is_var(u_bb):
                prob += pulp.lpSum(s_gk.values()) <= 1 - u_bb    # C5: no autosub value in a BB GW
            for i in gk_ids:
                prob += s_gk[i] <= x[(i, g)] - l[(i, g)]
            obj.append(w * pulp.lpSum(xp[(i, g)] * s_gk[i] for i in gk_ids))
            # Outfield: bench slots 1-3 with weights beta (assignment variables; continuous,
            # the assignment constraints are totally unimodular -> integral at the optimum)
            if beta and g in beta:
                b = pulp.LpVariable.dicts(f"b_{g}", [(i, k) for i in out_ids for k in range(3)],
                                          lowBound=0, upBound=1)
                for i in out_ids:
                    prob += pulp.lpSum(b[(i, k)] for k in range(3)) <= x[(i, g)] - l[(i, g)]
                for k in range(3):
                    prob += pulp.lpSum(b[(i, k)] for i in out_ids) <= 1 - u_bb   # C5: 0 in a BB GW
                obj.append(w * pulp.lpSum(beta[g][k] * xp[(i, g)] * b[(i, k)]
                                          for i in out_ids for k in range(3)))
                pool.setdefault('_bench', {})[g] = b

    # A: an unavailable player may be kept, but not started in the first GW
    for i in ids:
        if not pool['avail'][i]:
            prob += l[(i, t0)] == 0

    # --- M4: budget in every GW, with constant prices [assumption] ---
    # k[i,g] = "still holds the ORIGINAL copy of owned player i" (sells for sell[i]);
    # a re-bought copy costs price[i]. Only needed when sell < price.
    gain_ids = [i for i in S0 if i in pool['price'] and sell[i] < price[i] - 1e-9]
    k_var = pulp.LpVariable.dicts("k", [(i, g) for i in gain_ids for g in horizon], lowBound=0, upBound=1)
    # C4: kap[(i, g)] = still owns the original copy PERMANENTLY after GW g. A Free Hit squad
    # may leave i out for one GW without selling the original (same rule as y above):
    #   no FH in g: kap = k[g];  FH fixed: kap = kap[g-1];  FH variable: kap <= k + u_FH, kap <= kap[g-1].
    kap = {}
    for g in horizon:
        uf = u[('FH', g)]
        for i in gain_ids:
            prob += k_var[(i, g)] <= x[(i, g)]
            kap_prev = kap[(i, g - 1)] if g > t0 else 1
            if is_var(kap_prev):
                prob += k_var[(i, g)] <= kap_prev              # once sold, the original is gone
            if is_var(uf) and g != horizon[-1]:
                kv = pulp.LpVariable(f"kap_{i}_{g}", lowBound=0, upBound=1)
                prob += kv <= k_var[(i, g)] + uf
                if is_var(kap_prev):
                    prob += kv <= kap_prev
                kap[(i, g)] = kv
            elif not is_var(uf) and uf == 1:
                kap[(i, g)] = kap_prev                         # Free Hit fixed in g
            else:
                kap[(i, g)] = k_var[(i, g)]
        cost = []
        for i in ids:
            if i in gain_ids:
                cost.append(sell[i] * k_var[(i, g)] + price[i] * (x[(i, g)] - k_var[(i, g)]))
            else:
                cost.append((sell[i] if i in S0 else price[i]) * x[(i, g)])
        prob += pulp.lpSum(cost) <= budget

    # --- M2: free transfers and hits (linearisation, notes_06b §4) ---
    T = {g: pulp.lpSum(t_in[(i, g)] for i in ids) for g in horizon}
    F = {t0: ft0}
    h = {}
    obj.append(-TRANSFER_TIEBREAK * pulp.lpSum(T.values()))
    if chip_vars:
        obj.append(-CHIP_TIEBREAK * pulp.lpSum(chip_vars))   # C3: no chip without a gain
    for g in horizon:
        if g > t0:
            F[g] = pulp.LpVariable(f"ft_{g}", lowBound=1, upBound=MAX_FT, cat='Integer')
            ph = phi[g - 1]
            if g - 1 == t0 and ft0 is None:
                prob += F[g] <= 1                       # first deadline passed -> 1 FT
            elif not is_var(ph) and ph == 1:
                prob += F[g] <= F[g - 1]                # chip fixed: FTs kept, none added
            elif is_var(ph):
                # C2: WC/FH decided by the model (notes_07 §2). phi = 1 switches (2) off and
                # (3) gives F[g] <= F[g-1]; phi = 0 leaves (2) as in M2 and (3) is implied by it.
                prob += F[g] <= F[g - 1] - T[g - 1] + h[g - 1] + 1 + CHIP_TRANSFER_M * ph   # (2)
                prob += F[g] <= F[g - 1] + 1 - ph                                          # (3)
            else:
                prob += F[g] <= F[g - 1] - T[g - 1] + h[g - 1] + 1
        if (g == t0 and free_now) or (not is_var(phi[g]) and phi[g] == 1):
            h[g] = 0                                    # WC / FH fixed / before the first deadline
            continue
        h[g] = pulp.LpVariable(f"hits_{g}", lowBound=0, upBound=max_hits, cat='Integer')
        prob += h[g] >= T[g] - F[g] - CHIP_TRANSFER_M * phi[g]   # (1); C2: phi = 0 -> M2 as before
        obj.append(-weight[g] * HIT_COST * h[g])        # M2/A1: weighted like that GW's points

    prob += pulp.lpSum(obj)
    if warm:
        for var in prob.variables():
            if var.name in warm and warm[var.name] is not None:
                val = warm[var.name]
                # solvers return values like -1e-12 for a variable with lower bound 0;
                # PuLP rejects an initial value outside the bounds, so clip it first
                if var.lowBound is not None:
                    val = max(val, var.lowBound)
                if var.upBound is not None:
                    val = min(val, var.upBound)
                var.setInitialValue(round(val) if var.cat == 'Integer' else val)
    elif S0:
        # M7: starting solution for CBC = keep the current squad in every GW (always
        # feasible: no transfers, and it costs exactly the budget minus the bank), best
        # XI by xP, captain and vice-captain = the two highest xP of that XI. Without it,
        # most of the solve time was spent looking for a FIRST feasible plan (notes_06b §9).
        ft_run = ft0 if ft0 is not None else 1
        for g in horizon:
            xp_g = {i: (xp[(i, g)] if (g > t0 or pool['avail'][i]) else -1.0) for i in start_squad}
            xi = reference_xi(start_squad, pos, xp_g)
            ranked = sorted(xi, key=lambda i: -xp_g[i])
            for i in ids:
                x[(i, g)].setInitialValue(1 if i in S0 else 0)
                l[(i, g)].setInitialValue(1 if i in xi else 0)
                c[(i, g)].setInitialValue(1 if i == ranked[0] else 0)
                if '_vice' in pool and g in pool['_vice']:
                    pool['_vice'][g][i].setInitialValue(1 if i == ranked[1] else 0)
            for i in gain_ids:
                k_var[(i, g)].setInitialValue(1)     # original copies kept -> budget holds
                if isinstance(kap[(i, g)], pulp.LpVariable) and kap[(i, g)] is not k_var[(i, g)]:
                    kap[(i, g)].setInitialValue(1)   # C4
            for i in ids:
                if isinstance(y[(i, g)], pulp.LpVariable) and y[(i, g)] is not x[(i, g)]:
                    y[(i, g)].setInitialValue(1 if i in S0 else 0)   # C4: permanent squad = kept squad
            for cp in CHIPS:
                if is_var(u[(cp, g)]):
                    u[(cp, g)].setInitialValue(0)    # C3: start from "no chip played"
            if g > t0:
                F[g].setInitialValue(ft_run)
            if isinstance(h[g], pulp.LpVariable):
                h[g].setInitialValue(0)
            chip_fixed_free = not is_var(phi[g]) and phi[g] == 1   # WC/FH fixed in g: FTs kept
            if not ((g == t0 and ft0 is None) or chip_fixed_free):
                ft_run = min(MAX_FT, ft_run + 1)     # no transfer -> one more FT (cap 5)
            elif ft0 is None and g == t0:
                ft_run = 1
    out = run_solver(prob, solver, time_limit or SOLVER_TIME_LIMIT, bool(warm) or bool(S0))
    res = {'status': pulp.LpStatus[prob.status], 'objective': pulp.value(prob.objective),
           'values': {var.name: var.varValue for var in prob.variables()},
           'proven': out['proven'], 'solver': out['solver']}
    # sol_status: 1 = optimal (within MIP_REL_GAP), 2 = feasible plan found but the time
    # limit stopped the proof. Anything else (no plan found, infeasible) has no usable values.
    if prob.status != pulp.LpStatusOptimal or prob.sol_status not in (pulp.LpSolutionOptimal,
                                                                      pulp.LpSolutionIntegerFeasible):
        if prob.status == pulp.LpStatusOptimal:
            res['status'] = 'No plan found within the time limit'
        return res
    res['gap'] = None if out['bound'] is None else max(0.0, out['bound'] - res['objective'])

    # --- read the plan; FTs and hits are recomputed with the exact FPL rule ---
    plan, prev, ft = {}, list(start_squad), ft0
    for g in horizon:
        sq = [i for i in ids if is_chosen(x[(i, g)])]
        # C3: chip played in g (fixed or chosen by the model); None = no chip
        played = next((cp for cp in CHIPS if (is_chosen(u[(cp, g)]) if is_var(u[(cp, g)]) else u[(cp, g)] == 1)),
                      None)
        base = prev                                     # C4: permanent squad before g
        ins = [i for i in sq if i not in base]
        outs = [i for i in base if i not in sq]
        if (g == t0 and ft0 is None) or played in ('WC', 'FH'):
            hits, ft_next = 0, (1 if (g == t0 and ft0 is None) else ft)   # FTs kept, none added
        else:
            hits, ft_next = max(0, len(ins) - ft), min(MAX_FT, max(0, ft - len(ins)) + 1)
        xi = [i for i in sq if is_chosen(l[(i, g)])]
        cap = next(i for i in xi if is_chosen(c[(i, g)]))
        vice = None
        if '_vice' in pool and g in pool['_vice']:
            vice = next((i for i in xi if is_chosen(pool['_vice'][g][i])), None)
        bench_out = sorted((i for i in sq if i not in xi and pos[i] != 1), key=lambda i: -xp[(i, g)])
        if '_bench' in pool and g in pool['_bench']:
            b = pool['_bench'][g]                       # order by the slot the solver used
            bench_out = sorted(bench_out, key=lambda i: -sum((3 - k) * (b[(i, k)].varValue or 0) for k in range(3)))
        bench_gk = [i for i in sq if i not in xi and pos[i] == 1]
        plan[g] = {'squad': sq, 'xi': xi, 'captain': cap, 'vice': vice, 'bench': bench_gk + bench_out,
                   'in': ins, 'out': outs, 'ft': ft, 'hits': hits, 'chip': played}
        prev = base if played == 'FH' else sq           # C4: a Free Hit squad reverts
        ft = ft_next
    res['plan'] = plan
    pool.pop('_vice', None)
    pool.pop('_bench', None)
    return res


def plan_squad(pool, horizon, weight, start_squad, budget, ft0, chip='NONE', max_hits=None,
               use_vice=True, use_bench=True, time_limit=None, solver=None, chips_avail=None, chip_fix=None):
    """M5: fixed point for the outfield bench weights. Pass 1 uses the best XI of
    the current squad (or of the pool at GW1) as reference; each later pass uses the
    XI the solver chose, until the weights change by <= BENCH_BETA_TOL. Each pass
    starts CBC from the previous plan (warm start).
    Returns the result of the last pass that produced a plan (or the failed first one)."""
    pos, xp, papp = pool['pos'], pool['xp'], pool['papp']
    ref = start_squad if start_squad else pool['ids']
    beta = {g: bench_weights(reference_xi(ref, pos, {i: xp[(i, g)] for i in ref}), pos,
                             {i: papp[(i, g)] for i in ref}) for g in horizon} if use_bench else None
    best, seconds, history = None, 0.0, []
    for n_pass in range(1, BENCH_MAX_PASSES + 1):
        t_start = time.time()
        # M10: the time limit is for the whole run; each pass gets what is left
        left = (time_limit or SOLVER_TIME_LIMIT) - seconds
        if best is not None and left < MIN_PASS_SECONDS:
            break                                   # no time for another pass: keep the last plan
        res = solve_plan(pool, horizon, weight, start_squad, budget, ft0, chip, max_hits,
                         beta, use_vice, use_bench, warm=best['values'] if best else None,
                         time_limit=max(left, MIN_PASS_SECONDS), solver=solver,
                         chips_avail=chips_avail, chip_fix=chip_fix)   # C3
        seconds += time.time() - t_start
        history.append(beta)
        if 'plan' not in res:
            if best is None:                        # nothing found at all
                res['seconds'], res['passes'] = seconds, n_pass
                return res
            break                                   # keep the previous pass
        res['beta'], res['passes'] = beta, n_pass
        best = res
        if not use_bench:
            break
        implied = {g: bench_weights(res['plan'][g]['xi'], pos, {i: papp[(i, g)] for i in pool['ids']})
                   for g in horizon}
        best['beta_implied'] = implied
        best['beta_change'] = max(abs(implied[g][k] - beta[g][k]) for g in horizon for k in range(3))
        if best['beta_change'] <= BENCH_BETA_TOL:
            break
        beta = implied
    best['seconds'], best['beta_history'] = seconds, history
    return best


def gw_points(pg, g, pool, beta_g, use_bench=True):
    """C8: expected points of GW g of a plan, UNWEIGHTED (no 0.85^k), split into parts:
    XI (all 15 with Bench Boost) + captain x m + vice-captain x m x (1 - P_C) + autosubs
    (not with Bench Boost) - hits, with m = 2 for Triple Captain, else 1. Same terms as the
    objective of solve_plan for that GW, without the weight and the tie-breaks."""
    xp, papp, pos = pool['xp'], pool['papp'], pool['pos']
    bb = pg['chip'] == 'BB'
    mult = 2.0 if pg['chip'] == 'TC' else 1.0
    out = {'xi': sum(xp[(i, g)] for i in (pg['squad'] if bb else pg['xi'])),
           'captain': mult * xp[(pg['captain'], g)], 'vice': 0.0, 'autosubs': 0.0,
           'hits': -HIT_COST * pg['hits']}
    if pg['vice'] is not None:
        out['vice'] = mult * (1.0 - papp[(pg['captain'], g)]) * xp[(pg['vice'], g)]
    if use_bench and not bb:
        gk_start = next(i for i in pg['xi'] if pos[i] == 1)
        out['autosubs'] = sum((1.0 - papp[(gk_start, g)]) * xp[(i, g)] for i in pg['bench'] if pos[i] == 1)
        outs = [i for i in pg['bench'] if pos[i] != 1]
        if beta_g:
            out['autosubs'] += sum(beta_g[k] * xp[(i, g)] for k, i in enumerate(outs))
    out['total'] = sum(out.values())
    return out


def plan_points(res, pool, horizon, use_bench=True):
    """C8: unweighted expected points of a whole plan (sum of gw_points over the horizon)."""
    beta = res.get('beta') or {}
    return sum(gw_points(res['plan'][g], g, pool, beta.get(g), use_bench)['total'] for g in horizon)


def squad_value(pool, squad, g, chip='NONE', use_vice=True, use_bench=True, solver=None):
    """C9: best one-GW value (unweighted) of a FIXED 15-man squad: plan_squad on a pool that
    holds only these 15 players, so the XI, captain, vice-captain and autosubs come from the
    same model as the plan. Every player may start: an unavailable one has xP 0 and P(app) 0,
    so the model benches him or counts him as autosubbed (as the game would)."""
    sub = {k: pool[k] for k in ('pos', 'team', 'price', 'xp', 'papp')}
    sub['ids'] = list(squad)
    sub['avail'] = {i: True for i in squad}
    sub['sell'] = {i: pool['price'][i] for i in squad}           # no transfers: prices do not matter
    budget = sum(sub['sell'].values()) + 1.0
    res = plan_squad(sub, [g], {g: 1.0}, list(squad), budget, 1, chip, 0, use_vice, use_bench,
                     time_limit=30, solver=solver)
    return plan_points(res, sub, [g], use_bench) if 'plan' in res else None


def chip_report(pool, horizon, weight, plan, start_squad, budget, ft0, chip_windows, max_hits,
                use_vice, use_bench, time_left, solver=None):
    """C10: opportunity cost of each chip that can be played now (notes_07 §3, decision of the
    user: approach 'G', unweighted points).
      exact[c]    = Delta_c(t0): points of the best plan with chip c in t0 minus the best plan
                    without a chip, both solved with the full model over min(3, L) GWs and
                    scored UNWEIGHTED (plan_points).
      est[c][t]   = estimate of Delta_c(t) for t0 <= t <= end of the chip's window, on the
                    squad Q_t of the main plan (after the horizon: the squad of its last GW,
                    [assumption] no transfers after the horizon):
                    TC, BB: squad_value(Q_t, chip) - squad_value(Q_t);
                    FH: best one-GW Free Hit squad (whole pool) - squad_value(Q_t);
                    WC: not estimated after t0 ([assumption], no one-GW proxy exists).
    Returns a dict with the numbers; the caller prints them."""
    t0 = horizon[0]
    deadline = time.time() + time_left
    out = {'exact': {}, 'gap': {}, 'est': {}, 'window': {}, 'skipped': []}
    now_chips = [cp for cp in CHIPS for s, e in chip_windows.get(cp, []) if s <= t0 <= e]
    for cp in now_chips:
        out['window'][cp] = next(e for s, e in chip_windows[cp] if s <= t0 <= e)
    if not now_chips:
        return out
    # --- exact Delta_c(t0): 1 + len(now_chips) solves with the full model ---
    h3 = horizon[:3]
    w3 = {g: weight[g] for g in h3}
    runs = [('NONE', 4.0)] + [(cp, 4.0 if cp == 'WC' else 2.0) for cp in now_chips]   # time shares
    base = None
    for k, (cp, share) in enumerate(runs):
        left = deadline - time.time()
        # this solve's share of the remaining time, in proportion to the measured solve times
        # (WC ~2x the others, notes_07 §4), and at least MIN_PASS_SECONDS
        tl = max(MIN_PASS_SECONDS, left * share / sum(s for _, s in runs[k:]) * 0.75)
        if left < MIN_PASS_SECONDS or (cp != 'NONE' and base is None):
            out['skipped'].append(f"exact {cp}")
            continue
        res = plan_squad(pool, h3, w3, start_squad, budget, ft0, cp, max_hits, use_vice, use_bench,
                         time_limit=tl, solver=solver)
        if 'plan' not in res:
            out['skipped'].append(f"exact {cp} (no plan)")
            continue
        pts = plan_points(res, pool, h3, use_bench)
        if cp == 'NONE':
            base = (pts, res.get('gap'))
        else:
            out['exact'][cp] = pts - base[0]
            out['gap'][cp] = (res.get('gap') or 0.0) + (base[1] or 0.0)   # bound on the error of Delta
    # --- estimates Delta_c(t) up to the end of each window (one-GW solves) ---
    last = horizon[-1]
    for t in range(t0, max(out['window'].values()) + 1):
        if time.time() > deadline:
            out['skipped'].append(f"estimates from GW{t}")
            break
        q = plan[min(t, last)]['squad']
        chips_t = [cp for cp in now_chips if t <= out['window'][cp] and cp != 'WC']
        if not chips_t:
            continue
        v0 = squad_value(pool, q, t, 'NONE', use_vice, use_bench, solver)
        for cp in chips_t:
            if cp in ('TC', 'BB'):
                v = squad_value(pool, q, t, cp, use_vice, use_bench, solver)
            else:   # FH: whole pool, start from Q_t, sell = price ([assumption] constant prices)
                sub = dict(pool)
                sub['sell'] = {i: pool['sell'].get(i, pool['price'][i]) for i in q}
                sub['avail'] = {i: True for i in pool['ids']} if t > t0 else pool['avail']
                res = plan_squad(sub, [t], {t: 1.0}, list(q), budget, 1, 'FH', max_hits, use_vice, use_bench,
                                 time_limit=30, solver=solver)
                v = plan_points(res, sub, [t], use_bench) if 'plan' in res else None
            if v is not None and v0 is not None:
                out['est'].setdefault(cp, {})[t] = v - v0
    return out


def optimize_team(team_id, user_ft, chip_used, min_chance, lookahead, override_budget, max_hits=None,
                  explain=None, use_vice=True, use_bench=True, time_limit=None, solver=None, chip_suggest=True):
    print(f"[*] MILP Optimizer | Team: {team_id} | Lookahead: {lookahead} GWs")

    bootstrap = fetch_fpl_api(f"{FPL_API}/bootstrap-static/")
    current_gw = get_current_gw(bootstrap['events'])

    # G1: the planning horizon is a list of gameweeks, e.g. [8, 9, 10].
    # Every xP value and every lineup/captain decision below is indexed by these GWs.
    # M3: with a Free Hit the horizon is no longer cut to one GW: the FH squad only
    # exists in the current GW, and the later GWs are planned from the squad it reverts to.
    horizon = [g for g in range(current_gw, current_gw + lookahead) if g <= 38]
    # G1/M2: discount weight per GAMEWEEK: w_g = 0.85^(g - current_gw). Applied to the
    # points AND the hits of GW g (decision A1, notes_06b §4).
    # [assumption] 0.85 is a decision parameter (projections further ahead are less
    # reliable), not derived from data.
    gw_weight = {g: TIME_DECAY_RATE ** (g - current_gw) for g in horizon}
    print(f"[*] Planning horizon: GW{horizon[0]}-GW{horizon[-1]} | weights: "
          + ", ".join(f"GW{g}={w:.2f}" for g, w in gw_weight.items()))

    # X3: one request for every fixture of the season (past and future)
    all_fixtures = fetch_fpl_api(f"{FPL_API}/fixtures/")
    # C7: xP for EVERY remaining GW (not only the horizon): the chip estimates compare GWs up to
    # the end of each chip's window (GW19 / GW38). The MILP still uses only the horizon.
    ext_gws = list(range(current_gw, 39))
    fixtures_map = build_fixture_map(current_gw, len(ext_gws), all_fixtures)

    # X3/X4: the model's inputs come from finished GWs of this season only.
    hist = fetch_history(bootstrap, all_fixtures, current_gw)
    pos_of = {p['id']: p['element_type'] for p in bootstrap['elements']}
    price_of = {p['id']: p['now_cost'] / 10.0 for p in bootstrap['elements']}  # X8: current price
    # X9: GW from which each player is at his current club (bootstrap field team_join_date)
    deadlines = {e['id']: pd.Timestamp(e['deadline_time']) for e in bootstrap['events']
                 if e.get('deadline_time')}
    join_dates = {p['id']: p.get('team_join_date') for p in bootstrap['elements']}
    if deadlines and any(join_dates.values()):
        start_gw_of = first_gw_at_club(join_dates, deadlines)
    else:  # field missing from the API -> minutes model uses all GWs (chat-8a behaviour)
        print("[WARNING] deadline_time / team_join_date missing; minutes model uses all GWs.")
        start_gw_of = None
    feats, team_rating, league_xg = build_features(hist, pos_of, current_gw, price_of, start_gw_of)
    # 8c (notes_08c §8): if the GW before current_gw is not final yet (bonus points / corrections
    # still pending) it is missing from hist. The ages stay calendar ages, so W is smaller and
    # the minutes model shrinks more towards its prior; this is on purpose and only reported here.
    if not hist.empty and hist['gw'].max() < current_gw - 1:
        print(f"[INFO] GW{current_gw - 1} is not final yet: the minutes model uses data up to "
              f"GW{int(hist['gw'].max())} (its estimates are shrunk more towards the prior).")
    if league_xg is None:
        print("[WARNING] No finished GW yet: every player uses FPL's ep_next (fallback).")

    # G2: number of matches per (team, GW). 0 = blank (BGW), 2 = double (DGW).
    team_ids = [t['id'] for t in bootstrap['teams']]
    n_matches = {(t, g): 0 for t in team_ids for g in ext_gws}   # C7: all remaining GWs
    for t, matches in fixtures_map.items():
        for m in matches:
            n_matches[(t, m['gw'])] += 1
    # display only; .get() with fallbacks so a renamed API field cannot crash the run
    team_name = {t['id']: t.get('short_name') or t.get('name') or str(t['id']) for t in bootstrap['teams']}
    for g in horizon:
        blank = [team_name[t] for t in team_ids if n_matches[(t, g)] == 0]
        double = [team_name[t] for t in team_ids if n_matches[(t, g)] >= 2]
        if blank or double:
            print(f"[*] GW{g}: blank: {', '.join(blank) or '-'} | double: {', '.join(double) or '-'}")

    # A: fetch the current squad BEFORE filtering players (moved up from the
    # budget section), so the filter below knows who we already own.
    ft0 = None   # M8: None = unlimited free transfers (before the team's first deadline)
    if current_gw == 1:
        current_squad = []
        bank = 0.0
        sell_price = {}
    else:
        history = fetch_fpl_api(f"{FPL_API}/entry/{team_id}/history/")
        # R1 + S1: squad after a possible Free Hit revert, bank, selling prices
        current_squad, bank, sell_price = load_current_squad(team_id, current_gw, bootstrap, history)
        # M8: free transfers rebuilt from the GW history; --ft overrides
        ft_api, issues = free_transfers_from_history(history, current_gw)
        for msg in issues:
            print(f"[WARNING] Free-transfer rebuild: {msg}")
        if user_ft is not None:
            ft0 = user_ft
            print(f"[*] Free transfers: {ft0} (--ft; rebuilt from history: {ft_api})")
        elif issues:
            print("[ERROR] The rebuilt free-transfer count does not match the hit costs in the API. "
                  "Pass the number shown in the app with --ft.")
            sys.exit(1)
        else:
            ft0 = ft_api
            print(f"[*] Free transfers: {ft0} (rebuilt from history; --ft overrides)")
    if ft0 is None:
        print("[*] Transfers are unlimited this GW (first deadline of the team).")

    # C6: chip sets not played yet (bootstrap-static windows minus the entry history), for the
    # "Chip suggestions" section. The main plan does NOT get chip variables: with every chip free
    # the MILP stopped at the time limit with a gap of 117 points at lookahead 5 and a worse plan
    # than without chips (notes_07 §4). --chip forces a chip in the current GW, as before.
    chip_hist = history if current_gw > 1 else fetch_fpl_api(f"{FPL_API}/entry/{team_id}/history/")
    chip_windows, chip_rows, chip_warn, chip_src = available_chips(bootstrap, chip_hist, current_gw)
    for msg in chip_warn:
        print(f"[WARNING] Chips: {msg}")

    explain_names = {n.strip().lower() for n in explain.split(',')} if explain else set()
    explained = []
    players = []
    for p in bootstrap['elements']:
        chance = p['chance_of_playing_next_round']
        available = chance is None or chance >= min_chance  # None = no injury news
        # A: unavailable players are dropped from the pool EXCEPT our own.
        # Our own must stay so that (1) their price counts in the budget and
        # (2) the solver can keep them on the bench or sell them (shown in SELL).
        if not available and p['id'] not in current_squad:
            continue

        pos = p['element_type']
        team = p['team']
        ep_next = float(p.get('ep_next') or 0.0)
        f = feats.get(p['id'])
        # X5/X9: fallback for players with no GW yet at their current club (GW1,
        # new signings): FPL's ep_next per match. A player who has been at the club
        # all season without playing stays in the model (P(appear) ~ 0). It REPLACES the model for these players, it is never added to
        # it, so nothing is counted twice.
        source = 'model' if f is not None else 'FPL ep'
        n_next = n_matches.get((team, current_gw), 0)
        fallback_per_match = ep_next / n_next if n_next > 0 else float(p.get('points_per_game') or 0.0)

        # X4: xp_gw[g] = sum over the team's matches in GW g of match_xp.
        # Removed (chat 6 formula): ep_next, form, Understat xG/xA, FPL strength
        # ratings, rotation penalty, penalty/set-piece bonuses. Reasons in notes_08a.
        xp_gw = {g: 0.0 for g in ext_gws}            # C7: every remaining GW
        # M6: P(no appearance in GW g) = product over his matches of (1 - P(appear)).
        # [assumption] the matches of a DGW are independent. Blank GW -> product 1 -> P = 0.
        # M9/8c: per-match P(appear) for the vice-captain / autosub terms is the SAME shrunk
        # Papp that the xP uses (build_features); chat 6b shrunk it here only for these terms
        # (notes_06b §17). Fallback players (no GW at the club, W = 0) get the prior mean m.
        p_match = f['Papp'] if f is not None else APP_PRIOR_MEAN
        p_none = {g: 1.0 for g in ext_gws}
        comp_now = {}
        for match in fixtures_map.get(team, []):
            g = match['gw']
            p_none[g] *= 1.0 - p_match
            if f is not None:
                c = match_xp(f, pos, team_rating.get(team, NEUTRAL_TEAM),
                             team_rating.get(match['opp'], NEUTRAL_TEAM), match['is_home'], league_xg)
                mx = sum(c.values())
                if g == current_gw:
                    for k_, v_ in c.items():
                        comp_now[k_] = comp_now.get(k_, 0.0) + v_
            else:
                mx = fallback_per_match
            if g == current_gw and chance is not None:
                # X5: FPL's own probability (e.g. 75) scales the NEXT GW only.
                # Assumption: the percentage is taken at face value.
                mx *= chance / 100.0
            xp_gw[g] += mx

        papp_gw = {g: 1.0 - p_none[g] for g in ext_gws}
        if chance is not None:
            papp_gw[current_gw] *= chance / 100.0   # M6: same face-value use of chance as X5
        if not available:
            # A + G2: below --min-chance counts as not playing. chance_of_playing_next_round
            # only describes the next GW; we conservatively assume 0 for the whole horizon
            # (C7: and for the later GWs of the chip estimates).
            xp_gw = {g: 0.0 for g in ext_gws}
            papp_gw = {g: 0.0 for g in ext_gws}

        if p['web_name'].lower() in explain_names:
            explained.append((p, f, comp_now, xp_gw, source, chance))

        players.append({
            'id': p['id'], 'name': p['web_name'], 'team': team,
            # S2: 'cost' = what the player costs against the budget. Owned players: their
            # SELLING price (keeping one means not receiving that amount); others: now_cost.
            'position': pos, 'cost': sell_price.get(p['id'], p['now_cost'] / 10.0),
            'xp_gw': {g: round(x, 2) for g, x in xp_gw.items()},  # G2: dict GW -> xP
            'papp_gw': papp_gw,      # M6: dict GW -> P(appears at least once)
            'available': available,  # A: used for the lineup constraint and the output
            'price': p['now_cost'] / 10.0,  # M4: purchase price (owned players: re-buy price)
            'ep_next': ep_next, 'source': source,  # V1: logged as baseline / X5
        })

    # X7: --explain prints the components of xP for the named players (web_name)
    for p, f, comp, xg, source, chance in explained:
        print(f"\n[explain] {p['web_name']} (id {p['id']}, {team_name[p['team']]}, "
              f"pos {p['element_type']}, £{p['now_cost'] / 10:.1f}m) source={source} chance={chance}")
        if f is not None:
            print(f"   minutes model (shrunk; raw in brackets): P(appear)={f['Papp']:.2f} [{f['raw_Papp']:.2f}] "
                  f"P(60+)={f['P60']:.2f} [{f['raw_P60']:.2f}] E[min]={f['Emin']:.0f} [{f['raw_Emin']:.0f}] "
                  f"| evidence W={f['Wapp']:.2f} | season minutes={f['minutes_total']:.0f} "
                  f"| price factor={f['price_factor']:.2f}")  # 8c, X8
            print("   per 90: " + ", ".join(f"{k[:-2]}={f[k]:.2f}" for k in f if k.endswith('90')))
            print(f"   GW{current_gw}: " + ", ".join(f"{k}={v:+.2f}" for k, v in comp.items()))
        else:
            print(f"   no GW at current club yet -> ep_next={float(p.get('ep_next') or 0):.2f}")  # X9
        print("   xP by GW: " + ", ".join(f"GW{g}={xg[g]:.2f}" for g in horizon))   # C7: horizon only
    if explain_names:
        found = {e[0]['web_name'].lower() for e in explained}
        for n in sorted(explain_names - found):
            print(f"[explain] '{n}' not found (check web_name; unavailable players outside your squad are skipped)")

    df = pd.DataFrame(players)

    # Financial state calculations
    if current_gw == 1:
        total_budget = override_budget if override_budget else 100.0
    else:
        # A: current_squad and bank are now fetched before the player loop.
        # Unavailable squad players are in df, so their cost is included here.
        # S2: budget = selling value of the squad + bank (was: now_cost of the squad + bank,
        # which counted the full price rise instead of half). With cost = selling price
        # for owned players, 'sum of costs <= budget' means: each sale returns the
        # selling price and each purchase costs now_cost.
        squad_value = df[df['id'].isin(current_squad)]['cost'].sum()
        if override_budget:
            total_budget = override_budget
            print(f"[*] Budget (--override-budget): £{total_budget:.1f}m "
                  f"(computed: £{squad_value + bank:.1f}m)")
        else:
            total_budget = squad_value + bank
            print(f"[*] Budget: £{total_budget:.1f}m = squad selling value £{squad_value:.1f}m "
                  f"+ bank £{bank:.1f}m")

    # M: pool = plain dicts keyed by player id (the MILP needs fast lookups, not a DataFrame)
    ids = df['id'].tolist()
    pos_dict = dict(zip(df['id'], df['position']))
    cost_dict = dict(zip(df['id'], df['cost']))       # S2: selling price if owned, else now_cost
    avail_dict = dict(zip(df['id'], df['available']))  # A
    source_dict = dict(zip(df['id'], df['source']))    # X5
    name_of = dict(zip(df['id'], df['name']))
    team_of = dict(zip(df['id'], df['team']))
    pool = {
        'ids': ids, 'pos': pos_dict, 'team': team_of, 'avail': avail_dict,
        'price': dict(zip(df['id'], df['price'])),
        'sell': {i: cost_dict[i] for i in current_squad if i in cost_dict},
        # G2: key = (player_id, gameweek) tuple -> xP / P(appear) of that player in that GW
        'xp': {(i, g): x for i, xg in zip(df['id'], df['xp_gw']) for g, x in xg.items()},
        'papp': {(i, g): p_ for i, pg in zip(df['id'], df['papp_gw']) for g, p_ in pg.items()},
    }
    xp_dict = pool['xp']
    start_squad = [i for i in current_squad if i in cost_dict]

    res = plan_squad(pool, horizon, gw_weight, start_squad, total_budget, ft0, chip_used,
                     max_hits, use_vice, use_bench, time_limit, solver)

    # C: solve() does not raise on failure; it only sets the status.
    if 'plan' not in res:
        print(f"[ERROR] Solver status: {res['status']}. No valid plan exists under these constraints.")
        print("        Check --override-budget, --min-chance, --ft and --max-hits.")
        sys.exit(1)
    plan = res['plan']
    now = plan[current_gw]
    gap_txt = f"{res['gap']:.2f}" if res['gap'] is not None else "unknown"
    print(f"[*] Solved with {res['solver']} in {res['seconds']:.1f} s ({res['passes']} pass(es) for the "
          f"bench weights); proven within {gap_txt} weighted points of the best plan of this model")
    if res.get('beta_change', 0.0) > BENCH_BETA_TOL:
        print(f"[*] Bench weights still changed by {res['beta_change']:.2f} after {res['passes']} passes "
              f"(they depend on the chosen XI); the last plan is used.")
    if not res['proven']:
        # M7: time limit reached; the plan is feasible and its distance from the best
        # possible plan of this model is at most the gap (weighted points)
        print(f"[*] Time limit reached before the {MIP_REL_GAP:.2%} target: a longer --time-limit "
              f"can only improve the plan or tighten the gap.")

    print("\n" + "=" * 84)
    print(f" SQUAD PLAN | GW{current_gw} (horizon: GW{horizon[0]}-GW{horizon[-1]})")
    print("=" * 84)

    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    n_now = {i: n_matches[(team_of[i], current_gw)] for i in ids}
    final_squad = []
    for i in now['squad']:
        if i == now['captain']:
            role = 'Captain'
        elif i == now['vice']:
            role = 'Vice'                       # M6: logged too (counts as a starter in evaluate)
        elif i in now['xi']:
            role = 'Start'
        else:
            role = 'Bench'
        final_squad.append({
            'id': i, 'name': name_of[i], 'pos': pos_map[pos_dict[i]], 'cost': cost_dict[i],
            'xp': xp_dict[(i, current_gw)],
            'xp_h': sum(gw_weight[g] * xp_dict[(i, g)] for g in horizon),
            'n': n_now[i], 'role': role, 'action': 'KEEP' if i in current_squad else 'BUY',
            'pos_id': pos_dict[i],
            'bench_slot': now['bench'].index(i) if i in now['bench'] else -1,
        })
    # XI by position, then the bench in its order (GK first, then outfield slots 1-3)
    final_squad.sort(key=lambda p: (p['role'] == 'Bench', p['bench_slot'], p['pos_id']))

    print(f"   {'Player':<18} | Pos | Cost  | xP GW{current_gw:<3}| P(app) | xP horizon | Role")
    for p in final_squad:
        indicator = ">>" if p['action'] == 'BUY' else "  "
        flag = "" if avail_dict[p['id']] else "  (unavailable)"  # A
        gw_tag = "  (blank)" if p['n'] == 0 else ("  (DGW)" if p['n'] >= 2 else "")   # G4
        src_tag = "  (FPL ep)" if source_dict[p['id']] != 'model' else ""            # X5
        role = p['role']
        if role == 'Bench':
            role = 'Bench GK' if p['pos_id'] == 1 else f"Bench {p['bench_slot']}"   # M5: bench order
        print(f"{indicator} {p['name']:<18} | {p['pos']:<3} | £{p['cost']:<4.1f} | {p['xp']:<8.2f} | "
              f"{pool['papp'][(p['id'], current_gw)]:<6.2f} | {p['xp_h']:<10.2f} | {role}{flag}{gw_tag}{src_tag}")
    if current_gw > 1:
        print("   Cost = selling price for players you keep, current price for players you buy.")  # S3

    # M: the whole plan, one line per GW (only the current GW is a decision now;
    # later GWs are re-planned every week with new data)
    print("-" * 84)
    print(" Transfer plan (FT = free transfers available before that GW's transfers):")
    for g in horizon:
        pg = plan[g]
        by_pos = lambda lst: sorted(lst, key=lambda i: pos_dict[i])   # pair sales and buys by position
        moves = ", ".join(f"{name_of[o]} -> {name_of[n]}"
                          for o, n in zip(by_pos(pg['out']), by_pos(pg['in']))) or "none"
        chip_tag = f" [{pg['chip']}]" if pg['chip'] else ""          # C3: chip of that GW (fixed or chosen)
        ft_txt = "unlimited" if ((g == current_gw and ft0 is None) or pg['chip'] in ('WC', 'FH')) else pg['ft']
        vice_txt = f", VC {name_of[pg['vice']]}" if pg['vice'] is not None else ""
        print(f"   GW{g}{chip_tag}: FT {ft_txt} | {moves} | hits {pg['hits']} | "
              f"C {name_of[pg['captain']]}{vice_txt}")
        if pg['chip'] == 'FH' and g + 1 in plan:
            print(f"   Free Hit: GW{g + 1} starts again from the squad before GW{g}.")   # M3/C4

    if current_gw > 1:
        # S3: each sale of the current GW with the selling price it returns
        sold = [f"{name_of[i]} £{cost_dict[i]:.1f}m" + ("" if avail_dict[i] else " (unavailable)")
                for i in now['out']]
        if sold:
            print(f"Sales now (SELL): {', '.join(sold)}")
        else:
            print("Action now: roll the transfer (no change this GW).")

    # G4/M5/M6: current-GW expected points, split into their parts (chip effects included)
    bb = now['chip'] == 'BB'                      # C3: chip of the current GW in the plan
    mult = 2.0 if now['chip'] == 'TC' else 1.0
    xi_pts = sum(xp_dict[(i, current_gw)] for i in (now['squad'] if bb else now['xi']))
    cap_pts = mult * xp_dict[(now['captain'], current_gw)]
    vice_pts = 0.0
    if now['vice'] is not None:
        vice_pts = mult * (1.0 - pool['papp'][(now['captain'], current_gw)]) * xp_dict[(now['vice'], current_gw)]
    bench_pts = 0.0
    if use_bench and not bb:
        gk_start = next(i for i in now['xi'] if pos_dict[i] == 1)
        bench_gk = [i for i in now['bench'] if pos_dict[i] == 1]
        bench_pts += sum((1.0 - pool['papp'][(gk_start, current_gw)]) * xp_dict[(i, current_gw)] for i in bench_gk)
        outs = [i for i in now['bench'] if pos_dict[i] != 1]
        beta_now = res['beta'][current_gw]
        bench_pts += sum(beta_now[k] * xp_dict[(i, current_gw)] for k, i in enumerate(outs))
        print(f"   Bench weights GW{current_gw} (P(slot used)): "
              + ", ".join(f"{b:.2f}" for b in beta_now) + "  [approximation, notes_06b §8]")
    hit_pts = HIT_COST * now['hits']
    print(f"Expected points GW{current_gw}: XI {xi_pts:.2f} + captain {cap_pts:.2f} + vice {vice_pts:.2f} "
          f"+ autosubs {bench_pts:.2f} - hits {hit_pts} = {xi_pts + cap_pts + vice_pts + bench_pts - hit_pts:.2f}")
    print(f"Objective (weighted over horizon, hits included): {res['objective']:.2f}")
    n_fb = sum(1 for p in players if p['source'] != 'model')
    if n_fb:
        print(f"[*] {n_fb} player(s) with no GW at their current club yet use FPL's ep_next (marked '(FPL ep)').")  # X5

    # C10: chip suggestions (opportunity costs), notes_07 §3
    print("\n" + "=" * 84)
    print(f" CHIP SUGGESTIONS | GW{current_gw} (expected points, unweighted)")
    print("=" * 84)
    print(f"   Chips ({chip_src}): " + " | ".join(f"{cp} GW{s}-{e} {st}" for cp, s, e, st in chip_rows))
    if not chip_suggest:
        print("   Skipped (--no-chip-report).")
    elif chip_used != 'NONE':
        print(f"   Skipped: --chip {chip_used} is played this GW (one chip per GW).")
    else:
        time_left = (time_limit or SOLVER_TIME_LIMIT) - res['seconds']   # M10: one limit for the whole run
        t_rep = time.time()
        rep = chip_report(pool, horizon, gw_weight, plan, start_squad, total_budget, ft0, chip_windows,
                          max_hits, use_vice, use_bench, time_left, solver)
        if not rep['window']:
            print(f"   No chip can be played in GW{current_gw}.")
        else:
            h3 = horizon[:3]
            print(f"   Exact Delta now = best plan with the chip in GW{current_gw} minus best plan without a chip, "
                  f"full model, GW{h3[0]}-GW{h3[-1]}:")
            print("      " + " | ".join(f"{cp} {rep['exact'][cp]:+.2f} (error <= {rep['gap'][cp]:.2f})"
                                       if cp in rep['exact'] else f"{cp} not computed" for cp in rep['window']))
            print("   Estimates by GW (squad of the plan; after "
                  f"GW{horizon[-1]} its last squad, no transfers [assumption]); * = best GW to the end of the window:")
            for cp, est in rep['est'].items():
                best = max(est.values())
                cells = [f"GW{t} {d:+.1f}{'*' if d == best else ''}" for t, d in sorted(est.items())]
                for k in range(0, len(cells), 8):                 # 8 GWs per line
                    print(f"      {cp if k == 0 else '':<3} " + "  ".join(cells[k:k + 8]))
            # rule (notes_07 §3): play c now only if the estimate now is the maximum up to the end of
            # its window AND the exact Delta now is positive; one chip per GW -> the largest exact Delta
            now_ok = [cp for cp, est in rep['est'].items()
                      if est.get(current_gw) is not None and est[current_gw] >= max(est.values()) - 1e-9
                      and rep['exact'].get(cp, 0.0) > 0.0]
            for cp, est in rep['est'].items():
                if cp not in now_ok:
                    t_best = max(est, key=est.get)
                    now_txt = f"now {est[current_gw]:+.1f}, " if current_gw in est else ""
                    # the margin is printed because estimates of different GWs that differ by a few
                    # tenths are within the error of the xP model (RMSE ~2.5 per player-GW)
                    print(f"   {cp}: hold ({now_txt}best estimate GW{t_best} {est[t_best]:+.1f}; "
                          f"window ends GW{rep['window'][cp]})")
            if 'WC' in rep['window']:
                txt = f"{rep['exact']['WC']:+.2f}" if 'WC' in rep['exact'] else "not computed"
                print(f"   WC: exact Delta now {txt}; later GWs are not estimated [assumption] -> compare yourself "
                      f"(window ends GW{rep['window']['WC']}).")
            if now_ok:
                pick = max(now_ok, key=lambda cp: rep['exact'][cp])
                print(f">> Suggestion: play {pick} in GW{current_gw} (exact Delta {rep['exact'][pick]:+.2f}). "
                      f"Re-run with --chip {pick} for the plan that uses it.")
                if len(now_ok) > 1:
                    print(f"   Also best now: {', '.join(cp for cp in now_ok if cp != pick)} (one chip per GW).")
                # C11: WC is outside the rule (no estimates), but one chip per GW: playing the
                # suggested chip rules out a WC this GW. Shown when the WC gains more now
                # (first real run, GW8: FH +3.03 suggested while WC was +10.35, notes_07 §13).
                if pick != 'WC' and rep['exact'].get('WC', float('-inf')) > rep['exact'][pick]:
                    print(f"   Note: WC gains more now ({rep['exact']['WC']:+.2f} vs {rep['exact'][pick]:+.2f}); "
                          f"playing {pick} rules out WC this GW, and the WC timing is not estimated [assumption]. "
                          f"Compare with --chip WC.")
            else:
                print(">> Suggestion: no chip this GW.")
        if rep['skipped']:
            print(f"[*] Time limit: not computed: {', '.join(rep['skipped'])} (raise --time-limit).")
        print(f"[*] Chip analysis: {time.time() - t_rep:.1f} s")

    log_predictions(current_gw, players, final_squad)  # V1: whole pool, not only the XI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPL Engine v2.0 - Operations Research Optimization")
    parser.add_argument("--mode", type=str, choices=['optimize', 'evaluate'], default='optimize', help="Execution mode (Default: optimize)")
    parser.add_argument("--team", type=int, help="Official FPL Team ID (Required for optimize)")
    parser.add_argument("--ft", type=int, default=None,
                        help="Free transfers available now (1-5). Default: rebuilt from your GW history")  # M8
    # C6: the chip to play NOW (override). The suggestion comes from the "Chip suggestions" section;
    # the model does not choose chips inside the main plan (notes_07 §4).
    parser.add_argument("--chip", type=str, choices=['WC', 'FH', 'BB', 'TC', 'NONE'], default='NONE',
                        help="Chip played in the current GW (Default: NONE; see the Chip suggestions section)")
    parser.add_argument("--no-chip-report", action="store_true",
                        help="Skip the Chip suggestions section (saves its solve time)")   # C10
    parser.add_argument("--min-chance", type=int, default=75, help="Minimum availability probability percentage (Default: 75)")
    # M7: default 5. Timing on the offline harness (700 players, 2 cores): lookahead 3 ->
    # 10 s, 5 -> 88 s (proven optimal), 8 -> 244 s with a 0.45-point gap left (notes_06b §9).
    parser.add_argument("--lookahead", type=int, default=5,
                        help="Gameweeks planned, including the current one (Default: 5)")
    parser.add_argument("--override-budget", type=float, default=None, help="Total budget in GBPm = selling value of your squad + bank, as shown in the app "
                             "(e.g. 100.9). Default: computed from your transfers")  # S2
    parser.add_argument("--gw", type=int, default=None, help="Gameweek number to calibrate (Required for evaluate)")
    # M7: the multi-period model values waiting for a free transfer itself, so the cap
    # is no longer needed for correctness; it stays as an optional risk limit
    # (a hit is a certain -4, its gain is an estimate).
    parser.add_argument("--max-hits", type=int, default=None,
                        help="Optional cap on paid transfers (-4 each) per GW (Default: no cap)")
    parser.add_argument("--explain", type=str, default=None,
                        help="Comma-separated web names; print their xP components (e.g. Gabriel,Greaves)")  # X7
    parser.add_argument("--time-limit", type=int, default=SOLVER_TIME_LIMIT,
                        help=f"Seconds for the whole solve; the optimality gap is always printed "
                             f"(Default: {SOLVER_TIME_LIMIT})")  # M7/M10
    parser.add_argument("--solver", type=str, choices=['highs', 'cbc'], default=None,
                        help="MIP solver (Default: highs if highspy is installed, else cbc)")  # M10

    args = parser.parse_args()

    if args.mode == 'evaluate':
        if not args.gw:
            print("[ERROR] Evaluation mode requires target Gameweek. Example: --mode evaluate --gw 5")
            sys.exit(1)
        evaluate_model(args.gw)
    else:
        if not args.team:
            print("[ERROR] Optimization mode requires --team ID. Example: --team 10248663")
            sys.exit(1)
        if args.max_hits is not None and args.max_hits < 0:  # H: negative cap -> infeasible
            print("[ERROR] --max-hits must be 0 or greater.")
            sys.exit(1)
        if args.ft is not None and not 1 <= args.ft <= MAX_FT:  # M8: FPL stores 1 to 5
            print(f"[ERROR] --ft must be between 1 and {MAX_FT}.")
            sys.exit(1)
        if args.lookahead < 1:
            print("[ERROR] --lookahead must be 1 or greater.")
            sys.exit(1)
        if args.solver == 'highs' and highspy is None:  # M10
            print("[ERROR] --solver highs needs the highspy package: pip install highspy")
            sys.exit(1)
        optimize_team(args.team, args.ft, args.chip, args.min_chance, args.lookahead,
                      args.override_budget, args.max_hits, args.explain, time_limit=args.time_limit,
                      solver=args.solver, chip_suggest=not args.no_chip_report)
