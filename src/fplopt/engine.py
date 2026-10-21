"""One optimizer run: API data -> xP of every player -> multi-period MILP -> chip analysis.

optimize() returns an OptimizeResult (plain data, no printing): report.py turns it into text,
jsonout.py into JSON. Progress and warnings are sent to the 'fplopt' logger as they happen
(the command line prints them on stderr); warnings are also collected in result.warnings.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import xp_model as xpm
from .chips import advise, chip_report
from .data import build_fixture_map, fetch_history, get_current_gw
from .errors import SolverError, UserActionError
from .messages import collect
from .milp import BENCH_BETA_TOL, MIP_REL_GAP, SOLVER_TIME_LIMIT, plan_squad, require_pulp_3
from .predictions import PREDICTIONS_FILE, log_predictions
from .rules import HIT_COST
from .squad import available_chips, free_transfers_from_history, load_current_squad

log = logging.getLogger(__name__)

# Discount weight per gameweek AHEAD in the objective: w_g = 0.85^(g - current GW), applied to
# the points AND the hits of GW g (decision A1, notes_06b §4). [assumption] a decision
# parameter (projections further ahead are less reliable), not derived from data; its value
# is studied with the rolling-horizon backtest of chat 9b.
TIME_DECAY_RATE = 0.85
POSITION_NAME = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}


# --------------------------------------------------------------------------- input
@dataclass
class Options:
    """Everything a run depends on besides the API data (the command-line options)."""
    team_id: int
    ft: Optional[int] = None             # free transfers now; None = rebuilt from the history (M8)
    chip: str = 'NONE'                   # chip played in the current GW: WC, FH, BB, TC or NONE
    min_chance: int = 75                 # below this chance_of_playing a player counts as out
    lookahead: int = 5                   # gameweeks planned, the current one included
    override_budget: Optional[float] = None
    max_hits: Optional[int] = None       # cap on paid transfers per GW; None = no cap
    explain: Optional[str] = None        # comma-separated web names for the xP breakdown
    use_vice: bool = True                # vice-captain term in the objective (M6)
    use_bench: bool = True               # autosub value of the bench (M5)
    time_limit: Optional[int] = None     # seconds for the whole run; None = SOLVER_TIME_LIMIT
    solver: Optional[str] = None         # 'highs' or 'cbc'; None = HiGHS if installed
    chip_report: bool = True             # compute the Chip suggestions section
    write_log: bool = True               # append the predictions to log_path
    log_path: str = PREDICTIONS_FILE


# --------------------------------------------------------------------------- output
@dataclass
class SquadRow:
    """One player of the squad chosen for the current GW."""
    id: int
    name: str
    team: int
    position: str              # GK, DEF, MID, FWD
    pos_id: int                # 1-4
    cost: float                # selling price if kept, current price if bought (GBPm)
    xp_now: float              # xP in the current GW
    papp_now: float            # P(appears at least once in the current GW)
    xp_horizon: float          # weighted xP over the horizon
    n_matches_now: int         # 0 = blank, 2 = double gameweek
    role: str                  # Captain, Vice, Start, Bench
    bench_slot: int            # position in the bench order (0 = GK), -1 for starters
    action: str                # KEEP or BUY
    available: bool            # False = below --min-chance (may not start this GW)
    source: str                # 'model' or 'FPL ep' (fallback: no GW at his club yet)


@dataclass
class Transfer:
    out_id: int
    out_name: str
    in_id: int
    in_name: str


@dataclass
class PlanGW:
    """The plan of one gameweek of the horizon."""
    gw: int
    chip: Optional[str]
    ft: Optional[int]          # free transfers before this GW's transfers (None = unlimited)
    ft_unlimited: bool         # first deadline of the team, Wildcard or Free Hit
    transfers: list            # [Transfer], sales and buys paired by position
    hits: int
    captain_id: int
    captain_name: str
    vice_id: Optional[int]
    vice_name: Optional[str]
    squad: list                # 15 player ids
    xi: list                   # 11 player ids
    bench: list                # bench order: GK first, then outfield slots 1-3


@dataclass
class ExpectedPoints:
    """Expected points of the current GW, split into the terms of the objective (unweighted)."""
    xi: float
    captain: float
    vice: float
    autosubs: float
    hits: int                  # points deducted (4 per hit)
    total: float
    bench_weights: Optional[list] = None   # [b1, b2, b3] = P(bench slot k is used); None with BB


@dataclass
class SolverInfo:
    name: str
    seconds: float
    passes: int
    proven: bool               # stopped by the gap target, not by the time limit
    gap: Optional[float]       # weighted points; None = bound unknown
    objective: float           # weighted over the horizon, hits included
    beta_change: float


@dataclass
class ChipSection:
    """Chip suggestions (C6-C11)."""
    source: str                # where the chip windows came from (API or rule fallback)
    sets: list                 # [(chip, start GW, stop GW, state)], state: available / played GWn / expired
    status: str                # computed | no_chip_now | skipped_flag (--no-chip-report) | skipped_chip (--chip)
    window: dict = field(default_factory=dict)     # {chip: last GW of the window that contains now}
    exact: dict = field(default_factory=dict)      # {chip: Delta now}
    error: dict = field(default_factory=dict)      # {chip: bound on the error of Delta now}
    estimates: dict = field(default_factory=dict)  # {chip: {gw: estimated Delta}}
    advice: Optional[dict] = None                  # chips.advise()
    not_computed: list = field(default_factory=list)
    seconds: Optional[float] = None


@dataclass
class ExplainEntry:
    """--explain: the parts of one player's xP."""
    id: int
    name: str
    team_name: str
    pos_id: int
    price: float
    source: str
    chance: Optional[int]
    minutes_model: Optional[dict]   # Papp, P60, Emin (shrunk) and raw_*, Wapp, minutes_total, price_factor
    per90: list                     # [(stat, value per 90)]
    components: dict                # {component: points} in the current GW
    ep_next: float
    xp_by_gw: list                  # [(gw, xP)] over the horizon


@dataclass
class EngineState:
    """Internals of a run, for tests and the backtests of later chats (not part of the JSON)."""
    pool: dict
    plan_result: dict
    horizon: list
    weights: dict
    start_squad: list
    budget: float
    ft0: Optional[int]
    players: list


@dataclass
class OptimizeResult:
    options: Options
    current_gw: int
    horizon: list
    weights: dict                   # {gw: weight}
    history_gws: list               # finished GWs the model learned from
    history_rows: int
    gw_fixtures: list               # [(gw, [blank teams], [double teams])] where any
    ft: Optional[int]
    ft_source: str                  # history | override | unlimited
    ft_rebuilt: Optional[int]       # value rebuilt from the history (also when --ft is given)
    budget: float
    squad_value: Optional[float]    # selling value of the current squad (None at GW1)
    bank: float
    budget_source: str              # computed | override | default (GW1)
    squad: list                     # [SquadRow], XI by position then the bench in its order
    plan: list                      # [PlanGW] over the horizon
    sales_now: list                 # [(id, name, selling price, available)]
    expected_now: ExpectedPoints
    solver: SolverInfo
    chips: ChipSection
    fallback_players: int           # players using FPL's ep_next
    explain: list                   # [ExplainEntry]
    explain_not_found: list
    log_written: Optional[str]      # path of the predictions log, None if not written
    warnings: list                  # [{'code', 'message'}]
    timings: dict                   # data_s, solve_s, chips_s, total_s
    api_stats: dict                 # requests, retries, cache_hits, cache_writes of the client
    state: EngineState = field(repr=False, default=None)


def optimize(client, opts):
    """Runs the optimizer for opts.team_id with data from client (api.FplClient or a fake)."""
    require_pulp_3()
    with collect() as warnings:
        return _optimize(client, opts, warnings)


def _optimize(client, opts, warnings):
    t_start = time.time()
    team_id, user_ft, chip_used, min_chance = opts.team_id, opts.ft, opts.chip, opts.min_chance
    lookahead, override_budget, max_hits, explain = opts.lookahead, opts.override_budget, opts.max_hits, opts.explain
    use_vice, use_bench, time_limit, solver = opts.use_vice, opts.use_bench, opts.time_limit, opts.solver
    log.info(f"[*] MILP Optimizer | Team: {team_id} | Lookahead: {lookahead} GWs")

    bootstrap = client.bootstrap()
    current_gw = get_current_gw(bootstrap['events'])

    # G1: the planning horizon is a list of gameweeks, e.g. [8, 9, 10].
    # Every xP value and every lineup/captain decision below is indexed by these GWs.
    # M3: with a Free Hit the horizon is no longer cut to one GW: the FH squad only
    # exists in the current GW, and the later GWs are planned from the squad it reverts to.
    horizon = [g for g in range(current_gw, current_gw + lookahead) if g <= 38]
    gw_weight = {g: TIME_DECAY_RATE ** (g - current_gw) for g in horizon}
    log.info(f"[*] Planning horizon: GW{horizon[0]}-GW{horizon[-1]} | weights: "
             + ", ".join(f"GW{g}={w:.2f}" for g, w in gw_weight.items()))

    # X3: one request for every fixture of the season (past and future)
    all_fixtures = client.fixtures()
    # C7: xP for EVERY remaining GW (not only the horizon): the chip estimates compare GWs up to
    # the end of each chip's window (GW19 / GW38). The MILP still uses only the horizon.
    ext_gws = list(range(current_gw, 39))
    fixtures_map = build_fixture_map(current_gw, len(ext_gws), all_fixtures)

    # X3/X4: the model's inputs come from finished GWs of this season only.
    hist, history_gws = fetch_history(client, bootstrap, all_fixtures, current_gw)
    pos_of = {p['id']: p['element_type'] for p in bootstrap['elements']}
    price_of = {p['id']: p['now_cost'] / 10.0 for p in bootstrap['elements']}  # X8: current price
    # X9: GW from which each player is at his current club (bootstrap field team_join_date)
    deadlines = {e['id']: pd.Timestamp(e['deadline_time']) for e in bootstrap['events']
                 if e.get('deadline_time')}
    join_dates = {p['id']: p.get('team_join_date') for p in bootstrap['elements']}
    if deadlines and any(join_dates.values()):
        start_gw_of = xpm.first_gw_at_club(join_dates, deadlines)
    else:  # field missing from the API -> minutes model uses all GWs (chat-8a behaviour)
        log.warning("[WARNING] deadline_time / team_join_date missing; minutes model uses all GWs.",
                    extra={'code': 'join_dates_missing'})
        start_gw_of = None
    feats, team_rating, league_xg = xpm.build_features(hist, pos_of, current_gw, price_of, start_gw_of)
    # 8c (notes_08c §8): if the GW before current_gw is not final yet (bonus points / corrections
    # still pending) it is missing from hist. The ages stay calendar ages, so W is smaller and
    # the minutes model shrinks more towards its prior; this is on purpose and only reported here.
    if not hist.empty and hist['gw'].max() < current_gw - 1:
        log.info(f"[INFO] GW{current_gw - 1} is not final yet: the minutes model uses data up to "
                 f"GW{int(hist['gw'].max())} (its estimates are shrunk more towards the prior).",
                 extra={'code': 'previous_gw_not_final'})
    if league_xg is None:
        log.warning("[WARNING] No finished GW yet: every player uses FPL's ep_next (fallback).",
                    extra={'code': 'no_history'})

    # G2: number of matches per (team, GW). 0 = blank (BGW), 2 = double (DGW).
    team_ids = [t['id'] for t in bootstrap['teams']]
    n_matches = {(t, g): 0 for t in team_ids for g in ext_gws}   # C7: all remaining GWs
    for t, matches in fixtures_map.items():
        for m in matches:
            n_matches[(t, m['gw'])] += 1
    # display only; .get() with fallbacks so a renamed API field cannot crash the run
    team_name = {t['id']: t.get('short_name') or t.get('name') or str(t['id']) for t in bootstrap['teams']}
    gw_fixtures = []
    for g in horizon:
        blank = [team_name[t] for t in team_ids if n_matches[(t, g)] == 0]
        double = [team_name[t] for t in team_ids if n_matches[(t, g)] >= 2]
        if blank or double:
            gw_fixtures.append((g, blank, double))
            log.info(f"[*] GW{g}: blank: {', '.join(blank) or '-'} | double: {', '.join(double) or '-'}")

    # A: fetch the current squad BEFORE filtering players, so the filter below knows who we own.
    ft0 = None   # M8: None = unlimited free transfers (before the team's first deadline)
    ft_api, ft_source, history = None, 'unlimited', None
    if current_gw == 1:
        current_squad = []
        bank = 0.0
        sell_price = {}
    else:
        history = client.entry_history(team_id)
        # R1 + S1: squad after a possible Free Hit revert, bank, selling prices
        current_squad, bank, sell_price = load_current_squad(client, team_id, current_gw, bootstrap, history)
        # M8: free transfers rebuilt from the GW history; --ft overrides
        ft_api, issues = free_transfers_from_history(history, current_gw)
        for msg in issues:
            log.warning(f"[WARNING] Free-transfer rebuild: {msg}", extra={'code': 'ft_rebuild'})
        if user_ft is not None:
            ft0, ft_source = user_ft, 'override'
            log.info(f"[*] Free transfers: {ft0} (--ft; rebuilt from history: {ft_api})")
        elif issues:
            raise UserActionError("The rebuilt free-transfer count does not match the hit costs in the API. "
                                  "Pass the number shown in the app with --ft.")
        else:
            ft0, ft_source = ft_api, ('history' if ft_api is not None else 'unlimited')
            log.info(f"[*] Free transfers: {ft0} (rebuilt from history; --ft overrides)")
    if ft0 is None:
        log.info("[*] Transfers are unlimited this GW (first deadline of the team).")

    # C6: chip sets not played yet (bootstrap-static windows minus the entry history), for the
    # "Chip suggestions" section. The main plan does NOT get chip variables: with every chip free
    # the MILP stopped at the time limit with a gap of 117 points at lookahead 5 and a worse plan
    # than without chips (notes_07 §4). --chip forces a chip in the current GW, as before.
    chip_hist = history if current_gw > 1 else client.entry_history(team_id)
    chip_windows, chip_rows, chip_warn, chip_src = available_chips(bootstrap, chip_hist, current_gw)
    for msg in chip_warn:
        log.warning(f"[WARNING] Chips: {msg}", extra={'code': 'chips'})

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
        # X5/X9: fallback for players with no GW yet at their current club (GW1, new signings):
        # FPL's ep_next per match. A player who has been at the club all season without playing
        # stays in the model (P(appear) ~ 0). It REPLACES the model for these players, it is
        # never added to it, so nothing is counted twice.
        source = 'model' if f is not None else 'FPL ep'
        n_next = n_matches.get((team, current_gw), 0)
        fallback_per_match = ep_next / n_next if n_next > 0 else float(p.get('points_per_game') or 0.0)

        # X4: xp_gw[g] = sum over the team's matches in GW g of match_xp.
        xp_gw = {g: 0.0 for g in ext_gws}            # C7: every remaining GW
        # M6: P(no appearance in GW g) = product over his matches of (1 - P(appear)).
        # [assumption] the matches of a DGW are independent. Blank GW -> product 1 -> P = 0.
        # M9/8c: per-match P(appear) for the vice-captain / autosub terms is the SAME shrunk
        # Papp that the xP uses (build_features). Fallback players (no GW at the club, W = 0)
        # get the prior mean m.
        p_match = f['Papp'] if f is not None else xpm.APP_PRIOR_MEAN
        p_none = {g: 1.0 for g in ext_gws}
        comp_now = {}
        for match in fixtures_map.get(team, []):
            g = match['gw']
            p_none[g] *= 1.0 - p_match
            if f is not None:
                c = xpm.match_xp(f, pos, team_rating.get(team, xpm.NEUTRAL_TEAM),
                                 team_rating.get(match['opp'], xpm.NEUTRAL_TEAM), match['is_home'], league_xg)
                mx = sum(c.values())
                if g == current_gw:
                    for k_, v_ in c.items():
                        comp_now[k_] = comp_now.get(k_, 0.0) + v_
            else:
                mx = fallback_per_match
            if g == current_gw and chance is not None:
                # X5: FPL's own probability (e.g. 75) scales the NEXT GW only.
                # [assumption] the percentage is taken at face value.
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
            explained.append(_explain_entry(p, f, comp_now, xp_gw, source, chance, team_name, horizon))

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
    found = {e.name.lower() for e in explained}
    explain_not_found = sorted(explain_names - found)

    df = pd.DataFrame(players)

    # Financial state
    squad_sell_value = None
    if current_gw == 1:
        total_budget = override_budget if override_budget else 100.0
        budget_source = 'override' if override_budget else 'default'
    else:
        # S2: budget = selling value of the squad + bank. With cost = selling price for owned
        # players, 'sum of costs <= budget' means: each sale returns the selling price and each
        # purchase costs now_cost. Unavailable squad players are in df, so their cost counts.
        squad_sell_value = df[df['id'].isin(current_squad)]['cost'].sum()
        if override_budget:
            total_budget, budget_source = override_budget, 'override'
            log.info(f"[*] Budget (--override-budget): £{total_budget:.1f}m "
                     f"(computed: £{squad_sell_value + bank:.1f}m)")
        else:
            total_budget, budget_source = squad_sell_value + bank, 'computed'
            log.info(f"[*] Budget: £{total_budget:.1f}m = squad selling value £{squad_sell_value:.1f}m "
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
    t_solve = time.time()

    res = plan_squad(pool, horizon, gw_weight, start_squad, total_budget, ft0, chip_used,
                     max_hits, use_vice, use_bench, time_limit, solver)

    # C: solve() does not raise on failure; it only sets the status.
    if 'plan' not in res:
        raise SolverError(f"Solver status: {res['status']}. No valid plan exists under these constraints.\n"
                          f"        Check --override-budget, --min-chance, --ft and --max-hits.")
    plan = res['plan']
    now = plan[current_gw]
    gap_txt = f"{res['gap']:.2f}" if res['gap'] is not None else "unknown"
    log.info(f"[*] Solved with {res['solver']} in {res['seconds']:.1f} s ({res['passes']} pass(es) for the "
             f"bench weights); proven within {gap_txt} weighted points of the best plan of this model")
    if res.get('beta_change', 0.0) > BENCH_BETA_TOL:
        log.info(f"[*] Bench weights still changed by {res['beta_change']:.2f} after {res['passes']} passes "
                 f"(they depend on the chosen XI); the last plan is used.", extra={'code': 'bench_not_converged'})
    if not res['proven']:
        # M7: time limit reached; the plan is feasible and its distance from the best
        # possible plan of this model is at most the gap (weighted points)
        log.info(f"[*] Time limit reached before the {MIP_REL_GAP:.2%} target: a longer --time-limit "
                 f"can only improve the plan or tighten the gap.", extra={'code': 'time_limit'})

    # Squad of the current GW with roles
    n_now = {i: n_matches[(team_of[i], current_gw)] for i in ids}
    squad_rows = []
    for i in now['squad']:
        if i == now['captain']:
            role = 'Captain'
        elif i == now['vice']:
            role = 'Vice'                       # M6: logged too (counts as a starter in evaluate)
        elif i in now['xi']:
            role = 'Start'
        else:
            role = 'Bench'
        squad_rows.append(SquadRow(
            id=i, name=name_of[i], team=team_of[i], position=POSITION_NAME[pos_dict[i]], pos_id=pos_dict[i],
            cost=cost_dict[i], xp_now=xp_dict[(i, current_gw)], papp_now=pool['papp'][(i, current_gw)],
            xp_horizon=sum(gw_weight[g] * xp_dict[(i, g)] for g in horizon), n_matches_now=n_now[i],
            role=role, bench_slot=now['bench'].index(i) if i in now['bench'] else -1,
            action='KEEP' if i in current_squad else 'BUY', available=bool(avail_dict[i]),
            source=source_dict[i]))
    # XI by position, then the bench in its order (GK first, then outfield slots 1-3)
    squad_rows.sort(key=lambda r: (r.role == 'Bench', r.bench_slot, r.pos_id))

    # M: the whole plan (only the current GW is a decision now; later GWs are re-planned
    # every week with new data)
    plan_rows = []
    for g in horizon:
        pg = plan[g]
        by_pos = lambda lst: sorted(lst, key=lambda i: pos_dict[i])   # pair sales and buys by position
        moves = [Transfer(o, name_of[o], n, name_of[n]) for o, n in zip(by_pos(pg['out']), by_pos(pg['in']))]
        unlimited = (g == current_gw and ft0 is None) or pg['chip'] in ('WC', 'FH')
        plan_rows.append(PlanGW(
            gw=g, chip=pg['chip'], ft=None if unlimited else pg['ft'], ft_unlimited=unlimited, transfers=moves,
            hits=pg['hits'], captain_id=pg['captain'], captain_name=name_of[pg['captain']],
            vice_id=pg['vice'], vice_name=name_of[pg['vice']] if pg['vice'] is not None else None,
            squad=list(pg['squad']), xi=list(pg['xi']), bench=list(pg['bench'])))

    # S3: each sale of the current GW with the selling price it returns
    sales_now = [(i, name_of[i], cost_dict[i], bool(avail_dict[i])) for i in now['out']] if current_gw > 1 else []

    # G4/M5/M6: current-GW expected points, split into their parts (chip effects included)
    bb = now['chip'] == 'BB'                      # C3: chip of the current GW in the plan
    mult = 2.0 if now['chip'] == 'TC' else 1.0
    xi_pts = sum(xp_dict[(i, current_gw)] for i in (now['squad'] if bb else now['xi']))
    cap_pts = mult * xp_dict[(now['captain'], current_gw)]
    vice_pts = 0.0
    if now['vice'] is not None:
        vice_pts = mult * (1.0 - pool['papp'][(now['captain'], current_gw)]) * xp_dict[(now['vice'], current_gw)]
    bench_pts, beta_now = 0.0, None
    if use_bench and not bb:
        gk_start = next(i for i in now['xi'] if pos_dict[i] == 1)
        bench_gk = [i for i in now['bench'] if pos_dict[i] == 1]
        bench_pts += sum((1.0 - pool['papp'][(gk_start, current_gw)]) * xp_dict[(i, current_gw)] for i in bench_gk)
        outs = [i for i in now['bench'] if pos_dict[i] != 1]
        beta_now = res['beta'][current_gw]
        bench_pts += sum(beta_now[k] * xp_dict[(i, current_gw)] for k, i in enumerate(outs))
    hit_pts = HIT_COST * now['hits']
    expected = ExpectedPoints(xi=xi_pts, captain=cap_pts, vice=vice_pts, autosubs=bench_pts, hits=hit_pts,
                              total=xi_pts + cap_pts + vice_pts + bench_pts - hit_pts,
                              bench_weights=list(beta_now) if beta_now is not None else None)
    n_fb = sum(1 for p in players if p['source'] != 'model')
    if n_fb:
        log.info(f"[*] {n_fb} player(s) with no GW at their current club yet use FPL's ep_next (marked '(FPL ep)').")  # X5

    # C10: chip suggestions (opportunity costs), notes_07 §3
    chips = ChipSection(source=chip_src, sets=list(chip_rows), status='computed')
    t_chips = time.time()
    if not opts.chip_report:
        chips.status = 'skipped_flag'
    elif chip_used != 'NONE':
        chips.status = 'skipped_chip'
    else:
        time_left = (time_limit or SOLVER_TIME_LIMIT) - res['seconds']   # M10: one limit for the whole run
        rep = chip_report(pool, horizon, gw_weight, plan, start_squad, total_budget, ft0, chip_windows,
                          max_hits, use_vice, use_bench, time_left, solver)
        chips.window, chips.exact, chips.error, chips.estimates = rep['window'], rep['exact'], rep['gap'], rep['est']
        chips.not_computed = list(rep['skipped'])
        if not rep['window']:
            chips.status = 'no_chip_now'
        else:
            chips.advice = advise(rep, current_gw)
        if rep['skipped']:
            log.info(f"[*] Time limit: not computed: {', '.join(rep['skipped'])} (raise --time-limit).",
                     extra={'code': 'chips_not_computed'})
        chips.seconds = time.time() - t_chips
        log.info(f"[*] Chip analysis: {chips.seconds:.1f} s")
    chips_s = time.time() - t_chips

    log_written = None
    if opts.write_log:
        log_predictions(current_gw, players, {r.id: r.role for r in squad_rows}, opts.log_path)  # V1: whole pool
        log_written = opts.log_path

    return OptimizeResult(
        options=opts, current_gw=current_gw, horizon=horizon, weights=gw_weight, history_gws=history_gws,
        history_rows=len(hist), gw_fixtures=gw_fixtures, ft=ft0, ft_source=ft_source, ft_rebuilt=ft_api,
        budget=total_budget, squad_value=squad_sell_value, bank=bank, budget_source=budget_source,
        squad=squad_rows, plan=plan_rows, sales_now=sales_now, expected_now=expected,
        solver=SolverInfo(name=res['solver'], seconds=res['seconds'], passes=res['passes'], proven=res['proven'],
                          gap=res['gap'], objective=res['objective'], beta_change=res.get('beta_change', 0.0)),
        chips=chips, fallback_players=n_fb, explain=explained, explain_not_found=explain_not_found,
        log_written=log_written, warnings=warnings,
        timings={'data_s': t_solve - t_start, 'solve_s': res['seconds'], 'chips_s': chips_s,
                 'total_s': time.time() - t_start},
        api_stats=dict(getattr(client, 'stats', {})),
        state=EngineState(pool=pool, plan_result=res, horizon=horizon, weights=gw_weight, start_squad=start_squad,
                          budget=total_budget, ft0=ft0, players=players))


def _explain_entry(p, f, comp_now, xp_gw, source, chance, team_name, horizon):
    """X7: the data behind one --explain block (printed by report.py)."""
    mm = None
    if f is not None:
        mm = {k: f[k] for k in ('Papp', 'raw_Papp', 'P60', 'raw_P60', 'Emin', 'raw_Emin', 'Wapp',
                                'minutes_total', 'price_factor')}
    return ExplainEntry(
        id=p['id'], name=p['web_name'], team_name=team_name[p['team']], pos_id=p['element_type'],
        price=p['now_cost'] / 10, source=source, chance=chance, minutes_model=mm,
        per90=[(k[:-2], f[k]) for k in f if k.endswith('90')] if f is not None else [],
        components=dict(comp_now), ep_next=float(p.get('ep_next') or 0),
        xp_by_gw=[(g, xp_gw[g]) for g in horizon])
