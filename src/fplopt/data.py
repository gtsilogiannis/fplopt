"""Gameweeks, fixtures and the per-(player, GW) history that the xP model learns from."""
import logging

import pandas as pd

from .xp_model import HIST_COLS, HIST_STATS

log = logging.getLogger(__name__)


def get_current_gw(events):
    """The next gameweek (the one whose deadline is ahead); 1 before the season."""
    for event in events:
        if event['is_next']:
            return event['id']
    return 1


def build_fixture_map(current_gw, max_gw_lookahead, all_fixtures):
    """{team id: [{'opp', 'is_home', 'gw'}, ...]} for the unfinished fixtures of GWs
    current_gw .. current_gw + max_gw_lookahead - 1 (capped at GW38)."""
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


def fetch_history(client, bootstrap, all_fixtures, current_gw):
    """X3: per-(player, GW) stats of every FINISHED gameweek before current_gw,
    from /event/{gw}/live/ (one request per gameweek).
    Returns (DataFrame with HIST_COLS, list of the finished GWs)."""
    team_of = {p['id']: p['team'] for p in bootstrap['elements']}
    event_of = {e['id']: e for e in bootstrap['events']}
    gws = [e['id'] for e in bootstrap['events'] if e['finished'] and e['id'] < current_gw]
    n_matches = {}
    for f in all_fixtures:
        if f['event'] in gws:
            for t in (f['team_h'], f['team_a']):
                n_matches[(t, f['event'])] = n_matches.get((t, f['event']), 0) + 1
    rows = []
    missing = set()
    for gw in gws:
        live = client.event_live(gw, event_of.get(gw))
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
        log.warning(f"[WARNING] Live endpoint lacks fields {sorted(missing)}; treated as 0.",
                    extra={'code': 'live_fields_missing'})
    log.info(f"[*] History: {len(gws)} finished GWs, {len(rows)} player-GW rows")
    return pd.DataFrame(rows, columns=HIST_COLS), gws
