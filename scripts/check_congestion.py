"""X11: fixture congestion vs the minutes model: is there an error the model does not already see?

Usage (from scripts/):  python check_congestion.py OPENFOOTBALL_DIR [SEASON]
  OPENFOOTBALL_DIR: folder with clones of github.com/openfootball/england and .../champions-league
  SEASON: default 2024-25 (the only season with complete non-PL data in openfootball)

Sample: the walk-forward rows of check_minutes.table() (single-match GWs only, so one row = one
player in one Premier League match). Predictions: xp_model.shrink_minutes (production code).
Residuals: r_app = played - P(appear), r_60 = (60+ minutes) - P(60+).
Congestion: days since the team's previous match / until its next one, any competition
(PL + UCL/UEL/UECL + FA Cup + EFL Cup), in UK calendar days.
Standard errors are clustered by (team, PL match): the rotation decision is one per team per match.

Result, 2024/25 (rule fixed before the run: a correction is worth modelling only if the gap exceeds
2 s.e.): previous match outside the PL <= 4 days before vs previous match >= 6 days before,
P(appear) residual -0.008 +/- 0.006, P(60+) residual -0.003 +/- 0.006 -> no effect the walk-forward
minutes model misses; congestion is not part of the xP model.
"""
import os
import sys
import numpy as np
import pandas as pd
from fplopt import xp_model as F
from check_minutes import table, season_data, POS_NAME
from fixtures_txt import parse

# openfootball name (after normalise) -> FPL / vaastav short name; names equal after normalise are not listed
ALIAS = {'Brighton & Hove Albion': 'Brighton', 'Ipswich Town': 'Ipswich',
         'Leicester City': 'Leicester', 'Manchester City': 'Man City', 'Manchester United': 'Man Utd',
         'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'm Forest", 'Tottenham Hotspur': 'Spurs',
         'West Ham United': 'West Ham', 'Wolverhampton Wanderers': 'Wolves', 'Leeds United': 'Leeds'}
NON_PL = {  # competition key -> files (repository, file name); a missing qualifier file is skipped
    'UCL': [('champions-league', 'cl.txt'), ('champions-league', 'clq.txt')],
    'UEL': [('champions-league', 'el.txt'), ('champions-league', 'elq.txt')],
    'UECL': [('champions-league', 'conf.txt'), ('champions-league', 'confq.txt')],   # e.g. Chelsea, 2024/25 play-off round
    'FAC': [('england', 'facup.txt')], 'EFL': [('england', 'eflcup.txt')]}
EUROPE = ('UCL', 'UEL', 'UECL')


def normalise(name):
    """'Arsenal FC (ENG)' -> 'Arsenal'; 'AFC Bournemouth' -> 'Bournemouth'; then the FPL short name."""
    n = name.split(' (')[0].strip()             # drop the country tag of UEFA files
    for pre in ('AFC ',):
        if n.startswith(pre):
            n = n[len(pre):]
    for suf in (' FC', ' AFC'):
        if n.endswith(suf):
            n = n[:-len(suf)]
    return ALIAS.get(n, n)


def pl_fixtures(d):
    """vaastav per-player rows -> one row per (team, PL fixture): team name, fixture id, UK date, home/away."""
    teams = d[['team', 'team_id']].drop_duplicates()
    name_of = dict(zip(teams['team_id'], teams['team']))
    f = d[['fixture', 'team', 'opponent_team', 'was_home', 'kickoff_time']].drop_duplicates(['fixture', 'team'])
    f = f.copy()
    # UTC kickoff -> calendar date in the UK (all other sources give UK/European local dates)
    f['date'] = pd.to_datetime(f['kickoff_time'], utc=True).dt.tz_convert('Europe/London').dt.date
    f['opp'] = f['opponent_team'].map(name_of)
    return f[['team', 'fixture', 'date', 'was_home', 'opp']]


def check_pl(of_dir, season, f):
    """The openfootball PL file must give the same (date, home, away) set as vaastav: proves names + dates."""
    of = {(dt, normalise(h), normalise(a)) for dt, h, a in parse(os.path.join(of_dir, 'england', season, '1-premierleague.txt'))}
    home = f[f['was_home']]
    va = {(dt, h, a) for dt, h, a in zip(home['date'], home['team'], home['opp'])}
    only_of, only_va = of - va, va - of
    print(f'[check] PL matches: openfootball {len(of)}, vaastav {len(va)}, '
          f'differences {len(only_of)} / {len(only_va)}')
    for x in sorted(only_of)[:10]:
        print('   only openfootball:', x)
    for x in sorted(only_va)[:10]:
        print('   only vaastav     :', x)
    return not only_of and not only_va


def non_pl(of_dir, season, pl_teams):
    """Non-PL matches of PL clubs: rows (team, date, comp)."""
    rows = []
    for comp, files in NON_PL.items():
        for repo, fn in files:
            p = os.path.join(of_dir, repo, season, fn)
            if not os.path.exists(p) and fn.endswith('q.txt'):
                continue                           # qualifier file not published for this season
            for dt, h, a in parse(p):
                for t in (normalise(h), normalise(a)):
                    if t in pl_teams:
                        rows.append((t, dt, comp))
    out = pd.DataFrame(rows, columns=['team', 'date', 'comp'])
    print('[check] non-PL matches of PL clubs:', out.groupby('comp').size().to_dict())
    return out


def congestion(f, np_):
    """Per (team, PL fixture): days to the previous / next match of the team in any competition."""
    sched = pd.concat([f[['team', 'date']].assign(comp='PL', fixture=f['fixture']),
                       np_.assign(fixture=-1)], ignore_index=True).sort_values(['team', 'date'])
    dup = sched.duplicated(['team', 'date'], keep=False)
    if dup.any():                                  # two matches of one club on one day = a data error
        raise ValueError(f'same-day matches:\n{sched[dup]}')
    g = sched.groupby('team')
    sched['d_prev'] = (pd.to_datetime(sched['date']) - pd.to_datetime(g['date'].shift(1))).dt.days
    sched['prev_comp'] = g['comp'].shift(1)
    sched['d_next'] = (pd.to_datetime(g['date'].shift(-1)) - pd.to_datetime(sched['date'])).dt.days
    sched['next_comp'] = g['comp'].shift(-1)
    return sched[sched['comp'] == 'PL'].drop(columns='comp')


def build(of_dir, season):
    d, hist, pos_of = season_data(season)
    f = pl_fixtures(d)
    if not check_pl(of_dir, season, f):
        raise SystemExit('PL fixtures do not match: fix the name map / parser first')
    c = congestion(f, non_pl(of_dir, season, set(f['team'])))
    x = table(season)
    x = x[x['n_matches'] == 1].copy()
    # fixture + team of each (player, GW) row: single-match GWs have exactly one fixture
    fx = d.groupby(['element', 'gw']).agg(fixture=('fixture', 'first'), team=('team', 'last')).reset_index()
    x = x.merge(fx, left_on=['element', 't'], right_on=['element', 'gw'], how='left')
    x = x.merge(c, on=['team', 'fixture'], how='left')
    papp, p60, _ = F.shrink_minutes(x['W'], x['S_app'], x['S_60'], x['S_min'])
    x['Papp'], x['P60'] = np.asarray(papp), np.asarray(p60)
    x['r_app'], x['r_60'] = x['played'] - x['Papp'], x['y60'] - x['P60']
    x['cluster'] = x['team'] + '#' + x['fixture'].astype(str)
    x['prev_nonpl'] = x['prev_comp'].notna() & (x['prev_comp'] != 'PL')   # NaN (no previous match) -> False
    x['next_nonpl'] = x['next_comp'].notna() & (x['next_comp'] != 'PL')
    print(f'[check] rows {len(x)}, without a previous match {int(x["d_prev"].isna().sum())}')
    return x


def cmean(df, col):
    """Cluster-robust mean: r = sum R_c / sum n_c, Var = C/(C-1) * sum (R_c - r n_c)^2 / (sum n_c)^2."""
    g = df.groupby('cluster')[col].agg(['sum', 'size'])
    n, C = g['size'].sum(), len(g)
    r = g['sum'].sum() / n
    var = C / (C - 1) * ((g['sum'] - r * g['size']) ** 2).sum() / n ** 2 if C > 1 else np.nan
    return r, float(np.sqrt(var)), int(n), C


def contrast(name, a, b, cols=('r_app', 'r_60')):
    """Mean residual in group a minus group b (disjoint clusters: SEs add in quadrature)."""
    out = []
    for col in cols:
        ra, sa, na, ca = cmean(a, col)
        rb, sb, nb, cb = cmean(b, col)
        diff, se = ra - rb, float(np.hypot(sa, sb))
        out.append({'test': name, 'resid': col, 'mean A': ra, 'mean B': rb, 'diff': diff, 'SE': se,
                    'z': diff / se, 'rows A': na, 'matches A': ca, 'rows B': nb, 'matches B': cb})
    return out


def groups(x):
    """Player groups: all, regular starters, rotation players, each position."""
    yield 'all', x
    yield 'starter (Papp>=0.8)', x[x['Papp'] >= 0.8]
    yield 'rotation (0.2-0.8)', x[(x['Papp'] >= 0.2) & (x['Papp'] < 0.8)]
    for p, name in POS_NAME.items():
        yield name, x[x['pos'] == p]


def report(x):
    pd.set_option('display.width', 200)
    fmt = lambda rows: pd.DataFrame(rows).round(4).to_string(index=False)
    print('\n=== PRIMARY P1 / P2: previous match outside PL and <= 4 days before (A) vs previous match >= 6 days (B)')
    rows = []
    for gname, g in groups(x):
        rows += contrast(gname, g[g['prev_nonpl'] & (g['d_prev'] <= 4)], g[g['d_prev'] >= 6])
    print(fmt(rows))
    print('\n=== S1: days since the previous match (any competition), mean residual per bucket')
    x = x.copy()
    x['bucket'] = pd.cut(x['d_prev'], [0, 3, 4, 5, 400], labels=['<=3', '4', '5', '>=6'])
    rows = []
    for gname, g in groups(x):
        for b, gb in g.groupby('bucket', observed=True):
            for col in ('r_app', 'r_60'):
                r, s, n, C = cmean(gb, col)
                rows.append({'group': gname, 'd_prev': b, 'resid': col, 'mean': r, 'SE': s, 'rows': n, 'matches': C})
    print(fmt(rows))
    print('\n=== S2: next match outside PL within <= 4 days (A) vs next match >= 6 days (B)')
    rows = []
    for gname, g in groups(x):
        rows += contrast(gname, g[g['next_nonpl'] & (g['d_next'] <= 4)], g[g['d_next'] >= 6])
    print(fmt(rows))
    print('\n=== S3: P1 split by the previous competition (Europe vs domestic cups)')
    rows = []
    for gname, g in groups(x):
        for lab, comps in (('Europe', EUROPE), ('domestic cups', ('FAC', 'EFL'))):
            rows += contrast(f'{gname} | {lab}', g[g['prev_comp'].isin(comps) & (g['d_prev'] <= 4)], g[g['d_prev'] >= 6])
    print(fmt(rows))


if __name__ == '__main__':
    of_dir = sys.argv[1]
    season = sys.argv[2] if len(sys.argv) > 2 else '2024-25'
    report(build(of_dir, season))
