"""Small hand-checkable player pools for the MILP tests (port of the toy pools of
claude/harness_6b.py). No API: the pool dicts are built directly in the format the
solver expects ({'ids', 'pos', 'team', 'price', 'sell', 'xp', 'papp', 'avail'})."""


def toy_pool():
    """3 GK, 7 DEF, 7 MID, 5 FWD (ids 1-22), each on his own team, price 5.0."""
    pool = {'ids': [], 'pos': {}, 'team': {}, 'price': {}, 'sell': {}, 'xp': {}, 'papp': {}, 'avail': {}}
    pid = 0
    for p_, n_ in ((1, 3), (2, 7), (3, 7), (4, 5)):
        for _ in range(n_):
            pid += 1
            pool['ids'].append(pid)
            pool['pos'][pid] = p_
            pool['team'][pid] = pid
            pool['price'][pid] = 5.0
            pool['avail'][pid] = True
    return pool


def fill(pool, horizon, xp=2.0, papp=1.0):
    """Every player without a value gets xP 2.0 and P(app) 1.0 in every GW of the horizon."""
    for i in pool['ids']:
        for g in horizon:
            pool['xp'].setdefault((i, g), xp)
            pool['papp'].setdefault((i, g), papp)


def add(pool, i, pos, price, xp_by_g, papp_by_g=None, team=None):
    """Adds a target player i with the given xP per GW."""
    pool['ids'].append(i)
    pool['pos'][i] = pos
    pool['team'][i] = team or i
    pool['price'][i] = price
    pool['avail'][i] = True
    for g, v in xp_by_g.items():
        pool['xp'][(i, g)] = v
        pool['papp'][(i, g)] = (papp_by_g or {}).get(g, 1.0)


def base_squad(pool):
    """Owned squad: 2 GK, 5 DEF, 5 MID, 3 FWD of the toy pool. The last FWD gets 20 xP in every
    GW so that he is always the captain (the targets' gains are then not doubled)."""
    by = {p: [i for i in pool['ids'] if pool['pos'][i] == p] for p in (1, 2, 3, 4)}
    sq = by[1][:2] + by[2][:5] + by[3][:5] + by[4][:3]
    for i in sq:
        pool['sell'][i] = pool['price'][i]
    for (i, g) in list(pool['xp']):
        if i == sq[14]:
            pool['xp'][(i, g)] = 20.0
    return sq


def weights(horizon):
    """Discount weights 0.85^(g - first GW), as in the optimizer."""
    return {g: 0.85 ** (g - horizon[0]) for g in horizon}


def chip_gws(res):
    """GWs in which the plan plays a chip."""
    return [g for g, p in res['plan'].items() if p['chip']]
