"""Machine-readable output (--json): one JSON document on stdout, UTF-8.

Written for the C console frontend and the website (roadmap: GUI / frontend). Design rules:
  - exactly one document per run, also on failure ({"ok": false, "error": {...}});
  - no object whose keys are numbers: per-GW values are arrays of {"gw": ..., ...};
  - fixed field names; enums are short strings; null = not applicable / unknown;
  - numbers are not rounded (the text output rounds);
  - schema_version changes when a field is removed or changes meaning (adding fields does not).
The schema is documented in docs/json-output.md.
"""
import json
import math

from . import __version__
from .errors import EXIT_INTERNAL

SCHEMA_VERSION = "1.0"


def _header(mode, ok):
    return {'schema_version': SCHEMA_VERSION, 'engine_version': __version__, 'ok': ok, 'mode': mode}


def optimize_to_dict(r):
    """engine.OptimizeResult -> dict."""
    o, c, ep, s = r.options, r.chips, r.expected_now, r.solver
    adv = c.advice or {}
    doc = _header('optimize', True)
    doc.update({
        'input': {
            'team_id': o.team_id, 'ft': o.ft, 'chip': o.chip, 'min_chance': o.min_chance,
            'lookahead': o.lookahead, 'override_budget': o.override_budget, 'max_hits': o.max_hits,
            'time_limit': o.time_limit, 'solver': o.solver, 'chip_report': o.chip_report,
            'write_log': o.write_log,
            'check': o.check, 'check_gw': o.check_gw, 'captain': o.check_captain,   # decision check
        },
        'context': {
            'current_gw': r.current_gw,
            'horizon': list(r.horizon),
            'weights': [{'gw': g, 'weight': w} for g, w in r.weights.items()],
            'history_gws': list(r.history_gws),
            'history_rows': r.history_rows,
            'blank_double': [{'gw': g, 'blank': b, 'double': d} for g, b, d in r.gw_fixtures],
            'free_transfers': {'value': r.ft, 'source': r.ft_source, 'rebuilt': r.ft_rebuilt},
            'budget': {'total': r.budget, 'squad_value': r.squad_value, 'bank': r.bank, 'source': r.budget_source},
        },
        'solver': {
            'name': s.name, 'seconds': s.seconds, 'passes': s.passes, 'proven': s.proven,
            'gap': s.gap, 'objective': s.objective, 'bench_weights_change': s.beta_change,
        },
        'squad': [{
            'id': x.id, 'name': x.name, 'team': x.team, 'position': x.position, 'cost': x.cost,
            'xp_now': x.xp_now, 'papp_now': x.papp_now, 'xp_horizon': x.xp_horizon,
            'n_matches_now': x.n_matches_now, 'role': x.role,
            'bench_order': x.bench_slot if x.bench_slot >= 0 else None,
            'action': x.action, 'available': x.available, 'source': x.source,
        } for x in r.squad],
        'plan': [{
            'gw': pg.gw, 'chip': pg.chip, 'ft': pg.ft, 'ft_unlimited': pg.ft_unlimited, 'hits': pg.hits,
            'transfers': [{'out_id': t.out_id, 'out_name': t.out_name, 'in_id': t.in_id, 'in_name': t.in_name}
                          for t in pg.transfers],
            'captain': {'id': pg.captain_id, 'name': pg.captain_name},
            'vice': {'id': pg.vice_id, 'name': pg.vice_name} if pg.vice_id is not None else None,
            'squad': list(pg.squad), 'xi': list(pg.xi), 'bench': list(pg.bench),
        } for pg in r.plan],
        'sales_now': [{'id': i, 'name': n, 'price': p, 'available': a} for i, n, p, a in r.sales_now],
        'expected_points_now': {
            'xi': ep.xi, 'captain': ep.captain, 'vice': ep.vice, 'autosubs': ep.autosubs,
            'hits': ep.hits, 'total': ep.total, 'bench_weights': ep.bench_weights,
        },
        'chips': {
            'source': c.source,
            'sets': [{'chip': cp, 'start': st, 'stop': sp, 'state': state} for cp, st, sp, state in c.sets],
            'status': c.status,
            'exact': [{'chip': cp, 'window_end': end, 'delta': c.exact.get(cp), 'error_bound': c.error.get(cp)}
                      for cp, end in c.window.items()],
            'estimates': [{'chip': cp, 'gw': g, 'delta': d}
                          for cp, est in c.estimates.items() for g, d in sorted(est.items())],
            'suggestion': ({'chip': adv['pick'], 'delta': c.exact[adv['pick']]}
                           if adv.get('pick') is not None else None),
            'also_best_now': list(adv.get('also', [])),
            'holds': [{'chip': cp, 'now': now, 'best_gw': g, 'best': best, 'window_end': end}
                      for cp, now, g, best, end in adv.get('holds', [])],
            'incomplete': list(adv.get('incomplete', [])),   # C12
            'wc_note': ({'wc_delta': adv['wc_note'][0], 'suggested_delta': adv['wc_note'][1]}
                        if adv.get('wc_note') is not None else None),
            'not_computed': list(c.not_computed),
            'seconds': c.seconds,
        },
        'decision_check': _decision(r.decision),
        'explain': [{
            'id': e.id, 'name': e.name, 'team': e.team_name, 'pos_id': e.pos_id, 'price': e.price,
            'source': e.source, 'chance': e.chance, 'minutes_model': e.minutes_model,
            'per90': dict(e.per90), 'components': e.components, 'ep_next': e.ep_next,
            'xp_by_gw': [{'gw': g, 'xp': x} for g, x in e.xp_by_gw],
        } for e in r.explain],
        'explain_not_found': list(r.explain_not_found),
        'fallback_players': r.fallback_players,
        'predictions_log': r.log_written,
        'warnings': list(r.warnings),
        'timings': dict(r.timings),
        'api': dict(r.api_stats),
    })
    return doc


def _decision(d):
    """engine.DecisionSection -> dict; None without --check. Keys 'no_chip', 'best', 'decision' name
    Z0 (plan without the chip), Z+ (best version of the chip) and Z(D) (the user's decision)."""
    if d is None:
        return None
    x = d.result or {}                                    # {} when the check is invalid (nothing solved)
    def who(i):                                           # a player as {id, name}; None stays None
        return None if i is None else {'id': i, 'name': d.names.get(i)}

    def plan_gw(pg):                                      # the plan of the decision's GW
        if pg is None:
            return None
        return {'chip': pg['chip'], 'captain': who(pg['captain']), 'vice': who(pg['vice']), 'hits': pg['hits'],
                'transfers_in': [who(i) for i in pg['in']], 'transfers_out': [who(i) for i in pg['out']],
                'squad': list(pg['squad']), 'xi': list(pg['xi'])}

    keys = (('no_chip', 'z0'), ('best', 'z_best'), ('decision', 'z_decision'))
    t = x.get('timing')
    return {
        'status': d.status, 'chip': d.chip, 'gw': d.gw,
        'captain': {'id': d.captain_id, 'name': d.captain_name} if d.captain_id is not None else None,
        'violations': list(d.violations), 'notes': list(d.notes), 'verdict': list(d.verdict),
        'g': x.get('g'), 'g_low': x.get('g_low'), 'g_high': x.get('g_high'),
        'r': x.get('r'), 'r_low': x.get('r_low'), 'r_high': x.get('r_high'),
        'gain_action': x.get('gain_action'),
        'objective': {k: x.get(src) for k, src in keys},
        'gaps': {k: x.get('gaps', {}).get(src) for k, src in keys},
        'points_gw': {k: x.get('points_gw', {}).get(src) for k, src in keys},
        'best': plan_gw(x.get('plan_best')), 'decision': plan_gw(x.get('plan_decision')),
        'captain_bought': x.get('captain_bought'),
        'timing': None if t is None else {
            'window_end': t['window_end'], 'at_gw': t['at_gw'], 'best_gw': t['best_gw'], 'best': t['best'],
            'is_best': t['is_best'], 'estimates': [{'gw': g, 'delta': v} for g, v in sorted(t['estimates'].items())]},
        'not_computed': list(x.get('not_computed', [])),
        'seconds': x.get('seconds'),
    }


def evaluate_to_dict(r, warnings=()):
    """predictions.EvaluateResult -> dict."""
    doc = _header('evaluate', True)
    doc.update({
        'gw': r.gw, 'log_path': r.log_path,
        'rows': [{'label': x.label, 'n': x.n, 'mae': x.mae, 'mae_fpl': x.mae_fpl, 'bias': x.bias,
                  'bias_fpl': x.bias_fpl} for x in r.rows],
        'xi_captain': ({'predicted': r.xi_predicted, 'actual': r.xi_actual}
                       if r.xi_predicted is not None else None),
        'warnings': list(warnings),
    })
    return doc


def error_to_dict(exc, mode=None, warnings=()):
    """An exception -> {"ok": false, "error": {"exit_code", "type", "message"}}."""
    doc = _header(mode, False)
    doc['error'] = {'exit_code': getattr(exc, 'exit_code', EXIT_INTERNAL),
                    'type': getattr(exc, 'kind', 'internal'),
                    'message': str(exc) or type(exc).__name__}
    doc['warnings'] = list(warnings)
    return doc


def _clean(x):
    """Plain JSON types only: numpy scalars -> Python numbers, NaN/infinity -> null, tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, 'item') and not isinstance(x, (str, bytes)):   # numpy.int64, numpy.float64, numpy.bool_
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def dumps(doc):
    """UTF-8 text of the document (names such as Ødegaard are written as they are, not as \\u escapes)."""
    return json.dumps(_clean(doc), ensure_ascii=False, indent=2, allow_nan=False)
