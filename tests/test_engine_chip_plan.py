"""C13 (chat 16): the plan with the suggested chip. On the synthetic season (GW8 next) the rule suggests no chip,
so the suggestion is set by the test (engine.advise wrapped); the engine then solves the plan with that chip,
which must be the plan of a run with --chip X (same solve, same options)."""
import pytest

from fplopt import engine
from fplopt.engine import Options, optimize
from fpl_fake import FakeApi, build_manager
from conftest import FakeClient

BASE = dict(team_id=1, lookahead=2, write_log=False, time_limit=600, transfer_gain=False, formations=False)


def run(season, **kw):
    return optimize(FakeClient(FakeApi(season, build_manager(season))), Options(**dict(BASE, **kw)))


def suggest(monkeypatch, chip):
    """The chip analysis runs as usual; only its suggestion is replaced by 'chip'."""
    real = engine.advise
    monkeypatch.setattr(engine, 'advise', lambda rep, gw: dict(real(rep, gw), pick=chip))


@pytest.mark.parametrize("chip", ['BB', 'TC'])
def test_chip_plan_is_the_plan_of_a_run_with_the_chip(season, monkeypatch, chip):
    suggest(monkeypatch, chip)
    r = run(season)
    cp = r.chip_plan
    assert cp is not None and (cp.chip, cp.status) == (chip, 'computed')
    w = run(season, chip=chip, chip_report=False)               # what --chip X would show
    assert w.chip_plan is None                                  # no suggestion without the chip analysis
    assert cp.plan == w.plan and cp.squad == w.squad and cp.sales_now == w.sales_now
    assert cp.expected_now == w.expected_now
    assert cp.solver.objective == pytest.approx(w.solver.objective, abs=1e-9)
    assert cp.plan[0].chip == chip and r.plan[0].chip is None   # the main plan is unchanged


def test_no_suggestion_no_chip_plan(season):
    r = run(season)
    assert r.chips.advice['pick'] is None and r.chip_plan is None


def test_chip_plan_switched_off(season, monkeypatch):
    suggest(monkeypatch, 'BB')
    assert run(season, chip_plan=False).chip_plan is None


def test_chip_plan_not_computed_without_time(season, monkeypatch):
    """C12: with no time left the plan is not solved and nothing of it is shown."""
    suggest(monkeypatch, 'BB')
    monkeypatch.setattr(engine, 'MIN_PASS_SECONDS', 10 ** 9)
    cp = run(season).chip_plan
    assert (cp.chip, cp.status) == ('BB', 'not_computed')
    assert cp.plan == [] and cp.expected_now is None and cp.solver is None


def test_chip_plan_in_details_and_json(season, monkeypatch):
    import io
    import json
    from fplopt import jsonout, report
    suggest(monkeypatch, 'BB')
    r = run(season)
    out = io.StringIO()
    report.print_optimize(r, out)
    text = out.getvalue()
    assert ">> Suggestion: play BB in GW8" in text and "Its plan follows." in text
    assert " PLAN WITH BB | GW8 (the same plan as --chip BB)" in text
    doc = json.loads(jsonout.dumps(jsonout.optimize_to_dict(r)))
    cp = doc['chip_plan']
    assert (cp['chip'], cp['status']) == ('BB', 'computed')
    assert cp['plan'][0]['chip'] == 'BB' and len(cp['squad']) == 15
    assert cp['expected_points_now']['total'] == pytest.approx(r.chip_plan.expected_now.total)
    assert json.loads(jsonout.dumps(jsonout.optimize_to_dict(run(season, chip_plan=False))))['chip_plan'] is None


# --------------------------------------------------------------------------- summary (C13)
def test_summary_shows_the_chip_plan(season, monkeypatch):
    from fplopt.summary import build_summary
    suggest(monkeypatch, 'TC')
    r = run(season)
    s = build_summary(r)
    text = "\n".join(s['text'])
    assert "Suggested: Triple Captain this gameweek. Its plan is below." in text
    assert "run again with --chip TC" not in text
    i = s['text'].index("== WITH TRIPLE CAPTAIN ==")
    block = s['text'][i:s['text'].index("== EXPECTED POINTS ==")]
    cp, main = r.chip_plan, r.plan[0]
    same_moves = [(m.out_id, m.in_id) for m in cp.plan[0].transfers] == [(m.out_id, m.in_id) for m in main.transfers]
    assert any(l.startswith("Transfers:") and "Same as above" in l for l in block) == same_moves
    assert any(l.startswith("Captain:") and "as Triple Captain" in l for l in block)
    assert any(l.startswith("Expected:") and "without the chip" in l for l in block)
    d = s['chip_plan']
    assert (d['chip'], d['status'], d['same_transfers']) == ('TC', 'computed', same_moves)
    assert d['expected_points'] == pytest.approx(cp.expected_now.total)
    assert d['expected_points_without'] == pytest.approx(r.expected_now.total)
    assert d['captain']['id'] == cp.plan[0].captain_id
    assert d['transfers']['vs_no_transfer'] is None


def test_summary_chip_plan_not_computed(season, monkeypatch):
    from fplopt.summary import build_summary
    suggest(monkeypatch, 'BB')
    monkeypatch.setattr(engine, 'MIN_PASS_SECONDS', 10 ** 9)
    s = build_summary(run(season))
    i = s['text'].index("== WITH BENCH BOOST ==")
    assert s['text'][i + 1].startswith("Not computed in the time limit: run again with --chip BB")
    assert s['chip_plan'] == {'chip': 'BB', 'status': 'not_computed'}


def test_summary_without_suggestion_is_unchanged(season):
    from fplopt.summary import build_summary
    s = build_summary(run(season))
    assert s['chip_plan'] is None and not any(l.startswith("== WITH ") for l in s['text'])
