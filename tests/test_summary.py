"""Plain-English summary (summary.py, chat 15c): every fact equals its source in the result (S1),
the 'pays for' rule by hand (S2), numbers in the text equal the rounded fields (S3), no technical
words in the text (S4), the deadline in local time (S9). One engine run on the synthetic season
(GW8 next, lookahead 2, no chip analysis); the other cases change copies of that result."""
import json
import locale
import re
from dataclasses import replace
from zoneinfo import ZoneInfo

import pytest

from fplopt.engine import ChipSection, Options, optimize
from fplopt.rules import MAX_FT
from fplopt.summary import (RANGE_HIGH, RANGE_LOW, build_summary, format_deadline, half_up, pays_for,
                            signed)
from fpl_fake import FakeApi, build_manager
from conftest import FakeClient

FORBIDDEN = ['xP', 'P(app)', 'gap', 'objective', 'Delta', 'horizon', 'weighted', '[assumption]']


@pytest.fixture(scope="module")
def plain(season):
    return optimize(FakeClient(FakeApi(season, build_manager(season))),
                    Options(team_id=1, lookahead=2, chip_report=False, write_log=False))


def text(s):
    return "\n".join(s['text'])


def flat(s):
    """The text with every run of spaces and line breaks made one space (alignment and wrapping)."""
    return " ".join(text(s).split())


def with_chips(r, **kw):
    """The result with a hand-made chip section (status 'computed' unless given)."""
    return replace(r, chips=ChipSection(**dict(dict(source='API', sets=[], status='computed'), **kw)))


def advice(pick=None, also=(), holds=(), incomplete=()):
    return {'now_ok': [pick] if pick else [], 'pick': pick, 'also': list(also), 'holds': list(holds),
            'wc_note': None, 'incomplete': list(incomplete)}


# --------------------------------------------------------------------------- S2: pays for, by hand
@pytest.mark.parametrize("bank, deltas, expected", [
    (30, [-40, 14], [False, True]),          # K9 (chat 15): Cherki -> Fernandes needs Araujo -> Thomas
    (30, [-19, -2], [False, False]),         # GW9 plan of 2026-10-26: both purchases fit the bank
    (0, [-10, 10], [False, True]),           # exactly at the limit: 0 <= 0 counts as fitting
    (0, [-30, 20, 20], [False, True, True]),  # two sales, each needed
    (29, [0, 13, 4, -20, -16], [False, True, False, False, False]),   # synthetic season: 10 left
])
def test_pays_for_by_hand(bank, deltas, expected):
    assert pays_for(bank, deltas) == expected


# --------------------------------------------------------------------------- rounding
def test_rounding():
    assert half_up(62.5) == 63 and half_up(62.49) == 62 and half_up(64.33) == 64   # round(62.5) would be 62
    assert signed(0.04) == "no difference" and signed(-0.04) == "no difference"
    assert signed(1.76) == "+1.8" and signed(-4.04) == "-4.0"


# --------------------------------------------------------------------------- S1: fields = sources
def test_fields_equal_their_sources(plain):
    s = build_summary(plain)
    pg = plain.plan[0]
    assert s['gw'] == plain.current_gw
    assert s['captain']['name'] == pg.captain_name and s['vice']['name'] == pg.vice_name
    names = {r.id: r.name for r in plain.squad}
    assert s['lineup']['bench'] == [names[i] for i in pg.bench]            # bench order of the plan
    xi = [names[i] for i in pg.xi]
    assert sorted(sum((s['lineup'][p] for p in ('GK', 'DEF', 'MID', 'FWD')), [])) == sorted(xi)
    t = s['transfers']
    assert [(m['out']['id'], m['in']['id']) for m in t['moves']] == [(x.out_id, x.in_id) for x in pg.transfers]
    assert (t['free_transfers'], t['hits'], t['hit_points']) == (pg.ft, pg.hits, 4 * pg.hits)
    assert t['bank_before'] == pytest.approx(plain.budget - plain.squad_value, abs=1e-9)
    assert t['bank_after'] == pytest.approx(t['bank_before'] + sum(m['money'] for m in t['moves']), abs=1e-9)
    assert t['bank_after'] >= 0                                            # the plan fits the budget
    assert s['expected_points'] == pytest.approx(plain.expected_now.total)
    assert s['expected_range'] == pytest.approx([max(0.0, s['expected_points'] + RANGE_LOW),
                                                 s['expected_points'] + RANGE_HIGH])
    json.dumps(s)                                                          # plain types only


# --------------------------------------------------------------------------- S3: text = fields
def test_text_numbers_equal_fields(plain):
    s = build_summary(plain)
    x, f = text(s), flat(s)
    lo, hi = s['expected_range']
    assert f"About {half_up(s['expected_points'])} this gameweek" in x
    assert f"In 8 of 10 gameweeks: between {half_up(lo)} and {half_up(hi)}" in f
    assert f"C: {s['captain']['name']} (about {half_up(s['captain']['expected_points'])} expected points" in f
    assert f"VC: {s['vice']['name']}" in f
    t = s['transfers']
    if t['moves']:
        assert f"Bank: £{t['bank_before']:.1f}m now -> £{t['bank_after']:.1f}m after" in f
        for m in t['moves']:
            assert re.search(rf"^  {m['out']['name']} +-> {m['in']['name']}\b", x, re.M)
    assert x.endswith("Full numbers: run again with --details")


# --------------------------------------------------------------------------- S4: plain words
def variants(r):
    """The result and copies of it that reach every branch of the text."""
    pg = r.plan[0]
    yield r
    yield replace(r, plan=[replace(pg, transfers=[], hits=0)] + r.plan[1:])          # no transfer
    yield with_chips(r, window={'BB': 19, 'TC': 19, 'WC': 19}, exact={'FH': 1.76, 'BB': 9.1, 'TC': 10.8, 'WC': 3.0},
                     advice=advice('FH', also=['BB'], holds=[('TC', 10.8, 16, 11.0, 19)], incomplete=['BB']))
    yield replace(with_chips(r, status='skipped_chip'), options=replace(r.options, chip='TC'))
    yield with_chips(r, status='no_chip_now')
    yield replace(r, solver=replace(r.solver, proven=False),
                  warnings=[{'code': 'previous_gw_not_final', 'message': 'x'}])


def test_no_technical_words(plain):
    for r in variants(plain):
        x = text(build_summary(r))
        for w in FORBIDDEN:
            assert w not in x, (w, x)
        assert all(len(line) <= 80 for line in x.splitlines())


# --------------------------------------------------------------------------- branches
def test_no_transfer_saves_it(plain):
    pg = plain.plan[0]
    r = replace(plain, plan=[replace(pg, transfers=[], hits=0, ft=2)] + plain.plan[1:])
    assert "No transfer this gameweek: you will have 3 next gameweek." in flat(build_summary(r))
    r = replace(plain, plan=[replace(pg, transfers=[], hits=0, ft=MAX_FT)] + plain.plan[1:])
    assert f"{MAX_FT} is the most you can keep, so one is lost." in flat(build_summary(r))


def test_chip_texts(plain):
    r = with_chips(plain, window={'FH': 19, 'TC': 19}, exact={'FH': 1.76, 'TC': 10.82},
                   advice=advice('FH', holds=[('TC', 10.82, 16, 11.0, 19)]))
    s = build_summary(r)
    x = flat(s)
    assert "Suggested: Free Hit this gameweek (+1.8 points)." in x
    assert "does not use it: run again with --chip FH to see it." in x     # the plan shown is without it
    assert "Triple Captain: keep it (now +10.8, best GW16: +11.0)" in x
    assert "within the model's error" in x
    # one value column per section: the values start at the same position after labels of any length
    lines = [ln for ln in s['text'] if ln.startswith(('Suggested:', 'Triple Captain:'))]
    starts = {len(ln) - len(ln.split(':', 1)[1].lstrip()) for ln in lines}
    assert len(lines) == 2 and starts == {len("Triple Captain: ")}
    assert s['chip']['play'] == {'chip': 'FH', 'gain': 1.76}
    x = text(build_summary(replace(with_chips(plain, status='skipped_chip'), options=replace(plain.options, chip='TC'))))
    assert "the plan above uses it" in x
    r = with_chips(plain, window={'TC': 19}, advice=advice(incomplete=['TC']))
    s = build_summary(r)
    assert "Triple Captain: no advice" in text(s)
    assert 'chips_not_computed' in [a['code'] for a in s['alerts']]


def test_alerts(plain):
    assert build_summary(plain)['alerts'] == []
    r = replace(plain, solver=replace(plain.solver, proven=False),
                warnings=[{'code': 'previous_gw_not_final', 'message': 'x'}, {'code': 'api_retry', 'message': 'y'}])
    codes = [a['code'] for a in build_summary(r)['alerts']]
    assert codes == ['previous_gw_not_final', 'time_limit']               # api_retry stays in --details
    bench = next(x for x in plain.squad if x.role == 'Bench')
    squad = [replace(x, available=False, action='KEEP') if x is bench else x for x in plain.squad]
    s = build_summary(replace(plain, squad=squad))
    assert {'code': 'player_unavailable', 'message': f"{bench.name} is injured or doubtful and is on your bench."} \
        in s['alerts']
    assert f"{bench.name} (injured or doubtful" in text(s)


# --------------------------------------------------------------------------- S9: deadline
@pytest.mark.parametrize("utc, local", [
    ("2026-10-31T11:00:00Z", "Sat 31 Oct, 13:00"),     # Athens UTC+2 (winter time from 25 Oct)
    ("2026-10-24T10:00:00Z", "Sat 24 Oct, 13:00"),     # Athens UTC+3 (summer time)
])
def test_deadline_local_time(utc, local):
    assert format_deadline(utc, ZoneInfo("Europe/Athens")) == local


def test_deadline_ignores_the_locale():
    """English day and month names whatever the system language (strftime('%a') would follow it)."""
    old = locale.setlocale(locale.LC_TIME)
    try:
        for name in ('el_GR.UTF-8', 'el_GR.utf8', 'Greek_Greece.1253'):
            try:
                locale.setlocale(locale.LC_TIME, name)
                break
            except locale.Error:
                continue
        assert format_deadline("2026-10-31T11:00:00Z", ZoneInfo("Europe/Athens")) == "Sat 31 Oct, 13:00"
    finally:
        locale.setlocale(locale.LC_TIME, old)


def test_header_with_deadline(plain):
    r = replace(plain)
    r.deadline = "2026-10-31T11:00:00Z"        # the engine sets it from step 2 of 15c
    assert build_summary(r, tz=ZoneInfo("Europe/Athens"))['text'][:2] == \
        ["FPL plan for GW8", "Deadline: Sat 31 Oct, 13:00 (your time)"]
