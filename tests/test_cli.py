"""Command line: argument checks, exit codes, --json and the output encoding."""
import json
import os
import subprocess
import sys

import pytest

from fplopt import cli
from fplopt.errors import ApiError, NotFoundError
from fpl_fake import FakeApi, Season, build_manager
from fake_server import FakeServer
from conftest import FakeClient


def run_cli(argv, capsys):
    status = cli.main(argv)
    out, err = capsys.readouterr()
    return status, out, err


@pytest.mark.parametrize("argv, message", [
    ([], "Optimization mode requires --team ID"),
    (['--team', '1', '--ft', '7'], "--ft must be between 1 and 5."),
    (['--team', '1', '--max-hits', '-1'], "--max-hits must be 0 or greater."),
    (['--team', '1', '--lookahead', '0'], "--lookahead must be 1 or greater."),
    (['--team', '1', '--min-chance', '150'], "--min-chance must be between 0 and 100."),
    (['--team', '1', '--time-limit', '0'], "--time-limit must be 1 second or more."),
    (['--team', '1', '--override-budget', '-5'], "--override-budget must be a positive amount"),
    (['--team', '1', '--chip', 'AUTO'], "invalid choice: 'AUTO'"),
    (['--team', 'abc'], "invalid int value: 'abc'"),
    (['--team', '1', '--check-gw', '9'], "--check-gw and --captain are used only with --check."),
    (['--team', '1', '--captain', '5'], "--check-gw and --captain are used only with --check."),
    (['--team', '1', '--check', 'TC', '--captain', '5', '--check-gw', '40'], "--check-gw must be between 1 and 38."),
    (['--team', '1', '--check', 'WC'], "invalid choice: 'WC'"),
    (['--mode', 'evaluate'], "Evaluation mode requires target Gameweek"),
    (['--mode', 'evaluate', '--gw', '39'], "--gw must be between 1 and 38."),
])
def test_invalid_arguments(argv, message, capsys):
    status, out, err = run_cli(argv, capsys)
    assert status == 2
    assert out == ""
    assert err.startswith("[ERROR] ") and message in err
    status, out, _ = run_cli(argv + ['--json'], capsys)       # the same error as one JSON document
    doc = json.loads(out)
    assert status == 2
    assert (doc['ok'], doc['error']['type'], doc['error']['exit_code']) == (False, 'input', 2)
    assert message in doc['error']['message']


@pytest.fixture
def cli_client(monkeypatch, season):
    """The command line talks to the synthetic season instead of the FPL API."""
    def use(client_class=FakeClient):
        monkeypatch.setattr(cli, 'FplClient',
                            lambda base_url, **kw: client_class(FakeApi(season, build_manager(season))))
    return use


def test_optimize_json(cli_client, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                 # predictions_log.csv
    cli_client()
    status, out, err = run_cli(['--team', '1', '--lookahead', '1', '--no-chip-report', '--json'], capsys)
    assert status == 0
    doc = json.loads(out)                       # stdout holds exactly one JSON document
    assert (doc['schema_version'], doc['ok'], doc['mode']) == ('1.0', True, 'optimize')
    assert doc['context']['current_gw'] == 8 and doc['context']['horizon'] == [8]
    assert doc['context']['free_transfers'] == {'value': 3, 'source': 'history', 'rebuilt': 3}
    assert len(doc['squad']) == 15 and len(doc['plan']) == 1
    roles = [p['role'] for p in doc['squad']]
    assert roles.count('Captain') == 1 and roles.count('Vice') == 1 and roles.count('Bench') == 4
    assert [p['bench_order'] for p in doc['squad'] if p['role'] == 'Bench'] == [0, 1, 2, 3]
    assert doc['chips']['status'] == 'skipped_flag'
    assert doc['decision_check'] is None                        # no --check
    assert (doc['input']['check'], doc['input']['check_gw'], doc['input']['captain']) == (None, None, None)
    assert doc['warnings'] == []
    assert doc['predictions_log'] == 'predictions_log.csv' and (tmp_path / 'predictions_log.csv').is_file()
    assert "[*] Solved with" in err             # progress stays on stderr


def test_no_log(cli_client, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cli_client()
    status, out, err = run_cli(['--team', '1', '--lookahead', '1', '--no-chip-report', '--no-log', '--json'], capsys)
    assert status == 0
    assert json.loads(out)['predictions_log'] is None
    assert not (tmp_path / 'predictions_log.csv').exists()
    assert "[*] Predictions not written to predictions_log.csv (--no-log)." in err


def test_optimize_text(cli_client, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cli_client()
    status, out, err = run_cli(['--team', '1', '--lookahead', '1', '--no-chip-report'], capsys)
    assert status == 0
    assert " SQUAD PLAN | GW8 (horizon: GW8-GW8)" in out and "   Skipped (--no-chip-report)." in out
    assert not any(ln.startswith('[*]') for ln in out.splitlines())
    assert err.splitlines()[0] == "[*] MILP Optimizer | Team: 1 | Lookahead: 1 GWs"


CHECK = ['--team', '1', '--lookahead', '1', '--no-chip-report', '--no-log']


def test_check_tc_json(cli_client, capsys, season):
    """--check TC on an owned forward: one 'decision_check' object with every field of the schema."""
    cli_client()
    cap = build_manager(season)['truth'][0][14]
    status, out, err = run_cli(CHECK + ['--check', 'TC', '--captain', str(cap), '--json'], capsys)
    assert status == 0
    doc = json.loads(out)
    d = doc['decision_check']
    assert (doc['input']['check'], doc['input']['captain']) == ('TC', cap)
    assert (d['status'], d['chip'], d['gw'], d['captain']['id']) == ('computed', 'TC', 8, cap)
    assert d['g'] + d['r'] == pytest.approx(d['gain_action'], abs=1e-9)
    assert set(d['objective']) == set(d['gaps']) == set(d['points_gw']) == {'no_chip', 'best', 'decision'}
    assert d['decision']['captain'] == d['captain'] and d['decision']['chip'] == 'TC'
    assert [e['gw'] for e in d['timing']['estimates']] == list(range(8, 20))
    assert isinstance(d['timing']['is_best'], bool) and len(d['verdict']) >= 1
    assert d['not_computed'] == [] and d['violations'] == []
    assert "[*] Decision check: TC GW8:" in err


def test_check_bb_text(cli_client, capsys):
    cli_client()
    status, out, _ = run_cli(CHECK + ['--check', 'BB'], capsys)
    assert status == 0
    assert " DECISION CHECK | BB in GW8" in out
    assert "R = best BB - your decision:      0 (Bench Boost leaves nothing to choose)" in out
    assert ">> Verdict: " in out
    assert out.index(" DECISION CHECK") < out.index(" CHIP SUGGESTIONS")   # after the plan, before the chips


def test_check_invalid_text_and_json(monkeypatch, season, capsys):
    """TC already played: exit 0 (the plan is still given), the reason in the text and in the JSON."""
    manager = build_manager(season)
    manager['chips'].append({'name': '3xc', 'event': 3})
    monkeypatch.setattr(cli, 'FplClient', lambda base_url, **kw: FakeClient(FakeApi(season, manager)))
    status, out, _ = run_cli(CHECK + ['--check', 'TC', '--captain', '1'], capsys)
    assert status == 0
    assert "   Invalid: TC cannot be played in GW8: GW1-19 played GW3, GW20-38 available." in out
    assert ">> Verdict: this decision is not allowed by the rules of the game." in out
    status, out, _ = run_cli(CHECK + ['--check', 'TC', '--captain', '1', '--json'], capsys)
    d = json.loads(out)['decision_check']
    assert (d['status'], d['violations'][0]['code']) == ('invalid', 'chip_unavailable')
    assert d['g'] is None and d['timing'] is None                 # nothing solved


class NotFoundClient(FakeClient):
    def get(self, path):
        if path.startswith('/entry/'):
            raise NotFoundError(f"404 Not Found: Verify Team ID or Target Gameweek.\nEndpoint: {path}")
        return super().get(path)


class OfflineClient(FakeClient):
    def get(self, path):
        raise ApiError("Network/FPL API Handshake Failure: connection refused")


@pytest.mark.parametrize("client_class, status, kind", [(NotFoundClient, 4, 'not_found'), (OfflineClient, 3, 'network')])
def test_api_errors(cli_client, capsys, client_class, status, kind):
    cli_client(client_class)
    code, out, err = run_cli(['--team', '1', '--json'], capsys)
    doc = json.loads(out)
    assert code == status
    assert (doc['ok'], doc['error']['type'], doc['error']['exit_code']) == (False, kind, status)
    assert "[ERROR]" in err


def test_ft_mismatch_needs_the_user(monkeypatch, season, capsys):
    manager = build_manager(season)
    manager['current'][2]['event_transfers'] = 3      # GW3: 3 transfers with 2 FT, but cost 0 in the API
    monkeypatch.setattr(cli, 'FplClient', lambda base_url, **kw: FakeClient(FakeApi(season, manager)))
    code, out, err = run_cli(['--team', '1', '--json'], capsys)
    doc = json.loads(out)
    assert code == 6 and doc['error']['type'] == 'user_action'
    assert "Pass the number shown in the app with --ft." in doc['error']['message']
    assert doc['warnings'][0]['code'] == 'ft_rebuild'


def test_subprocess_utf8_output(tmp_path):
    """End to end in a separate process, through real HTTP (local fake server), with a non-UTF-8
    locale and PYTHONIOENCODING=cp1252, as on Windows when a frontend reads the output through a
    pipe: the output must still be valid UTF-8 and keep the names as they are."""
    season = Season()
    names = {39: "Ødegaard", 57: "Horníček"}            # two players of the test manager's squad
    for e in season.elements:
        if e['id'] in names:
            e['web_name'] = names[e['id']]
    env = dict(os.environ, PYTHONIOENCODING='cp1252', LC_ALL='C', LANG='C')
    with FakeServer(FakeApi(season, build_manager(season))) as url:
        env[cli.API_URL_ENV] = url
        for extra in (['--json'], []):
            proc = subprocess.run([sys.executable, '-m', 'fplopt', '--team', '1', '--lookahead', '1',
                                   '--no-chip-report', *extra], cwd=tmp_path, env=env, capture_output=True,
                                  timeout=300)
            assert proc.returncode == 0, proc.stderr.decode('utf-8', 'replace')
            out = proc.stdout.decode('utf-8')                # raises if the bytes are not UTF-8
            for name in names.values():
                assert name in out
            if extra:
                json.loads(out)
            else:
                assert "£" in out
