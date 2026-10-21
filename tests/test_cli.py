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
