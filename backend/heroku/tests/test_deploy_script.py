"""deploy.sh refuses to push to an app without DATABASE_URL (the service would answer every upload 503).

Nothing here can reach Heroku: `heroku` is a stand-in script first on PATH, HEROKU_API_KEY is a dummy,
git pushes are rewritten to a local bare repository, and git may use no transport but local files.
Needs bash (Git Bash on Windows); skipped without it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / 'deploy.sh'
APP = 'nosetouch-deploy-test'
FAKE_HEROKU = """#!/usr/bin/env bash
echo "heroku $*" >> "$FAKE_HEROKU_CALLS"
case "$1 $2" in
  "config:get DATABASE_URL")
    [[ "${FAKE_APP_EXISTS:-1}" == 1 ]] || { echo " !    Couldn't find that app." >&2; exit 1; }
    printf '%s\\n' "${FAKE_DATABASE_URL:-}" ;;
  "config:get ALLOW_EPHEMERAL_DB") printf '%s\\n' "${FAKE_ALLOW_EPHEMERAL_DB:-}" ;;
  "auth:token "*) echo "fake-token" ;;
  *) echo "unexpected: heroku $*" >&2; exit 3 ;;
esac
"""


def _bash() -> str | None:
    candidates: list[Path] = []
    if os.name == 'nt':  # Git Bash, not the WSL launcher in System32 / WindowsApps
        for base in {os.environ.get('ProgramFiles', r'C:\Program Files'), os.environ.get('ProgramW6432', '')}:
            if base:
                candidates.append(Path(base) / 'Git' / 'bin' / 'bash.exe')
    found = shutil.which('bash')
    if found:
        candidates.append(Path(found))
    for c in candidates:
        low = str(c).lower()
        if c.is_file() and 'system32' not in low and 'windowsapps' not in low:
            return str(c)
    return None


BASH = _bash()
pytestmark = pytest.mark.skipif(BASH is None or shutil.which('git') is None, reason='needs bash and git')


def _deploy(tmp_path: Path, *args: str, **fake: str) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
    """Runs deploy.sh APP *args; returns (result, the bare 'Heroku' repository, the heroku calls)."""
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir(exist_ok=True)
    (fake_bin / 'heroku').write_text(FAKE_HEROKU, encoding='utf-8', newline='\n')
    (fake_bin / 'heroku').chmod(0o755)
    remotes = tmp_path / 'remotes'
    bare = remotes / f'{APP}.git'
    if not bare.exists():
        subprocess.run(['git', 'init', '-q', '--bare', str(bare)], check=True)
    calls = tmp_path / 'heroku-calls.txt'
    calls.write_text('', encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GIT_', 'HEROKU_'))}
    env.update({
        'PATH': f'{fake_bin}{os.pathsep}{os.environ["PATH"]}',
        'HEROKU_API_KEY': 'dummy-key-for-tests',
        'TMPDIR': tmp_path.as_posix(),
        'FAKE_HEROKU_CALLS': calls.as_posix(),
        'GIT_ALLOW_PROTOCOL': 'file',  # no https: a push can only go to the local bare repository
        'GIT_CONFIG_COUNT': '1',
        'GIT_CONFIG_KEY_0': f'url.{remotes.as_posix()}/.insteadOf',
        'GIT_CONFIG_VALUE_0': 'https://git.heroku.com/',
        'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 'test@example.org',
        'GIT_COMMITTER_NAME': 'test', 'GIT_COMMITTER_EMAIL': 'test@example.org',
        **{f'FAKE_{k.upper()}': v for k, v in fake.items()},
    })
    result = subprocess.run([BASH, SCRIPT.as_posix(), APP, *args], env=env, capture_output=True, text=True,
                            timeout=180)
    return result, bare, calls.read_text(encoding='utf-8').splitlines()


def _pushed(bare: Path) -> bool:
    refs = subprocess.run(['git', '--git-dir', str(bare), 'for-each-ref'], capture_output=True, text=True, check=True)
    return 'refs/heads/main' in refs.stdout


def test_an_app_without_database_url_is_not_deployed(tmp_path):
    result, bare, calls = _deploy(tmp_path, database_url='')
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"'{APP}' has no DATABASE_URL" in result.stderr and 'heroku-postgresql' in result.stderr
    assert calls[0] == f'heroku config:get DATABASE_URL -a {APP}'  # the stand-in answered, not the real CLI
    assert not _pushed(bare)


def test_an_app_with_its_database_is_deployed(tmp_path):
    result, bare, calls = _deploy(tmp_path, database_url='postgres://u:s3cret-pw@db.example.org:5432/d1')
    assert result.returncode == 0, result.stdout + result.stderr
    assert _pushed(bare)
    assert 's3cret-pw' not in result.stdout + result.stderr  # the URL holds the database password
    assert calls[0] == f'heroku config:get DATABASE_URL -a {APP}'


def test_allow_ephemeral_db_deploys_without_a_database(tmp_path):
    result, bare, _ = _deploy(tmp_path, database_url='', allow_ephemeral_db='1')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ALLOW_EPHEMERAL_DB' in result.stderr and _pushed(bare)


def test_an_unknown_app_is_not_deployed(tmp_path):
    result, bare, _ = _deploy(tmp_path, app_exists='0')
    assert result.returncode == 1 and 'cannot read the config' in result.stderr
    assert not _pushed(bare)


def test_a_dry_run_checks_nothing_and_pushes_nothing(tmp_path):
    result, bare, calls = _deploy(tmp_path, '--dry-run', database_url='')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--dry-run: nothing pushed.' in result.stdout and 'Procfile' in result.stdout
    assert calls == [] and not _pushed(bare)
