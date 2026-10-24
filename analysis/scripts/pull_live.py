"""Download the live study data from the Heroku upload service into analysis/data/live/.

Usage (Python 3.10+, standard library only; from the repository root or from analysis/):
    python analysis/scripts/pull_live.py                    URL and token from %USERPROFILE%\\.nosetouch\\heroku.env
    python analysis/scripts/pull_live.py --url https://<app>.herokuapp.com --token <ADMIN_TOKEN>
    python analysis/scripts/pull_live.py --real-only        leave out simulator data
    python analysis/scripts/pull_live.py --participant P-XXXXXX --out analysis/data/live_P-XXXXXX

It downloads GET /api/export.zip (the backend folder layout <study>/<participant>/<session>/session.json
and trial_*.json, the same tree as server/receiver.py and the Apps Script backend) with the admin token
and unpacks it into --out (default analysis/data/live/, which git ignores). Each pull replaces the
folder completely, so a participant deleted on the server disappears locally too; the folder is marked
with a '.nosetouch-live' file and a folder without that mark is never replaced (unless --force).
Then, in Python: from motionlab.io import load_dataset; ds = load_dataset('data/live')

Credentials: --token, else NOSETOUCH_ADMIN_TOKEN, else ADMIN_TOKEN in heroku.env; the service: --url,
else NOSETOUCH_APP_URL, else APP_URL in heroku.env (lines KEY=VALUE). Exit code 0 on success, 1 when
the download or the archive failed, 2 on a usage error.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ANALYSIS_DIR / 'data' / 'live'
ENV_FILE = Path.home() / '.nosetouch' / 'heroku.env'
MARKER = '.nosetouch-live'
USER_AGENT = 'nosetouch-pull-live/1.0'
CHUNK = 1 << 20
# the only names a backend export may contain (anything else is refused: no path tricks, no surprises)
MEMBER_RE = re.compile(
    r'(?P<study>[A-Za-z0-9_-]{1,64})/(?P<participant>P-[A-HJ-NP-Z2-9]{6})/(?P<session>[A-Za-z0-9_-]{1,64})/'
    r'(?P<file>session\.json|trial_\d{3,}_a\d+_[A-Za-z0-9_-]{1,64}\.json)'
)
_KNOWN_SUFFIXES = ('/api/ingest', '/ingest', '/api/export.zip', '/api/export', '/admin', '/api/health')


class PullError(Exception):
    pass


# ───────────────────────────── configuration ─────────────────────────────


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """KEY=VALUE lines (# comments, optional 'export ', optional quotes). Missing file -> {}."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding='utf-8-sig')
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        if key.startswith('export '):
            key = key[len('export '):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        out[key] = value
    return out


def resolve(url: str | None, token: str | None, env_file: Path = ENV_FILE) -> tuple[str, str]:
    """(service base URL, admin token) from the arguments, the environment or the env file."""
    file_values = read_env_file(env_file)
    base = url or os.environ.get('NOSETOUCH_APP_URL') or file_values.get('APP_URL')
    tok = token or os.environ.get('NOSETOUCH_ADMIN_TOKEN') or file_values.get('ADMIN_TOKEN')
    if not base:
        raise PullError(f'no service URL: pass --url, set NOSETOUCH_APP_URL, or put APP_URL=... in {env_file}')
    if not tok:
        raise PullError(f'no admin token: pass --token, set NOSETOUCH_ADMIN_TOKEN, or put ADMIN_TOKEN=... in {env_file}')
    return base_url(base), tok


def base_url(url: str) -> str:
    """https://app.herokuapp.com/ (also from a pasted .../api/ingest or .../admin URL)."""
    u = url.strip().rstrip('/')
    for suffix in _KNOWN_SUFFIXES:
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            break
    parts = urllib.parse.urlsplit(u)
    local = parts.hostname in ('localhost', '127.0.0.1')
    if parts.scheme != 'https' and not (parts.scheme == 'http' and local):
        raise PullError(f'the service URL must start with https:// (got {url!r})')
    return u + '/'


def export_url(base: str, participant: str | None = None, since: str | None = None, real_only: bool = False) -> str:
    params: dict[str, str] = {}
    if participant:
        params['participant_id'] = participant
    if since:
        params['since'] = since
    if real_only:
        params['include_simulated'] = 'false'
    query = ('?' + urllib.parse.urlencode(params)) if params else ''
    return urllib.parse.urljoin(base, 'api/export.zip') + query


# ───────────────────────────── download ─────────────────────────────


def download(url: str, token: str, dest: Path, *, urlopen: Callable[..., Any] = urllib.request.urlopen,
             timeout: float = 300.0) -> int:
    """Streams the zip to `dest`; returns its size. Raises PullError with a readable reason."""
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}', 'User-Agent': USER_AGENT,
                                               'Accept': 'application/zip'})
    size = 0
    try:
        with urlopen(req, timeout=timeout) as resp, open(dest, 'wb') as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)
    except urllib.error.HTTPError as e:
        reason = {
            401: 'the service wants the admin token (check ADMIN_TOKEN)',
            403: 'the admin token was rejected (it must equal the app\'s ADMIN_TOKEN config var)',
            404: 'no export at this URL (is it the upload service\'s address?)',
            503: 'the service is up but its admin endpoints are off (ADMIN_TOKEN unset) or its database is down',
        }.get(e.code, 'unexpected answer')
        raise PullError(f'HTTP {e.code} from {redact(url)}: {reason}') from None
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise PullError(f'cannot download {redact(url)}: {getattr(e, "reason", e)}') from None
    return size


def redact(url: str) -> str:
    return re.sub(r'(token=)[^&]*', r'\1[redacted]', url)


# ───────────────────────────── unpacking ─────────────────────────────


def check_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """The archive's files, after checking that every name is a backend-layout path."""
    files = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        if not MEMBER_RE.fullmatch(info.filename):
            raise PullError(f'unexpected file in the export, nothing was unpacked: {info.filename!r}')
        files.append(info)
    return files


def unpack(zip_path: Path, out: Path, *, source: str, force: bool = False) -> dict[str, Any]:
    """Unpacks into a fresh folder next to `out`, then swaps it in for `out`. Returns counts."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise PullError('the download is not a zip archive') from None
    with zf:
        members = check_members(zf)
        if out.exists() and not (out / MARKER).exists() and any(out.iterdir()) and not force:
            raise PullError(f'{out} exists and was not created by pull_live.py: choose another --out or pass --force')
        out.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f'.{out.name}.new-', dir=out.parent))
        try:
            counts: dict[str, Any] = {'participants': {}, 'sessions': 0, 'trials': 0, 'bytes': 0}
            for info in members:
                target = stage.joinpath(*info.filename.split('/'))
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                m = MEMBER_RE.fullmatch(info.filename)
                assert m is not None
                per = counts['participants'].setdefault(m['participant'], {'sessions': 0, 'trials': 0})
                kind = 'sessions' if m['file'] == 'session.json' else 'trials'
                per[kind] += 1
                counts[kind] += 1
                counts['bytes'] += info.file_size
            now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')
            (stage / MARKER).write_text(f'source={source}\npulled_at={now}\n', encoding='utf-8')
            _swap(stage, out)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return counts


def _swap(stage: Path, out: Path) -> None:
    old = None
    if out.exists():
        old = out.parent / f'.{out.name}.old-{os.getpid()}-{int(time.time())}'
        os.replace(out, old)
    try:
        os.replace(stage, out)
    except OSError:
        if old is not None:
            os.replace(old, out)  # put the previous pull back
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


# ───────────────────────────── command line ─────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='pull_live.py', description='Download the live NoseTouch data (backend folder layout) '
                                                                 'into analysis/data/live/.')
    p.add_argument('--url', help='the service, e.g. https://<app>.herokuapp.com (default: NOSETOUCH_APP_URL or APP_URL in heroku.env)')
    p.add_argument('--token', help='the admin token (default: NOSETOUCH_ADMIN_TOKEN or ADMIN_TOKEN in heroku.env)')
    p.add_argument('--env-file', type=Path, default=ENV_FILE, help=f'KEY=VALUE file (default {ENV_FILE})')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT, help=f'target folder, replaced on every pull (default {DEFAULT_OUT})')
    p.add_argument('--participant', help='only this participant code (P-XXXXXX)')
    p.add_argument('--since', help='only records received since this ISO time (plus their sessions)')
    p.add_argument('--real-only', action='store_true', help='leave out simulator data')
    p.add_argument('--keep-zip', type=Path, help='also keep the downloaded zip at this path')
    p.add_argument('--force', action='store_true', help='replace --out even if pull_live.py did not create it')
    p.add_argument('--timeout', type=float, default=300.0, help='seconds (default 300)')
    return p


def main(argv: list[str] | None = None, urlopen: Callable[..., Any] = urllib.request.urlopen) -> int:
    args = build_parser().parse_args(argv)
    try:
        base, token = resolve(args.url, args.token, args.env_file)
    except PullError as e:
        print(f'pull_live: {e}', file=sys.stderr)
        return 2
    url = export_url(base, args.participant, args.since, args.real_only)
    out = args.out.resolve()
    print(f'downloading {url}')
    tmp_dir = Path(tempfile.mkdtemp(prefix='nosetouch-pull-'))
    try:
        zip_path = tmp_dir / 'export.zip'
        size = download(url, token, zip_path, urlopen=urlopen, timeout=args.timeout)
        if args.keep_zip:
            args.keep_zip.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(zip_path, args.keep_zip)
        counts = unpack(zip_path, out, source=url, force=args.force)
    except PullError as e:
        print(f'pull_live: {e}', file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    people = counts['participants']
    print(f'{len(people)} participants, {counts["sessions"]} sessions, {counts["trials"]} trials '
          f'({counts["bytes"] / 1e6:.1f} MB unpacked, {size / 1e6:.1f} MB downloaded) -> {out}')
    for pid in sorted(people):
        print(f'  {pid}: {people[pid]["sessions"]} sessions, {people[pid]["trials"]} trials')
    print("load it with:  from motionlab.io import load_dataset; ds = load_dataset(r'%s')" % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
