"""Admin authentication and the /admin page (one self-contained HTML page, no external assets).

Authentication: ADMIN_TOKEN (unset = admin disabled, every admin route answers 503).
  - API clients send 'Authorization: Bearer <token>' (compared in constant time).
  - Browsers sign in at /admin/login (a form that POSTs the token, or once /admin/login?token=...),
    which sets an HttpOnly, Secure, SameSite=Strict cookie holding an expiry and an HMAC of it keyed
    by the token: the token itself never sits in the cookie jar or in later URLs, and changing
    ADMIN_TOKEN signs every browser out.
Every string that comes from an upload is HTML-escaped before it reaches the page.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import html
import re
import time
from typing import Any
from urllib.parse import urlencode

from . import SERVER_VERSION, SERVICE_NAME
from .settings import Settings
from .validation import iso

COOKIE_NAME = 'nosetouch_admin'
REFRESH_SECONDS = 15
# The cookie's expiry: ASCII digits only. str.isdigit() also accepts characters such as '²' that int()
# refuses, and int() refuses strings of more than 4300 digits: either would raise before the HMAC check.
_EXPIRY_RE = re.compile(r'[0-9]{1,18}')

OK, DISABLED, MISSING, DENIED = 'ok', 'disabled', 'missing', 'denied'


def _sign(token: str, payload: str) -> str:
    return hmac.new(token.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()


class AdminAuth:
    def __init__(self, settings: Settings):
        self.token = settings.admin_token
        self.max_age = settings.admin_cookie_max_age

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def token_ok(self, candidate: str | None) -> bool:
        if not self.token or not isinstance(candidate, str) or not candidate:
            return False
        return hmac.compare_digest(candidate.strip().encode('utf-8', 'surrogatepass'), self.token.encode('utf-8'))

    def make_cookie(self, now_s: float | None = None) -> str:
        assert self.token
        exp = int(now_s if now_s is not None else time.time()) + self.max_age
        payload = f'v1.{exp}'
        return f'{payload}.{_sign(self.token, payload)}'

    def cookie_ok(self, value: str | None, now_s: float | None = None) -> bool:
        """Whether a cookie value is one of ours and still valid. Never raises: the value comes from the
        request, and this runs before any authentication."""
        if not self.token or not value:
            return False
        parts = value.split('.')
        if len(parts) != 3 or parts[0] != 'v1' or not _EXPIRY_RE.fullmatch(parts[1]):
            return False
        if int(parts[1]) < (now_s if now_s is not None else time.time()):
            return False
        expected = _sign(self.token, f'v1.{parts[1]}')
        return hmac.compare_digest(parts[2].encode('ascii', 'replace'), expected.encode('ascii'))

    def check(self, headers: Any, cookies: Any) -> str:
        """OK, DISABLED (no ADMIN_TOKEN: 503), MISSING (no or expired credentials: 401) or DENIED (403)."""
        if not self.enabled:
            return DISABLED
        auth = headers.get('authorization')
        if auth:
            scheme, _, credential = auth.partition(' ')
            if scheme.lower() == 'bearer':
                return OK if self.token_ok(credential) else DENIED
        cookie = cookies.get(COOKIE_NAME)
        if cookie:
            return OK if self.cookie_ok(cookie) else MISSING  # expired or from an old token: sign in again
        return MISSING


# ───────────────────────────── HTML helpers ─────────────────────────────


def esc(v: Any) -> str:
    return html.escape('' if v is None else str(v), quote=True)


def _fmt_time(t: _dt.datetime | None, now: _dt.datetime) -> str:
    if t is None:
        return '<span class="muted">never</span>'
    # SQLite returns naive UTC values, Postgres aware ones in the session time zone
    t = t.replace(tzinfo=_dt.timezone.utc) if t.tzinfo is None else t.astimezone(_dt.timezone.utc)
    secs = max(0, int((now - t).total_seconds()))
    if secs < 60:
        rel = f'{secs} s ago'
    elif secs < 3600:
        rel = f'{secs // 60} min ago'
    elif secs < 86400:
        rel = f'{secs // 3600} h ago'
    else:
        rel = f'{secs // 86400} d ago'
    return f'<time datetime="{esc(iso(t))}" title="{esc(t.strftime("%Y-%m-%d %H:%M:%S"))} UTC">{esc(rel)}</time>'


def _bytes(n: int | None) -> str:
    if n is None:
        return '—'
    size = float(n)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024
    return f'{n} B'


def _yes_no(v: Any) -> str:
    return 'yes' if v is True else 'no' if v is False else '—'


def _short(v: Any, n: int = 8) -> str:
    s = '' if v is None else str(v)
    return f'<span title="{esc(s)}">{esc(s[:n])}{"…" if len(s) > n else ""}</span>'


def _table(head: list[str], rows: list[str], empty: str) -> str:
    if not rows:
        return f'<p class="muted">{esc(empty)}</p>'
    th = ''.join(f'<th>{esc(h)}</th>' for h in head)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _td(*cells: str) -> str:
    return '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'


def _export_links(params: dict[str, str]) -> str:
    q = ('?' + urlencode(params)) if params else ''
    return f'<a href="/api/export.zip{esc(q)}">zip</a> · <a href="/api/export{esc(q)}">json</a>'


_STYLE = """
:root{--bg:#f6f7f9;--fg:#1d2330;--muted:#667085;--card:#fff;--line:#e3e6eb;--accent:#2457c5;--bad:#b42318;--ok:#067647}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e6e8eb;--muted:#9aa4b2;--card:#1a1f26;--line:#2b323c;--accent:#7aa2ff;--bad:#ff8a80;--ok:#6ce9a6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1280px;margin:0 auto;padding:16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.muted{color:var(--muted)}a{color:var(--accent)}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.tile b{display:block;font-size:22px}
.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-weight:600;color:var(--muted)}tr:last-child td{border-bottom:none}.num{text-align:right}
.tag{display:inline-block;padding:0 6px;border-radius:6px;border:1px solid var(--line);font-size:12px}
.bad{color:var(--bad)}.good{color:var(--ok)}form{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
button{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.danger{border-color:var(--bad);color:var(--bad)}
"""


def _page(title: str, body: str, nonce: str, script: str = '') -> str:
    js = f'<script nonce="{esc(nonce)}">{script}</script>' if script else ''
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">'
        f'<title>{esc(title)}</title><style nonce="{esc(nonce)}">{_STYLE}</style></head>'
        f'<body><main>{body}</main>{js}</body></html>'
    )


def csp(nonce: str) -> str:
    return (f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; connect-src 'self'; "
            "img-src 'self' data:; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")


# ───────────────────────────── pages ─────────────────────────────


def render_login(nonce: str, message: str | None = None, disabled: bool = False) -> str:
    if disabled:
        body = ('<h1>NoseTouch admin</h1><p class="bad">The admin page is switched off: set ADMIN_TOKEN '
                '(<code>heroku config:set ADMIN_TOKEN=…</code>) and reload.</p>')
        return _page('NoseTouch admin', body, nonce)
    note = f'<p class="bad">{esc(message)}</p>' if message else ''
    body = (
        '<h1>NoseTouch admin</h1><p class="muted">Sign in with the admin token (ADMIN_TOKEN).</p>' + note +
        '<form method="post" action="/admin/login" autocomplete="off">'
        '<input type="password" name="token" placeholder="admin token" required autofocus size="40">'
        '<button type="submit">Sign in</button></form>'
    )
    return _page('NoseTouch admin: sign in', body, nonce)


_SCRIPT = """
(function () {
  var PID = /^P-[A-HJ-NP-Z2-9]{6}$/;
  function busy() {
    var a = document.activeElement;
    return a && (a.tagName === 'INPUT' || a.tagName === 'BUTTON');
  }
  function later() {
    setTimeout(function () { if (busy()) { later(); } else { location.reload(); } }, %d);
  }
  later();
  var form = document.getElementById('delete-form');
  var out = document.getElementById('delete-result');
  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    var pid = form.elements.pid.value.trim().toUpperCase();
    if (!PID.test(pid)) { out.textContent = 'Not a participant code (P- and 6 characters).'; return; }
    if (!window.confirm('Delete ALL server data of ' + pid + '? This cannot be undone.')) { return; }
    out.textContent = 'Deleting ' + pid + ' ...';
    fetch('/api/participants/' + encodeURIComponent(pid), { method: 'DELETE', credentials: 'same-origin' })
      .then(function (r) { return r.json().then(function (b) { return [r.status, b]; }); })
      .then(function (res) {
        out.textContent = res[0] === 200 ? ('Deleted ' + pid + ': ' + JSON.stringify(res[1].deleted))
                                         : ('Failed: ' + (res[1].error || res[0]));
      })
      .catch(function (e) { out.textContent = 'Failed: ' + e; });
  });
})();
""" % (REFRESH_SECONDS * 1000)


def _val(v: Any) -> str:
    """A record value for a table cell: escaped, or a dash when missing."""
    if v is None or v == '' or isinstance(v, (dict, list)):
        return '<span class="muted">—</span>'
    return esc(v)


def _profile_cells(p: dict[str, Any]) -> list[str]:
    prof = p['profile'] if isinstance(p.get('profile'), dict) else {}
    lq = prof.get('ehi_lq')
    ehi = _val(prof.get('ehi_class')) + (f' <span class="muted">({esc(lq)})</span>' if lq is not None else '')
    return [_val(prof.get('name') or prof.get('nickname')), _val(prof.get('age_years')), _val(prof.get('sex')),
            _val(prof.get('handedness_self')), ehi, _val(prof.get('phone_hand'))]


def render_admin(d: dict[str, Any], nonce: str) -> str:
    now = d['now']
    c = d['counts']
    status = d['session_status']
    day = d['last_day']
    last_upload = max((t for t in (d['last_trial_at'], d['last_session_at']) if t is not None), default=None)
    tiles = [
        ('Participants', c['participants'], ''),
        ('Sessions', c['sessions'], ', '.join(f'{esc(k)} {v}' for k, v in sorted(status.items())) or '—'),
        ('Trials', c['trials'], f"{d['trials_accepted']} accepted · {d['trials_simulated']} simulated"),
        ('Last upload', _fmt_time(last_upload, now), ''),
        ('Last 24 h', sum(day.values()),
         ', '.join(f'{esc(k)} {v}' for k, v in sorted(day.items())) or 'no uploads'),
        ('Stored data', _bytes(d['trial_bytes']),
         f"database {_bytes(d['db_size'])}" if d['db_size'] is not None else esc(d['dialect'])),
    ]
    tile_html = ''.join(
        f'<div class="tile"><span class="muted">{esc(label)}</span><b>{value}</b>'
        f'<span class="muted">{extra}</span></div>' for label, value, extra in tiles
    )

    people = []
    for p in d['participants']:
        pid = p['participant_id']
        sim = ' <span class="tag">simulated</span>' if p['sessions'] and p['sessions_simulated'] == p['sessions'] else ''
        people.append(_td(
            f'<code>{esc(pid)}</code>{sim}', *_profile_cells(p),
            f'<span class="num">{p["sessions"]}</span>',
            f'{p["trials"]} <span class="muted">({p["accepted"]} acc.)</span>',
            f'L {p["left"]} / R {p["right"]} <span class="muted">(acc. {p["left_accepted"]}/{p["right_accepted"]})</span>',
            esc(p['platform'] or '—') + (f' <span class="muted">{esc(p["browser"])}</span>' if p['browser'] else ''),
            _fmt_time(p['first_seen'], now), _fmt_time(p['last_seen'], now),
            _export_links({'participant_id': pid}),
        ))

    def count(v: Any) -> str:
        return '?' if v is None else esc(v)

    sessions = [_td(
        _fmt_time(s['last_received_at'], now), f'<code>{esc(s["participant_id"])}</code>', _short(s['session_id']),
        _val(s['status']), _val(s['preset']),
        f'{count(s["n_accepted"])}/{count(s["n_planned"])} '
        f'<span class="muted">({count(s["n_attempts"])} attempts)</span>',
        _val(s['platform']), _val(s['browser']), _val(s['app_version']), _yes_no(s['simulated']),
    ) for s in d['recent_sessions']]

    trials = [_td(
        _fmt_time(t['received_at'], now), f'<code>{esc(t["participant_id"])}</code>', _short(t['session_id']),
        f'{esc(t["trial_index"])}<span class="muted"> a{esc(t["attempt"])}</span>', _val(t['hand']),
        _val(t['orientation']), _val(t['speed']), _val(t['variant']), _val(t['status']),
        '<span class="good">yes</span>' if t['accepted'] else '<span class="bad">no</span>',
        _bytes(t['size_bytes']), _yes_no(t['simulated']),
    ) for t in d['recent_trials']]

    log_rows = [_td(
        _fmt_time(r['at'], now),
        f'<span class="{"bad" if r["outcome"] == "rejected" else "muted"}">{esc(r["outcome"])}</span>',
        _val(r['code']), _val(r['kind']), _val(r['participant_id']), _short(r['record_id']),
        _val(r['app_version']), _bytes(r['bytes']),
    ) for r in d['log']]

    body = (
        f'<h1>NoseTouch uploads</h1><p class="muted">{esc(SERVICE_NAME)} {esc(SERVER_VERSION)} · '
        f'{esc(now.strftime("%Y-%m-%d %H:%M:%S"))} UTC · refreshes every {REFRESH_SECONDS} s · '
        '<a href="/admin/logout">sign out</a></p>'
        f'<div class="tiles">{tile_html}</div>'
        '<h2>Download</h2><p>All data: ' + _export_links({}) + ' · real data only (no simulator): ' +
        _export_links({'include_simulated': 'false'}) +
        '. The zip holds the backend folder layout (<code>&lt;study&gt;/&lt;participant&gt;/&lt;session&gt;/'
        '</code>); <code>analysis/scripts/pull_live.py</code> fetches it for the analysis kit. The json is the '
        "app's own export format.</p>"
        '<h2>Participants</h2>' + _table(
            ['Code', 'Name', 'Age', 'Sex', 'Handedness', 'EHI', 'Phone hand', 'Sessions', 'Trials',
             'Left / right', 'Platform', 'First seen', 'Last upload', 'Data'], people, 'No uploads yet.') +
        '<h2>Recent sessions</h2>' + _table(
            ['Received', 'Participant', 'Session', 'Status', 'Preset', 'Accepted / planned', 'Platform',
             'Browser', 'App', 'Simulated'], sessions, 'No sessions yet.') +
        '<h2>Recent trials</h2>' + _table(
            ['Received', 'Participant', 'Session', 'Trial', 'Hand', 'Orientation', 'Speed', 'Variant', 'Status',
             'Accepted', 'Size', 'Simulated'], trials, 'No trials yet.') +
        '<h2>Refused or ignored uploads</h2><p class="muted">Rejections (a wrong study key usually means a '
        'phone with an old config) and stale re-sends that did not replace a newer copy. Identifiers only.</p>' +
        _table(['When', 'Outcome', 'Code', 'Kind', 'Participant', 'Record', 'App', 'Size'], log_rows,
               'Nothing refused or ignored.') +
        '<h2>Delete a participant</h2><p class="muted">Deletion on request: removes the participant with all '
        "sessions, trials and log lines from this server. Delete the data on the participant's phone too "
        '(Data screen), or queued uploads will store it again.</p>'
        '<form id="delete-form" autocomplete="off"><input name="pid" placeholder="P-XXXXXX" size="12" required>'
        '<button class="danger" type="submit">Delete</button><span id="delete-result" class="muted"></span></form>'
    )
    return _page('NoseTouch admin', body, nonce, _SCRIPT)
