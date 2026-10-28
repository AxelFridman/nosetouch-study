"""Asset step of the NoseTouch report: copy the saved Stage 2 analysis outputs into report/. It never fits a model.

    python scripts/make_assets.py --config report.toml --mode draft|final      # collect, then count what is missing
    python scripts/make_assets.py --config report.toml --mode final --check    # only list missing assets (exit 1 if any)

latexmk runs it before LaTeX (latexmkrc); ``python -m stage2.pipeline report`` runs it, then latexmk.

What it writes (every file atomically: a temporary file in the same folder, then ``os.replace``, so a LaTeX run in
another process never reads half a file; an unchanged file is not rewritten, so latexmk sees no change):

* ``figures/<unit>/<NAME>.pdf`` (PNG when a figure has no PDF) for every figure listed in a ``captions.json`` under
  ``<runs.stage2>/figures/``; the root registry (the describe figures S2_*) goes to ``figures/core/``. Each unit
  folder also gets the merged ``captions.json`` (caption + the decision the figure supports) for the writers.
  ``figures/photo/`` and ``figures/static/`` are hand-placed and never touched.
* ``tables/<unit>/<name>.tex`` for every LaTeX table fragment under ``<runs.stage2>/tables/<unit>/``. Each tabular is
  wrapped in ``\\fitwidth{...}`` (main.tex), which shrinks it to the text width only when it is wider.
* ``tables/numbers.tex``: one definition per entry of every ``results/<unit>_numbers.json``:
  ``\\R{<unit>/<key>}`` prints the formatted value WITH its unit sign (``.1%`` -> ``89.3\\,\\%``),
  ``\\Rb{<unit>/<key>}`` the bare number (``89.3``); ``\\Rci`` / ``\\Rcib`` add ``[<key>_lo, <key>_hi]``.
* ``tables/numbers_index.tsv``: key, value as printed, description, source (for writers searching for a key).
* ``tables/provenance.tex``: ``\\datakind``, ``\\buildlabel``, ``\\provenanceline`` (git hash and dirty flag, the cache's
  data identity, persons and trials) and ``\\finalbuildtrue`` in final mode.

The check lists every ``\\R``-family key, ``\\genfig`` and ``\\gentab`` used in main.tex and sections/*.tex that has no
generated asset; in final mode a missing asset is an error (exit 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

REPORT = Path(__file__).resolve().parents[1]
MANAGED_FIG_EXT = ('.pdf', '.png')
HAND_PLACED = {'photo', 'static'}          # figures/ sub-folders the step never touches
CORE_UNIT = 'core'                         # folder of the figures registered at the figures/ root (describe S2_*)


# ───────────────────────────── small helpers ─────────────────────────────

def load_config(path: Path) -> dict:
    text = path.read_text(encoding='utf-8')
    if tomllib is not None:
        return tomllib.loads(text)
    cfg: dict = {}
    sec = cfg
    for line in text.splitlines():                       # minimal fallback: [section] and key = "value"
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('['):
            sec = cfg.setdefault(line.strip('[]').strip(), {})
        elif '=' in line:
            k, v = (s.strip() for s in line.split('=', 1))
            sec[k] = v.strip('"')
    return cfg


def write_bytes_atomic(path: Path, data: bytes) -> bool:
    """Write ``data`` to ``path`` through a temporary file + os.replace; skip (return False) when unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return False
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
        for attempt in range(20):                        # Windows: a reader may hold the file for a moment
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                import time
                time.sleep(0.1 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return True


def write_text_atomic(path: Path, text: str) -> bool:
    return write_bytes_atomic(path, text.encode('utf-8'))


def analysis_formatting(analysis_dir: Path):
    """The analysis' own formatters (stage2.report_export), so numbers print exactly as in the generated tables."""
    sys.path.insert(0, str(analysis_dir))
    from stage2 import report_export as RX            # noqa: E402  (light: no numpy at import)
    return RX


# ───────────────────────────── figures ─────────────────────────────

def collect_figures(stage2: Path, mode: str, log: list[str]) -> dict[str, dict]:
    """Copy every registered figure; return {'<unit>/<NAME>': {caption, decision, file}}."""
    fig_root = stage2 / 'figures'
    out_root = REPORT / 'figures'
    found: dict[str, dict] = {}
    wanted: dict[str, set[str]] = {}
    problems = []
    for reg in sorted(fig_root.rglob('captions.json')):
        rel = reg.parent.relative_to(fig_root)
        unit = CORE_UNIT if str(rel) == '.' else rel.as_posix()
        if unit.split('/')[0] in HAND_PLACED:
            continue
        caps = json.loads(reg.read_text(encoding='utf-8'))
        merged = {}
        for name, c in sorted(caps.items()):
            src = next((reg.parent / f'{name}{ext}' for ext in MANAGED_FIG_EXT if (reg.parent / f'{name}{ext}').exists()),
                       None)
            if src is None:
                problems.append(f'{reg.parent.name}/{name}: registered in captions.json but neither PDF nor PNG exists')
                continue
            dst = out_root / unit / src.name
            write_bytes_atomic(dst, src.read_bytes())
            wanted.setdefault(unit, set()).add(dst.name)
            merged[name] = {'caption': c.get('caption', ''), 'decision': c.get('decision', ''), 'file': src.name}
            found[f'{unit}/{name}'] = merged[name]
        write_text_atomic(out_root / unit / 'captions.json', json.dumps(merged, indent=1, ensure_ascii=False) + '\n')
        wanted.setdefault(unit, set()).add('captions.json')
    # remove figures of a managed unit folder that its registry no longer lists (never photo/ or static/)
    for unit, names in wanted.items():
        for f in (out_root / unit).iterdir():
            if f.is_file() and f.name not in names and (f.suffix in MANAGED_FIG_EXT or f.name == 'captions.json'):
                f.unlink()
    for p in problems:
        log.append('figure: ' + p)
    if problems and mode == 'final':
        raise SystemExit('make_assets: registered figures are missing (final mode):\n  ' + '\n  '.join(problems))
    return found


# ───────────────────────────── tables ─────────────────────────────

_TABULAR = re.compile(r'(\\begin\{tabular\}.*?\\end\{tabular\})', re.S)


def collect_tables(stage2: Path) -> set[str]:
    tab_root = stage2 / 'tables'
    out_root = REPORT / 'tables'
    found: set[str] = set()
    wanted: dict[str, set[str]] = {}
    for f in sorted(tab_root.rglob('*.tex')):
        rel = f.relative_to(tab_root)
        unit = rel.parent.as_posix() if str(rel.parent) != '.' else CORE_UNIT
        text = f.read_text(encoding='utf-8')
        text = _TABULAR.sub(lambda m: '\\fitwidth{%\n' + m.group(1) + '%\n}', text)
        text = _BARE_TABLE.sub(lambda m: 'Table~\\ref{tab:' + m.group(1).replace('\\_', '_') + '}', text)
        text = ('% copied by report/scripts/make_assets.py from analysis/reports/stage2/tables/'
                f'{rel.as_posix()}; tabulars wrapped in \\fitwidth. Do not edit: rerun the asset step.\n' + text)
        dst = out_root / unit / f.name
        write_text_atomic(dst, text)
        wanted.setdefault(unit, set()).add(f.name)
        found.add(f'{unit}/{f.stem}')
    for unit, names in wanted.items():
        for f in (out_root / unit).glob('*.tex'):
            if f.name not in names:
                f.unlink()
    return found


# ───────────────────────────── numbers ─────────────────────────────

#: a caption that names a generated table by its bare label ("Table x1\_sensitivity") gets a real reference
_BARE_TABLE = re.compile(r'Table ((?:x1|m1|m2|m3|m4|p1|final|sec|demo)\\_[A-Za-z0-9_\\]*[A-Za-z0-9])')

#: keys that hold a p-value (a token 'p', 'holm' or 'bh'); presentation rule: below 0.001 they print '< 0.001'
_P_TOKENS = {'p', 'holm', 'bh', 'pval'}


def _p_floor(key: str, entry: dict) -> dict:
    v = entry.get('value')
    if (_P_TOKENS & set(key.split('_')) and isinstance(v, (int, float)) and not isinstance(v, bool)
            and 0 <= v < 0.001):
        return {**entry, 'value': '< 0.001', 'format': ''}
    return entry


def derived_numbers(units: dict[str, dict]) -> dict[str, dict]:
    """Exact arithmetic on saved m4 keys (no fitting). A calibrated P(right) of exactly 0.5 is an abstention, so the
    share of recordings that show a hand, and the accuracy on them, leave those ties out. Skipped unless the saved
    accuracy on the answered is a pooled count (correct answers, a tie counting one half, a multiple of 0.5)."""
    out: dict[str, dict] = {}
    m4 = units.get('m4', {})
    for tag, n_key, ans_key, tie_key, acc_key, what in (
            ('prod', 'n_recordings', 'prod_n_answered', 'prod_n_answered_tie', 'prod_acc_answered', 'recordings'),
            ('prod_nat', 'nat_n', 'prod_nat_answered', 'prod_nat_answered_tie', 'prod_nat_acc_answered',
             'natural-use trials')):
        try:
            n = float(m4[n_key]['value'])
            n_ans = float(m4[ans_key]['value'])
            n_tie = float(m4[tie_key]['value'])
            correct = float(m4[acc_key]['value']) * n_ans
        except (KeyError, TypeError, ValueError):
            continue
        if abs(correct * 2 - round(correct * 2)) > 1e-6 or n_ans <= n_tie or n <= 0:
            continue
        shown = n_ans - n_tie
        right = round(correct * 2) / 2 - 0.5 * n_tie
        src = f'report/scripts/make_assets.py: m4/{ans_key} - m4/{tie_key}; m4/{acc_key}; m4/{n_key}'
        out[f'm4_{tag}_n_shown'] = {'value': int(shown), 'format': 'd', 'source': src,
                                    'description': f'cue-free production: {what} with a hand shown '
                                                   '(approach accepted and P(right) not exactly 0.5)'}
        out[f'm4_{tag}_shown_coverage'] = {'value': shown / n, 'format': '.1%', 'source': src,
                                           'description': f'cue-free production: share of {what} with a hand shown '
                                                          '(an output of exactly 0.5 is an abstention)'}
        out[f'm4_{tag}_shown_acc'] = {'value': right / shown, 'format': '.1%', 'source': src,
                                      'description': f'cue-free production: accuracy on the {what} with a hand '
                                                     'shown (for natural use: right-hand recall)'}
    return out


def _tex_value(RX, value, fmt: str) -> tuple[str, str]:
    """(printed with its unit sign, bare number) for one numbers entry."""
    s = RX.format_value(value, fmt)
    if s == RX.MISSING:
        return '--', '--'
    if fmt.endswith('%') and s.endswith('%'):
        bare = RX.latex_value(s[:-1], '')               # escaped, typographic minus
        return bare + r'\,\%', bare
    if fmt in ('', 's') or isinstance(value, str):
        t = RX.latex_escape(s)
        return t, t
    t = RX.latex_value(value, fmt)
    return t, t


def collect_numbers(stage2: Path, RX) -> dict[str, dict]:
    res = stage2 / 'results'
    lines = ['% generated by report/scripts/make_assets.py from analysis/reports/stage2/results/*_numbers.json;',
             '% do not edit. \\R{<unit>/<key>}: value with its unit sign; \\Rb{<unit>/<key>}: bare number.']
    index = ['key\tprinted\tdescription\tsource']
    keys: dict[str, dict] = {}
    units: dict[str, dict] = {}
    for f in sorted(res.glob('*_numbers.json')):
        units[f.name[: -len('_numbers.json')]] = json.loads(f.read_text(encoding='utf-8'))
    derived = derived_numbers(units)
    if derived:
        units['derived'] = derived
    for unit, raw in units.items():
        entries = {k: _p_floor(k, e) for k, e in raw.items()}
        lines.append(f'% --- {unit} ---')
        for k in sorted(entries):
            e = entries[k]
            full, bare = _tex_value(RX, e.get('value'), e.get('format', '') or '')
            key = f'{unit}/{k}'
            lines.append(f'\\expandafter\\def\\csname res@{key}\\endcsname{{{full}}}')
            if bare != full:
                lines.append(f'\\expandafter\\def\\csname resb@{key}\\endcsname{{{bare}}}')
            keys[key] = e
            desc = ' '.join(str(e.get('description', '')).split())
            index.append(f'{key}\t{full}\t{desc}\t{e.get("source", "")}')
    write_text_atomic(REPORT / 'tables' / 'numbers.tex', '\n'.join(lines) + '\n')
    write_text_atomic(REPORT / 'tables' / 'numbers_index.tsv', '\n'.join(index) + '\n')
    return keys


# ───────────────────────────── provenance ─────────────────────────────

def _git(*args: str) -> str:
    try:
        return subprocess.run(['git', *args], cwd=REPORT, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def write_provenance(cfg: dict, stage2: Path, keys: dict[str, dict], mode: str) -> str:
    head = _git('rev-parse', '--short', 'HEAD') or '?'
    dirty = '+dirty' if _git('status', '--porcelain') else ''
    meta_path = (REPORT / cfg.get('cache', {}).get('meta', '../analysis/data/stage2_cache/meta.json')).resolve()
    data_id = '?'
    if meta_path.exists():
        di = json.loads(meta_path.read_text(encoding='utf-8')).get('data_identity', {})
        data_id = str(di.get('data_id') or hashlib.sha256(json.dumps(di, sort_keys=True).encode()).hexdigest())[:8]
    persons = keys.get('describe/n_persons', {}).get('value', '?')
    trials = keys.get('describe/primary_trials', {}).get('value', '?')
    line = (f'git {head}{dirty} \\textperiodcentered{{}} data {data_id} \\textperiodcentered{{}} '
            f'{persons} persons, {trials} trials \\textperiodcentered{{}} leave-one-person-out')
    url = cfg.get('predictor', {}).get('url', '')
    text = ['% generated by report/scripts/make_assets.py; do not edit',
            '\\def\\datakind{real}',
            f'\\def\\buildlabel{{{"final" if mode == "final" else "preliminary draft"}}}',
            f'\\def\\provenanceline{{{line}}}']
    if url:
        text.append(f'\\def\\predictorURL{{{url}}}')
    if mode == 'final':
        text.append('\\finalbuildtrue')
    write_text_atomic(REPORT / 'tables' / 'provenance.tex', '\n'.join(text) + '\n')
    return line


# ───────────────────────────── check ─────────────────────────────

_USE = re.compile(r'\\(Rcib|Rci|Rb|R|genfig|gentab)\s*(?:\[[^\]]*\])?\s*\{([^{}#]+)\}')


def _strip_comments(text: str) -> str:
    return '\n'.join(re.sub(r'(?<!\\)%.*', '', line) for line in text.splitlines())


def check(keys: set[str], figs: set[str], tabs: set[str]) -> list[str]:
    files = sorted(REPORT.glob('*.tex')) + sorted((REPORT / 'sections').glob('*.tex'))
    missing = []
    for f in files:
        text = _strip_comments(f.read_text(encoding='utf-8'))
        for m in _USE.finditer(text):
            cmd, arg = m.group(1), m.group(2).strip()
            line = text.count('\n', 0, m.start()) + 1
            where = f'{f.relative_to(REPORT).as_posix()}:{line}'
            if cmd in ('R', 'Rb'):
                need = [arg]
            elif cmd in ('Rci', 'Rcib'):
                need = [arg, arg + '_lo', arg + '_hi']
            elif cmd == 'genfig':
                if arg not in figs and not any((REPORT / 'figures' / f'{arg}{e}').exists() for e in MANAGED_FIG_EXT):
                    missing.append(f'{where}: figure {arg}')
                continue
            else:
                if arg not in tabs and not (REPORT / 'tables' / f'{arg}.tex').exists():
                    missing.append(f'{where}: table {arg}')
                continue
            for k in need:
                if k not in keys:
                    missing.append(f'{where}: number {k}')
    return missing


def existing_assets() -> tuple[set[str], set[str], set[str]]:
    """What is already in report/ (for --check without collecting)."""
    figs = {f'{p.parent.relative_to(REPORT / "figures").as_posix()}/{p.stem}'
            for p in (REPORT / 'figures').rglob('*') if p.suffix in MANAGED_FIG_EXT}
    tabs = {p.relative_to(REPORT / 'tables').with_suffix('').as_posix() for p in (REPORT / 'tables').rglob('*.tex')}
    keys = set()
    nt = REPORT / 'tables' / 'numbers.tex'
    if nt.exists():
        keys = set(re.findall(r'\\csname res@([^\\]+)\\endcsname', nt.read_text(encoding='utf-8')))
    return keys, figs, tabs


# ───────────────────────────── main ─────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', default=str(REPORT / 'report.toml'))
    ap.add_argument('--mode', choices=('draft', 'final'), default=os.environ.get('REPORT_MODE', 'draft'))
    ap.add_argument('--check', action='store_true', help='only list the assets the .tex files use but that do not exist')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(argv)
    cfg_path = Path(a.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    cfg = load_config(cfg_path)
    stage2 = (cfg_path.parent / cfg.get('runs', {}).get('stage2', '../analysis/reports/stage2')).resolve()
    analysis = (cfg_path.parent / cfg.get('runs', {}).get('analysis', '../analysis')).resolve()

    if a.check:
        keys, figs, tabs = existing_assets()
        missing = check(keys, figs, tabs)
        print(f'make_assets --check: {len(missing)} missing asset(s)')
        for m in missing:
            print('  ' + m)
        return 1 if (missing and a.mode == 'final') else 0

    if not stage2.exists():
        msg = f'make_assets: the analysis outputs {stage2} do not exist (run the Stage 2 pipeline first)'
        if a.mode == 'final':
            raise SystemExit(msg)
        print(msg + '; the report builds with placeholders')
        return 0
    RX = analysis_formatting(analysis)
    log: list[str] = []
    figs = collect_figures(stage2, a.mode, log)
    tabs = collect_tables(stage2)
    keys = collect_numbers(stage2, RX)
    prov = write_provenance(cfg, stage2, keys, a.mode)
    missing = check(set(keys), set(figs), tabs)
    if not a.quiet:
        print(f'make_assets ({a.mode}): {len(figs)} figures, {len(tabs)} tables, {len(keys)} numbers from '
              f'{stage2.as_posix()}; {len(missing)} used asset(s) missing')
        for line in log + missing[:40]:
            print('  ' + line)
        if len(missing) > 40:
            print(f'  ... and {len(missing) - 40} more (python scripts/make_assets.py --check lists all)')
    if missing and a.mode == 'final':
        print('make_assets: final mode stops on missing assets', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
