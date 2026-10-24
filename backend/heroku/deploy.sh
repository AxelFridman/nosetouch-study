#!/usr/bin/env bash
# Deploy the upload service to Heroku without committing anything in the main repository.
#
#   backend/heroku/deploy.sh <heroku-app-name> [--dry-run]
#
# Copies this folder (without .venv, tests, caches, local databases and secrets) into a temporary
# directory, commits it there and force-pushes that commit to https://git.heroku.com/<app>.git (branch
# main). Heroku then builds it (Python buildpack: requirements.txt, .python-version, Procfile) and
# restarts the dyno. --dry-run builds the copy and lists it, but pushes nothing.
# Before pushing, the app must have its Postgres database (config var DATABASE_URL): without it the
# service stores nothing and answers every upload 503. ALLOW_EPHEMERAL_DB=1 on the app (a throwaway
# test app) skips that check; without the Heroku CLI it is skipped with a warning.
# Git authentication: HEROKU_API_KEY if set, else the Heroku CLI's token (`heroku auth:token`), handed
# to git by a one-off credential helper through an environment variable (never on a command line, never
# stored); without either, git uses whatever `heroku login` set up.
set -euo pipefail

usage() { echo "usage: $0 <heroku-app-name> [--dry-run]" >&2; exit 2; }

APP=""
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage ;;
    -*) echo "unknown option: $arg" >&2; usage ;;
    *) if [[ -z "$APP" ]]; then APP="$arg"; else usage; fi ;;
  esac
done
[[ -n "$APP" ]] || usage
if ! [[ "$APP" =~ ^[a-z][a-z0-9-]{1,29}$ ]]; then
  echo "not a Heroku app name: $APP" >&2
  exit 2
fi

# 0. The app's database. The service refuses to keep uploads in a dyno's memory (lost at every
#    restart), so without DATABASE_URL it would answer every upload 503 until Postgres is attached.
check_database() {
  if ! command -v heroku >/dev/null 2>&1; then
    echo "warning: no Heroku CLI here, so '$APP' was not checked for DATABASE_URL (without it every upload gets 503)." >&2
    return 0
  fi
  local db_url allow
  if ! db_url="$(heroku config:get DATABASE_URL -a "$APP" 2>/dev/null)"; then
    echo "cannot read the config of '$APP' (heroku config:get failed): does the app exist, and is the CLI logged in?" >&2
    exit 1
  fi
  if [[ -n "${db_url//[[:space:]]/}" ]]; then
    return 0  # (the URL holds the database password: never printed)
  fi
  allow="$(heroku config:get ALLOW_EPHEMERAL_DB -a "$APP" 2>/dev/null || true)"
  case "$(printf '%s' "$allow" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      echo "note: '$APP' has no DATABASE_URL and ALLOW_EPHEMERAL_DB is set: uploads go to a throwaway database." >&2
      return 0 ;;
  esac
  {
    echo "'$APP' has no DATABASE_URL, so the service would store nothing and answer every upload 503."
    echo "Attach Postgres first, wait until it is ready, then run this script again:"
    echo "  heroku addons:create heroku-postgresql:essential-1 -a $APP"
    echo "  heroku addons:wait -a $APP"
  } >&2
  exit 1
}
if [[ $DRY_RUN -eq 0 ]]; then
  check_database
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/nosetouch-deploy.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

# 1. The deploy copy: only what the dyno needs.
tar -C "$HERE" \
  --exclude=.venv --exclude=tests --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.pyc' \
  --exclude='*.db' --exclude='*.db-wal' --exclude='*.db-shm' --exclude='*.sqlite' --exclude='*.sqlite3' \
  --exclude=.env --exclude='*.env' --exclude=.git --exclude=deploy.sh --exclude=pytest.ini \
  --exclude=requirements-dev.txt \
  -cf - . | tar -C "$STAGE" -xf -

for f in Procfile requirements.txt .python-version app.py nosetouch_server/main.py; do
  [[ -f "$STAGE/$f" ]] || { echo "missing $f in the deploy copy" >&2; exit 1; }
done
# Heroku runs on Linux: no CR line endings (a Windows checkout may have them)
find "$STAGE" -type f \( -name '*.py' -o -name 'Procfile' -o -name '*.txt' -o -name '.python-version' \) \
  -exec sed -i 's/\r$//' {} +

SOURCE_REV="$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
if [[ -n "$(git -C "$HERE" status --porcelain -- . 2>/dev/null)" ]]; then
  SOURCE_REV="$SOURCE_REV + uncommitted changes"
fi

# 2. A one-commit repository holding the copy.
cd "$STAGE"
git init -q -b main 2>/dev/null || { git init -q && git symbolic-ref HEAD refs/heads/main; }
git -c core.autocrlf=false add -A
git -c core.autocrlf=false commit -q -m "NoseTouch upload service from ${SOURCE_REV} ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo "Deploy copy for '$APP' (commit $(git rev-parse --short HEAD)):"
git ls-files | sed 's/^/  /'
if [[ $DRY_RUN -eq 1 ]]; then
  echo "--dry-run: nothing pushed."
  exit 0
fi

# 3. Push (Heroku builds and releases on receipt).
REMOTE="https://git.heroku.com/${APP}.git"
TOKEN="${HEROKU_API_KEY:-}"
if [[ -z "$TOKEN" ]] && command -v heroku >/dev/null 2>&1; then
  TOKEN="$(heroku auth:token 2>/dev/null | tail -n 1 || true)"
fi
if [[ -n "$TOKEN" ]]; then
  export NOSETOUCH_GIT_TOKEN="$TOKEN"
  # shellcheck disable=SC2016  # $1 and the token are expanded by the helper's own shell
  git -c credential.helper= \
      -c 'credential.helper=!f() { test "$1" = get && echo username=heroku && echo "password=$NOSETOUCH_GIT_TOKEN"; }; f' \
      push --force "$REMOTE" HEAD:main
else
  git push --force "$REMOTE" HEAD:main
fi

echo
echo "Pushed. Heroku builds and restarts the app now. Then check:"
echo "  heroku apps:info -a $APP            (web URL)"
echo "  curl -s <web URL>api/health"
echo "  heroku logs --tail -a $APP"
