#!/usr/bin/env bash
# Start the backend (port 8000) and the frontend dev server (port 5173). Ctrl-C stops both.
#
#   scripts/dev.sh                # plain machine: http://localhost:5173/
#   scripts/dev.sh                # inside code-server on Open OnDemand: served under /absproxy/5173/
#
# Override: VITE_BASE=/some/prefix/  BACKEND_PORT=8000  FRONTEND_PORT=5173
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-5173}

if ! command -v node >/dev/null 2>&1; then
  [ -f ~/.bashrc ] && source ~/.bashrc >/dev/null 2>&1 || true
  command -v module >/dev/null 2>&1 && module load nodejs/24.5.0 >/dev/null 2>&1 || true
fi
command -v node >/dev/null 2>&1 || { echo "node not found (try: module load nodejs/24.5.0)"; exit 1; }
[ -x "$ROOT/backend/.venv/bin/uvicorn" ] || { echo "backend venv missing: cd backend && python -m venv .venv && .venv/bin/pip install -e '.[dev]'"; exit 1; }

# Behind code-server (VSCODE_PROXY_URI is set) the browser reaches port P at .../absproxy/P/ with the
# path kept, and .../proxy/P/ with the path stripped. The frontend needs the kept path as its base;
# the backend gets --root-path so its /docs page builds URLs that survive the stripping proxy.
ROOT_PATH=""
if [ -n "${VSCODE_PROXY_URI:-}" ]; then
  BASE=${VITE_BASE:-/absproxy/$FRONTEND_PORT/}
  FRONT_URL=${VSCODE_PROXY_URI//proxy\/\{\{port\}\}\//absproxy/$FRONTEND_PORT/}
  DOCS_URL=${VSCODE_PROXY_URI//\{\{port\}\}/$BACKEND_PORT}docs
  ROOT_PATH=$(printf '%s' "$VSCODE_PROXY_URI" | sed -E 's#^https?://[^/]+##; s#\{\{port\}\}#'"$BACKEND_PORT"'#; s#/$##')
else
  BASE=${VITE_BASE:-/}
  FRONT_URL=http://localhost:$FRONTEND_PORT$BASE
  DOCS_URL=http://localhost:$BACKEND_PORT/docs
fi

python3 "$ROOT/backend/app/services/scan.py" --data "$ROOT/data/sample" --out "$ROOT/frontend/public/data"

trap 'kill 0 2>/dev/null' INT TERM EXIT
(cd "$ROOT/backend" && exec .venv/bin/uvicorn app.main:app --port "$BACKEND_PORT" --reload ${ROOT_PATH:+--root-path "$ROOT_PATH"}) &
(cd "$ROOT/frontend" && VITE_BASE=$BASE VITE_API_TARGET=http://localhost:$BACKEND_PORT exec npx vite --port "$FRONTEND_PORT" --strictPort) &
sleep 2
echo
echo "  frontend : $FRONT_URL"
echo "  API docs : $DOCS_URL"
echo
wait
