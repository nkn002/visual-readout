#!/usr/bin/env bash
# Start the backend (port 8000) and the frontend dev server (port 5173). Ctrl-C stops both.
#
#   scripts/dev.sh      # laptop: http://localhost:5173/
#   scripts/dev.sh      # code-server on Open OnDemand: served under /absproxy/5173/, URLs printed
#
# Override: VITE_BASE=/some/prefix/  BACKEND_PORT=8000  FRONTEND_PORT=5173
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-5173}

# --- node: on the UVA HPC it comes from an environment module ------------------------------------
if ! command -v node >/dev/null 2>&1; then
  if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh "${LMOD_PKG:-/usr/share/lmod/lmod}/init/bash"; do
      [ -f "$f" ] && { set +e; source "$f"; set -e; break; }
    done
  fi
  if type module >/dev/null 2>&1; then
    echo "loading nodejs module"
    module load nodejs/24.5.0
  fi
fi
command -v node >/dev/null 2>&1 || { echo "node not found. Install Node 22+ or run: module load nodejs/24.5.0"; exit 1; }
[ -x "$ROOT/backend/.venv/bin/uvicorn" ] || { echo "backend venv missing. Run: cd backend && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"; exit 1; }
[ -d "$ROOT/frontend/node_modules" ] || { echo "frontend deps missing. Run: cd frontend && npm ci"; exit 1; }

# --- ports ----------------------------------------------------------------------------------------
busy() { (command -v ss >/dev/null && ss -ltn | grep -q ":$1 ") || (command -v lsof >/dev/null && lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1); }
for p in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if busy "$p"; then
    echo "port $p is already in use. Stop the other copy first:"
    echo "  pkill -f 'vite --port $FRONTEND_PORT'; pkill -f 'uvicorn app.main'"
    exit 1
  fi
done

# --- URLs -----------------------------------------------------------------------------------------
# Behind code-server on Open OnDemand (VSCODE_PROXY_URI is set) two proxies sit in front of us. The
# outer one strips /rnode/<host>/<port>; code-server's /absproxy/P/ keeps its own prefix and
# /proxy/P/ strips it. So the browser-facing prefix and the prefix a server receives differ:
#   frontend: browser .../rnode/<host>/<port>/absproxy/5173/  ->  Vite receives /absproxy/5173/
#   backend : browser .../rnode/<host>/<port>/proxy/8000/     ->  FastAPI receives /
# Vite gets both prefixes (VITE_BASE, VITE_SERVER_PREFIX); FastAPI gets --root-path.
ROOT_PATH=""
SERVER_PREFIX=""
if [ -n "${VSCODE_PROXY_URI:-}" ]; then
  ORIGIN=$(printf '%s' "$VSCODE_PROXY_URI" | sed -E 's#^(https?://[^/]+).*#\1#')
  PUBLIC_PREFIX=$(printf '%s' "$VSCODE_PROXY_URI" | sed -E 's#^https?://[^/]+##; s#proxy/\{\{port\}\}/$##')
  BASE=${VITE_BASE:-${PUBLIC_PREFIX}absproxy/$FRONTEND_PORT/}
  SERVER_PREFIX=/absproxy/$FRONTEND_PORT/
  FRONT_URL=$ORIGIN$BASE
  DOCS_URL=${ORIGIN}${PUBLIC_PREFIX}proxy/$BACKEND_PORT/docs
  ROOT_PATH=${PUBLIC_PREFIX}proxy/$BACKEND_PORT
else
  BASE=${VITE_BASE:-/}
  FRONT_URL=http://localhost:$FRONTEND_PORT$BASE
  DOCS_URL=http://localhost:$BACKEND_PORT/docs
fi

"$ROOT/backend/.venv/bin/python" "$ROOT/backend/app/services/scan.py" --data "$ROOT/data/sample" --out "$ROOT/frontend/public/data"

trap 'kill 0 2>/dev/null' INT TERM EXIT
(cd "$ROOT/backend" && exec .venv/bin/uvicorn app.main:app --port "$BACKEND_PORT" --reload ${ROOT_PATH:+--root-path "$ROOT_PATH"}) &
(cd "$ROOT/frontend" && exec env VITE_BASE="$BASE" ${SERVER_PREFIX:+VITE_SERVER_PREFIX="$SERVER_PREFIX"} VITE_API_TARGET="http://localhost:$BACKEND_PORT" npx vite --port "$FRONTEND_PORT" --strictPort) &
sleep 3
echo
echo "  frontend : $FRONT_URL"
echo "  API docs : $DOCS_URL"
echo
echo "  Open the frontend URL above, not the Local: link Vite prints."
echo "  Ctrl-C stops both servers."
wait
