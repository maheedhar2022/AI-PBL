#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=== Starting AI Traffic Safety Analyzer ==="

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Please create it and install requirements."
    exit 1
fi

# Stop existing instances if running
bash ./stop.sh 2>/dev/null || true

# Start Backend using setsid to completely detach process session
echo "Starting Backend (FastAPI on port 8000)..."
setsid "$DIR/.venv/bin/uvicorn" backend.main:app --host 0.0.0.0 --port 8000 </dev/null >"$DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > .backend.pid
disown -a 2>/dev/null || true

# Start Frontend using setsid
echo "Starting Frontend (Vite on port 5173)..."
cd "$DIR/frontend"
setsid npm run dev -- --host 0.0.0.0 </dev/null >"$DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$DIR/.frontend.pid"
cd "$DIR"
disown -a 2>/dev/null || true

echo "Waiting for backend to be ready..."
for i in {1..25}; do
    if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "✓ Backend is healthy at http://localhost:8000"
        break
    fi
    sleep 1
done

echo "Waiting for frontend to be ready..."
for i in {1..15}; do
    if curl -s http://127.0.0.1:5173/ >/dev/null 2>&1; then
        echo "✓ Frontend is ready at http://localhost:5173"
        break
    fi
    sleep 1
done

echo "=== Both services running successfully ==="
echo "Backend:  http://localhost:8000 (Docs: http://localhost:8000/docs)"
echo "Frontend: http://localhost:5173"

