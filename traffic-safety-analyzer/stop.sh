#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "Stopping any existing services..."

if [ -f .backend.pid ]; then
    PID=$(cat .backend.pid)
    kill $PID 2>/dev/null || true
    rm -f .backend.pid
fi

if [ -f .frontend.pid ]; then
    PID=$(cat .frontend.pid)
    kill $PID 2>/dev/null || true
    rm -f .frontend.pid
fi

# Clean up any lingering processes on ports 8000 and 5173
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 5173/tcp 2>/dev/null || true
sleep 1
echo "Services stopped."

