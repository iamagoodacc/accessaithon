#!/bin/bash
set -e

echo "============================================"
echo "  Project Setup and Launch"
echo "============================================"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed."
    echo "Install it from https://www.python.org/downloads/"
    exit 1
fi

# Check for Node.js / npm
if ! command -v npm &> /dev/null; then
    echo "[ERROR] Node.js/npm is not installed."
    echo "Install it from https://nodejs.org/"
    exit 1
fi

# ---- Python API Setup ----
echo "[1/4] Setting up Python virtual environment..."
cd hand_tracking

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "[2/4] Installing Python dependencies..."
echo "       (This may take a while on first run due to large packages like torch)"

if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found in hand_tracking folder!"
    exit 1
fi

# Remove the broken 'dotenv' package line and install from cleaned file
grep -v "^dotenv==" requirements.txt > requirements_clean.txt
pip install -r requirements_clean.txt
rm requirements_clean.txt

echo "Python dependencies installed successfully."

echo "[3/4] Starting Python API..."
python api.py &
API_PID=$!

# ---- Next.js App Setup ----
cd ../app

echo "[4/4] Installing npm dependencies and starting Next.js app..."
npm install

npx next dev &
NPM_PID=$!

# Wait for dev server to start, then open browser
echo ""
echo "Waiting for dev server to start..."
sleep 5
open http://localhost:3000

echo ""
echo "============================================"
echo "  Both services started!"
echo "  Browser opened to http://localhost:3000"
echo "  Python API  (PID: $API_PID)"
echo "  Next.js dev (PID: $NPM_PID)"
echo "  Press Ctrl+C to stop both."
echo "============================================"

trap "echo 'Shutting down...'; kill $API_PID $NPM_PID 2>/dev/null; exit" SIGINT SIGTERM

wait
