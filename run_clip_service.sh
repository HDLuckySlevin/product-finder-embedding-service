#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------
# 🐍 Python Virtual Environment Setup
# -----------------------------------

# 1. Locate Python 3
PYTHON_BIN="$(command -v python3)"
if [ -z "$PYTHON_BIN" ]; then
  echo "❌ Error: python3 not found. Please install Python 3." >&2
  exit 1
fi
echo "🐍 Using Python interpreter: $PYTHON_BIN"

# 2. Create venv if missing
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating virtual environment in $VENV_DIR..."
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

# 3. Activate venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated: $(which python)"

# 4. Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# -----------------------------------
# 📦 Requirements Installation
# -----------------------------------
echo "📋 Verifying required packages..."
while IFS= read -r requirement || [ -n "$requirement" ]; do
  # Ignore comments and empty lines
  [[ "$requirement" =~ ^#.*$ || -z "$requirement" ]] && continue

  pkg="${requirement%%[*}" # remove [extras] if present
  if ! pip show "$pkg" > /dev/null 2>&1; then
    echo "➕ Installing missing requirement: $requirement"
    pip install "$requirement"
  else
    echo "✅ $pkg already installed"
  fi
done < requirements.txt

# -----------------------------------
# 🌍 Environment Variable Injection
# -----------------------------------
if [ -f ".env" ]; then
  echo "🔁 Loading environment variables from .env"
  export $(grep -v '^#' .env | xargs)
else
  echo "⚠️  No .env file found – using default settings"
fi

# -----------------------------------
# 🚀 Run the FastAPI Service
# -----------------------------------
SERVICE_PORT="${PORT:-1337}"
echo "🚀 Starting service on port $SERVICE_PORT..."
exec "$VENV_DIR/bin/python" clip_service.py
