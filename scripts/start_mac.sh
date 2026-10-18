#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ ! -d .venv ]; then
  if command -v python3.12 >/dev/null 2>&1; then PY=python3.12
  elif command -v python3.11 >/dev/null 2>&1; then PY=python3.11
  else PY=python3; fi
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run academic_v2_demo.py --server.port 8501
