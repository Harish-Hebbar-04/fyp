"""
run.py
───────
Starts the FastAPI backend and Streamlit UI simultaneously.

Usage:
  python run.py              → starts both API + UI
  python run.py --api-only   → API only  (port 8000)
  python run.py --ui-only    → UI only   (port 8501)
"""

import sys
import time
import argparse
import subprocess
import threading
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON      = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# Resolve executables inside the venv
VENV_BIN    = ROOT / ".venv" / "Scripts"
UVICORN     = str(VENV_BIN / "uvicorn.exe")   if (VENV_BIN / "uvicorn.exe").exists()    else "uvicorn"
STREAMLIT   = str(VENV_BIN / "streamlit.exe") if (VENV_BIN / "streamlit.exe").exists()  else "streamlit"

API_HOST    = "0.0.0.0"
API_PORT    = 8000
UI_PORT     = 8501


def stream_output(proc: subprocess.Popen, prefix: str):
    """Thread target: stream stdout from *proc* with a prefix label."""
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="")


def start_api() -> subprocess.Popen:
    print(f"[API] Starting FastAPI on http://localhost:{API_PORT} …")
    proc = subprocess.Popen(
        [
            UVICORN, "api.main:app",
            "--host", API_HOST,
            "--port", str(API_PORT),
            "--reload",
        ],
        cwd    = str(ROOT),
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text   = True,
        bufsize= 1,
    )
    t = threading.Thread(target=stream_output, args=(proc, "API"), daemon=True)
    t.start()
    return proc


def start_ui() -> subprocess.Popen:
    print(f"[UI ] Starting Streamlit on http://localhost:{UI_PORT} …")
    proc = subprocess.Popen(
        [
            STREAMLIT, "run", str(ROOT / "ui" / "app.py"),
            "--server.port",       str(UI_PORT),
            "--server.address",    "0.0.0.0",
            "--server.headless",   "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd    = str(ROOT),
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text   = True,
        bufsize= 1,
    )
    t = threading.Thread(target=stream_output, args=(proc, "UI "), daemon=True)
    t.start()
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--ui-only",  action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Stutter Detection System – Launcher")
    print("="*60)

    procs = []

    if not args.ui_only:
        procs.append(start_api())
        time.sleep(2)   # give API a head start

    if not args.api_only:
        procs.append(start_ui())

    print("\n" + "─"*60)
    if not args.ui_only:
        print(f"  API  →  http://localhost:{API_PORT}")
        print(f"  Docs →  http://localhost:{API_PORT}/docs")
    if not args.api_only:
        print(f"  UI   →  http://localhost:{UI_PORT}")
    print("─"*60)
    print("  Press Ctrl+C to stop all services.\n")

    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down …")
        for p in procs:
            p.terminate()
        print("  Done.")


if __name__ == "__main__":
    main()
