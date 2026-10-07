"""Launcher script for the Streamlit Financial Dashboard."""

import os
import subprocess
import sys

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print(f"Starting PetroPulse Dashboard with command: {' '.join(cmd)}")
    subprocess.run(cmd)
