#!/usr/bin/env python
"""
Run Streamlit app from project root to ensure proper module resolution.
"""
import sys
import subprocess
import os
from pathlib import Path

# Get project root
project_root = Path(__file__).parent

# Run streamlit with project root in PYTHONPATH
env = dict(os.environ)
pythonpath = str(project_root)
if env.get('PYTHONPATH'):
    pythonpath = f"{pythonpath};{env['PYTHONPATH']}"
env['PYTHONPATH'] = pythonpath

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "src/app/streamlit_app.py"],
    cwd=str(project_root),
    env=env
)
