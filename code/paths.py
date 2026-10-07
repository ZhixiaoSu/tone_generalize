"""
Central path resolution for the tone_generalize project.

Set the environment variable TONE_RESULTS_DIR to override where checkpoints
and logs are saved and loaded.  If the variable is not set, falls back to a
'results/' directory relative to the current working directory.

Example (PowerShell):
    $env:TONE_RESULTS_DIR = 'C:\\Users\\zhixi\\Documents\\data\\results\\tone_generalize'

Example (.env at project root, loaded by python-dotenv):
    TONE_RESULTS_DIR=C:\\Users\\zhixi\\Documents\\data\\results\\tone_generalize
"""

import os
from pathlib import Path


def results_dir() -> Path:
    """Return the root results directory, controlled by TONE_RESULTS_DIR."""
    return Path(os.environ.get('TONE_RESULTS_DIR', 'results'))
