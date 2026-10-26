"""
Run bifurcation analysis on transfer-learning checkpoints.

Usage:
    python run_bifurcation_analysis.py [extra args forwarded to bifurcation_analysis.py]

Examples:
    python run_bifurcation_analysis.py
    python run_bifurcation_analysis.py --n-alpha 11 --show
    python run_bifurcation_analysis.py --n-iters 4000
"""
import os
import sys
import subprocess
from pathlib import Path

# ── load TONE_RESULTS_DIR from project .env ───────────────────────────────────
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            key, _, val = line.partition('=')
            os.environ.setdefault(key.strip(), val.strip())

results_dir = Path(os.environ.get('TONE_RESULTS_DIR', ''))
if not results_dir:
    sys.exit('TONE_RESULTS_DIR is not set. Add it to .env or export it.')

# ── checkpoints to analyse: (ckpt_path, task_a, task_b) ─────────────────────
CHECKPOINTS = [
    # # pure tone → HCT transfer model
    # (results_dir / 'transfer_pure_to_harmonic/context+sensory'
    #                '/harmonic_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint_init.pkl',
    #  'pure', 'harmonic'),

    # # white-noise → HCT transfer model
    # (results_dir / 'transfer_wn_to_harmonic/context+sensory'
    #                '/harmonic_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint_init.pkl',
    #  'wn', 'harmonic'),
    
    # (results_dir / 'transfer_pure_to_harmonic/context+sensory/harmonic/harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_init.pkl',
    #  'pure', 'harmonic'),
    # (results_dir / 'transfer_pure_to_harmonic/context+sensory/harmonic/harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_10000.pkl',
    #  'pure', 'harmonic'),

    # (results_dir / 'pure-harmonic/pure-harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_8000.pkl',
    #  'pure', 'harmonic'),

    # (results_dir / 'pure-harmonic/pure-harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_20000.pkl',
    #  'pure', 'harmonic'),
    # (results_dir / 'transfer_pure_to_harmonic/context+sensory/harmonic/harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_10000.pkl',
    #  'pure', 'harmonic'),
    # (results_dir / 'transfer_pure_to_harmonic/all/harmonic/harmonic_0_lr0.0001_bs64_noise0.1_nrnn128_src4000/checkpoint_10000.pkl',
    #  'pure', 'harmonic'),
    # (results_dir / 'transfer_pure_to_harmonic/context+sensory/harmonic/harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_10000.pkl',
    #  'pure', 'harmonic'),
    (results_dir / 'transfer_wn_to_harmonic/all/harmonic/harmonic_0_lr0.0001_bs64_noise0.1_nrnn128_src4000/checkpoint_10000.pkl',
      'wn', 'harmonic'),
  
]

# ── run ───────────────────────────────────────────────────────────────────────
script     = Path(__file__).parent / 'bifurcation_analysis.py'
extra_args = sys.argv[1:]   # forward everything after this script's name

print(f'Results dir : {results_dir}')
print(f'Checkpoints : {len(CHECKPOINTS)}')
print()

failed = []

for ckpt, task_a, task_b in CHECKPOINTS:
    if not ckpt.exists():
        print(f'SKIP (not found): {ckpt}')
        failed.append((ckpt, task_a, task_b))
        continue

    print('=' * 68)
    print(f'Running: {ckpt}  ({task_a} → {task_b})')
    print('=' * 68)

    ret = subprocess.run(
        [sys.executable, str(script), str(ckpt), '--tasks', task_a, task_b] + extra_args
    )
    if ret.returncode != 0:
        print(f'WARNING: analysis failed for {ckpt} (exit {ret.returncode})')
        failed.append((ckpt, task_a, task_b))

# ── summary ───────────────────────────────────────────────────────────────────
print()
print(f'Done. {len(CHECKPOINTS)} checkpoints attempted.')
if failed:
    print('Failed or skipped:')
    for f, a, b in failed:
        print(f'  [{a}→{b}] {f}')
