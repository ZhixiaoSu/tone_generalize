"""
Run fixed-point analysis on a list of checkpoints.

Usage:
    python run_fixed_point_analysis.py [extra args forwarded to fixed_point_analysis.py]

Examples:
    python run_fixed_point_analysis.py
    python run_fixed_point_analysis.py --n-iters 12000 --show
"""
import os
import sys
import subprocess
from pathlib import Path

# ── load TONE_RESULTS_DIR from .env if not already set ───────────────────────
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

# ── checkpoints to analyse ────────────────────────────────────────────────────
CHECKPOINTS = [
    # # pure tone model
    # results_dir / 'pure/pure_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint_8000.pkl',

    # # white-noise model
    # results_dir / 'wn/wn_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint_6000.pkl',

    # # harmonic complex tone (HCT) model
    # results_dir / 'harmonic/harmonic_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint_6000.pkl',

    # # train all together models
    # results_dir / 'pure/pure_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint.pkl',
    # results_dir / 'pure/pure_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint_8000.pkl'
    # results_dir / 'transfer_pure_to_harmonic/context+sensory/harmonic_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint.pkl',
    # results_dir / 'transfer_pure_to_harmonic/context+sensory/harmonic_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint_init.pkl',
    # results_dir / 'wn/wn_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_14000.pkl',
    # results_dir / 'pure/pure_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_14000.pkl',
    # results_dir / 'harmonic/harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_14000.pkl',
    # results_dir /'harmonic/harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_8000.pkl',
    # results_dir / 'pure/pure_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_8000.pkl',
    # results_dir / 'wn/wn_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_8000.pkl',
    # results_dir / 'pure-harmonic/pure-harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_10000.pkl',
    # results_dir / 'pure-harmonic/pure-harmonic_0_lr0.0001_bs64_noise0.1_nrnn128/checkpoint_20000.pkl',
    results_dir / 'transfer_pure_to_harmonic/all/harmonic/harmonic_0_lr0.0001_bs64_noise0.1_nrnn128_src4000/checkpoint_10000.pkl'
]

# ── run ───────────────────────────────────────────────────────────────────────
script = Path(__file__).parent / 'fixed_point_analysis.py'
extra_args = sys.argv[1:]   # forward everything after this script's name

print(f'Results dir : {results_dir}')
print(f'Checkpoints : {len(CHECKPOINTS)}')
print()

failed = []

for ckpt in CHECKPOINTS:
    if not ckpt.exists():
        print(f'SKIP (not found): {ckpt}')
        failed.append(ckpt)
        continue

    print('=' * 68)
    print(f'Running: {ckpt}')
    print('=' * 68)

    ret = subprocess.run([sys.executable, str(script), str(ckpt)] + extra_args)
    if ret.returncode != 0:
        print(f'WARNING: analysis failed for {ckpt} (exit {ret.returncode})')
        failed.append(ckpt)

# ── summary ───────────────────────────────────────────────────────────────────
print()
print(f'Done. {len(CHECKPOINTS)} checkpoints attempted.')
if failed:
    print('Failed or skipped:')
    for f in failed:
        print(f'  {f}')
