#!/usr/bin/env bash
# Run fixed-point analysis on a list of checkpoints.
# Usage: bash run_fixed_point_analysis.sh [extra args passed to fixed_point_analysis.py]
#
# Add or remove entries in CHECKPOINTS to control which models are analysed.
# Any extra CLI argument is forwarded to fixed_point_analysis.py, e.g.:
#   bash run_fixed_point_analysis.sh --n-iters 12000 --show

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── load TONE_RESULTS_DIR from project .env ───────────────────────────────────
ENV_FILE="$(dirname "$SCRIPT_DIR")/.env"
if [[ -f "$ENV_FILE" ]]; then
    # strip Windows-style carriage returns, export key=value lines only
    set -a
    # shellcheck disable=SC1090
    source <(grep -E '^[A-Z_]+=.*' "$ENV_FILE" | tr -d '\r')
    set +a
fi

: "${TONE_RESULTS_DIR:?TONE_RESULTS_DIR is not set. Add it to .env or export it.}"

# ── checkpoints to analyse ────────────────────────────────────────────────────
CHECKPOINTS=(
    # pure tone model
    "$TONE_RESULTS_DIR/pure/pure_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint.pkl"

    # white-noise model
    "$TONE_RESULTS_DIR/wn/wn_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint.pkl"

    # harmonic complex tone (HCT) model
    "$TONE_RESULTS_DIR/harmonic/harmonic_0_lr5e-05_bs128_noise0.5_nrnn128/checkpoint.pkl"
)

# ── run ───────────────────────────────────────────────────────────────────────
echo "Results dir : $TONE_RESULTS_DIR"
echo "Checkpoints : ${#CHECKPOINTS[@]}"
echo ""

FAILED=()

for CKPT in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$CKPT" ]]; then
        echo "SKIP (not found): $CKPT"
        FAILED+=("$CKPT")
        continue
    fi

    echo "════════════════════════════════════════════════════════════════"
    echo "Running: $CKPT"
    echo "════════════════════════════════════════════════════════════════"

    python "$SCRIPT_DIR/fixed_point_analysis.py" "$CKPT" "$@"
done

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Done. ${#CHECKPOINTS[@]} checkpoints attempted."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Skipped (missing):"
    for F in "${FAILED[@]}"; do echo "  $F"; done
fi
