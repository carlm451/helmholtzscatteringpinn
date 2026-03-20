#!/usr/bin/env bash
# High-ka sweep: ka=2pi and ka=3pi with tuned hyperparameters.
# Previous sweep used uniform 10K Adam + 200 L-BFGS which failed for high ka
# (PDE loss explodes due to k^2 gradient scaling).
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash run_sweep_highka.sh

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
PYTHON=".venv/bin/python"
COMMON="--outer-boundary circle --abc-order 2 --device $DEVICE"

echo "=== High-ka Sweep ==="
echo "Device: $DEVICE"
echo ""

# --- ka=2pi, Run A: Conservative ---
echo "[1/4] ka=2pi Run A (conservative): lr=5e-4, 20K Adam, 300 L-BFGS"
$PYTHON main.py --ka 6.2832 $COMMON \
    --adam-lr 5e-4 \
    --adam-epochs 20000 \
    --lbfgs-epochs 300 \
    --n-interior 20000 \
    --n-outer 600 \
    --fourier-features 128 \
    --hidden-layers 6 \
    --hidden-neurons 256 \
    --run-name "highka-2pi-A"

# --- ka=2pi, Run B: Aggressive ---
echo "[2/4] ka=2pi Run B (aggressive): lr=3e-4, 30K Adam, 500 L-BFGS"
$PYTHON main.py --ka 6.2832 $COMMON \
    --adam-lr 3e-4 \
    --adam-epochs 30000 \
    --lbfgs-epochs 500 \
    --n-interior 30000 \
    --n-outer 800 \
    --fourier-features 128 \
    --hidden-layers 8 \
    --hidden-neurons 256 \
    --run-name "highka-2pi-B"

# --- ka=3pi, Run A: Conservative ---
echo "[3/4] ka=3pi Run A (conservative): lr=3e-4, 25K Adam, 400 L-BFGS"
$PYTHON main.py --ka 9.4248 $COMMON \
    --adam-lr 3e-4 \
    --adam-epochs 25000 \
    --lbfgs-epochs 400 \
    --n-interior 25000 \
    --n-outer 800 \
    --fourier-features 128 \
    --hidden-layers 6 \
    --hidden-neurons 512 \
    --run-name "highka-3pi-A"

# --- ka=3pi, Run B: Aggressive ---
echo "[4/4] ka=3pi Run B (aggressive): lr=2e-4, 40K Adam, 600 L-BFGS"
$PYTHON main.py --ka 9.4248 $COMMON \
    --adam-lr 2e-4 \
    --adam-epochs 40000 \
    --lbfgs-epochs 600 \
    --n-interior 40000 \
    --n-outer 1000 \
    --fourier-features 256 \
    --hidden-layers 8 \
    --hidden-neurons 512 \
    --run-name "highka-3pi-B"

echo ""
echo "=== High-ka sweep complete ==="
