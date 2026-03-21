#!/usr/bin/env bash
# Production sweep: 5 ka values across 4 GPUs, circle boundary + BGT2 ABC, L=3.
# Checkpoints get unique timestamps — no collisions.
#
# GPU layout:
#   GPU 0: ka=0.5 (~10min) then ka=3pi (~90min)
#   GPU 1: ka=1.0 (~10min)
#   GPU 2: ka=pi  (~20min)
#   GPU 3: ka=2pi (~75min)
# Total wall time: ~1.5-2 hours
#
# Usage: bash run_prod_sweep.sh
#        DEVICE=mps bash run_prod_sweep.sh   # single-GPU fallback (sequential)

set -euo pipefail
source .env

PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
COMMON="--outer-boundary circle --abc-order 2 --L 3.0 --wandb-project helmholtz-pinn-prod"

echo "=== HelmholtzPINN Production Sweep ==="
echo "Config: L=3, circle boundary, BGT2 ABC, device=$DEVICE"
echo "Start: $(date)"
echo ""

if [ "$DEVICE" = "cuda" ]; then
    echo "4-GPU parallel mode"
    echo ""

    # GPU 0: fastest then hardest (sequential on same GPU)
    (
        echo ">>> [GPU 0] ka=0.5 — sub-wavelength"
        CUDA_VISIBLE_DEVICES=0 $PYTHON main.py --ka 0.5 $COMMON --device cuda \
            --hidden-neurons 256 --hidden-layers 4 --fourier-features 64 \
            --n-interior 10000 --n-boundary 200 --n-outer 400 \
            --adam-epochs 10000 --lbfgs-epochs 200 \
            --run-name "prod-ka0.50-circ-bgt2" \
        && \
        echo ">>> [GPU 0] ka=3pi — 3 wavelengths/radius (loss-rebalanced)" && \
        CUDA_VISIBLE_DEVICES=0 $PYTHON main.py --ka 9.42478 $COMMON --device cuda \
            --hidden-neurons 512 --hidden-layers 6 --fourier-features 128 \
            --n-interior 20000 --n-boundary 400 --n-outer 600 \
            --adam-lr 3e-4 --adam-epochs 30000 --lbfgs-epochs 300 \
            --lambda-pde 0.1 --lambda-bc 20.0 --lambda-abc 1.0 \
            --run-name "prod-ka_3pi-circ-bgt2"
    ) 2>&1 | sed 's/^/[GPU0] /' &

    # GPU 1: ka=1.0
    (
        echo ">>> [GPU 1] ka=1.0 — transition regime"
        CUDA_VISIBLE_DEVICES=1 $PYTHON main.py --ka 1.0 $COMMON --device cuda \
            --hidden-neurons 256 --hidden-layers 4 --fourier-features 64 \
            --n-interior 10000 --n-boundary 200 --n-outer 400 \
            --adam-epochs 10000 --lbfgs-epochs 200 \
            --run-name "prod-ka1.00-circ-bgt2"
    ) 2>&1 | sed 's/^/[GPU1] /' &

    # GPU 2: ka=pi
    (
        echo ">>> [GPU 2] ka=pi — 1 wavelength/radius"
        CUDA_VISIBLE_DEVICES=2 $PYTHON main.py --ka 3.14159 $COMMON --device cuda \
            --hidden-neurons 256 --hidden-layers 4 --fourier-features 64 \
            --n-interior 10000 --n-boundary 200 --n-outer 400 \
            --adam-epochs 10000 --lbfgs-epochs 200 \
            --run-name "prod-ka_pi-circ-bgt2"
    ) 2>&1 | sed 's/^/[GPU2] /' &

    # GPU 3: ka=2pi
    (
        echo ">>> [GPU 3] ka=2pi — 2 wavelengths/radius (loss-rebalanced)"
        CUDA_VISIBLE_DEVICES=3 $PYTHON main.py --ka 6.28318 $COMMON --device cuda \
            --hidden-neurons 384 --hidden-layers 6 --fourier-features 96 \
            --n-interior 15000 --n-boundary 300 --n-outer 500 \
            --adam-lr 5e-4 --adam-epochs 20000 --lbfgs-epochs 300 \
            --lambda-pde 0.25 --lambda-bc 15.0 --lambda-abc 1.0 \
            --run-name "prod-ka_2pi-circ-bgt2"
    ) 2>&1 | sed 's/^/[GPU3] /' &

    echo "All 5 runs launched. Waiting for completion..."
    wait
else
    echo "Single-device sequential mode (device=$DEVICE)"
    echo ""

    echo ">>> [1/5] ka=0.5 — sub-wavelength"
    $PYTHON main.py --ka 0.5 $COMMON --device $DEVICE \
        --hidden-neurons 256 --hidden-layers 4 --fourier-features 64 \
        --n-interior 10000 --n-boundary 200 --n-outer 400 \
        --adam-epochs 10000 --lbfgs-epochs 200 \
        --run-name "prod-ka0.50-circ-bgt2"

    echo ">>> [2/5] ka=1.0 — transition regime"
    $PYTHON main.py --ka 1.0 $COMMON --device $DEVICE \
        --hidden-neurons 256 --hidden-layers 4 --fourier-features 64 \
        --n-interior 10000 --n-boundary 200 --n-outer 400 \
        --adam-epochs 10000 --lbfgs-epochs 200 \
        --run-name "prod-ka1.00-circ-bgt2"

    echo ">>> [3/5] ka=pi — 1 wavelength/radius"
    $PYTHON main.py --ka 3.14159 $COMMON --device $DEVICE \
        --hidden-neurons 256 --hidden-layers 4 --fourier-features 64 \
        --n-interior 10000 --n-boundary 200 --n-outer 400 \
        --adam-epochs 10000 --lbfgs-epochs 200 \
        --run-name "prod-ka_pi-circ-bgt2"

    echo ">>> [4/5] ka=2pi — 2 wavelengths/radius (loss-rebalanced)"
    $PYTHON main.py --ka 6.28318 $COMMON --device $DEVICE \
        --hidden-neurons 384 --hidden-layers 6 --fourier-features 96 \
        --n-interior 15000 --n-boundary 300 --n-outer 500 \
        --adam-lr 5e-4 --adam-epochs 20000 --lbfgs-epochs 300 \
        --lambda-pde 0.25 --lambda-bc 15.0 --lambda-abc 1.0 \
        --run-name "prod-ka_2pi-circ-bgt2"

    echo ">>> [5/5] ka=3pi — 3 wavelengths/radius (loss-rebalanced)"
    $PYTHON main.py --ka 9.42478 $COMMON --device $DEVICE \
        --hidden-neurons 512 --hidden-layers 6 --fourier-features 128 \
        --n-interior 20000 --n-boundary 400 --n-outer 600 \
        --adam-lr 3e-4 --adam-epochs 30000 --lbfgs-epochs 300 \
        --lambda-pde 0.1 --lambda-bc 20.0 --lambda-abc 1.0 \
        --run-name "prod-ka_3pi-circ-bgt2"
fi

echo ""
echo "=== Production sweep complete at $(date) ==="
echo "Checkpoints saved in checkpoints/ directory"
echo "Check wandb dashboard for training curves"
