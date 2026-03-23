#!/bin/bash
# Ablation experiment: Fourier features vs plain MLP
# Circle boundary, BGT2 ABC, param-matched networks at each ka
set -e

source .env
COMMON="--outer-boundary circle --abc-order 2 --wandb-project helmholtz-pinn-ablation"

echo "============================================"
echo "  ka=pi  (4 layers, ~296k params each)"
echo "============================================"

echo "=== FF-PINN ka=pi (Fourier features, hidden=256) ==="
.venv/bin/python main.py $COMMON --run-name "ff-pinn-ka-pi"

echo ""
echo "=== Plain MLP ka=pi (no Fourier features, hidden=272) ==="
.venv/bin/python main.py $COMMON --no-fourier --hidden-neurons 272 --run-name "plain-mlp-ka-pi"

echo ""
echo "============================================"
echo "  ka=2pi  (6 layers, ~428k params each)"
echo "============================================"

echo "=== FF-PINN ka=2pi (Fourier features, hidden=256) ==="
.venv/bin/python main.py --ka 6.2832 $COMMON --run-name "ff-pinn-ka-2pi"

echo ""
echo "=== Plain MLP ka=2pi (no Fourier features, hidden=266) ==="
.venv/bin/python main.py --ka 6.2832 $COMMON --no-fourier --hidden-neurons 266 --run-name "plain-mlp-ka-2pi"

echo ""
echo "=== Ablation runs complete. Check wandb project: helmholtz-pinn-ablation ==="
echo "Generate comparison plots with:"
echo "  source .env && .venv/bin/python scripts/plot_ablation.py"
