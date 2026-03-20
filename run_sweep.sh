#!/bin/bash
set -e
source .env

PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

echo "=== HelmholtzPINN Production Runs: Circle+BGT2 across ka values ==="
echo "Config: L=3, circle boundary, 2nd-order ABC, 10K Adam + 200 L-BFGS, device=$DEVICE"
echo ""

# ka=0.5 — sub-wavelength
echo ">>> Run 1/5: ka=0.5"
$PYTHON main.py --ka 0.5 --L 3.0 --n-interior 10000 --n-outer 400 \
  --outer-boundary circle --abc-order 2 --device $DEVICE \
  --adam-epochs 10000 --lbfgs-epochs 200 --run-name "prod-ka0.5-circle-abc2"

# ka=1.0 — transition regime
echo ">>> Run 2/5: ka=1.0"
$PYTHON main.py --ka 1.0 --L 3.0 --n-interior 10000 --n-outer 400 \
  --outer-boundary circle --abc-order 2 --device $DEVICE \
  --adam-epochs 10000 --lbfgs-epochs 200 --run-name "prod-ka1.0-circle-abc2"

# ka=pi — 1 wavelength per radius (our main benchmark)
echo ">>> Run 3/5: ka=pi"
$PYTHON main.py --ka 3.14159 --L 3.0 --n-interior 10000 --n-outer 400 \
  --outer-boundary circle --abc-order 2 --device $DEVICE \
  --adam-epochs 10000 --lbfgs-epochs 200 --run-name "prod-ka_pi-circle-abc2"

# ka=2pi — 2 wavelengths per radius
echo ">>> Run 4/5: ka=2pi"
$PYTHON main.py --ka 6.28318 --L 3.0 --n-interior 15000 --n-outer 500 \
  --outer-boundary circle --abc-order 2 --device $DEVICE \
  --hidden-layers 6 \
  --adam-epochs 10000 --lbfgs-epochs 200 --run-name "prod-ka_2pi-circle-abc2"

# ka=3pi — 3 wavelengths per radius
echo ">>> Run 5/5: ka=3pi"
$PYTHON main.py --ka 9.42478 --L 3.0 --n-interior 20000 --n-outer 600 \
  --outer-boundary circle --abc-order 2 --device $DEVICE \
  --hidden-layers 6 --hidden-neurons 512 --fourier-features 128 \
  --adam-epochs 10000 --lbfgs-epochs 200 --run-name "prod-ka_3pi-circle-abc2"

echo ""
echo "All production runs complete. Check wandb dashboard."
