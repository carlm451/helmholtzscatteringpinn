#!/bin/bash
# Intensive honeycomb + report-strengthening runs
# Target: unicorn (3 CUDA GPUs)
# Runs 4 jobs in parallel across GPUs 0/1/2
# Expected wall time: ~5.5 hours
set -e

source .env

HC_COMMON="--honeycomb --outer-boundary circle --abc-order 2 \
  --hidden-neurons 384 --hidden-layers 6 --fourier-features 128 \
  --n-interior 25000 --n-boundary 300 --n-outer 700 \
  --lambda-pde 5.0 --lambda-bc 10.0 \
  --adam-lr 0.0005 --adam-epochs 50000 --lbfgs-epochs 500 \
  --sampling cluster_bias --use-rad \
  --wandb-project helmholtz-pinn-honeycomb \
  --device cuda"

echo "============================================"
echo "  Launching 4 runs on 3 GPUs"
echo "  GPU 0: prod-ka2.00 -> prod-ka3pi-long"
echo "  GPU 1: hc-ka2.00-intensive-long"
echo "  GPU 2: hc-ka3.14-intensive-long"
echo "============================================"

# GPU 0: Quick prod ka=2 solid cylinder (~30 min), then ka=3pi long (~5 hr)
(
  echo "=== [GPU 0] prod-ka2.00-circ-bgt2 ==="
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py --ka 2.0 \
    --outer-boundary circle --abc-order 2 \
    --adam-epochs 15000 --lbfgs-epochs 200 \
    --wandb-project helmholtz-pinn-prod --device cuda \
    --run-name "prod-ka2.00-circ-bgt2" \
  && \
  echo "=== [GPU 0] prod-ka_3pi-circ-bgt2-long ===" && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py --ka 9.4248 \
    --outer-boundary circle --abc-order 2 \
    --hidden-neurons 512 --hidden-layers 6 --fourier-features 128 \
    --n-interior 25000 --n-boundary 400 --n-outer 700 \
    --adam-lr 0.0005 --adam-epochs 50000 --lbfgs-epochs 500 \
    --wandb-project helmholtz-pinn-prod --device cuda \
    --run-name "prod-ka_3pi-circ-bgt2-long"
) &
PID_GPU0=$!

# GPU 1: Honeycomb ka=2.0 intensive (~4-5 hr)
(
  echo "=== [GPU 1] hc-ka2.00-intensive-long ==="
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python main.py --ka 2.0 --L 4.0 $HC_COMMON \
    --run-name "hc-ka2.00-intensive-long"
) &
PID_GPU1=$!

# GPU 2: Honeycomb ka=pi intensive (~4-5 hr)
(
  echo "=== [GPU 2] hc-ka3.14-intensive-long ==="
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python main.py --ka 3.14159 --L 4.0 $HC_COMMON \
    --run-name "hc-ka3.14-intensive-long"
) &
PID_GPU2=$!

echo ""
echo "PIDs: GPU0=$PID_GPU0  GPU1=$PID_GPU1  GPU2=$PID_GPU2"
echo "Monitor: wandb projects helmholtz-pinn-prod and helmholtz-pinn-honeycomb"
echo ""

wait
echo "============================================"
echo "  All runs complete!"
echo "============================================"
