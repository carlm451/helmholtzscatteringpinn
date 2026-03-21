#!/usr/bin/env bash
# Post-sweep evaluation: run eval_suite on all production checkpoints.
# Must be run AFTER run_prod_sweep.sh completes.
#
# Usage: bash run_prod_eval.sh

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
CKPT_DIR="checkpoints"

echo "=== Production Evaluation Suite ==="
echo ""

for KA_LABEL in "ka0.50" "ka1.00" "ka3.14" "ka6.28" "ka9.42"; do
    # Get latest matching checkpoint (by modification time)
    CKPT=$(ls -t ${CKPT_DIR}/helmholtz_${KA_LABEL}*_lbfgs.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "WARNING: No checkpoint found for $KA_LABEL, skipping"
        continue
    fi
    echo ">>> Evaluating: $CKPT"
    $PYTHON scripts/eval_suite.py "$CKPT" \
        --run-id "prod-${KA_LABEL}" \
        --outer-boundary circle --abc-order 2 \
        --device $DEVICE
    echo ""
done

echo "=== All evaluations complete ==="
echo "Outputs in: outputs/eval_prod-ka*/"
