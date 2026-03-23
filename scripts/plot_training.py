"""Training curve visualization from wandb run history.

Usage:
    source .env && .venv/bin/python scripts/plot_training.py --run-path cmerrigan-alynix/helmholtz-pinn/le0ttokl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import wandb
except ImportError:
    print("wandb not installed. Install with: pip install wandb")
    sys.exit(1)

from helmholtz.visualize import _LAYOUT, _TITLE_STYLE


def fetch_history(run_path):
    """Fetch full run history from wandb."""
    api = wandb.Api()
    run = api.run(run_path)

    rows = []
    for row in run.scan_history():
        rows.append(dict(row))

    print(f"Fetched {len(rows)} history rows from {run_path}")
    return rows, run


def plot_loss_curves(rows, run_id, resample_every=2000):
    """4-panel loss components (log scale) with resampling event markers."""
    steps, total, pde, bc, abc = [], [], [], [], []

    for r in rows:
        s = r.get("_step")
        lt = r.get("loss/total")
        if s is not None and lt is not None:
            steps.append(s)
            total.append(lt)
            pde.append(r.get("loss/pde"))
            bc.append(r.get("loss/bc"))
            abc.append(r.get("loss/abc"))

    # Sort by step to prevent wraparound lines
    if steps:
        sorted_data = sorted(zip(steps, total, pde, bc, abc))
        steps, total, pde, bc, abc = [list(x) for x in zip(*sorted_data)]

    fig = go.Figure()
    for name, vals, color in [
        ("Total", total, "#1a56db"),
        ("PDE", pde, "#dc2626"),
        ("BC", bc, "#059669"),
        ("ABC", abc, "#d97706"),
    ]:
        fig.add_trace(go.Scatter(
            x=steps, y=vals, mode="lines", name=name,
            line=dict(color=color, width=1.5),
        ))

    # Add resampling event lines
    if resample_every and steps:
        max_step = max(steps)
        resample_steps = list(range(resample_every, int(max_step) + 1, resample_every))
        for rs in resample_steps:
            fig.add_vline(x=rs, line=dict(color="gray", dash="dot", width=0.8),
                          annotation_text="resample" if rs == resample_steps[0] else None,
                          annotation_position="top right")

    fig.update_layout(**_LAYOUT,
                      title=dict(text=f"Training Loss — {run_id}", **_TITLE_STYLE),
                      xaxis_title="Step", yaxis_title="Loss",
                      yaxis_type="log", width=900, height=500)
    return fig


def plot_eval_metrics(rows, run_id):
    """L2_rel and max_err over training steps."""
    steps, l2_rel, max_err = [], [], []

    for r in rows:
        s = r.get("_step")
        l2 = r.get("eval/l2_rel")
        if s is not None and l2 is not None:
            steps.append(s)
            l2_rel.append(l2)
            max_err.append(r.get("eval/max_err"))

    # Sort by step to prevent wraparound lines
    if steps:
        sorted_data = sorted(zip(steps, l2_rel, max_err))
        steps, l2_rel, max_err = [list(x) for x in zip(*sorted_data)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=steps, y=l2_rel, mode="lines+markers", name="L2_rel",
        line=dict(color="#1a56db", width=2), marker=dict(size=4),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=steps, y=max_err, mode="lines+markers", name="max_err",
        line=dict(color="#dc2626", width=2, dash="dash"), marker=dict(size=4),
    ), secondary_y=True)

    fig.update_layout(**_LAYOUT,
                      title=dict(text=f"Evaluation Metrics — {run_id}", **_TITLE_STYLE),
                      xaxis_title="Step", width=900, height=500)
    fig.update_yaxes(title_text="L2 Relative Error", secondary_y=False)
    fig.update_yaxes(title_text="Max Error", secondary_y=True)
    return fig


def plot_lr_gradnorm(rows, run_id):
    """LR schedule and gradient norm."""
    steps_lr, lr_vals = [], []
    steps_gn, gn_vals = [], []

    for r in rows:
        s = r.get("_step")
        if s is None:
            continue
        lr = r.get("lr")
        gn = r.get("grad_norm")
        if lr is not None:
            steps_lr.append(s)
            lr_vals.append(lr)
        if gn is not None:
            steps_gn.append(s)
            gn_vals.append(gn)

    # Sort by step to prevent wraparound lines
    if steps_lr:
        steps_lr, lr_vals = zip(*sorted(zip(steps_lr, lr_vals)))
        steps_lr, lr_vals = list(steps_lr), list(lr_vals)
    if steps_gn:
        steps_gn, gn_vals = zip(*sorted(zip(steps_gn, gn_vals)))
        steps_gn, gn_vals = list(steps_gn), list(gn_vals)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=steps_lr, y=lr_vals, mode="lines", name="Learning Rate",
        line=dict(color="#1a56db", width=2),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=steps_gn, y=gn_vals, mode="lines", name="Grad Norm",
        line=dict(color="#d97706", width=1.5),
    ), secondary_y=True)

    fig.update_layout(**_LAYOUT,
                      title=dict(text=f"LR Schedule & Gradient Norm — {run_id}", **_TITLE_STYLE),
                      xaxis_title="Step", width=900, height=500)
    fig.update_yaxes(title_text="Learning Rate", secondary_y=False)
    fig.update_yaxes(title_text="Gradient Norm", type="log", secondary_y=True)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from wandb")
    parser.add_argument("--run-path", type=str, required=True,
                        help="wandb run path: entity/project/run_id")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--resample-every", type=int, default=2000,
                        help="Resampling interval for vertical markers")
    args = parser.parse_args()

    run_id = args.run_path.split("/")[-1]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Fetching history for {args.run_path}...")
    rows, run = fetch_history(args.run_path)

    if not rows:
        print("No history rows found.")
        sys.exit(1)

    # Loss curves
    fig = plot_loss_curves(rows, run_id, args.resample_every)
    path = os.path.join(args.output_dir, f"{run_id}_loss_curves.html")
    fig.write_html(path)
    print(f"Saved: {path}")

    # Eval metrics
    fig = plot_eval_metrics(rows, run_id)
    path = os.path.join(args.output_dir, f"{run_id}_eval_metrics.html")
    fig.write_html(path)
    print(f"Saved: {path}")

    # LR + grad norm
    fig = plot_lr_gradnorm(rows, run_id)
    path = os.path.join(args.output_dir, f"{run_id}_lr_gradnorm.html")
    fig.write_html(path)
    print(f"Saved: {path}")

    print(f"\nAll training plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
