"""Ablation comparison: Fourier features vs plain MLP.

Fetches runs from wandb project 'helmholtz-pinn-ablation' and generates
overlay plots comparing convergence and final accuracy.

Usage:
    source .env && .venv/bin/python scripts/plot_ablation.py               # all ka values
    source .env && .venv/bin/python scripts/plot_ablation.py --ka ka-pi    # specific ka
    source .env && .venv/bin/python scripts/plot_ablation.py --ka ka-2pi
"""

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

WANDB_PROJECT = "helmholtz-pinn-ablation"
FF_COLOR = "#1a56db"   # blue
MLP_COLOR = "#dc2626"  # red
OUTPUT_DIR = "outputs"


def fetch_runs(project, ka_filter=None):
    """Fetch ablation runs, optionally filtered by ka suffix (e.g. 'ka-pi', 'ka-2pi')."""
    api = wandb.Api()
    entity = api.default_entity
    runs = api.runs(f"{entity}/{project}")

    # Group runs by ka value
    groups = {}
    for run in runs:
        name = run.name or ""
        if "ff-pinn" not in name and "plain-mlp" not in name:
            continue
        # Extract ka suffix from run name (e.g. "ff-pinn-ka-pi" -> "ka-pi")
        for part in ["ka-3pi", "ka-2pi", "ka-pi"]:
            if part in name:
                ka_key = part
                break
        else:
            continue
        if ka_filter and ka_key != ka_filter:
            continue
        if ka_key not in groups:
            groups[ka_key] = {}
        if "ff-pinn" in name:
            groups[ka_key]["ff"] = run
        elif "plain-mlp" in name:
            groups[ka_key]["mlp"] = run

    # Validate each group has both runs
    valid = {}
    for ka_key, run_map in groups.items():
        if "ff" in run_map and "mlp" in run_map:
            valid[ka_key] = run_map
        else:
            print(f"  Warning: {ka_key} missing a run, skipping")

    if not valid:
        all_names = [r.name for r in runs]
        print(f"Found runs: {all_names}")
        print("Need paired runs with 'ff-pinn-<ka>' and 'plain-mlp-<ka>' naming")
        sys.exit(1)

    return valid


def get_history(run):
    """Extract history rows from a run."""
    rows = list(run.scan_history())
    print(f"  {run.name}: {len(rows)} rows")
    return rows


def extract_series(rows, key):
    """Extract (step, value) pairs for a given metric key, sorted by step."""
    steps, vals = [], []
    for r in rows:
        s = r.get("_step")
        v = r.get(key)
        if s is not None and v is not None:
            steps.append(s)
            vals.append(v)
    if steps:
        steps, vals = zip(*sorted(zip(steps, vals)))
        steps, vals = list(steps), list(vals)
    return steps, vals


def plot_loss_comparison(ff_rows, mlp_rows):
    """Overlay total loss curves for both variants."""
    fig = go.Figure()

    for rows, name, color, dash in [
        (ff_rows, "FF-PINN (Fourier)", FF_COLOR, "solid"),
        (mlp_rows, "Plain MLP", MLP_COLOR, "solid"),
    ]:
        steps, vals = extract_series(rows, "loss/total")
        fig.add_trace(go.Scatter(
            x=steps, y=vals, mode="lines", name=name,
            line=dict(color=color, width=2, dash=dash),
        ))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Ablation: Total Loss Convergence", **_TITLE_STYLE),
        xaxis_title="Epoch", yaxis_title="Total Loss",
        yaxis_type="log", width=900, height=500,
    )
    return fig


def plot_l2_comparison(ff_rows, mlp_rows):
    """Overlay L2 relative error curves."""
    fig = go.Figure()

    for rows, name, color in [
        (ff_rows, "FF-PINN (Fourier)", FF_COLOR),
        (mlp_rows, "Plain MLP", MLP_COLOR),
    ]:
        steps, vals = extract_series(rows, "eval/l2_rel")
        fig.add_trace(go.Scatter(
            x=steps, y=vals, mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=5),
        ))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Ablation: L2 Relative Error vs Analytic", **_TITLE_STYLE),
        xaxis_title="Epoch", yaxis_title="L2 Relative Error",
        yaxis_type="log", width=900, height=500,
    )
    return fig


def plot_final_bar(ff_rows, mlp_rows):
    """Bar chart of final L2 and max errors."""
    metrics = {}
    for label, rows in [("FF-PINN", ff_rows), ("Plain MLP", mlp_rows)]:
        # Get last eval row
        last_l2 = None
        last_max = None
        for r in rows:
            l2 = r.get("eval/l2_rel")
            mx = r.get("eval/max_err")
            if l2 is not None:
                last_l2 = l2
                last_max = mx
        metrics[label] = {"l2_rel": last_l2, "max_err": last_max}

    labels = list(metrics.keys())
    l2_vals = [metrics[l]["l2_rel"] for l in labels]
    max_vals = [metrics[l]["max_err"] for l in labels]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("L2 Relative Error", "Max Error"))

    fig.add_trace(go.Bar(
        x=labels, y=l2_vals, marker_color=[FF_COLOR, MLP_COLOR],
        text=[f"{v:.2e}" for v in l2_vals], textposition="outside",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=labels, y=max_vals, marker_color=[FF_COLOR, MLP_COLOR],
        text=[f"{v:.2e}" for v in max_vals], textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Ablation: Final Error Comparison (ka=pi)", **_TITLE_STYLE),
        width=900, height=450,
    )
    return fig, metrics


def plot_pde_bc_comparison(ff_rows, mlp_rows):
    """Overlay PDE and BC loss components."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PDE Loss", "BC Loss"))

    for rows, name, color in [
        (ff_rows, "FF-PINN", FF_COLOR),
        (mlp_rows, "Plain MLP", MLP_COLOR),
    ]:
        steps, vals = extract_series(rows, "loss/pde")
        fig.add_trace(go.Scatter(
            x=steps, y=vals, mode="lines", name=name,
            line=dict(color=color, width=1.5),
        ), row=1, col=1)

        steps, vals = extract_series(rows, "loss/bc")
        fig.add_trace(go.Scatter(
            x=steps, y=vals, mode="lines", name=name,
            line=dict(color=color, width=1.5),
            showlegend=False,
        ), row=1, col=2)

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Ablation: PDE and BC Loss Components", **_TITLE_STYLE),
        width=1000, height=450,
    )
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot ablation comparison from wandb")
    parser.add_argument("--ka", type=str, default=None,
                        help="Filter by ka value: ka-pi, ka-2pi, ka-3pi (default: all)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Fetching runs from wandb project: {WANDB_PROJECT}")
    groups = fetch_runs(WANDB_PROJECT, ka_filter=args.ka)

    all_metrics = {}
    for ka_key in sorted(groups.keys()):
        run_map = groups[ka_key]
        print(f"\n=== {ka_key} ===")
        print("Downloading history...")
        ff_rows = get_history(run_map["ff"])
        mlp_rows = get_history(run_map["mlp"])

        suffix = f"_{ka_key}"

        # 1. Total loss comparison
        fig = plot_loss_comparison(ff_rows, mlp_rows)
        fig.update_layout(title=dict(text=f"Ablation: Total Loss Convergence ({ka_key})", **_TITLE_STYLE))
        path = os.path.join(args.output_dir, f"ablation_loss_comparison{suffix}.html")
        fig.write_html(path)
        print(f"Saved: {path}")

        # 2. L2 error comparison
        fig = plot_l2_comparison(ff_rows, mlp_rows)
        fig.update_layout(title=dict(text=f"Ablation: L2 Relative Error ({ka_key})", **_TITLE_STYLE))
        path = os.path.join(args.output_dir, f"ablation_l2_comparison{suffix}.html")
        fig.write_html(path)
        print(f"Saved: {path}")

        # 3. Final bar chart
        fig, metrics = plot_final_bar(ff_rows, mlp_rows)
        fig.update_layout(title=dict(text=f"Ablation: Final Error Comparison ({ka_key})", **_TITLE_STYLE))
        path = os.path.join(args.output_dir, f"ablation_final_bar{suffix}.html")
        fig.write_html(path)
        print(f"Saved: {path}")

        # 4. PDE + BC component comparison
        fig = plot_pde_bc_comparison(ff_rows, mlp_rows)
        fig.update_layout(title=dict(text=f"Ablation: PDE and BC Loss Components ({ka_key})", **_TITLE_STYLE))
        path = os.path.join(args.output_dir, f"ablation_components{suffix}.html")
        fig.write_html(path)
        print(f"Saved: {path}")

        all_metrics[ka_key] = metrics

    # Summary
    print("\n=== Ablation Summary ===")
    for ka_key, metrics in sorted(all_metrics.items()):
        print(f"\n  {ka_key}:")
        for label, m in metrics.items():
            print(f"    {label:12s}  L2_rel={m['l2_rel']:.4e}  max_err={m['max_err']:.4e}")
        ff_l2 = metrics["FF-PINN"]["l2_rel"]
        mlp_l2 = metrics["Plain MLP"]["l2_rel"]
        if ff_l2 and mlp_l2 and ff_l2 < mlp_l2:
            ratio = mlp_l2 / ff_l2
            print(f"    -> Fourier features reduce L2 error by {ratio:.1f}x")

    print(f"\nAll ablation plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
