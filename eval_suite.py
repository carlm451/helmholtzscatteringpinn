"""Comprehensive post-training evaluation suite for HelmholtzPINN.

Usage:
    .venv/bin/python eval_suite.py checkpoints/helmholtz_ka3.14_lbfgs.pt [--run-id le0ttokl]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import HelmholtzConfig
from domain import ScatteringDomain
from network import HelmholtzPINN
from analytic import scattered_field, scattered_field_gradient
from evaluate import evaluate_against_analytic, check_symmetry
from visualize import (
    _LAYOUT, _TITLE_STYLE, _get_fields_on_grid, _scatterer_circle_trace,
    create_zoom_report,
)

# ──────────────────────────────────────────────────────────────
# Evaluation regions
# ──────────────────────────────────────────────────────────────

REGIONS = {
    "full": None,  # uses default (-L, L)
    "lit_side": {"x_range": None, "y_range": None},  # set dynamically
    "shadow_side": {"x_range": None, "y_range": None},
    "near_surface": {"x_range": (-1.5, 1.5), "y_range": (-1.5, 1.5)},
}

ANNULAR_BINS = [(1.0, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]


def _get_regions(config):
    L = config.L
    return {
        "full": {"x_range": (-L, L), "y_range": (-L, L)},
        "lit_side": {"x_range": (-L, -0.5), "y_range": (-L, L)},
        "shadow_side": {"x_range": (1.0, L), "y_range": (-L, L)},
        "near_surface": {"x_range": (-1.5, 1.5), "y_range": (-1.5, 1.5)},
    }


# ──────────────────────────────────────────────────────────────
# Regional evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_regions(model, domain, config, analytic_fn, grid_size=400):
    """Compute L2_rel and max_err per named region."""
    regions = _get_regions(config)
    results = {}
    for name, rng in regions.items():
        metrics = evaluate_against_analytic(
            model, domain, config, analytic_fn,
            x_range=rng["x_range"], y_range=rng["y_range"],
            grid_size=grid_size,
        )
        results[name] = metrics
    return results


# ──────────────────────────────────────────────────────────────
# Spatial error map
# ──────────────────────────────────────────────────────────────

def compute_spatial_error_map(model, domain, config, analytic_fn, grid_size=400):
    """Return 2D arrays: error_magnitude, u_err, v_err, x_arr, y_arr."""
    fields = _get_fields_on_grid(
        model, domain, config, analytic_fn,
        (-config.L, config.L), (-config.L, config.L), grid_size,
    )
    u_err = fields["u_pred"] - fields["u_exact"]
    v_err = fields["v_pred"] - fields["v_exact"]
    err_mag = np.sqrt(u_err**2 + v_err**2)

    x_arr = np.linspace(-config.L, config.L, grid_size)
    y_arr = np.linspace(-config.L, config.L, grid_size)
    return err_mag, u_err, v_err, x_arr, y_arr


# ──────────────────────────────────────────────────────────────
# Radial error profile
# ──────────────────────────────────────────────────────────────

def compute_radial_error_profile(err_mag, config, bins=None):
    """Bin error by radius from origin. Returns list of dicts with r_lo, r_hi, mean, max, count."""
    if bins is None:
        bins = ANNULAR_BINS
    gs = err_mag.shape[0]
    x_arr = np.linspace(-config.L, config.L, gs)
    y_arr = np.linspace(-config.L, config.L, gs)
    xx, yy = np.meshgrid(x_arr, y_arr, indexing="xy")
    rr = np.sqrt(xx**2 + yy**2)

    results = []
    for r_lo, r_hi in bins:
        mask = (rr >= r_lo) & (rr < r_hi) & ~np.isnan(err_mag)
        if mask.sum() == 0:
            results.append({"r_lo": r_lo, "r_hi": r_hi, "mean": float("nan"),
                            "max": float("nan"), "count": 0})
            continue
        vals = err_mag[mask]
        results.append({
            "r_lo": r_lo, "r_hi": r_hi,
            "mean": float(np.mean(vals)),
            "max": float(np.max(vals)),
            "count": int(mask.sum()),
        })
    return results


# ──────────────────────────────────────────────────────────────
# Angular error profile
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_angular_error_profile(model, config, analytic_fn, r_values, n_angles=360):
    """Error vs theta at fixed radii. Returns dict {r: {theta, error}}."""
    model.eval()
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    results = {}
    for r in r_values:
        x_np = r * np.cos(theta)
        y_np = r * np.sin(theta)
        x_t = torch.tensor(x_np, dtype=torch.float32, device=config.device)
        y_t = torch.tensor(y_np, dtype=torch.float32, device=config.device)

        u_pred, v_pred = model(x_t, y_t)
        u_pred = u_pred.cpu().numpy()
        v_pred = v_pred.cpu().numpy()

        phi_exact = analytic_fn(x_np, y_np, config.k, config.a, config.n_series_terms)
        u_exact = np.real(phi_exact)
        v_exact = np.imag(phi_exact)

        err = np.sqrt((u_pred - u_exact)**2 + (v_pred - v_exact)**2)
        results[r] = {"theta": theta, "error": err}
    model.train()
    return results


# ──────────────────────────────────────────────────────────────
# BC satisfaction
# ──────────────────────────────────────────────────────────────

def check_bc_satisfaction(model, config, n_points=1000):
    """Neumann BC residual vs theta on scatterer surface using autograd."""
    model.train()  # need grad
    a = config.a
    k = config.k
    theta = torch.linspace(0, 2 * np.pi, n_points, device=config.device)
    x = (a * torch.cos(theta)).requires_grad_(True)
    y = (a * torch.sin(theta)).requires_grad_(True)
    nx = torch.cos(theta)
    ny = torch.sin(theta)

    u, v = model(x, y)
    ones = torch.ones_like(u)

    du_dx = torch.autograd.grad(u, x, ones, create_graph=False, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, ones, create_graph=False, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, ones, create_graph=False, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, ones, create_graph=False, retain_graph=True)[0]

    du_dn = nx * du_dx + ny * du_dy
    dv_dn = nx * dv_dx + ny * dv_dy

    # Target: -d(phi_inc)/dn
    kx = k * x
    target_real = nx * k * torch.sin(kx)
    target_imag = -nx * k * torch.cos(kx)

    res_real = (du_dn - target_real).detach().cpu().numpy()
    res_imag = (dv_dn - target_imag).detach().cpu().numpy()
    res_mag = np.sqrt(res_real**2 + res_imag**2)

    return {
        "theta": theta.detach().cpu().numpy(),
        "residual_real": res_real,
        "residual_imag": res_imag,
        "residual_mag": res_mag,
        "mean": float(np.mean(res_mag)),
        "max": float(np.max(res_mag)),
    }


# ──────────────────────────────────────────────────────────────
# ABC diagnostic — analytic solution violation
# ──────────────────────────────────────────────────────────────

def check_abc_satisfaction_analytic(config, n_points=500):
    """Check how much the analytic scattered field violates the first-order ABC
    at the outer boundary r=L (approximated on the square boundary).

    ABC: dphi_s/dn - ik*phi_s = 0
    Split: (du/dn + kv) = 0, (dv/dn - ku) = 0

    Returns per-side and overall residual statistics.
    """
    L = config.L
    k = config.k
    a = config.a
    n_terms = config.n_series_terms

    # Sample points along each side of the square boundary
    t = np.linspace(-L, L, n_points)
    sides = {
        "bottom": (t, np.full_like(t, -L), np.zeros_like(t), -np.ones_like(t)),
        "top":    (t, np.full_like(t, L),  np.zeros_like(t), np.ones_like(t)),
        "left":   (np.full_like(t, -L), t, -np.ones_like(t), np.zeros_like(t)),
        "right":  (np.full_like(t, L),  t, np.ones_like(t),  np.zeros_like(t)),
    }

    all_res_mag = []
    all_theta = []
    side_results = {}

    for side_name, (x, y, nx, ny) in sides.items():
        phi = scattered_field(x, y, k, a, n_terms)
        u = np.real(phi)
        v = np.imag(phi)

        dphi_dx, dphi_dy = scattered_field_gradient(x, y, k, a, n_terms)
        du_dn = nx * np.real(dphi_dx) + ny * np.real(dphi_dy)
        dv_dn = nx * np.imag(dphi_dx) + ny * np.imag(dphi_dy)

        res_real = du_dn + k * v
        res_imag = dv_dn - k * u
        res_mag = np.sqrt(res_real**2 + res_imag**2)

        side_results[side_name] = {
            "mean": float(np.nanmean(res_mag)),
            "max": float(np.nanmax(res_mag)),
        }
        all_res_mag.append(res_mag)

        # Compute angle for plotting
        angle = np.arctan2(y, x)
        all_theta.append(angle)

    all_res = np.concatenate(all_res_mag)
    all_theta = np.concatenate(all_theta)

    return {
        "sides": side_results,
        "overall_mean": float(np.nanmean(all_res)),
        "overall_max": float(np.nanmax(all_res)),
        "theta": all_theta,
        "residual": all_res,
    }


def check_abc_satisfaction_pinn(model, config, n_points=500):
    """Check PINN's ABC residual on the outer boundary."""
    model.train()
    L = config.L
    k = config.k

    t = torch.linspace(-L, L, n_points, device=config.device)
    sides = [
        (t, torch.full_like(t, -L), torch.zeros_like(t), -torch.ones_like(t)),  # bottom
        (t, torch.full_like(t, L), torch.zeros_like(t), torch.ones_like(t)),     # top
        (torch.full_like(t, -L), t, -torch.ones_like(t), torch.zeros_like(t)),   # left
        (torch.full_like(t, L), t, torch.ones_like(t), torch.zeros_like(t)),     # right
    ]

    all_res = []
    for x, y, nx, ny in sides:
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        u, v = model(x, y)
        ones = torch.ones_like(u)

        du_dx = torch.autograd.grad(u, x, ones, create_graph=False, retain_graph=True)[0]
        du_dy = torch.autograd.grad(u, y, ones, create_graph=False, retain_graph=True)[0]
        dv_dx = torch.autograd.grad(v, x, ones, create_graph=False, retain_graph=True)[0]
        dv_dy = torch.autograd.grad(v, y, ones, create_graph=False, retain_graph=True)[0]

        du_dn = nx * du_dx + ny * du_dy
        dv_dn = nx * dv_dx + ny * dv_dy

        res_real = du_dn + k * v
        res_imag = dv_dn - k * u
        res_mag = torch.sqrt(res_real**2 + res_imag**2)
        all_res.append(res_mag.detach().cpu().numpy())

    all_res = np.concatenate(all_res)
    return {
        "mean": float(np.mean(all_res)),
        "max": float(np.max(all_res)),
    }


# ──────────────────────────────────────────────────────────────
# Plot functions (all Plotly, matching _LAYOUT)
# ──────────────────────────────────────────────────────────────

def plot_regional_bar(regional_metrics):
    """Bar chart of L2_rel by region."""
    regions = list(regional_metrics.keys())
    l2_vals = [regional_metrics[r]["l2_rel"] for r in regions]
    max_vals = [regional_metrics[r]["max_err"] for r in regions]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["L2 Relative Error", "Max Error"])

    fig.add_trace(go.Bar(
        x=regions, y=l2_vals, marker_color=_LAYOUT["colorway"][:len(regions)],
        text=[f"{v:.4f}" for v in l2_vals], textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=regions, y=max_vals, marker_color=_LAYOUT["colorway"][:len(regions)],
        text=[f"{v:.4f}" for v in max_vals], textposition="outside",
    ), row=1, col=2)

    fig.update_layout(**_LAYOUT, title=dict(text="Regional Error Breakdown", **_TITLE_STYLE),
                      width=900, height=450, showlegend=False)
    return fig


def plot_radial_error(radial_profile):
    """Bar chart of mean/max error per annular ring."""
    labels = [f"{b['r_lo']:.2f}-{b['r_hi']:.2f}" for b in radial_profile]
    means = [b["mean"] for b in radial_profile]
    maxes = [b["max"] for b in radial_profile]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Mean Error", x=labels, y=means))
    fig.add_trace(go.Bar(name="Max Error", x=labels, y=maxes))

    fig.update_layout(**_LAYOUT, barmode="group",
                      title=dict(text="Annular Error Profile", **_TITLE_STYLE),
                      xaxis_title="Radial bin (r)", yaxis_title="Error",
                      width=800, height=450)
    return fig


def plot_spatial_error(err_mag, x_arr, y_arr, config):
    """Heatmap of |error(x,y)|."""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=err_mag, x=x_arr, y=y_arr,
        colorscale="Hot_r", colorbar=dict(title="|error|"),
    ))
    for cx, cy, r in config.scatterers:
        fig.add_trace(_scatterer_circle_trace(cx, cy, r))

    fig.update_layout(**_LAYOUT,
                      title=dict(text="Spatial Error Heatmap |pred - exact|", **_TITLE_STYLE),
                      xaxis_title="x", yaxis_title="y",
                      xaxis=dict(scaleanchor="y", constrain="domain"),
                      width=700, height=600)
    return fig


def plot_angular_error(angular_profiles):
    """Error vs theta at multiple radii."""
    fig = go.Figure()
    for r, data in angular_profiles.items():
        theta_deg = np.degrees(data["theta"])
        fig.add_trace(go.Scatter(
            x=theta_deg, y=data["error"],
            mode="lines", name=f"r={r:.1f}",
        ))

    fig.update_layout(**_LAYOUT,
                      title=dict(text="Angular Error Profile", **_TITLE_STYLE),
                      xaxis_title="theta (degrees)", yaxis_title="|error|",
                      width=800, height=450)
    return fig


def plot_bc_residual(bc_data):
    """BC Neumann residual vs theta."""
    theta_deg = np.degrees(bc_data["theta"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theta_deg, y=bc_data["residual_mag"],
                             mode="lines", name="|residual|"))
    fig.add_trace(go.Scatter(x=theta_deg, y=bc_data["residual_real"],
                             mode="lines", name="Re residual", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=theta_deg, y=bc_data["residual_imag"],
                             mode="lines", name="Im residual", line=dict(dash="dot")))

    fig.update_layout(**_LAYOUT,
                      title=dict(text=f"Neumann BC Residual (mean={bc_data['mean']:.2e}, max={bc_data['max']:.2e})",
                                 **_TITLE_STYLE),
                      xaxis_title="theta (degrees)", yaxis_title="Residual",
                      width=800, height=450)
    return fig


def plot_abc_comparison(abc_analytic, abc_pinn):
    """Compare analytic vs PINN ABC violation."""
    # Sort by theta for clean lines
    idx = np.argsort(abc_analytic["theta"])
    theta_deg = np.degrees(abc_analytic["theta"][idx])
    res = abc_analytic["residual"][idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theta_deg, y=res,
        mode="lines", name="Analytic ABC violation",
    ))

    title = (f"ABC Diagnostic: Analytic violation at L={abc_analytic.get('L', '?')}"
             f" (mean={abc_analytic['overall_mean']:.3e}, max={abc_analytic['overall_max']:.3e})"
             f" | PINN ABC (mean={abc_pinn['mean']:.3e}, max={abc_pinn['max']:.3e})")

    fig.update_layout(**_LAYOUT,
                      title=dict(text="ABC Diagnostic: Analytic Solution Violation", **_TITLE_STYLE),
                      xaxis_title="theta (degrees)", yaxis_title="|ABC residual|",
                      width=900, height=450)

    # Add annotation with stats
    fig.add_annotation(
        text=(f"Analytic: mean={abc_analytic['overall_mean']:.3e}, max={abc_analytic['overall_max']:.3e}<br>"
              f"PINN: mean={abc_pinn['mean']:.3e}, max={abc_pinn['max']:.3e}"),
        xref="paper", yref="paper", x=0.98, y=0.98,
        showarrow=False, font=dict(size=11),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="#ccc", borderwidth=1,
        align="right", xanchor="right", yanchor="top",
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Summary JSON generation
# ──────────────────────────────────────────────────────────────

def generate_summary_json(regional, radial, bc, abc_analytic, abc_pinn, symmetry,
                          resolution_results, config):
    """Produce a summary dict with all metrics + automated diagnosis."""
    diagnosis_parts = []

    # Check if ABC is the bottleneck
    if abc_analytic["overall_mean"] > 0.05:
        diagnosis_parts.append(
            f"HIGH ABC mismatch: analytic solution violates first-order ABC "
            f"with mean residual {abc_analytic['overall_mean']:.3e} at L={config.L}. "
            f"This is likely the primary accuracy bottleneck. Increase L to reduce."
        )
    elif abc_analytic["overall_mean"] > 0.01:
        diagnosis_parts.append(
            f"MODERATE ABC mismatch: analytic ABC violation mean={abc_analytic['overall_mean']:.3e} "
            f"at L={config.L}. May be limiting accuracy."
        )
    else:
        diagnosis_parts.append(
            f"LOW ABC mismatch: analytic ABC violation mean={abc_analytic['overall_mean']:.3e}. "
            f"ABC is not the primary bottleneck."
        )

    # Check regional pattern
    full_l2 = regional["full"]["l2_rel"]
    shadow_l2 = regional.get("shadow_side", {}).get("l2_rel", 0)
    if shadow_l2 > 2 * full_l2:
        diagnosis_parts.append(
            f"Shadow side error ({shadow_l2:.4f}) is >2x full domain ({full_l2:.4f}). "
            f"Network struggles with diffraction/shadow region."
        )

    # Check near-surface
    near_l2 = regional.get("near_surface", {}).get("l2_rel", 0)
    if near_l2 > 2 * full_l2:
        diagnosis_parts.append(
            f"Near-surface error ({near_l2:.4f}) is elevated vs full domain ({full_l2:.4f})."
        )

    # Check BC
    if bc["max"] > 0.01:
        diagnosis_parts.append(f"BC residual max={bc['max']:.3e} — may need higher lambda_bc.")

    # Check symmetry
    if symmetry["symmetry_u_mse"] > 1e-4:
        diagnosis_parts.append(f"Symmetry violation: u MSE={symmetry['symmetry_u_mse']:.3e}.")

    summary = {
        "config": {
            "ka": config.ka,
            "L": config.L,
            "k": config.k,
        },
        "regional": regional,
        "radial_profile": radial,
        "bc": {"mean": bc["mean"], "max": bc["max"]},
        "abc_analytic": {
            "overall_mean": abc_analytic["overall_mean"],
            "overall_max": abc_analytic["overall_max"],
            "sides": abc_analytic["sides"],
        },
        "abc_pinn": abc_pinn,
        "symmetry": symmetry,
        "resolution_sensitivity": resolution_results,
        "diagnosis": " | ".join(diagnosis_parts),
    }
    return summary


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="HelmholtzPINN evaluation suite")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--run-id", type=str, default="eval", help="Run ID for output dir naming")
    parser.add_argument("--ka", type=float, default=None, help="Override ka (default: infer from checkpoint name)")
    parser.add_argument("--L", type=float, default=None, help="Override domain size L")
    parser.add_argument("--grid-size", type=int, default=400, help="Evaluation grid size")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    import math
    ka = args.ka
    if ka is None:
        # Try to infer from checkpoint filename
        import re
        m = re.search(r"ka([\d.]+)", args.checkpoint)
        ka = float(m.group(1)) if m else math.pi
    kwargs = {}
    if args.L is not None:
        kwargs["L"] = args.L
    if args.device:
        kwargs["device"] = args.device
    kwargs["use_wandb"] = False

    if abs(ka - math.pi) < 0.01:
        config = HelmholtzConfig.ka_pi(**kwargs)
    elif abs(ka - 2 * math.pi) < 0.01:
        config = HelmholtzConfig.ka_2pi(**kwargs)
    else:
        config = HelmholtzConfig(ka=ka, **kwargs)

    print(f"Eval suite — ka={config.ka:.4f}, L={config.L}, device={config.device}")

    # Load model
    model = HelmholtzPINN(config).to(config.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=config.device, weights_only=True))
    print(f"Loaded checkpoint: {args.checkpoint}")

    domain = ScatteringDomain(config.L, config.scatterers, config.device)
    analytic_fn = scattered_field
    gs = args.grid_size

    out_dir = os.path.join(args.output_dir, f"eval_{args.run_id}")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Regional L2 breakdown ──
    print("\n[1/9] Regional evaluation...")
    regional = evaluate_regions(model, domain, config, analytic_fn, gs)
    for name, m in regional.items():
        print(f"  {name:15s}  L2_rel={m['l2_rel']:.6e}  max_err={m['max_err']:.6e}")

    fig = plot_regional_bar(regional)
    fig.write_html(os.path.join(out_dir, "regional_bar.html"))

    # ── 2. Spatial error heatmap ──
    print("[2/9] Spatial error map...")
    err_mag, u_err, v_err, x_arr, y_arr = compute_spatial_error_map(
        model, domain, config, analytic_fn, gs)

    fig = plot_spatial_error(err_mag, x_arr, y_arr, config)
    fig.write_html(os.path.join(out_dir, "spatial_error.html"))

    # ── 3. Annular error profile ──
    print("[3/9] Radial error profile...")
    # Filter bins to fit within domain
    valid_bins = [(lo, hi) for lo, hi in ANNULAR_BINS if lo < config.L]
    radial = compute_radial_error_profile(err_mag, config, valid_bins)
    for b in radial:
        print(f"  r=[{b['r_lo']:.2f}, {b['r_hi']:.2f})  mean={b['mean']:.4e}  max={b['max']:.4e}  n={b['count']}")

    fig = plot_radial_error(radial)
    fig.write_html(os.path.join(out_dir, "radial_error.html"))

    # ── 4. Angular error at fixed r ──
    print("[4/9] Angular error profiles...")
    r_values = [r for r in [1.5, 2.0, 2.5] if r < config.L]
    angular = compute_angular_error_profile(model, config, analytic_fn, r_values)

    fig = plot_angular_error(angular)
    fig.write_html(os.path.join(out_dir, "angular_error.html"))

    # ── 5. BC satisfaction ──
    print("[5/9] BC satisfaction check...")
    bc = check_bc_satisfaction(model, config)
    print(f"  BC residual: mean={bc['mean']:.4e}, max={bc['max']:.4e}")

    fig = plot_bc_residual(bc)
    fig.write_html(os.path.join(out_dir, "bc_residual.html"))

    # ── 6. ABC diagnostic (KEY!) ──
    print("[6/9] ABC diagnostic...")
    abc_analytic = check_abc_satisfaction_analytic(config)
    abc_analytic["L"] = config.L
    abc_pinn = check_abc_satisfaction_pinn(model, config)
    print(f"  Analytic ABC violation: mean={abc_analytic['overall_mean']:.4e}, max={abc_analytic['overall_max']:.4e}")
    print(f"  PINN ABC residual:     mean={abc_pinn['mean']:.4e}, max={abc_pinn['max']:.4e}")
    for side, stats in abc_analytic["sides"].items():
        print(f"    {side:8s}  mean={stats['mean']:.4e}  max={stats['max']:.4e}")

    fig = plot_abc_comparison(abc_analytic, abc_pinn)
    fig.write_html(os.path.join(out_dir, "abc_comparison.html"))

    # ── 7. Resolution sensitivity ──
    print("[7/9] Resolution sensitivity...")
    resolution_results = {}
    for res in [200, 400, 600]:
        m = evaluate_against_analytic(model, domain, config, analytic_fn, grid_size=res)
        resolution_results[res] = m["l2_rel"]
        print(f"  grid={res}x{res}: L2_rel={m['l2_rel']:.6e}")

    # ── 8. Symmetry check ──
    print("[8/9] Symmetry check...")
    symmetry = check_symmetry(model, config)
    print(f"  u symmetry MSE: {symmetry['symmetry_u_mse']:.4e}")
    print(f"  v antisymmetry MSE: {symmetry['symmetry_v_mse']:.4e}")

    # ── 9. Zoom report ──
    print("[9/9] Zoom report...")
    create_zoom_report(model, domain, config, analytic_fn, out_dir)

    # ── Summary JSON ──
    summary = generate_summary_json(
        regional, radial, bc, abc_analytic, abc_pinn, symmetry,
        resolution_results, config,
    )
    summary_path = os.path.join(out_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print(f"\nDIAGNOSIS: {summary['diagnosis']}")
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
