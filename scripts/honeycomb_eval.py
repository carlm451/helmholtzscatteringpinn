#!/usr/bin/env python3
"""Honeycomb PINN post-training diagnostics (no analytic reference).

Generates 12 interactive HTML plots + summary JSON for physics validation
of the honeycomb structured scatterer.

Usage:
    .venv/bin/python scripts/honeycomb_eval.py checkpoints/helmholtz_ka3.00_lbfgs.pt
    .venv/bin/python scripts/honeycomb_eval.py checkpoints/helmholtz_ka3.00_lbfgs.pt --output-dir outputs/hc_eval
"""

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helmholtz.config import HelmholtzConfig
from helmholtz.domain import ScatteringDomain
from helmholtz.network import HelmholtzPINN
from helmholtz.analytic import scattered_field as analytic_scattered
from helmholtz.analytic import total_field as analytic_total
from helmholtz.visualize import _LAYOUT, _TITLE_STYLE


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _add_outlines(fig, scatterers, row=None, col=None, color="white", width=1.5):
    """Add scatterer circle outlines to a figure (optionally into a subplot)."""
    theta = np.linspace(0, 2 * np.pi, 100)
    kw = {}
    if row is not None:
        kw = {"row": row, "col": col}
    for cx, cy, r in scatterers:
        fig.add_trace(go.Scatter(
            x=cx + r * np.cos(theta), y=cy + r * np.sin(theta),
            mode="lines", line=dict(color=color, width=width),
            showlegend=False, hoverinfo="skip",
        ), **kw)


def _heatmap_fig(panels, suptitle, scatterers, outline_width=1.5):
    """Multi-panel heatmap figure.

    panels: list of dicts with keys z, x, y, colorscale, title,
            and optional zmid, zmin, zmax, showscale.
    """
    n = len(panels)
    titles = [p["title"] for p in panels]
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles,
                        horizontal_spacing=max(0.04, 0.14 / n))
    for i, p in enumerate(panels):
        hm_kw = {}
        for k in ("zmid", "zmin", "zmax"):
            if k in p and p[k] is not None:
                hm_kw[k] = p[k]
        fig.add_trace(go.Heatmap(
            z=p["z"], x=p["x"], y=p["y"], colorscale=p["colorscale"],
            showscale=p.get("showscale", True), **hm_kw,
        ), row=1, col=i + 1)
        _add_outlines(fig, scatterers, row=1, col=i + 1, width=outline_width)
    fig.update_layout(**_LAYOUT, title=dict(text=suptitle, **_TITLE_STYLE),
                      width=max(650, 480 * n), height=500)
    for i in range(1, n + 1):
        ax = f"y{i}" if i > 1 else "y"
        fig.update_xaxes(scaleanchor=ax, constrain="domain", row=1, col=i)
    return fig


@torch.no_grad()
def _eval_grid(model, domain, config, x_range, y_range, grid_size):
    """Evaluate PINN scattered + total fields on a regular grid."""
    model.eval()
    x_flat, y_flat, valid = domain.sample_test_grid(grid_size, x_range, y_range)
    if config.outer_boundary == "circle":
        valid = valid & (x_flat ** 2 + y_flat ** 2 <= config.L ** 2)
    u, v = model(x_flat, y_flat)
    model.train()
    u, v = u.cpu().numpy(), v.cpu().numpy()
    vn = valid.cpu().numpy()
    u[~vn] = np.nan
    v[~vn] = np.nan
    x_np = x_flat.cpu().numpy()
    inc_re = np.cos(config.k * x_np)
    inc_im = np.sin(config.k * x_np)
    ut = np.where(vn, u + inc_re, np.nan)
    vt = np.where(vn, v + inc_im, np.nan)
    gs = grid_size
    return dict(
        u=u.reshape(gs, gs), v=v.reshape(gs, gs),
        ut=ut.reshape(gs, gs), vt=vt.reshape(gs, gs),
        mag_s=np.sqrt(u ** 2 + v ** 2).reshape(gs, gs),
        mag_t=np.sqrt(ut ** 2 + vt ** 2).reshape(gs, gs),
        x=np.linspace(x_range[0], x_range[1], gs),
        y=np.linspace(y_range[0], y_range[1], gs),
    )


@torch.no_grad()
def _eval_circle(model, config, r, n=720):
    """Evaluate PINN scattered field on a circle of given radius."""
    model.eval()
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_np, y_np = r * np.cos(theta), r * np.sin(theta)
    xt = torch.tensor(x_np, dtype=torch.float32, device=config.device)
    yt = torch.tensor(y_np, dtype=torch.float32, device=config.device)
    u, v = model(xt, yt)
    model.train()
    return theta, u.cpu().numpy(), v.cpu().numpy(), x_np, y_np


@torch.no_grad()
def _eval_line(model, domain, config, x_np, y_np):
    """Evaluate PINN scattered field on arbitrary points, masking scatterer interiors."""
    model.eval()
    xt = torch.tensor(x_np, dtype=torch.float32, device=config.device)
    yt = torch.tensor(y_np, dtype=torch.float32, device=config.device)
    inside = domain.is_inside_any_scatterer(xt, yt)
    u, v = model(xt, yt)
    model.train()
    u, v = u.cpu().numpy(), v.cpu().numpy()
    m = inside.cpu().numpy()
    u[m] = np.nan
    v[m] = np.nan
    return u, v


def _pde_res_batch(model, x, y, k):
    """Pointwise PDE residual magnitude (requires grad context)."""
    u, v = model(x, y)
    ones = torch.ones_like(u)
    du_dx = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, ones, create_graph=True, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, ones, create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, ones, create_graph=True, retain_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, ones, create_graph=False, retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, ones, create_graph=False, retain_graph=True)[0]
    d2v_dx2 = torch.autograd.grad(dv_dx, x, ones, create_graph=False, retain_graph=True)[0]
    d2v_dy2 = torch.autograd.grad(dv_dy, y, ones, create_graph=False)[0]
    ru = d2u_dx2 + d2u_dy2 + k ** 2 * u
    rv = d2v_dx2 + d2v_dy2 + k ** 2 * v
    return torch.sqrt(ru ** 2 + rv ** 2).detach()


def _compute_pde_grid(model, domain, config, x_range, y_range, grid_size,
                      batch_size=5000):
    """PDE residual on a regular grid using autograd (batched)."""
    model.train()
    k = config.k
    x_lin = torch.linspace(x_range[0], x_range[1], grid_size, device=config.device)
    y_lin = torch.linspace(y_range[0], y_range[1], grid_size, device=config.device)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing="xy")
    x_flat, y_flat = xx.reshape(-1), yy.reshape(-1)
    valid = ~domain.is_inside_any_scatterer(x_flat, y_flat)
    if config.outer_boundary == "circle":
        valid = valid & (x_flat ** 2 + y_flat ** 2 <= config.L ** 2)
    result = np.full(grid_size * grid_size, np.nan)
    idx_all = torch.where(valid)[0]
    for i in range(0, len(idx_all), batch_size):
        idx = idx_all[i:i + batch_size]
        xb = x_flat[idx].clone().requires_grad_(True)
        yb = y_flat[idx].clone().requires_grad_(True)
        res = _pde_res_batch(model, xb, yb, k)
        result[idx.cpu().numpy()] = res.cpu().numpy()
    return result.reshape(grid_size, grid_size)


def _compute_bc_all(model, config, n_per=200):
    """Neumann BC residual on all scatterer surfaces."""
    model.train()
    k = config.k
    all_x, all_y, all_res = [], [], []
    per_scat = []

    for i, (cx, cy, rs) in enumerate(config.scatterers):
        theta = torch.linspace(0, 2 * np.pi, n_per, device=config.device)
        x = (cx + rs * torch.cos(theta)).requires_grad_(True)
        y = (cy + rs * torch.sin(theta)).requires_grad_(True)
        nx = torch.cos(theta)
        ny = torch.sin(theta)

        u, v = model(x, y)
        ones = torch.ones_like(u)
        du_dx = torch.autograd.grad(u, x, ones, create_graph=False, retain_graph=True)[0]
        du_dy = torch.autograd.grad(u, y, ones, create_graph=False, retain_graph=True)[0]
        dv_dx = torch.autograd.grad(v, x, ones, create_graph=False, retain_graph=True)[0]
        dv_dy = torch.autograd.grad(v, y, ones, create_graph=False)[0]

        du_dn = nx * du_dx + ny * du_dy
        dv_dn = nx * dv_dx + ny * dv_dy

        kx = k * x
        target_re = nx * k * torch.sin(kx)
        target_im = -nx * k * torch.cos(kx)

        res_re = (du_dn - target_re).detach()
        res_im = (dv_dn - target_im).detach()
        res_mag = torch.sqrt(res_re ** 2 + res_im ** 2).cpu().numpy()

        all_x.append(x.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        all_res.append(res_mag)
        per_scat.append({"idx": i, "cx": cx, "cy": cy,
                         "mean": float(np.mean(res_mag)),
                         "max": float(np.max(res_mag))})

    return {
        "x": np.concatenate(all_x),
        "y": np.concatenate(all_y),
        "residual": np.concatenate(all_res),
        "per_scatterer": per_scat,
        "overall_mean": float(np.mean(np.concatenate(all_res))),
        "overall_max": float(np.max(np.concatenate(all_res))),
    }


def _compute_abc_circle(model, config, n_points=500):
    """ABC residual on circular outer boundary r=L."""
    model.train()
    k, L = config.k, config.L
    theta = torch.linspace(0, 2 * np.pi, n_points, device=config.device)
    x = (L * torch.cos(theta)).requires_grad_(True)
    y = (L * torch.sin(theta)).requires_grad_(True)
    nx, ny = torch.cos(theta), torch.sin(theta)

    u, v = model(x, y)
    ones = torch.ones_like(u)
    du_dx = torch.autograd.grad(u, x, ones, create_graph=False, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, ones, create_graph=False, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, ones, create_graph=False, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, ones, create_graph=False)[0]

    du_dr = nx * du_dx + ny * du_dy
    dv_dr = nx * dv_dx + ny * dv_dy

    if config.abc_order == 2:
        res_re = du_dr + k * v + u / (2 * L)
        res_im = dv_dr - k * u + v / (2 * L)
    else:
        res_re = du_dr + k * v
        res_im = dv_dr - k * u

    res_mag = torch.sqrt(res_re ** 2 + res_im ** 2).detach().cpu().numpy()
    return {
        "theta": theta.detach().cpu().numpy(),
        "residual": res_mag,
        "mean": float(np.mean(res_mag)),
        "max": float(np.max(res_mag)),
    }


def _hline_bands(scatterers, y_val):
    """x-ranges where horizontal line y=y_val crosses scatterers."""
    bands = []
    for cx, cy, r in scatterers:
        dy = abs(cy - y_val)
        if dy < r:
            dx = math.sqrt(r ** 2 - dy ** 2)
            bands.append((cx - dx, cx + dx))
    return bands


def _vline_bands(scatterers, x_val):
    """y-ranges where vertical line x=x_val crosses scatterers."""
    bands = []
    for cx, cy, r in scatterers:
        dx = abs(cx - x_val)
        if dx < r:
            dy = math.sqrt(r ** 2 - dx ** 2)
            bands.append((cy - dy, cy + dy))
    return bands


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def _download_from_wandb(entity="carlm451", project="helmholtz-pinn-honeycomb",
                         artifact_path=None, download_dir="checkpoints"):
    """Download a honeycomb checkpoint from wandb.

    If artifact_path is given, fetch that exact artifact.
    Otherwise, find the latest finished run and download its model artifact.
    Returns local path to the .pt file.
    """
    try:
        import wandb
    except ImportError:
        sys.exit("wandb is not installed — cannot use --from-wandb")

    api = wandb.Api()

    if artifact_path:
        print(f"Fetching artifact: {artifact_path}")
        artifact = api.artifact(artifact_path)
    else:
        # Find latest finished run with a logged model artifact
        print(f"Searching {entity}/{project} for latest finished run...")
        runs = api.runs(f"{entity}/{project}",
                        filters={"state": "finished"}, order="-created_at")
        artifact = None
        for run in runs:
            arts = [a for a in run.logged_artifacts() if a.type == "model"]
            if arts:
                artifact = arts[-1]  # last logged = lbfgs checkpoint
                print(f"  Found run: {run.name} ({run.id})")
                print(f"  Artifact:  {artifact.name}:{artifact.version}")
                break
        if artifact is None:
            sys.exit("No finished runs with model artifacts found in "
                     f"{entity}/{project}")

    os.makedirs(download_dir, exist_ok=True)
    local_dir = artifact.download(root=download_dir)
    # Find the .pt file in downloaded dir
    pt_files = [f for f in os.listdir(local_dir) if f.endswith(".pt")]
    if not pt_files:
        sys.exit(f"No .pt files found in downloaded artifact: {local_dir}")
    path = os.path.join(local_dir, pt_files[0])
    print(f"  Downloaded: {path}\n")
    return path


def parse_args():
    p = argparse.ArgumentParser(description="Honeycomb PINN post-training diagnostics")
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="Path to .pt checkpoint (optional if --from-wandb)")
    p.add_argument("--from-wandb", action="store_true",
                   help="Download latest honeycomb checkpoint from wandb")
    p.add_argument("--wandb-artifact", type=str, default=None,
                   help="Explicit wandb artifact path (e.g. entity/project/name:version)")
    p.add_argument("--ka", type=float, default=None,
                   help="Override ka (default: infer from filename)")
    p.add_argument("--output-dir", default="outputs/hc_eval",
                   help="Output directory (default: outputs/hc_eval)")
    p.add_argument("--grid-size", type=int, default=400,
                   help="Evaluation grid resolution (default: 400)")
    p.add_argument("--device", type=str, default=None, help="Device override")
    args = p.parse_args()
    if not args.checkpoint and not args.from_wandb and not args.wandb_artifact:
        p.error("provide a checkpoint path, --from-wandb, or --wandb-artifact")
    return args


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve checkpoint path
    checkpoint = args.checkpoint
    if args.wandb_artifact:
        checkpoint = _download_from_wandb(artifact_path=args.wandb_artifact)
    elif args.from_wandb:
        checkpoint = _download_from_wandb()

    # Infer ka from filename
    ka = args.ka
    if ka is None:
        m = re.search(r"ka(\d+\.?\d*)", checkpoint)
        ka = float(m.group(1)) if m else 3.0

    # Build config
    cfg_kw = {"use_wandb": False}
    if args.device:
        cfg_kw["device"] = args.device
    config = HelmholtzConfig.honeycomb_ka3(ka=ka, **cfg_kw)

    # Load model
    model = HelmholtzPINN(config).to(config.device)
    model.load_state_dict(torch.load(checkpoint, map_location=config.device,
                                     weights_only=True))
    domain = ScatteringDomain(config.L, config.scatterers, config.device)

    L = config.L
    k = config.k
    gs = args.grid_size
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    scats = config.scatterers
    r_s = config.honeycomb_r_s or 0.15 * config.a
    d = config.honeycomb_d or 0.4 * config.a

    print(f"Honeycomb eval — ka={config.ka:.2f}, k={k:.2f}, L={L}, "
          f"{len(scats)} scatterers, device={config.device}")
    print(f"Output: {out}/\n")

    summary = {"config": {"ka": config.ka, "k": k, "L": L, "a": config.a,
                          "r_s": r_s, "d": d, "gap": d - 2 * r_s,
                          "n_scatterers": len(scats)}}

    # ══════════════════════════════════════════════════════════
    # [1/11] Full-Domain Field Maps
    # ══════════════════════════════════════════════════════════
    print("[1/11] Full-domain scattered & total field maps...")
    f = _eval_grid(model, domain, config, (-L, L), (-L, L), gs)

    # Scattered field: Re | Im | |phi_s|
    vmax_s = max(np.nanmax(np.abs(f["u"])), np.nanmax(np.abs(f["v"])), 1e-6)
    fig = _heatmap_fig([
        {"z": f["u"], "x": f["x"], "y": f["y"], "colorscale": "RdBu_r",
         "zmid": 0, "zmin": -vmax_s, "zmax": vmax_s, "showscale": False,
         "title": "Re(phi_s)"},
        {"z": f["v"], "x": f["x"], "y": f["y"], "colorscale": "RdBu_r",
         "zmid": 0, "zmin": -vmax_s, "zmax": vmax_s,
         "title": "Im(phi_s)"},
        {"z": f["mag_s"], "x": f["x"], "y": f["y"], "colorscale": "Viridis",
         "title": "|phi_s|"},
    ], f"Honeycomb Scattered Field — ka={config.ka:.2f}", scats)
    fig.write_html(os.path.join(out, "hc_scattered_fields.html"))
    print("  -> hc_scattered_fields.html")

    # Total field: Re | Im | |phi_total|
    vmax_t = max(np.nanmax(np.abs(f["ut"])), np.nanmax(np.abs(f["vt"])), 1e-6)
    fig = _heatmap_fig([
        {"z": f["ut"], "x": f["x"], "y": f["y"], "colorscale": "RdBu_r",
         "zmid": 0, "zmin": -vmax_t, "zmax": vmax_t, "showscale": False,
         "title": "Re(phi_total)"},
        {"z": f["vt"], "x": f["x"], "y": f["y"], "colorscale": "RdBu_r",
         "zmid": 0, "zmin": -vmax_t, "zmax": vmax_t,
         "title": "Im(phi_total)"},
        {"z": f["mag_t"], "x": f["x"], "y": f["y"], "colorscale": "Viridis",
         "title": "|phi_total|"},
    ], f"Honeycomb Total Field — ka={config.ka:.2f}", scats)
    fig.write_html(os.path.join(out, "hc_total_fields.html"))
    print("  -> hc_total_fields.html")

    # ══════════════════════════════════════════════════════════
    # [2/11] Cluster Zoom
    # ══════════════════════════════════════════════════════════
    print("[2/11] Cluster zoom...")
    z = _eval_grid(model, domain, config, (-1.5, 1.5), (-1.5, 1.5), 500)
    fig = _heatmap_fig([
        {"z": z["ut"], "x": z["x"], "y": z["y"], "colorscale": "RdBu_r",
         "zmid": 0, "title": "Re(phi_total)"},
        {"z": z["mag_t"], "x": z["x"], "y": z["y"], "colorscale": "Viridis",
         "title": "|phi_total|"},
    ], f"Honeycomb Cluster Zoom — ka={config.ka:.2f}", scats, outline_width=2.5)
    fig.write_html(os.path.join(out, "hc_cluster_zoom.html"))
    print("  -> hc_cluster_zoom.html")

    # ══════════════════════════════════════════════════════════
    # [3/11] PDE Residual Heatmap
    # ══════════════════════════════════════════════════════════
    print("[3/11] PDE residual (autograd, may take a moment)...")
    pde_gs = 300
    res_full = _compute_pde_grid(model, domain, config, (-L, L), (-L, L), pde_gs)
    res_zoom = _compute_pde_grid(model, domain, config, (-1.5, 1.5), (-1.5, 1.5), pde_gs)

    res_log_full = np.log10(np.clip(res_full, 1e-10, None))
    res_log_zoom = np.log10(np.clip(res_zoom, 1e-10, None))

    fig = _heatmap_fig([
        {"z": res_log_full, "x": np.linspace(-L, L, pde_gs),
         "y": np.linspace(-L, L, pde_gs), "colorscale": "Hot_r",
         "title": "Full Domain log10(|res|)"},
        {"z": res_log_zoom, "x": np.linspace(-1.5, 1.5, pde_gs),
         "y": np.linspace(-1.5, 1.5, pde_gs), "colorscale": "Hot_r",
         "title": "Cluster Zoom log10(|res|)"},
    ], f"PDE Residual |nabla^2 phi_s + k^2 phi_s| — ka={config.ka:.2f}", scats)
    fig.write_html(os.path.join(out, "hc_pde_residual.html"))

    res_valid = res_full[~np.isnan(res_full)]
    pde_stats = {"mean": float(np.mean(res_valid)), "max": float(np.max(res_valid)),
                 "median": float(np.median(res_valid))}
    summary["pde_residual"] = pde_stats
    print(f"  PDE residual: mean={pde_stats['mean']:.3e}, "
          f"max={pde_stats['max']:.3e}, median={pde_stats['median']:.3e}")
    print("  -> hc_pde_residual.html")

    # ══════════════════════════════════════════════════════════
    # [4/11] BC Residual on All 19 Scatterers
    # ══════════════════════════════════════════════════════════
    print("[4/11] BC residual on all scatterers...")
    bc = _compute_bc_all(model, config, n_per=200)

    log_res = np.log10(np.clip(bc["residual"], 1e-10, None))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bc["x"], y=bc["y"], mode="markers",
        marker=dict(size=3, color=log_res, colorscale="Hot_r",
                    colorbar=dict(title="log10(|res|)"),
                    cmin=np.percentile(log_res, 5),
                    cmax=np.percentile(log_res, 95)),
        hovertext=[f"|res|={r:.2e}" for r in bc["residual"]],
        showlegend=False,
    ))
    _add_outlines(fig, scats, color="rgba(100,100,100,0.5)", width=1)
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"Neumann BC Residual — ka={config.ka:.2f}", **_TITLE_STYLE),
        xaxis_title="x", yaxis_title="y",
        xaxis=dict(scaleanchor="y", constrain="domain"),
        width=700, height=650,
    )

    # Annotation with stats
    worst3 = sorted(bc["per_scatterer"], key=lambda s: s["max"], reverse=True)[:3]
    ann = (f"Overall: mean={bc['overall_mean']:.2e}, max={bc['overall_max']:.2e}<br>"
           + "<br>".join(f"Scat {s['idx']} ({s['cx']:.2f},{s['cy']:.2f}): "
                         f"max={s['max']:.2e}" for s in worst3))
    fig.add_annotation(
        text=ann, xref="paper", yref="paper", x=0.98, y=0.02,
        showarrow=False, font=dict(size=10),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1,
        align="left", xanchor="right", yanchor="bottom",
    )
    fig.write_html(os.path.join(out, "hc_bc_residual.html"))

    summary["bc_residual"] = {
        "overall_mean": bc["overall_mean"], "overall_max": bc["overall_max"],
        "per_scatterer": bc["per_scatterer"],
    }
    print(f"  BC residual: mean={bc['overall_mean']:.3e}, max={bc['overall_max']:.3e}")
    print("  -> hc_bc_residual.html")

    # ══════════════════════════════════════════════════════════
    # [5/11] Far-Field Scattering Pattern
    # ══════════════════════════════════════════════════════════
    print("[5/11] Far-field scattering pattern...")
    r_ff = 0.9 * L
    theta_ff, u_ff, v_ff, x_ff, y_ff = _eval_circle(model, config, r_ff, n=720)
    pinn_amp = np.sqrt(u_ff ** 2 + v_ff ** 2)

    # Solid cylinder analytic at same radius
    phi_solid = analytic_scattered(x_ff, y_ff, k, a=config.cluster_radius,
                                   n_terms=config.n_series_terms)
    solid_amp = np.abs(phi_solid)

    theta_deg = np.degrees(theta_ff)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pinn_amp, theta=theta_deg, mode="lines",
        name="Honeycomb PINN", line=dict(width=2),
    ))
    fig.add_trace(go.Scatterpolar(
        r=solid_amp, theta=theta_deg, mode="lines",
        name=f"Solid cylinder (a={config.cluster_radius})",
        line=dict(width=2, dash="dash"),
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"Far-Field |phi_s| at r={r_ff:.1f} — ka={config.ka:.2f}",
                   **_TITLE_STYLE),
        polar=dict(radialaxis=dict(visible=True, title="|phi_s|")),
        width=700, height=600,
    )
    fig.write_html(os.path.join(out, "hc_far_field.html"))
    print("  -> hc_far_field.html")

    # ══════════════════════════════════════════════════════════
    # [6/11] Line Cuts Through Cluster
    # ══════════════════════════════════════════════════════════
    print("[6/11] Line cuts through cluster...")
    n_line = 800
    x_line = np.linspace(-L, L, n_line)
    y_gap = d / 2  # line through gap between ring-0 and ring-1

    cuts = [
        ("y=0 (through center)", x_line, np.zeros(n_line), True),
        ("x=0 (transverse)", np.zeros(n_line), np.linspace(-L, L, n_line), False),
        (f"y={y_gap:.2f} (through gap)", x_line, np.full(n_line, y_gap), False),
    ]

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=[c[0] for c in cuts],
                        vertical_spacing=0.08)
    colors = _LAYOUT["colorway"]

    for row, (label, xc, yc, overlay_analytic) in enumerate(cuts, 1):
        u_s, v_s = _eval_line(model, domain, config, xc, yc)
        inc = np.exp(1j * k * xc)
        u_tot = u_s + np.real(inc)
        v_tot = v_s + np.imag(inc)
        mag_tot = np.sqrt(u_tot ** 2 + v_tot ** 2)
        re_tot = u_tot

        # x-axis of subplot: use the varying coordinate
        coord = xc if row != 2 else yc
        coord_label = "x" if row != 2 else "y"

        fig.add_trace(go.Scatter(
            x=coord, y=mag_tot, mode="lines", name=f"|phi_total| {label}",
            line=dict(color=colors[0], width=2), showlegend=(row == 1),
            legendgroup="mag",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=coord, y=re_tot, mode="lines", name=f"Re(phi_total) {label}",
            line=dict(color=colors[1], width=1.5, dash="dot"),
            showlegend=(row == 1), legendgroup="re",
        ), row=row, col=1)

        if overlay_analytic and row == 1:
            # Solid cylinder analytic for y=0 cut
            phi_a = analytic_total(xc, yc, k, a=config.cluster_radius,
                                   n_terms=config.n_series_terms)
            fig.add_trace(go.Scatter(
                x=coord, y=np.abs(phi_a), mode="lines",
                name="Solid cyl |phi_total|",
                line=dict(color=colors[2], width=1.5, dash="dash"),
            ), row=row, col=1)

        # Shaded scatterer bands
        if row != 2:
            bands = _hline_bands(scats, yc[0] if hasattr(yc, '__len__') else yc)
        else:
            bands = _vline_bands(scats, xc[0] if hasattr(xc, '__len__') else xc)
        for lo, hi in bands:
            fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(150,150,150,0.25)",
                          line_width=0, row=row, col=1)

        fig.update_xaxes(title_text=coord_label, row=row, col=1)
        fig.update_yaxes(title_text="Field", row=row, col=1)

    fig.update_layout(**_LAYOUT,
                      title=dict(text=f"Line Cuts — ka={config.ka:.2f}", **_TITLE_STYLE),
                      width=900, height=900)
    fig.write_html(os.path.join(out, "hc_line_cuts.html"))
    print("  -> hc_line_cuts.html")

    # ══════════════════════════════════════════════════════════
    # [7/11] ABC Satisfaction
    # ══════════════════════════════════════════════════════════
    print("[7/11] ABC residual on outer boundary...")
    abc = _compute_abc_circle(model, config, n_points=500)
    abc_label = "BGT2" if config.abc_order == 2 else "1st-order"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.degrees(abc["theta"]), y=abc["residual"],
        mode="lines", name="|ABC residual|",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"{abc_label} ABC Residual at r=L={L} — ka={config.ka:.2f}",
                   **_TITLE_STYLE),
        xaxis_title="theta (degrees)", yaxis_title="|residual|",
        width=800, height=450,
    )
    fig.add_annotation(
        text=f"mean={abc['mean']:.3e}, max={abc['max']:.3e}",
        xref="paper", yref="paper", x=0.98, y=0.98,
        showarrow=False, font=dict(size=11),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="#ccc", borderwidth=1,
        xanchor="right", yanchor="top",
    )
    fig.write_html(os.path.join(out, "hc_abc_residual.html"))

    summary["abc_residual"] = {"mean": abc["mean"], "max": abc["max"],
                               "order": config.abc_order}
    print(f"  ABC residual ({abc_label}): mean={abc['mean']:.3e}, max={abc['max']:.3e}")
    print("  -> hc_abc_residual.html")

    # ══════════════════════════════════════════════════════════
    # [8/11] Radial Decay Check
    # ══════════════════════════════════════════════════════════
    print("[8/11] Radial decay check...")
    radii = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    radii = radii[radii < L]  # filter to domain
    pinn_mean = []
    solid_mean = []

    for r in radii:
        theta_r, u_r, v_r, xr, yr = _eval_circle(model, config, r, n=720)
        pinn_mean.append(np.mean(np.sqrt(u_r ** 2 + v_r ** 2)))
        phi_a = analytic_scattered(xr, yr, k, a=config.cluster_radius,
                                   n_terms=config.n_series_terms)
        solid_mean.append(np.mean(np.abs(phi_a)))

    pinn_mean = np.array(pinn_mean)
    solid_mean = np.array(solid_mean)

    # Reference: C/sqrt(r), fitted to first PINN point
    C_ref = pinn_mean[0] * np.sqrt(radii[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=radii, y=pinn_mean, mode="lines+markers", name="Honeycomb PINN",
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=radii, y=solid_mean, mode="lines+markers", name="Solid cylinder (analytic)",
        line=dict(width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=radii, y=C_ref / np.sqrt(radii), mode="lines",
        name="C/sqrt(r) reference", line=dict(width=1.5, dash="dot", color="gray"),
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"Radial Decay of Mean |phi_s| — ka={config.ka:.2f}",
                   **_TITLE_STYLE),
        xaxis_title="r", yaxis_title="Mean |phi_s|",
        xaxis_type="log", yaxis_type="log",
        width=700, height=450,
    )
    fig.write_html(os.path.join(out, "hc_radial_decay.html"))

    summary["radial_decay"] = {
        "radii": radii.tolist(),
        "pinn_mean": pinn_mean.tolist(),
        "solid_mean": solid_mean.tolist(),
    }
    print("  -> hc_radial_decay.html")

    # ══════════════════════════════════════════════════════════
    # [9/11] Honeycomb vs Solid Cylinder Comparison
    # ══════════════════════════════════════════════════════════
    print("[9/11] Honeycomb vs solid cylinder comparison...")
    # Reuse full-domain grid from step 1 for honeycomb
    # Compute analytic solid cylinder total field on same grid
    x_2d, y_2d = np.meshgrid(f["x"], f["y"], indexing="xy")
    phi_solid_total = analytic_total(x_2d.ravel(), y_2d.ravel(), k,
                                     a=config.cluster_radius,
                                     n_terms=config.n_series_terms)
    re_solid = np.real(phi_solid_total).reshape(gs, gs)

    vmax_cmp = max(np.nanmax(np.abs(f["ut"])), np.nanmax(np.abs(re_solid)), 1e-6)
    fig = _heatmap_fig([
        {"z": f["ut"], "x": f["x"], "y": f["y"], "colorscale": "RdBu_r",
         "zmid": 0, "zmin": -vmax_cmp, "zmax": vmax_cmp,
         "title": "Honeycomb Re(phi_total)"},
        {"z": re_solid, "x": f["x"], "y": f["y"], "colorscale": "RdBu_r",
         "zmid": 0, "zmin": -vmax_cmp, "zmax": vmax_cmp,
         "title": f"Solid Cylinder Re(phi_total) (a={config.cluster_radius})"},
    ], f"Honeycomb vs Solid Cylinder — ka={config.ka:.2f}", scats)
    # Add solid cylinder outline to panel 2
    theta_c = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=config.cluster_radius * np.cos(theta_c),
        y=config.cluster_radius * np.sin(theta_c),
        mode="lines", line=dict(color="white", width=2, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=2)
    fig.write_html(os.path.join(out, "hc_vs_solid.html"))
    print("  -> hc_vs_solid.html")

    # ══════════════════════════════════════════════════════════
    # [10/11] Energy Flux Consistency
    # ══════════════════════════════════════════════════════════
    print("[10/11] Energy flux consistency...")
    flux_radii = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    flux_radii = flux_radii[flux_radii < L]
    mean_intensity = []

    for r in flux_radii:
        theta_e, u_e, v_e, xe, ye = _eval_circle(model, config, r, n=720)
        inc_e = np.exp(1j * k * xe)
        ut_e = u_e + np.real(inc_e)
        vt_e = v_e + np.imag(inc_e)
        mean_intensity.append(float(np.mean(ut_e ** 2 + vt_e ** 2)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=flux_radii, y=mean_intensity, mode="lines+markers",
        marker=dict(size=8), line=dict(width=2),
        text=[f"{v:.4f}" for v in mean_intensity], textposition="top center",
    ))
    # Reference: mean of all intensities
    avg_I = np.mean(mean_intensity)
    fig.add_hline(y=avg_I, line_dash="dash", line_color="gray",
                  annotation_text=f"mean={avg_I:.4f}")
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"Mean |phi_total|^2 on Concentric Circles — ka={config.ka:.2f}",
                   **_TITLE_STYLE),
        xaxis_title="r", yaxis_title="<|phi_total|^2>",
        width=700, height=450, showlegend=False,
    )
    fig.write_html(os.path.join(out, "hc_energy_flux.html"))

    variation = float(np.std(mean_intensity) / np.mean(mean_intensity))
    summary["energy_flux"] = {
        "radii": flux_radii.tolist(),
        "mean_intensity": mean_intensity,
        "coefficient_of_variation": variation,
    }
    print(f"  Energy flux CoV: {variation:.4f}")
    print("  -> hc_energy_flux.html")

    # ══════════════════════════════════════════════════════════
    # [11/11] Lattice Geometry Visualization
    # ══════════════════════════════════════════════════════════
    print("[11/11] Lattice geometry diagram...")
    lam = 2 * np.pi / k  # wavelength

    fig = go.Figure()

    # Filled scatterer circles
    theta_c = np.linspace(0, 2 * np.pi, 100)
    for i, (cx, cy, rs) in enumerate(scats):
        fig.add_trace(go.Scatter(
            x=cx + rs * np.cos(theta_c), y=cy + rs * np.sin(theta_c),
            fill="toself", fillcolor="rgba(150,150,150,0.5)",
            line=dict(color="rgba(80,80,80,1)", width=1),
            showlegend=False, hovertext=f"Scatterer {i} ({cx:.2f},{cy:.2f})",
        ))

    # Cluster boundary (dashed)
    a_c = config.cluster_radius
    fig.add_trace(go.Scatter(
        x=a_c * np.cos(theta_c), y=a_c * np.sin(theta_c),
        mode="lines", line=dict(color="#dc2626", width=1.5, dash="dash"),
        name=f"Cluster radius a={a_c}",
    ))

    # Lattice constant d (line from center to first ring-1 neighbor)
    fig.add_trace(go.Scatter(
        x=[0, d], y=[0, 0], mode="lines+markers",
        line=dict(color="#1a56db", width=2),
        marker=dict(size=5, color="#1a56db"),
        name=f"d={d:.2f}",
    ))
    fig.add_annotation(x=d / 2, y=-0.06, text=f"d={d:.2f}",
                       showarrow=False, font=dict(size=11, color="#1a56db"))

    # Gap annotation (between center and ring-1 neighbor along x-axis)
    gap = d - 2 * r_s
    fig.add_shape(type="line", x0=r_s, y0=0.12, x1=d - r_s, y1=0.12,
                  line=dict(color="#059669", width=2))
    fig.add_annotation(x=(r_s + d - r_s) / 2, y=0.16,
                       text=f"gap={gap:.2f}", showarrow=False,
                       font=dict(size=10, color="#059669"))

    # Wavelength scale bar
    bar_y = -(a_c + 0.3)
    bar_x0 = -lam / 2
    fig.add_shape(type="line", x0=bar_x0, y0=bar_y, x1=bar_x0 + lam, y1=bar_y,
                  line=dict(color="#7c3aed", width=3))
    fig.add_annotation(x=bar_x0 + lam / 2, y=bar_y - 0.08,
                       text=f"lambda={lam:.2f}", showarrow=False,
                       font=dict(size=11, color="#7c3aed"))

    # r_s annotation on one scatterer
    fig.add_annotation(x=r_s, y=0, text=f"r_s={r_s:.2f}", showarrow=True,
                       ax=40, ay=-30, font=dict(size=10))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"Honeycomb Lattice: 19 circles, ka={config.ka:.2f}", **_TITLE_STYLE),
        xaxis=dict(scaleanchor="y", constrain="domain", title="x"),
        yaxis=dict(title="y"),
        width=650, height=650,
    )
    fig.write_html(os.path.join(out, "hc_geometry.html"))
    print("  -> hc_geometry.html")

    # ══════════════════════════════════════════════════════════
    # Summary JSON + Diagnosis
    # ══════════════════════════════════════════════════════════
    diagnosis = []
    if pde_stats["mean"] > 1e-2:
        diagnosis.append(f"HIGH PDE residual mean={pde_stats['mean']:.3e} — network under-trained")
    elif pde_stats["mean"] > 1e-3:
        diagnosis.append(f"MODERATE PDE residual mean={pde_stats['mean']:.3e}")
    else:
        diagnosis.append(f"LOW PDE residual mean={pde_stats['mean']:.3e} — good convergence")

    if bc["overall_max"] > 1e-1:
        diagnosis.append(f"HIGH BC residual max={bc['overall_max']:.3e} — increase lambda_bc or n_boundary")
    elif bc["overall_max"] > 1e-2:
        diagnosis.append(f"MODERATE BC residual max={bc['overall_max']:.3e}")

    if abc["max"] > 1e-1:
        diagnosis.append(f"HIGH ABC residual max={abc['max']:.3e} — increase L or check outer boundary")
    elif abc["max"] > 1e-2:
        diagnosis.append(f"MODERATE ABC residual max={abc['max']:.3e}")

    if variation > 0.1:
        diagnosis.append(f"Energy flux CoV={variation:.3f} — poor energy conservation")
    elif variation > 0.03:
        diagnosis.append(f"Energy flux CoV={variation:.3f} — moderate energy conservation")
    else:
        diagnosis.append(f"Energy flux CoV={variation:.3f} — good energy conservation")

    summary["diagnosis"] = " | ".join(diagnosis)

    summary_path = os.path.join(out, "hc_eval_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSummary saved: {summary_path}")
    print(f"DIAGNOSIS: {summary['diagnosis']}")
    print(f"\nAll outputs ({12} files) in: {out}/")


if __name__ == "__main__":
    main()
