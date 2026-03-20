"""Multi-ka analytic visualization for Helmholtz scattering.

Generates rich HTML pages showing how scattering physics changes across ka regimes,
plus a comparison template for later PINN validation.
"""

import math
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytic import incident_field, scattered_field, total_field

# Consistent plot styling across all outputs
_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, Helvetica, Arial, sans-serif"
_TITLE_STYLE = dict(font=dict(size=15, color="#1a1a2e"), x=0.01, xanchor="left")
_LAYOUT = dict(
    font=dict(family=_FONT, size=13, color="#1a1a2e"),
    paper_bgcolor="#f8f8fa",
    plot_bgcolor="#ffffff",
    margin=dict(t=56, b=48, l=56, r=28),
    colorway=["#1a56db", "#dc2626", "#059669", "#d97706", "#7c3aed"],
)

# ka values spanning sub-wavelength to geometric optics
KA_VALUES = [0.5, 1.0, math.pi, 2 * math.pi, 3 * math.pi]
KA_LABELS = {
    0.5: "ka=0.5 (Rayleigh)",
    1.0: "ka=1.0 (Transition)",
    math.pi: "ka=pi (Moderate)",
    2 * math.pi: "ka=2pi (Strong)",
    3 * math.pi: "ka=3pi (Geometric)",
}

L = 5.0          # Domain half-size (breathing room for high ka)
A = 1.0          # Scatterer radius
GRID_SIZE = 300  # Points per axis


def _n_terms(ka):
    """Series truncation matching config.py logic."""
    return int(30 + ka ** 1.01) + 5


def generate_ka_sweep_grid(ka_values, L, grid_size):
    """Compute analytic fields for all ka values on a shared grid.

    Returns dict: {ka: {"phi_total": 2D, "phi_scattered": 2D, "x": 1D, "y": 1D}}
    """
    x = np.linspace(-L, L, grid_size)
    y = np.linspace(-L, L, grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    fields = {}
    for ka in ka_values:
        k = ka / A
        n_terms = _n_terms(ka)
        print(f"  Computing ka={ka:.4f} (k={k:.4f}, {2*n_terms+1} terms)...")
        phi_t = total_field(xx, yy, k, a=A, n_terms=n_terms)
        phi_s = scattered_field(xx, yy, k, a=A, n_terms=n_terms)
        fields[ka] = {
            "phi_total": phi_t,
            "phi_scattered": phi_s,
            "x": x,
            "y": y,
        }
    return fields


def _scatterer_traces(row, col):
    """White circle trace for the scatterer."""
    theta = np.linspace(0, 2 * np.pi, 100)
    return go.Scatter(
        x=A * np.cos(theta),
        y=A * np.sin(theta),
        mode="lines",
        line=dict(color="white", width=2),
        showlegend=False,
    )


def plot_ka_sweep(fields_dict):
    """Build 5-row x 3-col subplot: Re(phi_total) | |phi_total| | |phi_scattered|."""
    ka_list = sorted(fields_dict.keys())
    n_rows = len(ka_list)

    row_labels = [KA_LABELS.get(ka, f"ka={ka:.2f}") for ka in ka_list]
    col_labels = ["Re(phi_total)", "|phi_total|", "|phi_scattered|"]
    subplot_titles = []
    for rl in row_labels:
        for cl in col_labels:
            subplot_titles.append(f"{rl}<br>{cl}")

    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.04,
    )

    for i, ka in enumerate(ka_list):
        r = i + 1
        f = fields_dict[ka]
        x, y = f["x"], f["y"]
        phi_t = f["phi_total"]
        phi_s = f["phi_scattered"]

        data_sets = [
            (np.real(phi_t), "RdBu_r", 0),
            (np.abs(phi_t), "Viridis", None),
            (np.abs(phi_s), "Viridis", None),
        ]

        for c, (data, colorscale, zmid) in enumerate(data_sets, 1):
            heatmap_kwargs = dict(
                z=data, x=x, y=y,
                colorscale=colorscale,
                showscale=(i == 0),  # Only top row gets colorbars
            )
            if zmid is not None:
                heatmap_kwargs["zmid"] = zmid
            fig.add_trace(go.Heatmap(**heatmap_kwargs), row=r, col=c)

            # Scatterer circle
            fig.add_trace(_scatterer_traces(r, c), row=r, col=c)

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Analytic Helmholtz Scattering — ka Sweep", **_TITLE_STYLE),
        width=1500,
        height=400 * n_rows,
    )

    # Equal aspect ratio for all subplots
    for i in range(1, n_rows * 3 + 1):
        axis_suffix = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"xaxis{axis_suffix}": dict(scaleanchor=f"y{axis_suffix}", constrain="domain"),
        })

    return fig


def plot_ka_line_cuts(ka_values, L):
    """Overlaid 1D line cuts of |phi_total| along y=0 for all ka values."""
    x = np.linspace(-L, L, 1000)
    y_val = np.zeros_like(x)

    fig = go.Figure()

    colors = _LAYOUT["colorway"]
    for i, ka in enumerate(sorted(ka_values)):
        k = ka / A
        n_terms = _n_terms(ka)
        phi_t = total_field(x, y_val, k, a=A, n_terms=n_terms)
        # Mask interior
        mask = np.abs(x) < A
        mag = np.abs(phi_t)
        mag[mask] = np.nan

        label = KA_LABELS.get(ka, f"ka={ka:.2f}")
        fig.add_trace(go.Scatter(
            x=x, y=mag,
            mode="lines",
            name=label,
            line=dict(width=2, color=colors[i % len(colors)]),
        ))

    # Add scatterer shading
    fig.add_vrect(x0=-A, x1=A, fillcolor="gray", opacity=0.3,
                  line_width=0, annotation_text="scatterer")

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Line Cuts Along y = 0 — |phi_total| vs x", **_TITLE_STYLE),
        xaxis_title="x",
        yaxis_title="|phi_total|",
        width=1000,
        height=500,
        legend=dict(x=0.02, y=0.98, font=dict(size=12)),
    )
    return fig


def plot_multi_ka_comparison(ka_model_dict, analytic_fields):
    """PINN vs Analytic comparison grid. Reusable when checkpoints are ready.

    Args:
        ka_model_dict: {ka: model} or {ka: None} for placeholder
        analytic_fields: dict from generate_ka_sweep_grid
    """
    ka_list = sorted(analytic_fields.keys())
    n_rows = len(ka_list)
    col_labels = ["PINN Re(phi_s)", "Analytic Re(phi_s)", "|Error|"]
    subplot_titles = []
    row_labels = [KA_LABELS.get(ka, f"ka={ka:.2f}") for ka in ka_list]
    for rl in row_labels:
        for cl in col_labels:
            subplot_titles.append(f"{rl}<br>{cl}")

    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.04,
    )

    for i, ka in enumerate(ka_list):
        r = i + 1
        f = analytic_fields[ka]
        x, y = f["x"], f["y"]
        phi_s = f["phi_scattered"]
        analytic_re = np.real(phi_s)
        vmax = np.nanmax(np.abs(analytic_re))

        model = ka_model_dict.get(ka)

        if model is not None:
            # TODO: evaluate PINN model on grid, compute error
            pinn_re = np.full_like(analytic_re, np.nan)
            error = np.full_like(analytic_re, np.nan)
        else:
            pinn_re = np.full_like(analytic_re, np.nan)
            error = np.full_like(analytic_re, np.nan)

        # Col 1: PINN (placeholder for now)
        fig.add_trace(go.Heatmap(
            z=pinn_re, x=x, y=y,
            colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
            showscale=False,
        ), row=r, col=1)

        if model is None:
            fig.add_annotation(
                text="Awaiting checkpoint",
                x=0, y=0, xref=f"x{(r-1)*3+1}", yref=f"y{(r-1)*3+1}",
                showarrow=False, font=dict(size=16, color="white"),
            )

        # Col 2: Analytic
        fig.add_trace(go.Heatmap(
            z=analytic_re, x=x, y=y,
            colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
            showscale=(i == 0),
        ), row=r, col=2)

        # Col 3: Error (empty for now)
        fig.add_trace(go.Heatmap(
            z=error, x=x, y=y,
            colorscale="Hot_r",
            showscale=False,
        ), row=r, col=3)

        # Scatterer circles
        for c in range(1, 4):
            fig.add_trace(_scatterer_traces(r, c), row=r, col=c)

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Multi-ka PINN vs Analytic Comparison", **_TITLE_STYLE),
        width=1500,
        height=400 * n_rows,
    )
    return fig


def main():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating multi-ka analytic sweep...")
    print(f"  ka values: {[f'{ka:.4f}' for ka in KA_VALUES]}")
    print(f"  Domain: [-{L}, {L}]^2, grid: {GRID_SIZE}x{GRID_SIZE}")

    # Compute all fields
    fields = generate_ka_sweep_grid(KA_VALUES, L, GRID_SIZE)

    # Figure 1: 5x3 sweep grid
    print("\nPlotting ka sweep grid (5x3)...")
    fig1 = plot_ka_sweep(fields)
    path1 = os.path.join(output_dir, "analytic_ka_sweep.html")
    fig1.write_html(path1)
    print(f"  Saved: {path1}")

    # Figure 2: PINN comparison template
    print("\nPlotting comparison template...")
    ka_model_dict = {ka: None for ka in KA_VALUES}  # No models yet
    fig2 = plot_multi_ka_comparison(ka_model_dict, fields)
    path2 = os.path.join(output_dir, "analytic_ka_comparison_template.html")
    fig2.write_html(path2)
    print(f"  Saved: {path2}")

    # Figure 3: Line cuts
    print("\nPlotting line cuts along y=0...")
    fig3 = plot_ka_line_cuts(KA_VALUES, L)
    path3 = os.path.join(output_dir, "analytic_line_cuts.html")
    fig3.write_html(path3)
    print(f"  Saved: {path3}")

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
