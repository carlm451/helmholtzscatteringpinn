"""Plotly visualization for HelmholtzPINN results."""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

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


def _scatterer_circle_trace(cx, cy, r, n_pts=100):
    """Create a Plotly trace for a scatterer circle outline."""
    theta = np.linspace(0, 2 * np.pi, n_pts)
    return go.Scatter(
        x=cx + r * np.cos(theta),
        y=cy + r * np.sin(theta),
        mode="lines",
        line=dict(color="white", width=2),
        showlegend=False,
    )


def plot_field_2d(values, x_range, y_range, title, scatterers=None,
                  colorscale="RdBu_r", zmid=0):
    """Plot a 2D field as a heatmap with scatterer circles."""
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=values,
        x=np.linspace(x_range[0], x_range[1], values.shape[1]),
        y=np.linspace(y_range[0], y_range[1], values.shape[0]),
        colorscale=colorscale,
        zmid=zmid,
        colorbar=dict(title="Field"),
    ))

    if scatterers:
        for cx, cy, r in scatterers:
            fig.add_trace(_scatterer_circle_trace(cx, cy, r))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text=title, **_TITLE_STYLE),
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(scaleanchor="y", constrain="domain"),
        width=700, height=600,
    )
    return fig


def _get_fields_on_grid(model, domain, config, analytic_fn, x_range, y_range, grid_size=200):
    """Compute PINN and analytic fields on a grid."""
    x_flat, y_flat, valid = domain.sample_test_grid(grid_size, x_range, y_range)

    # PINN
    with torch.no_grad():
        model.eval()
        u_pred, v_pred = model(x_flat, y_flat)
        model.train()
    u_pred = u_pred.cpu().numpy()
    v_pred = v_pred.cpu().numpy()

    # Analytic
    x_np = x_flat.cpu().numpy()
    y_np = y_flat.cpu().numpy()
    phi_analytic = analytic_fn(x_np, y_np, config.k, config.a, config.n_series_terms)
    u_exact = np.real(phi_analytic)
    v_exact = np.imag(phi_analytic)

    # Mask invalid points
    valid_np = valid.cpu().numpy()
    u_pred[~valid_np] = np.nan
    v_pred[~valid_np] = np.nan
    u_exact[~valid_np] = np.nan
    v_exact[~valid_np] = np.nan

    # Reshape to 2D grid
    def reshape(arr):
        return arr.reshape(grid_size, grid_size)

    return {
        "u_pred": reshape(u_pred), "v_pred": reshape(v_pred),
        "u_exact": reshape(u_exact), "v_exact": reshape(v_exact),
        "valid": reshape(valid_np),
    }


def plot_comparison(model, domain, config, analytic_fn,
                    x_range=None, y_range=None, grid_size=200, component="real"):
    """3-panel comparison: PINN | Analytic | Error."""
    if x_range is None:
        x_range = (-config.L, config.L)
    if y_range is None:
        y_range = (-config.L, config.L)

    fields = _get_fields_on_grid(model, domain, config, analytic_fn, x_range, y_range, grid_size)

    if component == "real":
        pred, exact = fields["u_pred"], fields["u_exact"]
        label = "Re(phi_s)"
    elif component == "imag":
        pred, exact = fields["v_pred"], fields["v_exact"]
        label = "Im(phi_s)"
    else:
        pred = np.sqrt(fields["u_pred"] ** 2 + fields["v_pred"] ** 2)
        exact = np.sqrt(fields["u_exact"] ** 2 + fields["v_exact"] ** 2)
        label = "|phi_s|"

    error = np.abs(pred - exact)

    x_arr = np.linspace(x_range[0], x_range[1], grid_size)
    y_arr = np.linspace(y_range[0], y_range[1], grid_size)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[f"PINN {label}", f"Analytic {label}", "Abs Error"],
                        horizontal_spacing=0.08)

    vmax = max(np.nanmax(np.abs(pred)), np.nanmax(np.abs(exact)))
    for col, data in enumerate([pred, exact], 1):
        fig.add_trace(go.Heatmap(
            z=data, x=x_arr, y=y_arr,
            colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
            showscale=(col == 2),
        ), row=1, col=col)

    fig.add_trace(go.Heatmap(
        z=error, x=x_arr, y=y_arr,
        colorscale="Hot_r",
        showscale=True,
    ), row=1, col=3)

    # Add scatterer circles to each subplot
    for col in range(1, 4):
        for cx, cy, r in config.scatterers:
            theta = np.linspace(0, 2 * np.pi, 100)
            fig.add_trace(go.Scatter(
                x=cx + r * np.cos(theta), y=cy + r * np.sin(theta),
                mode="lines", line=dict(color="white", width=1.5),
                showlegend=False,
            ), row=1, col=col)

    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"Helmholtz Scattering ka={config.ka:.2f} — {label}", **_TITLE_STYLE),
        width=1500, height=500,
    )
    for i in range(1, 4):
        fig.update_xaxes(scaleanchor=f"y{i if i > 1 else ''}", constrain="domain", row=1, col=i)

    return fig


def plot_line_cut(model, analytic_fn, config, x_values, y_value=0.0):
    """1D line cut comparing PINN vs analytic."""
    x_t = torch.tensor(x_values, dtype=torch.float32, device=config.device)
    y_t = torch.full_like(x_t, y_value)

    with torch.no_grad():
        model.eval()
        u_pred, v_pred = model(x_t, y_t)
        model.train()

    u_pred = u_pred.cpu().numpy()
    v_pred = v_pred.cpu().numpy()
    pred_mag = np.sqrt(u_pred ** 2 + v_pred ** 2)

    from analytic import scattered_field, total_field
    phi_total = total_field(x_values, np.full_like(x_values, y_value),
                            config.k, config.a, config.n_series_terms)

    # PINN total = PINN scattered + incident
    inc = np.exp(1j * config.k * x_values)
    pinn_total = (u_pred + 1j * v_pred) + inc

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=np.abs(phi_total),
                             mode="lines", name="Analytic |phi_total|",
                             line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x_values, y=np.abs(pinn_total),
                             mode="lines", name="PINN |phi_total|",
                             line=dict(width=2, dash="dash")))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"Line cut y={y_value} — ka={config.ka:.2f}", **_TITLE_STYLE),
        xaxis_title="x", yaxis_title="|phi_total|",
        width=800, height=400,
    )
    return fig


def create_zoom_report(model, domain, config, analytic_fn, output_dir="outputs"):
    """Generate HTML files for each zoom level."""
    os.makedirs(output_dir, exist_ok=True)
    L = config.L
    ka_str = f"ka{config.ka:.2f}"
    saved = []

    # 1. Global view
    fig = plot_comparison(model, domain, config, analytic_fn,
                          (-L, L), (-L, L), 200, "real")
    path = os.path.join(output_dir, f"{ka_str}_global_real.html")
    fig.write_html(path)
    saved.append(path)

    fig = plot_comparison(model, domain, config, analytic_fn,
                          (-L, L), (-L, L), 200, "magnitude")
    path = os.path.join(output_dir, f"{ka_str}_global_mag.html")
    fig.write_html(path)
    saved.append(path)

    # 2. Shadow boundary
    fig = plot_comparison(model, domain, config, analytic_fn,
                          (1, 5), (-3, 3), 200, "real")
    path = os.path.join(output_dir, f"{ka_str}_shadow_boundary.html")
    fig.write_html(path)
    saved.append(path)

    # 3. Near surface
    fig = plot_comparison(model, domain, config, analytic_fn,
                          (-2, -0.9), (-0.5, 0.5), 200, "real")
    path = os.path.join(output_dir, f"{ka_str}_near_surface.html")
    fig.write_html(path)
    saved.append(path)

    # 4. Poisson bright spot — line cut along y=0 behind scatterer
    x_vals = np.linspace(1.05, min(10, L), 300)
    fig = plot_line_cut(model, analytic_fn, config, x_vals, y_value=0.0)
    fig.update_layout(title=dict(text=f"Poisson Bright Spot — ka={config.ka:.2f}", **_TITLE_STYLE))
    path = os.path.join(output_dir, f"{ka_str}_poisson_bright_spot.html")
    fig.write_html(path)
    saved.append(path)

    # 5. Wake fringes (clamp to domain)
    x_hi = min(8, L)
    y_hi = min(4, L)
    if x_hi > 3:
        fig = plot_comparison(model, domain, config, analytic_fn,
                              (3, x_hi), (-y_hi, y_hi), 200, "real")
        path = os.path.join(output_dir, f"{ka_str}_wake_fringes.html")
        fig.write_html(path)
        saved.append(path)

    # 6. Overview with zoom rectangles
    fig = plot_comparison(model, domain, config, analytic_fn,
                          (-L, L), (-L, L), 300, "real")
    # Add rectangles for zoom regions
    zoom_regions = [
        ((1, -3), (5, 3), "Shadow boundary"),
        ((-2, -0.5), (-0.9, 0.5), "Near surface"),
    ]
    for (x0, y0), (x1, y1), label in zoom_regions:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="lime", width=2), row=1, col=1)
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="lime", width=2), row=1, col=2)
    fig.update_layout(title=dict(text=f"Overview with Zoom Regions — ka={config.ka:.2f}", **_TITLE_STYLE))
    path = os.path.join(output_dir, f"{ka_str}_overview.html")
    fig.write_html(path)
    saved.append(path)

    print(f"Zoom report saved ({len(saved)} files) to {output_dir}/")
    for p in saved:
        print(f"  {p}")
    return saved
