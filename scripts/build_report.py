#!/usr/bin/env python3
"""Build HTML report (slides + dashboard) for HelmholtzPINN project.

Generates:
  docs/slides.html  -- Self-contained HTML slide deck (arrow-key nav)
  docs/index.html   -- Interactive dashboard for GitHub Pages

Usage:
    source .env && .venv/bin/python scripts/build_report.py
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helmholtz.visualize import _LAYOUT, _TITLE_STYLE

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb unavailable. Charts requiring history will be skipped.")


# ════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════

PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.4.0.min.js"
MATHJAX_CDN = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
DOCS = "docs"
PLOTS = os.path.join(DOCS, "plots")
OUTPUTS = "outputs"

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "system-ui, Helvetica, Arial, sans-serif")

C = dict(blue="#1a56db", red="#dc2626", green="#059669",
         amber="#d97706", purple="#7c3aed", teal="#0891b2",
         text="#1a1a2e", text2="#52525b", text3="#94949e",
         bg="#f8f8fa", card="#ffffff", rule="#e0e0e4")

PROD = [
    dict(ka=0.5, label="0.5", l2="8.23", mx="4.04", me="1.90",
         net="256 / 4L / 64ff", ep="10K+200", t="12 min",
         wn="prod-ka0.50-circ-bgt2", c=C["blue"], slug="ka0.50"),
    dict(ka=1.0, label="1.0", l2="3.57", mx="2.99", me="1.34",
         net="256 / 4L / 64ff", ep="10K+200", t="13 min",
         wn="prod-ka1.00-circ-bgt2", c=C["green"], slug="ka1.00"),
    dict(ka=3.14, label="\u03c0", l2="2.41", mx="2.96", me="1.09",
         net="256 / 4L / 64ff", ep="10K+200", t="17 min",
         wn="prod-ka_pi-circ-bgt2", c=C["amber"], slug="ka3.14"),
    dict(ka=6.28, label="2\u03c0", l2="2.00", mx="4.37", me="0.93",
         net="384 / 6L / 96ff", ep="50K+300", t="259 min",
         wn="prod-ka_2pi-circ-bgt2-long", c=C["red"], slug="ka6.28"),
]


# ════════════════════════════════════════════════════════════════
#  Data Collection
# ════════════════════════════════════════════════════════════════

def _extract(rows, key):
    s, v = [], []
    for r in rows:
        si, vi = r.get("_step"), r.get(key)
        if si is not None and vi is not None:
            s.append(si)
            v.append(vi)
    if s:
        s, v = zip(*sorted(zip(s, v)))
        s, v = list(s), list(v)
    return s, v


def fetch_wandb():
    if not HAS_WANDB:
        return {}, {}
    api = wandb.Api()
    ent = api.default_entity
    prod, abl = {}, {}
    for proj, tgt in [(f"{ent}/helmholtz-pinn-prod", prod),
                      (f"{ent}/helmholtz-pinn-ablation", abl)]:
        try:
            print(f"  Fetching {proj} ...")
            for run in api.runs(proj):
                nm = run.name or run.id
                rows = list(run.scan_history())
                print(f"    {nm}: {len(rows)} rows")
                if tgt is abl:
                    # Categorize ablation runs by ka regime and architecture
                    if "ka-pi" in nm:
                        k = "ff" if "ff-pinn" in nm else "mlp"
                        tgt[k] = dict(rows=rows, name=nm)
                    elif "ka-2pi" in nm:
                        k = "ff_2pi" if "ff-pinn" in nm else "mlp_2pi"
                        tgt[k] = dict(rows=rows, name=nm)
                    else:
                        continue
                else:
                    tgt[nm] = dict(rows=rows)
        except Exception as e:
            print(f"  Warning: {e}")
    return prod, abl


# ════════════════════════════════════════════════════════════════
#  Figure Extraction & Generation
# ════════════════════════════════════════════════════════════════

def extract_figure_div(html_path, new_div_id, width="100%", height="480px"):
    """Extract Plotly chart from an existing write_html() output file.

    Finds the plotly-graph-div and its associated script block (after the
    bundled plotly.js), swaps the UUID for *new_div_id*, and returns an
    HTML fragment that works with an external CDN plotly.js.
    """
    import re
    try:
        with open(html_path) as f:
            content = f.read()
    except FileNotFoundError:
        return f'<p class="small">Figure not found: {os.path.basename(html_path)}</p>'

    div_pat = re.compile(
        r'<div id="([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'
        r'[0-9a-f]{4}-[0-9a-f]{12})" class="plotly-graph-div"')
    divs = list(div_pat.finditer(content))
    if not divs:
        return '<p class="small">No Plotly figure found in file</p>'

    last = divs[-1]
    old_id = last.group(1)
    start = last.start()
    script_end = content.find("</script>", start)
    chunk = content[start:script_end + len("</script>")]

    # Swap UUID → our div id and resize
    chunk = chunk.replace(old_id, new_div_id)
    chunk = re.sub(r'style="height:\d+px;\s*width:\d+px;"',
                   f'style="height:{height};width:{width};"', chunk)
    return chunk


def generate_analytic_gallery():
    """Create a 2×2 subplot of analytic scattered fields for 4 ka values.

    Pure numpy/scipy — no model loading needed.
    """
    import numpy as np
    from helmholtz.analytic import scattered_field

    kas = [0.5, 1.0, 3.14159, 6.28318]
    labels = ["ka = 0.5", "ka = 1.0", "ka = π", "ka = 2π"]
    N = 150  # grid resolution (smaller than 200 for speed)
    L = 3.0

    fig = make_subplots(rows=2, cols=2, subplot_titles=labels,
                        horizontal_spacing=0.08, vertical_spacing=0.1)

    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)
    xf, yf = X.ravel(), Y.ravel()
    r = np.sqrt(xf**2 + yf**2)

    for idx, ka in enumerate(kas):
        k = ka / 1.0  # a = 1
        n_terms = int(ka + 20)
        phi = scattered_field(xf, yf, k, 1.0, n_terms)
        vals = np.real(phi).reshape(N, N)
        vals[r.reshape(N, N) < 1.0] = np.nan  # mask scatterer interior

        row, col = divmod(idx, 2)
        vmax = float(np.nanmax(np.abs(vals)))
        fig.add_trace(go.Heatmap(
            z=vals, x=x, y=y, colorscale="RdBu_r",
            zmid=0, zmin=-vmax, zmax=vmax,
            showscale=(idx == 1),
            colorbar=dict(len=0.45, y=0.78) if idx == 1 else None,
        ), row=row+1, col=col+1)

        # Scatterer circle
        theta = np.linspace(0, 2*np.pi, 80)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta), mode="lines",
            line=dict(color="white", width=1.5), showlegend=False,
        ), row=row+1, col=col+1)

    fig.update_layout(**_lay(
        title=dict(text="Analytic Scattered Field — Increasing Wavenumber",
                   **_TITLE_STYLE),
        width=800, height=700,
        margin=dict(t=55, b=30, l=40, r=30)))

    for i in range(1, 5):
        fig.update_xaxes(scaleanchor=f"y{i if i > 1 else ''}",
                         constrain="domain", showticklabels=False, row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(showticklabels=False, row=(i-1)//2+1, col=(i-1)%2+1)

    return fig


def generate_error_gallery():
    """Create a 2x2 subplot of absolute error |PINN - analytic| for 4 ka values.

    Loads production checkpoints and evaluates against the Bessel/Hankel series.
    """
    import torch
    import numpy as np
    from helmholtz.config import HelmholtzConfig
    from helmholtz.network import HelmholtzPINN
    from helmholtz.analytic import scattered_field

    specs = [
        dict(ka=0.5, label="ka = 0.5  |  L2 = 8.23%",
             ckpt="checkpoints/helmholtz_ka0.50_L3.0_circ_bgt2_0321_1309_lbfgs.pt",
             neurons=256, layers=4, ff=64),
        dict(ka=1.0, label="ka = 1.0  |  L2 = 3.57%",
             ckpt="checkpoints/helmholtz_ka1.00_L3.0_circ_bgt2_0321_1419_lbfgs.pt",
             neurons=256, layers=4, ff=64),
        dict(ka=3.14159, label="ka = \u03c0  |  L2 = 2.41%",
             ckpt="checkpoints/helmholtz_ka3.14_L3.0_circ_bgt2_0321_1314_lbfgs.pt",
             neurons=256, layers=4, ff=64),
        dict(ka=6.28318, label="ka = 2\u03c0  |  L2 = 2.00%",
             ckpt="checkpoints/helmholtz_ka6.28_L3.0_circ_bgt2_0321_1855_lbfgs.pt",
             neurons=384, layers=6, ff=96),
    ]

    N = 150
    L = 3.0
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)
    xf, yf = X.ravel(), Y.ravel()
    r = np.sqrt(xf**2 + yf**2)
    mask = r < 1.0  # inside scatterer
    circ_mask = (xf**2 + yf**2) > L**2  # outside circular domain

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[s["label"] for s in specs],
        horizontal_spacing=0.08, vertical_spacing=0.12)

    for idx, sp in enumerate(specs):
        row, col = divmod(idx, 2)
        ka = sp["ka"]
        k = ka / 1.0

        # Build model with matching config
        cfg = HelmholtzConfig(
            ka=ka, a=1.0, L=L,
            outer_boundary="circle", abc_order=2,
            n_hidden_neurons=sp["neurons"],
            n_hidden_layers=sp["layers"],
            n_fourier_features=sp["ff"],
            device="cpu",
        )
        model = HelmholtzPINN(cfg)

        if not os.path.exists(sp["ckpt"]):
            print(f"    Warning: checkpoint not found: {sp['ckpt']}")
            continue

        state = torch.load(sp["ckpt"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # PINN prediction
        xt = torch.tensor(xf, dtype=torch.float32)
        yt = torch.tensor(yf, dtype=torch.float32)
        with torch.no_grad():
            u, v = model(xt, yt)
        u_pred = u.numpy()
        v_pred = v.numpy()

        # Analytic
        phi = scattered_field(xf, yf, k, 1.0, int(ka + 20))
        u_exact = np.real(phi)
        v_exact = np.imag(phi)

        # Absolute error (magnitude of complex error)
        err = np.sqrt((u_pred - u_exact)**2 + (v_pred - v_exact)**2)
        err[mask] = np.nan
        err[circ_mask] = np.nan
        err = err.reshape(N, N)

        emax = float(np.nanpercentile(err, 99))  # clip outliers
        fig.add_trace(go.Heatmap(
            z=err, x=x, y=y, colorscale="Inferno",
            zmin=0, zmax=emax,
            showscale=(idx == 1),
            colorbar=dict(len=0.42, y=0.78,
                          title=dict(text="Error", font=dict(size=10))
                          ) if idx == 1 else None,
        ), row=row+1, col=col+1)

        # Scatterer circle
        theta = np.linspace(0, 2*np.pi, 80)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta), mode="lines",
            line=dict(color="white", width=1.5), showlegend=False,
        ), row=row+1, col=col+1)

    fig.update_layout(**_lay(
        title=dict(text="Absolute Error  |PINN \u2212 Analytic|", **_TITLE_STYLE),
        width=800, height=700,
        margin=dict(t=55, b=30, l=40, r=30)))

    for i in range(1, 5):
        r, c = (i-1)//2+1, (i-1)%2+1
        fig.update_xaxes(scaleanchor=f"y{i if i > 1 else ''}",
                         constrain="domain", showticklabels=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, row=r, col=c)

    return fig


def generate_honeycomb_residuals():
    """Generate a 2-panel figure: PDE residual heatmap + BC residual on 19 circles.

    Uses the production ka=2.0 honeycomb checkpoint.  PDE residual via finite
    differences (no autograd needed); BC residual via autograd normal derivatives.
    """
    import torch
    import numpy as np
    from helmholtz.config import HelmholtzConfig
    from helmholtz.network import HelmholtzPINN
    from helmholtz.honeycomb import generate_honeycomb_lattice

    ckpt = "checkpoints/helmholtz_hc_ka2.00_L4.0_circ_bgt2_0321_1729_lbfgs.pt"
    if not os.path.exists(ckpt):
        print(f"    Warning: {ckpt} not found, skipping")
        return None

    ka, a, L = 2.0, 1.0, 4.0
    k = ka / a
    scatterers = generate_honeycomb_lattice(a=a, r_s=0.15, d=0.4)

    cfg = HelmholtzConfig(
        ka=ka, a=a, L=L,
        outer_boundary="circle", abc_order=2,
        n_hidden_neurons=256, n_hidden_layers=6,
        n_fourier_features=128, device="cpu",
    )
    cfg.scatterers = scatterers

    model = HelmholtzPINN(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ── Panel 1: PDE residual via finite differences ──
    N = 200
    h = 2 * L / (N - 1)
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)

    with torch.no_grad():
        xt = torch.tensor(X.ravel(), dtype=torch.float32)
        yt = torch.tensor(Y.ravel(), dtype=torch.float32)
        u, v = model(xt, yt)
    U = u.numpy().reshape(N, N)
    V = v.numpy().reshape(N, N)

    # 5-point Laplacian stencil
    lap_u = np.full_like(U, np.nan)
    lap_v = np.full_like(V, np.nan)
    s = slice(1, -1)
    lap_u[s, s] = (U[2:, s] + U[:-2, s] + U[s, 2:] + U[s, :-2] - 4*U[s, s]) / h**2
    lap_v[s, s] = (V[2:, s] + V[:-2, s] + V[s, 2:] + V[s, :-2] - 4*V[s, s]) / h**2

    pde_res = np.sqrt((lap_u + k**2 * U)**2 + (lap_v + k**2 * V)**2)

    # Mask scatterer interiors + buffer + outside circle
    for cx, cy, r in scatterers:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        pde_res[dist < r + 2*h] = np.nan
    pde_res[X**2 + Y**2 > L**2] = np.nan

    # ── Panel 2: BC residual on 19 circles via autograd ──
    n_pts = 100  # points per circle
    bc_x_all, bc_y_all, bc_res_all = [], [], []

    for cx, cy, r in scatterers:
        theta = torch.linspace(0, 2 * np.pi, n_pts + 1)[:-1]
        bx = cx + r * torch.cos(theta)
        by = cy + r * torch.sin(theta)
        nx = torch.cos(theta)  # outward normal
        ny = torch.sin(theta)

        bx.requires_grad_(True)
        by.requires_grad_(True)
        u_bc, v_bc = model(bx, by)

        # Normal derivatives via autograd
        du_dx = torch.autograd.grad(u_bc.sum(), bx, retain_graph=True)[0]
        du_dy = torch.autograd.grad(u_bc.sum(), by, retain_graph=True)[0]
        dv_dx = torch.autograd.grad(v_bc.sum(), bx, retain_graph=True)[0]
        dv_dy = torch.autograd.grad(v_bc.sum(), by)[0]

        du_dn = du_dx * nx + du_dy * ny
        dv_dn = dv_dx * nx + dv_dy * ny

        # Target: -d(phi_inc)/dn where phi_inc = exp(ikx)
        # d(phi_inc)/dx = ik * exp(ikx)
        # d(phi_inc)/dn = ik * exp(ikx) * nx  (since phi_inc only depends on x)
        kx = k * bx.detach()
        target_re = nx * k * torch.sin(kx)    # Re(-ik * exp(ikx) * nx)
        target_im = -nx * k * torch.cos(kx)   # Im(-ik * exp(ikx) * nx)

        res_re = (du_dn.detach() - target_re)
        res_im = (dv_dn.detach() - target_im)
        res_mag = torch.sqrt(res_re**2 + res_im**2).numpy()

        bc_x_all.append(bx.detach().numpy())
        bc_y_all.append(by.detach().numpy())
        bc_res_all.append(res_mag)

    bc_x = np.concatenate(bc_x_all)
    bc_y = np.concatenate(bc_y_all)
    bc_res = np.concatenate(bc_res_all)

    # ── Build 2-panel figure ──
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "PDE Residual |&nabla;&sup2;&phi;<sub>s</sub> + k&sup2;&phi;<sub>s</sub>|",
            "Neumann BC Residual (19 surfaces)",
        ],
        horizontal_spacing=0.1)

    # Panel 1: PDE residual heatmap
    emax = float(np.nanpercentile(pde_res, 97))
    fig.add_trace(go.Heatmap(
        z=pde_res, x=x, y=y, colorscale="Inferno",
        zmin=0, zmax=emax, showscale=True,
        colorbar=dict(len=0.9, x=0.44,
                      title=dict(text="PDE res", font=dict(size=9))),
    ), row=1, col=1)

    theta_c = np.linspace(0, 2*np.pi, 60)
    for cx, cy, r in scatterers:
        fig.add_trace(go.Scatter(
            x=cx + r*np.cos(theta_c), y=cy + r*np.sin(theta_c),
            mode="lines", line=dict(color="white", width=0.6),
            showlegend=False,
        ), row=1, col=1)

    # Panel 2: BC residual as colored scatter on circle boundaries
    bc_vmax = float(np.percentile(bc_res, 98))
    fig.add_trace(go.Scatter(
        x=bc_x, y=bc_y, mode="markers",
        marker=dict(size=3.5, color=bc_res, colorscale="Inferno",
                    cmin=0, cmax=bc_vmax, showscale=True,
                    colorbar=dict(len=0.9, x=1.02,
                                  title=dict(text="BC res", font=dict(size=9)))),
        showlegend=False,
    ), row=1, col=2)

    # Scatterer circle outlines on panel 2
    for cx, cy, r in scatterers:
        fig.add_trace(go.Scatter(
            x=cx + r*np.cos(theta_c), y=cy + r*np.sin(theta_c),
            mode="lines", line=dict(color="rgba(255,255,255,0.25)", width=0.5),
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(**_lay(
        title=dict(text="Honeycomb Residual Diagnostics (ka=2)", **_TITLE_STYLE),
        width=950, height=450,
        margin=dict(t=55, b=30, l=35, r=50)))

    for col in [1, 2]:
        fig.update_xaxes(scaleanchor=f"y{'' if col == 1 else '2'}",
                         constrain="domain", showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)

    return fig


def generate_collocation_figure():
    """Sample actual collocation points using the domain code and plot them.

    Shows PDE interior points, Neumann BC points, and ABC points with
    distinct markers — illustrating the training data structure.
    """
    import torch
    from helmholtz.domain import ScatteringDomain

    L = 3.0
    a = 1.0
    scatterers = [(0.0, 0.0, a)]
    domain = ScatteringDomain(L, scatterers, device="cpu")

    # Sample each point set (smaller counts for visual clarity)
    with torch.no_grad():
        interior = domain.sample_interior(2000, strategy="uniform",
                                          outer_boundary="circle")
        bc = domain.sample_scatterer_boundary(120)
        abc = domain.sample_circular_outer_boundary(200)

    xi = interior["x"].detach().numpy()
    yi = interior["y"].detach().numpy()
    xb = bc["x"].detach().numpy()
    yb = bc["y"].detach().numpy()
    xa = abc["x"].detach().numpy()
    ya = abc["y"].detach().numpy()

    import numpy as np
    fig = go.Figure()

    # PDE collocation points (small, semi-transparent)
    fig.add_trace(go.Scatter(
        x=xi, y=yi, mode="markers", name="PDE collocation (interior)",
        marker=dict(size=2.5, color=C["blue"], opacity=0.35),
    ))

    # Neumann BC points (on scatterer surface)
    fig.add_trace(go.Scatter(
        x=xb, y=yb, mode="markers", name="Neumann BC (r = a)",
        marker=dict(size=5, color=C["red"], symbol="diamond"),
    ))

    # ABC points (on outer boundary)
    fig.add_trace(go.Scatter(
        x=xa, y=ya, mode="markers", name="ABC (r = L)",
        marker=dict(size=4, color=C["green"], symbol="square"),
    ))

    # Scatterer fill
    theta = np.linspace(0, 2*np.pi, 80)
    fig.add_trace(go.Scatter(
        x=a*np.cos(theta), y=a*np.sin(theta), fill="toself",
        fillcolor="rgba(26,26,46,0.15)", line=dict(color=C["text"], width=1.5),
        showlegend=False,
    ))

    # ABC boundary circle (dashed)
    fig.add_trace(go.Scatter(
        x=L*np.cos(theta), y=L*np.sin(theta), mode="lines",
        line=dict(color=C["text3"], width=1, dash="dash"), showlegend=False,
    ))

    fig.update_layout(**_lay(
        title=dict(text="Training Collocation Points", **_TITLE_STYLE),
        xaxis=dict(scaleanchor="y", constrain="domain",
                   range=[-3.6, 3.6], title="x"),
        yaxis=dict(range=[-3.6, 3.6], title="y"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    font=dict(size=11)),
        width=520, height=520,
        margin=dict(t=45, b=40, l=45, r=15)))

    return fig


def generate_collocation_3panel():
    """Generate 3 separate collocation point figures, one per point type.

    Returns dict with keys: 'pde', 'bc', 'abc' -> Plotly Figure objects.
    """
    import torch
    import numpy as np
    from helmholtz.domain import ScatteringDomain

    L = 3.0
    a = 1.0
    scatterers = [(0.0, 0.0, a)]
    domain = ScatteringDomain(L, scatterers, device="cpu")

    with torch.no_grad():
        interior = domain.sample_interior(3000, strategy="uniform",
                                          outer_boundary="circle")
        bc = domain.sample_scatterer_boundary(200)
        abc = domain.sample_circular_outer_boundary(300)

    theta = np.linspace(0, 2*np.pi, 80)

    def _base_fig(title, w=420, h=420):
        fig = go.Figure()
        # Scatterer fill
        fig.add_trace(go.Scatter(
            x=a*np.cos(theta), y=a*np.sin(theta), fill="toself",
            fillcolor="rgba(99,102,241,0.08)",
            line=dict(color="rgba(99,102,241,0.4)", width=1.5),
            showlegend=False))
        # ABC boundary (dashed)
        fig.add_trace(go.Scatter(
            x=L*np.cos(theta), y=L*np.sin(theta), mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dash"),
            showlegend=False))
        lay = dict(_DARK_LAYOUT)
        lay.update(
            title=dict(text=title, font=dict(size=12, color="#e8e6f0"),
                       x=0.5, xanchor="center"),
            xaxis=dict(scaleanchor="y", constrain="domain",
                       range=[-3.8, 3.8], showticklabels=False,
                       gridcolor="rgba(255,255,255,0.04)",
                       zerolinecolor="rgba(255,255,255,0.06)"),
            yaxis=dict(range=[-3.8, 3.8], showticklabels=False,
                       gridcolor="rgba(255,255,255,0.04)",
                       zerolinecolor="rgba(255,255,255,0.06)"),
            width=w, height=h,
            margin=dict(t=40, b=15, l=15, r=15))
        fig.update_layout(**lay)
        return fig

    # PDE collocation
    fig_pde = _base_fig("PDE Interior Points")
    fig_pde.add_trace(go.Scatter(
        x=interior["x"].detach().numpy(), y=interior["y"].detach().numpy(),
        mode="markers", name="PDE collocation",
        marker=dict(size=2.5, color="#818cf8", opacity=0.5),
        showlegend=False))

    # Neumann BC points
    fig_bc = _base_fig("Neumann BC Points (r = a)")
    fig_bc.add_trace(go.Scatter(
        x=bc["x"].detach().numpy(), y=bc["y"].detach().numpy(),
        mode="markers", name="Neumann BC",
        marker=dict(size=5, color="#f87171", symbol="diamond"),
        showlegend=False))

    # ABC points
    fig_abc = _base_fig("ABC Points (r = L)")
    fig_abc.add_trace(go.Scatter(
        x=abc["x"].detach().numpy(), y=abc["y"].detach().numpy(),
        mode="markers", name="ABC boundary",
        marker=dict(size=4, color="#2dd4bf", symbol="square"),
        showlegend=False))

    return {"pde": fig_pde, "bc": fig_bc, "abc": fig_abc}


# ════════════════════════════════════════════════════════════════
#  Chart Builders
# ════════════════════════════════════════════════════════════════

def _lay(**kw):
    """Merge _LAYOUT with overrides (avoids duplicate keyword errors)."""
    d = dict(_LAYOUT)
    d.update(kw)
    return d


# Dark-themed layout for slide-embedded charts
_DARK_LAYOUT = dict(
    font=dict(family="DM Sans, sans-serif", size=12, color="#c4c0d8"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,15,26,0.5)",
    colorway=["#818cf8", "#2dd4bf", "#fbbf24", "#f87171", "#a78bfa", "#34d399"],
    title=dict(font=dict(size=13, color="#e8e6f0"), x=0.01, xanchor="left"),
)

def _dark(**kw):
    """Dark layout for slide charts."""
    d = dict(_DARK_LAYOUT)
    d.update(kw)
    return d


def _div(fig, did):
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id=did)


def chart_prod_loss(prod):
    fig = go.Figure()
    for p in PROD:
        if p["wn"] in prod:
            s, v = _extract(prod[p["wn"]]["rows"], "loss/total")
            fig.add_trace(go.Scatter(x=s, y=v, mode="lines",
                                     name=f'ka={p["label"]}',
                                     line=dict(color=p["c"], width=2)))
    # Add ka=3π (not converged) if available
    for wn in ["prod-ka_3pi-circ-bgt2-long", "prod-ka_3pi-circ-bgt2"]:
        if wn in prod:
            s, v = _extract(prod[wn]["rows"], "loss/total")
            fig.add_trace(go.Scatter(x=s, y=v, mode="lines",
                                     name="ka=3\u03c0 (68% L2 \u2717)",
                                     line=dict(color="#6b7280", width=2,
                                               dash="dash")))
            break
    fig.update_layout(**_lay(
        title=dict(text="Training Convergence", **_TITLE_STYLE),
        xaxis_title="Epoch", yaxis_title="Total Loss",
        yaxis_type="log", legend=dict(x=0.7, y=0.96),
        width=850, height=400,
        margin=dict(t=50, b=45, l=55, r=20)))
    return fig


def chart_l2_bar():
    fig = go.Figure(go.Bar(
        x=[f'ka = {p["label"]}' for p in PROD],
        y=[float(p["l2"]) for p in PROD],
        marker_color=[p["c"] for p in PROD],
        text=[f'{p["l2"]}%' for p in PROD],
        textposition="outside", textfont=dict(size=13)))
    fig.update_layout(**_lay(
        title=dict(text="L2 Relative Error by Wavenumber", **_TITLE_STYLE),
        yaxis_title="L2 Rel Error (%)",
        yaxis=dict(range=[0, 11]),
        width=550, height=370,
        margin=dict(t=50, b=45, l=55, r=20)))
    return fig


def chart_max_err_bar():
    """Max pointwise error bar chart across production ka values."""
    fig = go.Figure(go.Bar(
        x=[f'ka = {p["label"]}' for p in PROD],
        y=[float(p["mx"]) for p in PROD],
        marker_color=[p["c"] for p in PROD],
        text=[f'{p["mx"]}%' for p in PROD],
        textposition="outside", textfont=dict(size=13)))
    fig.update_layout(**_lay(
        title=dict(text="Max Pointwise Error by Wavenumber", **_TITLE_STYLE),
        yaxis_title="Max Error (%)",
        yaxis=dict(range=[0, 6]),
        width=550, height=370,
        margin=dict(t=50, b=45, l=55, r=20)))
    return fig


def chart_mean_err_bar():
    """Mean pointwise error bar chart across production ka values."""
    fig = go.Figure(go.Bar(
        x=[f'ka = {p["label"]}' for p in PROD],
        y=[float(p["me"]) for p in PROD],
        marker_color=[p["c"] for p in PROD],
        text=[f'{p["me"]}%' for p in PROD],
        textposition="outside", textfont=dict(size=13)))
    fig.update_layout(**_lay(
        title=dict(text="Mean Pointwise Error by Wavenumber", **_TITLE_STYLE),
        yaxis_title="Mean Error (%)",
        yaxis=dict(range=[0, 3]),
        width=550, height=370,
        margin=dict(t=50, b=45, l=55, r=20)))
    return fig


def chart_errors_combined():
    """Combined 3-panel bar chart: L2 rel, mean, max error."""
    xlabs = [f'ka={p["label"]}' for p in PROD]
    colors = [p["c"] for p in PROD]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("L2 Relative (%)", "Mean (%)", "Max (%)"),
                        horizontal_spacing=0.08)
    for col, key, rng in [(1, "l2", [0, 11]), (2, "me", [0, 3]), (3, "mx", [0, 6])]:
        vals = [float(p[key]) for p in PROD]
        fig.add_trace(go.Bar(
            x=xlabs, y=vals, marker_color=colors,
            text=[f'{v:.1f}' if v < 10 else f'{v:.0f}' for v in vals],
            textposition="outside", textfont=dict(size=11),
            showlegend=False), row=1, col=col)
        fig.update_yaxes(range=rng, row=1, col=col)
    fig.update_layout(**_lay(
        width=620, height=310,
        margin=dict(t=40, b=35, l=40, r=10)))
    fig.update_annotations(font_size=11)
    return fig


def chart_abl_loss(abl):
    fig = go.Figure()
    for k, lbl, clr in [("ff", "FF-PINN (Fourier)", C["blue"]),
                         ("mlp", "Plain MLP", C["red"])]:
        if k in abl:
            s, v = _extract(abl[k]["rows"], "loss/total")
            fig.add_trace(go.Scatter(x=s, y=v, mode="lines", name=lbl,
                                     line=dict(color=clr, width=2)))
    fig.update_layout(**_lay(
        title=dict(text="Ablation: Loss Convergence (ka=\u03c0)", **_TITLE_STYLE),
        xaxis_title="Epoch", yaxis_title="Total Loss",
        yaxis_type="log", width=550, height=370,
        margin=dict(t=50, b=45, l=55, r=20)))
    return fig


def chart_abl_bar(abl):
    m = {}
    for k, lbl in [("ff", "FF-PINN"), ("mlp", "Plain MLP")]:
        if k not in abl:
            continue
        ll, lm = None, None
        for r in abl[k]["rows"]:
            if r.get("eval/l2_rel") is not None:
                ll, lm = r["eval/l2_rel"], r.get("eval/max_err", 0)
        m[lbl] = (ll, lm)
    if not m or any(v[0] is None for v in m.values()):
        return None
    labs = list(m.keys())
    l2v = [m[l][0] * 100 for l in labs]
    mxv = [m[l][1] * 100 for l in labs]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("L2 Relative Error (%)", "Max Error (%)"))
    fig.add_trace(go.Bar(x=labs, y=l2v, marker_color=[C["blue"], C["red"]],
                         text=[f"{v:.2f}%" for v in l2v], textposition="outside",
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=labs, y=mxv, marker_color=[C["blue"], C["red"]],
                         text=[f"{v:.2f}%" for v in mxv], textposition="outside",
                         showlegend=False), row=1, col=2)
    fig.update_layout(**_lay(
        title=dict(text="Ablation: Final Error (ka=\u03c0)", **_TITLE_STYLE),
        width=650, height=340,
        margin=dict(t=65, b=40, l=50, r=20)))
    return fig


def chart_abl_components(abl):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PDE Loss", "BC Loss"))
    for k, lbl, clr in [("ff", "FF-PINN", C["blue"]),
                         ("mlp", "Plain MLP", C["red"])]:
        if k not in abl:
            continue
        rows = abl[k]["rows"]
        s, v = _extract(rows, "loss/pde")
        fig.add_trace(go.Scatter(x=s, y=v, mode="lines", name=lbl,
                                 line=dict(color=clr, width=1.5)), row=1, col=1)
        s, v = _extract(rows, "loss/bc")
        fig.add_trace(go.Scatter(x=s, y=v, mode="lines", name=lbl,
                                 line=dict(color=clr, width=1.5),
                                 showlegend=False), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_layout(**_lay(
        title=dict(text="Ablation: Loss Components (ka=\u03c0)", **_TITLE_STYLE),
        width=850, height=370,
        margin=dict(t=65, b=40, l=50, r=20)))
    return fig


def chart_abl_loss_2pi(abl):
    """Loss convergence curves for ka=2π ablation."""
    fig = go.Figure()
    for k, lbl, clr in [("ff_2pi", "FF-PINN (Fourier)", C["blue"]),
                         ("mlp_2pi", "Plain MLP", C["red"])]:
        if k in abl:
            s, v = _extract(abl[k]["rows"], "loss/total")
            fig.add_trace(go.Scatter(x=s, y=v, mode="lines", name=lbl,
                                     line=dict(color=clr, width=2)))
    fig.update_layout(**_lay(
        title=dict(text="Ablation: Loss Convergence (ka=2\u03c0)", **_TITLE_STYLE),
        xaxis_title="Epoch", yaxis_title="Total Loss",
        yaxis_type="log", width=550, height=370,
        margin=dict(t=50, b=45, l=55, r=20)))
    return fig


def chart_abl_bar_2pi(abl):
    """Final error bar chart for ka=2π ablation."""
    m = {}
    for k, lbl in [("ff_2pi", "FF-PINN"), ("mlp_2pi", "Plain MLP")]:
        if k not in abl:
            continue
        ll, lm = None, None
        for r in abl[k]["rows"]:
            if r.get("eval/l2_rel") is not None:
                ll, lm = r["eval/l2_rel"], r.get("eval/max_err", 0)
        m[lbl] = (ll, lm)
    if not m or any(v[0] is None for v in m.values()):
        return None
    labs = list(m.keys())
    l2v = [m[l][0] * 100 for l in labs]
    mxv = [m[l][1] * 100 for l in labs]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("L2 Relative Error (%)", "Max Error (%)"))
    fig.add_trace(go.Bar(x=labs, y=l2v, marker_color=[C["blue"], C["red"]],
                         text=[f"{v:.1f}%" for v in l2v], textposition="outside",
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=labs, y=mxv, marker_color=[C["blue"], C["red"]],
                         text=[f"{v:.1f}%" for v in mxv], textposition="outside",
                         showlegend=False), row=1, col=2)
    fig.update_layout(**_lay(
        title=dict(text="Ablation: Final Error (ka=2\u03c0)", **_TITLE_STYLE),
        width=650, height=340,
        margin=dict(t=65, b=40, l=50, r=20)))
    return fig


def chart_abl_components_2pi(abl):
    """PDE and BC loss components for ka=2π ablation."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PDE Loss", "BC Loss"))
    for k, lbl, clr in [("ff_2pi", "FF-PINN", C["blue"]),
                         ("mlp_2pi", "Plain MLP", C["red"])]:
        if k not in abl:
            continue
        rows = abl[k]["rows"]
        s, v = _extract(rows, "loss/pde")
        fig.add_trace(go.Scatter(x=s, y=v, mode="lines", name=lbl,
                                 line=dict(color=clr, width=1.5)), row=1, col=1)
        s, v = _extract(rows, "loss/bc")
        fig.add_trace(go.Scatter(x=s, y=v, mode="lines", name=lbl,
                                 line=dict(color=clr, width=1.5),
                                 showlegend=False), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_layout(**_lay(
        title=dict(text="Ablation: Loss Components (ka=2\u03c0)", **_TITLE_STYLE),
        width=850, height=370,
        margin=dict(t=65, b=40, l=50, r=20)))
    return fig


# ════════════════════════════════════════════════════════════════
#  Results Table HTML (shared)
# ════════════════════════════════════════════════════════════════

def results_table():
    rows = ""
    for i, p in enumerate(PROD):
        best = ' class="best"' if p["l2"] == "2.00" else ""
        rows += f"""<tr{best}>
  <td>ka = {p['label']}</td>
  <td class="mono">{p['l2']}%</td><td class="mono">{p['me']}%</td><td class="mono">{p['mx']}%</td>
  <td class="mono">{p['net']}</td><td class="mono">{p['ep']}</td>
  <td class="mono">{p['t']}</td>
</tr>\n"""
    return f"""<table>
<tr><th>ka</th><th>L2 Rel Error</th><th>Mean Error</th><th>Max Error</th>
    <th>Network</th><th>Epochs</th><th>Runtime</th></tr>
{rows}</table>"""


# ════════════════════════════════════════════════════════════════
#  Scattering Diagram SVG (shared)
# ════════════════════════════════════════════════════════════════

SCATTER_SVG = """<svg viewBox="-5 -3.5 10 7" width="380" height="265"
  style="display:block;margin:auto;">
<defs><marker id="arw" viewBox="0 0 10 10" refX="8" refY="5"
  markerWidth="5" markerHeight="5" orient="auto">
  <path d="M0,1 L8,5 L0,9z" fill="#818cf8"/></marker></defs>
<circle cx="0" cy="0" r="3" fill="none" stroke="#4b4868"
  stroke-width="0.05" stroke-dasharray="0.15,0.08"/>
<text x="1.8" y="2.8" font-size="0.34" fill="#6b6880"
  font-family="DM Sans,system-ui">ABC boundary (r = L)</text>
<g stroke="#818cf8" stroke-width="0.06">
  <line x1="-4.5" y1="-1.5" x2="-1.3" y2="-1.5" marker-end="url(#arw)"/>
  <line x1="-4.5" y1="-0.5" x2="-1.3" y2="-0.5" marker-end="url(#arw)"/>
  <line x1="-4.5" y1="0.5" x2="-1.3" y2="0.5" marker-end="url(#arw)"/>
  <line x1="-4.5" y1="1.5" x2="-1.3" y2="1.5" marker-end="url(#arw)"/>
</g>
<text x="-4.3" y="-2.3" font-size="0.4" fill="#a5b4fc"
  font-family="DM Sans,system-ui" font-style="italic">Incident wave</text>
<circle cx="0" cy="0" r="0.8" fill="#1e1b35" stroke="#6366f1"
  stroke-width="0.04"/>
<text x="0" y="0.12" text-anchor="middle" font-size="0.35"
  fill="#a5b4fc" font-family="DM Sans,system-ui">a</text>
<g fill="none" stroke="#2dd4bf">
  <circle cx="0" cy="0" r="1.3" stroke-width="0.04"
    stroke-dasharray="0.2,0.15" opacity="0.7"/>
  <circle cx="0" cy="0" r="1.8" stroke-width="0.03"
    stroke-dasharray="0.2,0.15" opacity="0.45"/>
  <circle cx="0" cy="0" r="2.3" stroke-width="0.02"
    stroke-dasharray="0.2,0.15" opacity="0.25"/>
</g>
<text x="1.5" y="-1.7" font-size="0.4" fill="#5eead4"
  font-family="DM Sans,system-ui" font-style="italic">Scattered</text>
</svg>"""


# ════════════════════════════════════════════════════════════════
#  Architecture Diagram HTML (shared)
# ════════════════════════════════════════════════════════════════

ARCH_HTML = """<div class="arch-flow">
  <div class="arch-node">(x, y)<span class="detail">2D coords</span></div>
  <div class="arch-arrow">&rarr;</div>
  <div class="arch-node node-ff">Fourier Features
    <span class="detail">&sigma; = k, 128-dim</span></div>
  <div class="arch-arrow">&rarr;</div>
  <div class="arch-node node-res">Residual MLP
    <span class="detail">256 &times; 4 layers</span></div>
  <div class="arch-arrow">&rarr;</div>
  <div class="arch-node node-out">(u, v)
    <span class="detail">Re(&phi;<sub>s</sub>), Im(&phi;<sub>s</sub>)</span></div>
</div>"""


# ════════════════════════════════════════════════════════════════
#  Slide Deck
# ════════════════════════════════════════════════════════════════

SLIDE_CSS = r"""
/* ── Reset & base ── */
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden;background:#0c0f1a}
@media(prefers-reduced-motion:reduce){.slide{transition:none !important}}

/* ── Slide frame ── */
.slide{
  display:none;flex-direction:column;width:100vw;height:100vh;
  padding:2.2rem 3.5rem 2rem;
  font-family:'DM Sans',sans-serif;color:#e8e6f0;
  background:#0c0f1a;line-height:1.55;
  opacity:0;transition:opacity .18s ease-out;
  overflow:hidden;position:relative;
}
.slide.active{display:flex;opacity:1}

/* Subtle top-edge accent line per slide */
.slide::before{
  content:'';position:absolute;top:0;left:3.5rem;right:3.5rem;height:2px;
  background:linear-gradient(90deg,#6366f1 0%,#2dd4bf 50%,transparent 100%);
  opacity:.5;
}

/* ── Title slide ── */
.slide.title{justify-content:center;align-items:flex-start;
  text-align:left;padding:3.5rem 4rem}
.slide.title::before{display:none}
.slide.title h1{
  font-family:'Space Grotesk',sans-serif;
  font-size:clamp(2.6rem,5.5vw,4rem);font-weight:700;
  letter-spacing:-.045em;line-height:1.05;
  color:#f0eef8;margin-bottom:1.4rem;max-width:75%;
}
.slide.title .sub{
  font-size:1.25rem;color:#b8b4cc;margin-bottom:2.5rem;
  max-width:60%;line-height:1.45;
}
.slide.title .author{font-size:.95rem;color:#9b97b0}
.slide.title .title-accent{
  position:absolute;right:3rem;top:50%;transform:translateY(-50%);
  opacity:.14;
}

/* ── Section headings — varied by slide type ── */
.slide h2{
  font-family:'Space Grotesk',sans-serif;
  font-size:1.55rem;font-weight:700;letter-spacing:-.03em;
  color:#f0eef8;margin-bottom:1rem;
}
/* Physics slides get an indigo left accent */
.slide.physics h2{border-left:4px solid #6366f1;padding-left:.9rem}
/* Results slides get a teal left accent */
.slide.results h2{border-left:4px solid #2dd4bf;padding-left:.9rem}
/* Method slides get a subtle bottom rule */
.slide.method h2{padding-bottom:.5rem;
  border-bottom:1px solid rgba(255,255,255,.15)}

/* ── Content area ── */
.content{flex:1;min-height:0;overflow-y:auto}
.content::-webkit-scrollbar{width:4px}
.content::-webkit-scrollbar-thumb{background:rgba(255,255,255,.15);border-radius:2px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:2rem;height:100%}

/* ── Equations ── */
.eq{
  background:rgba(99,102,241,.06);
  border:1px solid rgba(99,102,241,.18);border-radius:4px;
  padding:.8rem 1.2rem;margin:.6rem 0;text-align:center;
  font-size:1.05em;
}
.eq-hero{
  background:rgba(99,102,241,.1);
  border:1px solid rgba(99,102,241,.3);border-radius:4px;
  padding:1.1rem 1.8rem;margin:.8rem 0;text-align:center;
  font-size:1.2em;
}

/* ── Tables ── */
table{border-collapse:collapse;width:100%;font-size:.95rem}
th{background:rgba(255,255,255,.04);font-size:.78rem;
  text-transform:uppercase;letter-spacing:.07em;color:#9b97b0;
  border-bottom:1px solid rgba(255,255,255,.12);border-top:none;
  border-left:none;border-right:none}
td{border:none;border-bottom:1px solid rgba(255,255,255,.06)}
th,td{padding:.5rem .8rem;text-align:center}
td:first-child{text-align:left;font-weight:600;color:#e8e6f0}
.mono{font-family:'Space Mono',monospace;font-size:.92rem;color:#c4c0d8}
.best td{background:rgba(45,212,191,.1)}

/* ── Architecture flow ── */
.arch-flow{display:flex;align-items:center;justify-content:center;
  gap:.8rem;margin:1.4rem 0;flex-wrap:wrap}
.arch-node{border:1.5px solid rgba(255,255,255,.2);border-radius:6px;
  padding:.6rem 1rem;text-align:center;font-size:.92rem;
  font-weight:600;min-width:115px;color:#e8e6f0;
  background:rgba(255,255,255,.03)}
.arch-node .detail{display:block;font-size:.75rem;font-weight:400;
  color:#9b97b0;margin-top:.15rem}
.arch-arrow{font-size:1.4rem;color:#4b4868}
.node-ff{background:rgba(99,102,241,.12);border-color:rgba(99,102,241,.4);color:#a5b4fc}
.node-res{background:rgba(45,212,191,.1);border-color:rgba(45,212,191,.35);color:#99f6e4}
.node-out{background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.3);color:#fde68a}

/* ── Progress bar ── */
#progress{
  position:fixed;top:0;left:0;height:4px;
  background:#6366f1;transition:width .25s ease-out;z-index:100;
}

/* ── Slide counter ── */
#counter{position:fixed;bottom:1rem;left:2rem;
  font-family:'Space Mono',monospace;font-size:.75rem;color:#4b4868;
  pointer-events:none}
#nav-hint{position:fixed;bottom:1rem;right:2rem;
  font-family:'Space Mono',monospace;font-size:.68rem;color:#3a3754;
  pointer-events:none}

/* ── Callouts ── */
.highlight{
  background:rgba(45,212,191,.08);
  border-left:3px solid #2dd4bf;
  padding:.7rem 1rem;font-size:.95rem;margin:.7rem 0;
  color:#c4f0ea;
}
.highlight strong{color:#2dd4bf}
.callout-result{
  background:rgba(99,102,241,.08);
  border-left:3px solid #6366f1;
  padding:.7rem 1rem;font-size:.95rem;margin:.7rem 0;
  color:#c7c4f0;
}

/* ── Typography ── */
ul{margin-left:1.4rem;font-size:1rem;color:#c4c0d8}
li{margin-bottom:.45rem}
li strong{color:#e8e6f0}
.small{font-size:.88rem;color:#9b97b0}
.tiny{font-size:.78rem;color:#6b6880}
h3{font-family:'Space Grotesk',sans-serif;font-size:1.05rem;
  font-weight:600;margin:1rem 0 .45rem;color:#c4c0d8;
  letter-spacing:-.01em}
p{font-size:1rem;margin-bottom:.5rem;color:#b8b4cc}
a{color:#818cf8;text-decoration:none}
a:hover{color:#a5b4fc;text-decoration:underline}

/* ── Big result numbers ── */
.big-result{font-family:'Space Grotesk',sans-serif;font-size:2.2rem;
  font-weight:700;color:#2dd4bf;line-height:1}
.big-result .unit{font-size:1rem;font-weight:400;color:#9b97b0;margin-left:.15rem}

/* ── Field plot links ── */
.field-links{display:flex;gap:.5rem;flex-wrap:wrap;margin:.8rem 0}
.field-links a{
  display:inline-block;padding:.4rem .85rem;
  border:1px solid rgba(255,255,255,.1);border-radius:3px;
  font-size:.85rem;color:#818cf8;
  transition:border-color .15s,background .15s;
}
.field-links a:hover{border-color:#6366f1;background:rgba(99,102,241,.08);
  text-decoration:none}

/* ── Code ── */
code{background:rgba(255,255,255,.06);padding:.15rem .45rem;border-radius:3px;
  font-family:'Space Mono',monospace;font-size:.88rem;color:#c4c0d8}

/* ── Conclusion takeaways ── */
.takeaway{
  display:grid;grid-template-columns:auto 1fr;gap:.6rem 1rem;
  align-items:baseline;margin:.6rem 0;
}
.takeaway .num{
  font-family:'Space Grotesk',sans-serif;font-size:1.8rem;
  font-weight:700;color:#6366f1;line-height:1;
}
.takeaway p{margin:0;font-size:1rem;color:#c4c0d8}
.takeaway strong{color:#e8e6f0}
""".replace("FONT", FONT)

SLIDE_JS = """
let cur=0;
const sl=document.querySelectorAll('.slide'),
      ct=document.getElementById('counter'),
      pb=document.getElementById('progress'),
      tot=sl.length;
function show(n){
  if(n<0||n>=tot)return;
  sl[cur].classList.remove('active');
  sl[cur].style.opacity='0';
  cur=n;
  sl[cur].classList.add('active');
  /* Let the display:flex take effect, then fade in */
  requestAnimationFrame(()=>{sl[cur].style.opacity='1'});
  ct.textContent=(cur+1)+' / '+tot;
  pb.style.width=((cur+1)/tot*100)+'%';
  requestAnimationFrame(()=>window.dispatchEvent(new Event('resize')));
}
document.addEventListener('keydown',function(e){
  if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){e.preventDefault();show(cur+1)}
  else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();show(cur-1)}
  else if(e.key==='Home')show(0);
  else if(e.key==='End')show(tot-1);
});
sl[0].classList.add('active');sl[0].style.opacity='1';
ct.textContent='1 / '+tot;
pb.style.width=(1/tot*100)+'%';
"""


def _darkify(fig):
    """Apply dark theme to a Plotly figure for slide embedding."""
    fig.update_layout(**_DARK_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)",
                     zerolinecolor="rgba(255,255,255,0.1)",
                     tickfont=dict(color="#9b97b0"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)",
                     zerolinecolor="rgba(255,255,255,0.1)",
                     tickfont=dict(color="#9b97b0"))
    return fig


def build_slides(prod, abl):
    # Build chart divs (dark-themed for slides)
    ch = {}
    ch["l2bar"] = _div(_darkify(chart_l2_bar()), "sl-l2")
    ch["mxbar"] = _div(_darkify(chart_max_err_bar()), "sl-mx")
    ch["mebar"] = _div(_darkify(chart_mean_err_bar()), "sl-me")
    ch["errcomb"] = _div(_darkify(chart_errors_combined()), "sl-errcomb")
    if prod:
        ch["ploss"] = _div(_darkify(chart_prod_loss(prod)), "sl-ploss")
    if abl:
        ch["aloss"] = _div(_darkify(chart_abl_loss(abl)), "sl-aloss")
        ab = chart_abl_bar(abl)
        if ab:
            ch["abar"] = _div(_darkify(ab), "sl-abar")
        if "ff_2pi" in abl:
            ch["aloss_2pi"] = _div(_darkify(chart_abl_loss_2pi(abl)), "sl-aloss2")
            ab2 = chart_abl_bar_2pi(abl)
            if ab2:
                ch["abar_2pi"] = _div(_darkify(ab2), "sl-abar2")

    # Generate collocation point figure
    print("    Generating collocation point figure...")
    ch["colloc"] = _div(_darkify(generate_collocation_figure()), "sl-colloc")

    # Generate analytic gallery (2×2 multi-ka figure)
    print("    Generating analytic gallery...")
    ch["gallery"] = _div(_darkify(generate_analytic_gallery()), "sl-gallery")

    # Generate error gallery (2×2 absolute error from checkpoints)
    print("    Generating error gallery from checkpoints...")
    ch["errors"] = _div(_darkify(generate_error_gallery()), "sl-errors")

    # Generate honeycomb residual diagnostics
    print("    Generating honeycomb residual diagnostics...")
    hc_res_fig = generate_honeycomb_residuals()
    if hc_res_fig:
        ch["hc_resid"] = _div(_darkify(hc_res_fig), "sl-hc-resid")

    # Extract field comparison figures from existing HTML files
    print("    Extracting field comparison figures...")
    ch["field_pi"] = extract_figure_div(
        os.path.join(OUTPUTS, "prod_eval_ka3.14", "ka3.14_global_real.html"),
        "sl-field-pi", width="100%", height="320px")
    ch["field_2pi"] = extract_figure_div(
        os.path.join(OUTPUTS, "prod_eval_ka6.28", "ka6.28_global_real.html"),
        "sl-field-2pi", width="100%", height="320px")
    ch["hc_field"] = extract_figure_div(
        os.path.join(OUTPUTS, "hc_summary", "hc_ka2_mag_total.html"),
        "sl-hc-field", width="100%", height="400px")

    S = []  # slide list

    # ── 0  Title ──
    S.append(f"""<section class="slide title">
  <h1>Physics-Informed Neural Network for 2D Acoustic Scattering</h1>
  <p class="sub">Solving the Helmholtz equation across wavenumbers
  ka&nbsp;=&nbsp;0.5 to 2&pi; with 2&ndash;8% L2 error, under 3% for ka&nbsp;&ge;&nbsp;&pi;</p>
  <p class="author">Carl Merrigan &mdash; March 2026</p>
  <div class="title-accent">{SCATTER_SVG.replace('fill="#94949e"', 'fill="#3a3754"').replace('stroke="#94949e"', 'stroke="#2a2844"').replace('fill="#1a56db"', 'fill="#4b4868"').replace('stroke="#1a56db"', 'stroke="#4b4868"').replace('fill="#1a1a2e"', 'fill="#1e1b35"').replace('fill="white"', 'fill="#4b4868"').replace('stroke="#dc2626"', 'stroke="#3a3754"').replace('fill="#dc2626"', 'fill="#3a3754"').replace('width="380"', 'width="520"').replace('height="265"', 'height="365"')}</div>
</section>""")

    # ── 1  Problem ──
    S.append(f"""<section class="slide physics">
  <h2>Problem: Acoustic Wave Scattering</h2>
  <div class="content two-col">
    <div>
      <p>A plane wave impinges on a rigid circular cylinder. We predict the
      <strong>scattered field</strong> &mdash; the perturbation caused by the obstacle.</p>
      <div class="eq">\\[{r"\nabla^2 \phi_s + k^2 \phi_s = 0"}\\]</div>
      <p>The dimensionless parameter <strong>ka</strong> (wavenumber &times; radius)
      controls difficulty: higher ka = more oscillatory field = finer spatial features.</p>
      <p>We train from ka&nbsp;=&nbsp;0.5 (sub-wavelength) through ka&nbsp;=&nbsp;2&pi;
      (2 wavelengths per radius), spanning a 12&times; range of spatial frequency.</p>
      <h3>Exact Solution (Validation)</h3>
      <p>The Jacobi-Anger identity expands the incident wave in cylindrical harmonics:</p>
      <div class="eq">\\[{r"\phi_{\text{inc}} = e^{ikr\cos\theta} = \sum_{n=-\infty}^{\infty} i^n J_n(kr)\,e^{in\theta}"}\\]</div>
      <p>The sound-hard Neumann condition \\(\\partial\\phi_{{\\text{{total}}}}/\\partial r\\big|_{{r=a}} = 0\\) yields
      the exact scattered field:</p>
      <div class="eq-hero">\\[{r"\phi_s = -\sum_{n=-\infty}^{\infty} i^n \frac{J_n'(ka)}{H_n^{(1)\prime}(ka)}\, H_n^{(1)}(kr)\, e^{in\theta}"}\\]</div>
      <p class="small">Converges for N &asymp; ka&nbsp;+&nbsp;20 terms. This exact solution provides
      pixel-level validation of the PINN &mdash; no numerical reference solver needed.</p>
    </div>
    <div style="display:flex;align-items:center">{SCATTER_SVG}</div>
  </div>
</section>""")

    # ── 2  Formulation ──
    S.append(f"""<section class="slide physics">
  <h2>Scattered-Field Formulation</h2>
  <div class="content">
    <p>The network outputs \\((u,v) = (\\text{{Re}}(\\phi_s),\\;\\text{{Im}}(\\phi_s))\\).
    The total field is reconstructed as \\(\\phi_{{\\text{{total}}}} = e^{{ikx}} + \\phi_s\\).
    Three loss terms enforce physics:</p>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin:1rem 0">
      <div class="eq">
        <p class="tiny" style="margin-bottom:.3rem">PDE RESIDUAL</p>
        \\[{r"\nabla^2 \phi_s + k^2 \phi_s = 0"}\\]
      </div>
      <div class="eq">
        <p class="tiny" style="margin-bottom:.3rem">NEUMANN BC (r = a)</p>
        \\[{r"\frac{\partial \phi_s}{\partial n} = -\frac{\partial \phi_{\text{inc}}}{\partial n}"}\\]
      </div>
      <div class="eq">
        <p class="tiny" style="margin-bottom:.3rem">BGT2 ABC (r = L)</p>
        \\[{r"\frac{\partial \phi_s}{\partial r} - ik\phi_s + \frac{\phi_s}{2r} = 0"}\\]
      </div>
    </div>
    <div class="eq" style="max-width:500px;margin:1rem auto">
      <p class="tiny" style="margin-bottom:.3rem">TOTAL LOSS</p>
      \\[{r"\mathcal{L} = \lambda_{\text{pde}}\,\mathcal{L}_{\text{pde}} + \lambda_{\text{bc}}\,\mathcal{L}_{\text{bc}} + \lambda_{\text{abc}}\,\mathcal{L}_{\text{abc}}"}\\]
    </div>
    <p class="small">Scattered-field formulation lets the network focus on the unknown perturbation,
    avoiding the need to learn the known incident plane wave.</p>
  </div>
</section>""")

    # ── 3  Architecture ──
    S.append(f"""<section class="slide method">
  <h2>Network Architecture</h2>
  <div class="content">
    {ARCH_HTML}
    <div class="two-col" style="margin-top:1.2rem">
      <div>
        <h3>Random Fourier Features</h3>
        <div class="eq">\\[{r"\gamma(\mathbf{x}) = [\sin(B\mathbf{x}),\;\cos(B\mathbf{x})]"}\\]</div>
        <p>Overcomes spectral bias &mdash; standard MLPs struggle with high-frequency
        functions in low-dimensional domains
        <span class="tiny">[Tancik et al., NeurIPS 2020]</span></p>
        <p>Bandwidth \\(\\sigma\\) set to wavenumber \\(k\\), encoding the
        physical length scale directly into the feature space.</p>
      </div>
      <div>
        <h3>Design Choices</h3>
        <ul>
          <li><strong>Residual blocks:</strong> skip connections stabilize deep training</li>
          <li><strong>Xavier init:</strong> balanced gradient flow at initialization</li>
          <li><strong>Tanh activation:</strong> smooth, matches solution regularity</li>
          <li><strong>Complex output:</strong> (u, v) avoids branch cuts in polar form</li>
        </ul>
        <p class="small" style="margin-top:.8rem">~296K parameters (FF-PINN) vs ~298K (plain MLP baseline)</p>
      </div>
    </div>
  </div>
</section>""")

    # ── 4  Training Strategy & Loss Functions (merged) ──
    S.append(f"""<section class="slide method">
  <h2>Training Strategy &amp; Loss Functions</h2>
  <div class="content two-col">
    <div>
      <h3>Collocation Sampling</h3>
      {ch["colloc"]}
      <h3 style="margin-top:.6rem">Two-Phase Optimization</h3>
      <ul>
        <li><strong>Phase 1 &mdash; Adam</strong> (10K&ndash;50K epochs)<br>
          <span class="small">Cosine annealing LR; resampling points every 2K epochs</span></li>
        <li><strong>Phase 2 &mdash; L-BFGS</strong> (200&ndash;300 iters)<br>
          <span class="small">Strong Wolfe line search; fixed points for stable Hessian</span></li>
      </ul>
      <p class="small" style="margin-top:.5rem">High-ka: wider network (384&ndash;512n),
      more Fourier features (96&ndash;128), 50K Adam epochs.</p>
    </div>
    <div>
      <h3>Loss Components</h3>
      <table>
        <tr><th>Component</th><th>Definition</th><th>&lambda;</th></tr>
        <tr><td>\\(\\mathcal{{L}}_{{\\text{{pde}}}}\\)</td>
            <td style="text-align:left" class="small">\\(\\frac{{1}}{{N}}\\sum|\\nabla^2\\phi_s + k^2\\phi_s|^2\\)</td>
            <td class="mono">1.0</td></tr>
        <tr><td>\\(\\mathcal{{L}}_{{\\text{{bc}}}}\\)</td>
            <td style="text-align:left" class="small">\\(\\frac{{1}}{{N}}\\sum|\\partial_n\\phi_s + \\partial_n\\phi_{{\\text{{inc}}}}|^2\\)</td>
            <td class="mono">10.0</td></tr>
        <tr><td>\\(\\mathcal{{L}}_{{\\text{{abc}}}}\\)</td>
            <td style="text-align:left" class="small">\\(\\frac{{1}}{{N}}\\sum|\\partial_r\\phi_s - ik\\phi_s + \\phi_s/2r|^2\\)</td>
            <td class="mono">1.0</td></tr>
      </table>
      <h3 style="margin-top:.6rem">Evaluation Metrics</h3>
      <p class="small" style="margin-bottom:.3rem">vs analytic Bessel/Hankel series, 200&times;200 grid.</p>
      <table>
        <tr><th>Metric</th><th>Definition</th></tr>
        <tr><td>L2 Relative</td>
            <td style="text-align:left" class="small">\\(\\|\\phi_s^{{\\text{{PINN}}}} - \\phi_s^{{\\text{{exact}}}}\\|_2 / \\|\\phi_s^{{\\text{{exact}}}}\\|_2\\)
            &mdash; primary, normalized</td></tr>
        <tr><td>Max Error</td>
            <td style="text-align:left" class="small">\\(\\max|\\phi_s^{{\\text{{PINN}}}} - \\phi_s^{{\\text{{exact}}}}|\\)
            &mdash; worst-case</td></tr>
        <tr><td>Mean Error</td>
            <td style="text-align:left" class="small">\\(\\bar{{e}} = \\frac{{1}}{{N}}\\sum|\\phi_s^{{\\text{{PINN}}}} - \\phi_s^{{\\text{{exact}}}}|\\)
            &mdash; typical</td></tr>
      </table>
      <p class="small" style="margin-top:.4rem">All errors use complex magnitude
      \\(|\\phi| = \\sqrt{{u^2 + v^2}}\\).</p>
    </div>
  </div>
</section>""")

    # ── 5  Ablation: ka=π ──
    abl_charts_pi = ""
    if "aloss" in ch:
        abl_charts_pi += f'<div>{ch["aloss"]}</div>'
    if "abar" in ch:
        abl_charts_pi += f'<div>{ch["abar"]}</div>'
    if not abl_charts_pi:
        abl_charts_pi = '<p class="small">Ablation charts unavailable (wandb data not loaded)</p>'

    abl_charts_2pi = ""
    if "aloss_2pi" in ch:
        abl_charts_2pi += f'<div>{ch["aloss_2pi"]}</div>'
    if "abar_2pi" in ch:
        abl_charts_2pi += f'<div>{ch["abar_2pi"]}</div>'

    # ── 5  Ablation: Fourier Features (merged tables + dynamics) ──
    abl_dyn_html = ""
    if abl_charts_pi or abl_charts_2pi:
        abl_dyn_html = f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-top:.4rem">
      {abl_charts_pi}
    </div>"""
        if abl_charts_2pi:
            abl_dyn_html += f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-top:.4rem">
      {abl_charts_2pi}
    </div>"""

    S.append(f"""<section class="slide method">
  <h2>Ablation: Fourier Features</h2>
  <div class="content">
    <p class="small">Parameter-matched comparison &mdash;
    FF-PINN (64 features, 256 hidden, 296K params) vs plain MLP (272 hidden, 298K params).
    Identical 10K Adam + 200 L-BFGS schedules (shorter than production) to isolate architecture.</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-top:.3rem">
      <div>
        <h3>ka = &pi; (low frequency)</h3>
        <table>
          <tr><th></th><th>FF-PINN</th><th>Plain MLP</th></tr>
          <tr><td>L2 relative</td><td class="mono">2.42%</td><td class="mono">2.42%</td></tr>
          <tr><td>Max error</td><td class="mono">2.97%</td><td class="mono">2.99%</td></tr>
        </table>
        <p class="small" style="margin-top:.3rem">Identical &mdash; spectral bias not a bottleneck at low frequency.</p>
      </div>
      <div>
        <h3>ka = 2&pi; (high frequency)</h3>
        <table>
          <tr><th></th><th>FF-PINN</th><th>Plain MLP</th><th>Ratio</th></tr>
          <tr><td>L2 relative</td><td class="mono">10.1%</td><td class="mono">11.7%</td><td class="mono">1.15&times;</td></tr>
          <tr class="best"><td>Max error</td><td class="mono">22.7%</td><td class="mono">47.3%</td><td class="mono">2.1&times;</td></tr>
        </table>
        <p class="small" style="margin-top:.3rem">Max error doubles without Fourier features.</p>
      </div>
    </div>
    <div class="highlight" style="margin-top:.4rem">
      <strong>Key insight:</strong> The plain MLP&rsquo;s L-BFGS phase <em>increased</em> error
      (10.9%&rarr;11.7%), while the FF-PINN&rsquo;s L-BFGS <em>halved</em> it (19.6%&rarr;10.1%).
      Fourier features create a smoother loss landscape that second-order optimizers can exploit.
    </div>
    {abl_dyn_html}
  </div>
</section>""")

    # ── 6  Boundary Conditions ──
    S.append(f"""<section class="slide physics">
  <h2>Absorbing Boundary Conditions</h2>
  <div class="content two-col">
    <div>
      <h3>First-Order ABC</h3>
      <div class="eq">\\[{r"\frac{\partial \phi_s}{\partial n} - ik\phi_s = 0"}\\]</div>
      <p>Assumes purely outgoing plane waves at the boundary. Works on any shape but
      ignores wavefront curvature.</p>
      <h3 style="margin-top:1rem">BGT2 (Second-Order)</h3>
      <div class="eq">\\[{r"\frac{\partial \phi_s}{\partial r} - ik\phi_s + \frac{\phi_s}{2r} = 0"}\\]</div>
      <p>Adds a curvature correction \\(\\phi_s/2r\\) that accounts for the \\(1/\\sqrt{{r}}\\)
      amplitude decay of cylindrical waves
      <span class="tiny">[Bayliss, Gunzburger &amp; Turkel, 1982]</span></p>
    </div>
    <div>
      <h3>Why Circular Domain?</h3>
      <ul>
        <li><strong>Symmetry match:</strong> circular boundary matches the cylindrical
        geometry of the scatterer &mdash; uniform distance to the obstacle from all points</li>
        <li><strong>Well-defined normals:</strong> square boundaries have corner singularities
        where the outward normal is undefined &rarr; ABC is ill-conditioned</li>
        <li><strong>BGT2 requires curvature:</strong> the \\(\\phi_s/2r\\) correction is derived
        for circular boundaries; it has no square-boundary analogue</li>
      </ul>
      <div class="highlight" style="margin-top:1rem">
        <strong>Result:</strong> Circle + BGT2 gives the most physically consistent setup
        for this cylindrical scattering problem.
      </div>
    </div>
  </div>
</section>""")

    # ── 7  Results ──
    S.append(f"""<section class="slide results">
  <h2>Production Results</h2>
  <div class="content">
    <div style="display:flex;gap:2rem;margin-bottom:.6rem">
      <div><div class="big-result">2.00<span class="unit">%</span></div>
        <p class="small" style="margin-top:.2rem">Best L2 error (ka=2&pi;)</p></div>
      <div><div class="big-result" style="color:#818cf8">4&times;</div>
        <p class="small" style="margin-top:.2rem">ka values validated</p></div>
    </div>
    {results_table()}
    <div style="margin-top:.5rem">{ch["errcomb"]}</div>
    <p class="small" style="margin-top:.3rem">All runs: circle boundary, BGT2 ABC,
    L&nbsp;=&nbsp;3.0, trained on 4&times; NVIDIA RTX 4000 Ada.
    Mean error &lt;&nbsp;2% everywhere; max error peaks at 4.4% for ka=2&pi;
    (localized to the shadow boundary).</p>
    <p style="margin-top:.4rem;font-size:.9rem">L2 relative error is highest at low ka because
    the scattered field amplitude is small (smaller normalization denominator) and
    these runs received 4&times; less compute (10K vs 50K epochs, smaller network).
    The low-ka errors could likely be reduced with comparable training budgets.</p>
  </div>
</section>""")

    # ── 8  Multi-ka Gallery ──
    S.append(f"""<section class="slide physics">
  <h2>Multi-Scale Scattering Physics</h2>
  <div class="content" style="display:grid;grid-template-columns:2fr 1fr;gap:1.5rem">
    <div>{ch["gallery"]}</div>
    <div style="display:flex;flex-direction:column;justify-content:center">
      <p>Analytic scattered field Re(&phi;<sub>s</sub>) across four wavenumbers,
      showing how the scattering pattern evolves from a simple dipole at ka&nbsp;=&nbsp;0.5
      to a complex multi-lobe pattern at ka&nbsp;=&nbsp;2&pi;.</p>
      <ul style="margin-top:.6rem">
        <li><strong>ka = 0.5:</strong> Sub-wavelength &mdash; weak, dipolar scattering</li>
        <li><strong>ka = 1.0:</strong> Transition regime &mdash; shadow begins to form</li>
        <li><strong>ka = &pi;:</strong> Resonance &mdash; strong diffraction, clear shadow</li>
        <li><strong>ka = 2&pi;:</strong> Multi-wavelength &mdash; complex interference fringes</li>
      </ul>
      <p class="small" style="margin-top:.8rem">The PINN must resolve increasingly fine spatial
      features as ka grows &mdash; motivating Fourier features with &sigma;&nbsp;=&nbsp;k.</p>
    </div>
  </div>
</section>""")

    # ── 9  PINN vs Analytic: Field Comparisons (merged ka=π + ka=2π) ──
    S.append(f"""<section class="slide results">
  <h2>PINN vs Analytic: Field Comparisons</h2>
  <div class="content">
    <div style="display:flex;align-items:baseline;gap:.8rem;margin-bottom:.2rem">
      <h3 style="margin:0">ka = &pi;</h3>
      <div class="big-result" style="font-size:1.4rem">2.41<span class="unit">% L2</span></div>
      <span class="small" style="color:#818cf8">2.96% max</span>
    </div>
    <div>{ch["field_pi"]}</div>
    <div style="display:flex;align-items:baseline;gap:.8rem;margin-top:.3rem;margin-bottom:.2rem">
      <h3 style="margin:0">ka = 2&pi;</h3>
      <div class="big-result" style="font-size:1.4rem">2.00<span class="unit">% L2</span></div>
      <span class="small" style="color:#f87171">from 49%</span>
      <span class="small">&mdash; 384n/6L/96ff, 50K Adam + 300 L-BFGS</span>
    </div>
    <div>{ch["field_2pi"]}</div>
  </div>
</section>""")

    # ── 10  Error Gallery ──
    S.append(f"""<section class="slide results">
  <h2>Spatial Error Distribution</h2>
  <div class="content" style="display:grid;grid-template-columns:2fr 1fr;gap:1.5rem">
    <div>{ch["errors"]}</div>
    <div style="display:flex;flex-direction:column;justify-content:center">
      <p>Absolute error |PINN &minus; analytic| across all four production wavenumbers.
      Error is concentrated near the scatterer surface and in the shadow region
      behind the cylinder &mdash; the regions with the steepest field gradients.</p>
      <ul style="margin-top:.6rem">
        <li><strong>ka = 0.5, 1.0:</strong> errors uniformly low, smooth fields
        are easy for the network</li>
        <li><strong>ka = &pi;:</strong> shadow boundary shows slightly elevated error
        where the field transitions sharply</li>
        <li><strong>ka = 2&pi;:</strong> error peaks near the surface where
        the oscillation wavelength is smallest &mdash; addressed by more
        Fourier features and longer training (50K epochs)</li>
      </ul>
    </div>
  </div>
</section>""")

    # ── 11  Training Curves ──
    tchart = ch.get("ploss", '<p class="small">Training curve chart unavailable</p>')
    S.append(f"""<section class="slide method">
  <h2>Training Convergence</h2>
  <div class="content" style="display:grid;grid-template-columns:3fr 2fr;gap:1.5rem">
    <div>{tchart}</div>
    <div style="display:flex;flex-direction:column;justify-content:center">
      <p>All runs show the Adam &rarr; L-BFGS transition (sharp loss drop at phase
      boundary). Higher ka requires progressively more compute.</p>
      <h3 style="margin-top:.8rem">Scaling Across ka</h3>
      <table style="margin-top:.4rem">
        <tr><th></th><th>ka &le; &pi;</th><th>ka = 2&pi;</th><th style="color:#f87171">ka = 3&pi;</th></tr>
        <tr><td>Neurons</td><td class="mono">256</td>
            <td class="mono">384</td><td class="mono">512</td></tr>
        <tr><td>Fourier feat.</td><td class="mono">64</td>
            <td class="mono">96</td><td class="mono">128</td></tr>
        <tr><td>Adam epochs</td><td class="mono">10K</td>
            <td class="mono">50K</td><td class="mono">50K</td></tr>
        <tr><td>Wall time</td><td class="mono">12&ndash;17 min</td>
            <td class="mono">259 min</td><td class="mono">10+ hrs</td></tr>
        <tr><td>L2 error</td><td class="mono" style="color:#2dd4bf">2&ndash;8%</td>
            <td class="mono" style="color:#2dd4bf">2.0%</td>
            <td class="mono" style="color:#f87171">68%</td></tr>
      </table>
      <p class="small" style="margin-top:.5rem">ka=3&pi;: loss drops 5 orders of magnitude
      but L2 only halves &mdash; the architecture cannot resolve ~3 wavelengths per radius.
      Needs multi-scale decomposition or adaptive methods, not just more compute.</p>
    </div>
  </div>
</section>""")

    # ── 13a  Honeycomb Extension: Field ──
    S.append(f"""<section class="slide results">
  <h2>Extension: Honeycomb Acoustic Shield</h2>
  <div class="content" style="display:grid;grid-template-columns:3fr 2fr;gap:2rem">
    <div>
      {ch["hc_field"]}
    </div>
    <div style="display:flex;flex-direction:column;justify-content:center">
      <p>19-circle honeycomb cluster at ka&nbsp;=&nbsp;2, trained as a pure PINN
      with no analytic reference solution.</p>
      <table style="margin-top:1rem">
        <tr><th>Loss</th><th>Value</th></tr>
        <tr><td>PDE</td><td class="mono" style="color:#2dd4bf">4.92e-6</td></tr>
        <tr><td>BC (19)</td><td class="mono" style="color:#2dd4bf">2.00e-7</td></tr>
        <tr><td>ABC</td><td class="mono" style="color:#2dd4bf">1.41e-7</td></tr>
        <tr><td><strong>Total</strong></td><td class="mono" style="color:#2dd4bf"><strong>9.06e-6</strong></td></tr>
      </table>
      <div class="highlight" style="margin-top:1rem">
        <strong>Shielding:</strong> |&phi;<sub>total</sub>|&nbsp;&lt;&nbsp;0.003 inside cluster.
      </div>
    </div>
  </div>
</section>""")

    # ── 13b  Honeycomb Extension: Residual Diagnostics ──
    hc_resid_div = ch.get("hc_resid",
        '<p class="small">Residual diagnostics unavailable</p>')
    S.append(f"""<section class="slide results">
  <h2>Honeycomb: Residual Diagnostics</h2>
  <div class="content" style="display:grid;grid-template-columns:1fr 1fr;gap:2rem">
    <div>
      {hc_resid_div}
    </div>
    <div style="display:flex;flex-direction:column;justify-content:center">
      <p><strong>PDE residual</strong> (left panel): finite-difference Laplacian, 200&times;200 grid.
      Near-zero in inter-circle gaps. Elevated near boundaries due to steep field gradients
      &mdash; inherent to the physics.</p>
      <p style="margin-top:.8rem"><strong>BC residual</strong> (right panel): autograd normal derivatives,
      100 pts/circle (1,900 total). Neumann condition satisfied to O(10<sup>&minus;3</sup>) across
      all 19 surfaces.</p>
    </div>
  </div>
</section>""")

    # ── 12  Conclusions ──
    S.append("""<section class="slide results">
  <h2>Conclusion: PINNs as a Multi-Scale Probe</h2>
  <div class="content">
    <div style="background:rgba(45,212,191,.1);border:1.5px solid rgba(45,212,191,.35);
      border-radius:6px;padding:1.2rem 1.5rem;margin-bottom:1rem">
      <p style="font-size:1.15rem;color:#e8e6f0;margin:0;line-height:1.5">
        The assignment asks whether a PINN can <strong>dynamically resolve physical fields
        at arbitrary length scales</strong> &mdash; effectively zooming in to reveal
        finer spatial structure. These results show that it can, but with a clear tradeoff:
        <strong style="color:#2dd4bf">resolving shorter wavelengths demands
        exponentially more compute.</strong></p>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:2rem">
      <div>
        <h3 style="color:#2dd4bf">Where PINNs Excel</h3>
        <p>For <strong>low-to-moderate spatial frequency</strong> (ka&nbsp;&le;&nbsp;&pi;),
        the PINN converges to 2&ndash;3% L2 error in 12&ndash;17 minutes. At these scales,
        PINNs offer genuine advantages over mesh-based solvers: continuous field access
        at any point, no mesh generation, and physics enforced by construction. The network
        acts as a differentiable, resolution-independent surrogate for the field.</p>
        <div class="takeaway" style="margin-top:.8rem">
          <div class="num">1</div>
          <p><strong>Multi-scale accuracy.</strong> 2&ndash;8% L2 across a 12&times;
          range of spatial frequency, single architecture.</p>
        </div>
        <div class="takeaway">
          <div class="num">2</div>
          <p><strong>Physics-aware features.</strong> Fourier features with &sigma;=k
          defeat spectral bias at the source.</p>
        </div>
        <div class="takeaway">
          <div class="num">3</div>
          <p><strong>Symmetry-matched BCs.</strong> Circle + BGT2 &mdash;
          physics guides the numerics.</p>
        </div>
      </div>
      <div>
        <h3 style="color:#f87171">The Compute Wall</h3>
        <p>As ka grows, training cost scales steeply: the ka=2&pi; run needed
        <strong>5&times; more epochs and a 50% wider network</strong> (259 min vs 12 min).
        At ka=3&pi;, a 10+ hour run plateaued at 68% L2 despite loss dropping 5 orders
        of magnitude &mdash; the architecture cannot resolve ~3 wavelengths per radius.</p>
        <p style="margin-top:.5rem">This is not surprising: zooming in to finer features is equivalent to
        probing higher spatial frequencies, and neural networks face a fundamental
        resolution&ndash;compute tradeoff analogous to the Nyquist limit in signal processing.</p>
        <div class="takeaway" style="margin-top:.8rem">
          <div class="num" style="color:#f87171">4</div>
          <p><strong>Patience over engineering.</strong> The convergent ka=2&pi; run used
          default loss weights but 2.5&times; more epochs &mdash; 49%&rarr;2%.
          At the frontier, compute budget is the binding constraint.</p>
        </div>
        <h3 style="margin-top:.6rem">Next Steps</h3>
        <ul style="font-size:.88rem">
          <li>Multi-scale / domain decomposition</li>
          <li>Curriculum training: low&rarr;high ka</li>
          <li>Residual-adaptive sampling (RAD)</li>
        </ul>
      </div>
    </div>
  </div>
</section>""")

    # ── 13  References ──
    S.append("""<section class="slide">
  <h2>References</h2>
  <div class="content">
    <ol style="font-size:.82rem;line-height:1.7">
      <li>M. Raissi, P. Perdikaris, G.E. Karniadakis.
        <em>Physics-informed neural networks: A deep learning framework for solving forward
        and inverse problems involving nonlinear partial differential equations.</em>
        J. Comput. Phys. 378, 686&ndash;707 (2019).</li>
      <li>M. Tancik, P. Srinivasan, B. Mildenhall et al.
        <em>Fourier features let networks learn high frequency functions in low dimensional
        domains.</em> NeurIPS (2020).</li>
      <li>A. Bayliss, M. Gunzburger, E. Turkel.
        <em>Boundary conditions for the numerical solution of elliptic equations in
        exterior regions.</em> SIAM J. Appl. Math. 42(2), 430&ndash;451 (1982).</li>
      <li>L. Lu, X. Meng, Z. Mao, G.E. Karniadakis.
        <em>DeepXDE: A deep learning library for solving differential equations.</em>
        SIAM Rev. 63(1), 208&ndash;228 (2021).</li>
      <li>S. Wang, X. Yu, P. Perdikaris.
        <em>When and why PINNs fail to train: A neural tangent kernel perspective.</em>
        J. Comput. Phys. 449, 110768 (2022).</li>
      <li>P.M. Morse, K.U. Ingard.
        <em>Theoretical Acoustics.</em>
        Princeton University Press (1968). Ch.&nbsp;8: rigid cylinder scattering series solution.</li>
    </ol>
    <p style="margin-top:2rem;font-size:.78rem;color:#94949e">
      Code: <a href="https://github.com/carlm451/helmholtzscatteringpinn">github.com/carlm451/helmholtzscatteringpinn</a>
    </p>
  </div>
</section>""")

    # Assemble
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HelmholtzPINN &mdash; Presentation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Space+Grotesk:wght@500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="{PLOTLY_CDN}"></script>
<script>
MathJax = {{
  tex: {{inlineMath: [['\\\\(','\\\\)']],displayMath: [['\\\\[','\\\\]']]}},
  chtml: {{scale: 1.05}}
}};
</script>
<script src="{MATHJAX_CDN}" async></script>
<style>{SLIDE_CSS}</style>
</head>
<body>
<div id="progress"></div>
{''.join(S)}
<div id="counter"></div>
<div id="nav-hint">&larr; &rarr; navigate</div>
<script>{SLIDE_JS}</script>
<script>
/* Hide Plotly toolbar and titles that duplicate slide headings */
document.addEventListener('DOMContentLoaded',function(){{
  document.querySelectorAll('.plotly-graph-div').forEach(function(el){{
    var obs=new MutationObserver(function(){{
      var tb=el.querySelector('.modebar-container');
      if(tb)tb.style.display='none';
    }});
    obs.observe(el,{{childList:true,subtree:true}});
  }});
}});
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════
#  Dashboard
# ════════════════════════════════════════════════════════════════

DASH_CSS = r"""
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:'DM Sans',sans-serif;color:#e8e6f0;background:#0c0f1a;line-height:1.6}

/* ── Top tab bar ── */
.tab-bar{position:sticky;top:0;z-index:50;background:#0c0f1a;
  border-bottom:1px solid rgba(255,255,255,.08);padding:.6rem 2rem .5rem}
.tab-bar::before{content:'';position:absolute;top:0;left:2rem;right:2rem;height:2px;
  background:linear-gradient(90deg,#6366f1 0%,#2dd4bf 50%,transparent 100%);opacity:.5}
.tab-bar-inner{max-width:1200px;margin:0 auto;display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.tab-bar .brand{font-family:'Space Grotesk',sans-serif;font-size:.9rem;font-weight:700;
  color:#f0eef8;margin-right:1.5rem;letter-spacing:-.02em;white-space:nowrap}
.tab-btn{display:inline-block;padding:.35rem .75rem;
  border:1px solid rgba(255,255,255,.1);border-radius:3px;
  font-size:.75rem;color:#9b97b0;cursor:pointer;background:none;
  font-family:'DM Sans',sans-serif;transition:border-color .15s,background .15s,color .15s}
.tab-btn:hover{border-color:#6366f1;background:rgba(99,102,241,.08);color:#c4c0d8}
.tab-btn.active{border-color:#6366f1;background:rgba(99,102,241,.15);color:#a5b4fc;font-weight:600}
.tab-bar .slides-link{margin-left:auto;font-size:.72rem;color:#818cf8;text-decoration:none;
  font-family:'Space Mono',monospace;letter-spacing:.05em}
.tab-bar .slides-link:hover{color:#a5b4fc;text-decoration:underline}

/* ── Tab content ── */
.tab-content{display:none;max-width:1200px;margin:0 auto;padding:1.5rem 2rem 4rem}
.tab-content.active{display:block}

/* ── Section headings ── */
.tab-content h2{font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:700;
  letter-spacing:-.03em;color:#f0eef8;margin-bottom:1rem;padding-bottom:.4rem;
  border-bottom:1px solid rgba(255,255,255,.1)}
.tab-content h3{font-family:'Space Grotesk',sans-serif;font-size:.9rem;font-weight:600;
  margin:1.4rem 0 .5rem;color:#c4c0d8}
.tab-content p{font-size:.85rem;margin-bottom:.6rem;color:#b8b4cc}

/* ── Metrics row ── */
.metrics{display:flex;gap:.7rem;flex-wrap:wrap;margin:.8rem 0}
.metric{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
  border-radius:6px;padding:.6rem .9rem;min-width:130px;flex:1}
.metric .label{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:#9b97b0}
.metric .value{font-size:1.2rem;font-weight:700;margin-top:.1rem;color:#e8e6f0}
.metric .value.good{color:#2dd4bf}

/* ── Tables ── */
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.83rem}
th{background:rgba(255,255,255,.04);font-size:.68rem;text-transform:uppercase;
  letter-spacing:.07em;color:#9b97b0;
  border-bottom:1px solid rgba(255,255,255,.1);border-top:none;border-left:none;border-right:none}
td{border:none;border-bottom:1px solid rgba(255,255,255,.06)}
th,td{padding:.45rem .7rem;text-align:center}
td:first-child{text-align:left;font-weight:600;color:#e8e6f0}
.mono{font-family:'Space Mono',monospace;font-size:.8rem;color:#c4c0d8}
.best td{background:rgba(45,212,191,.08)}

/* ── Plot grids ── */
.plot-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
  gap:.8rem;margin:.8rem 0}
.plot-grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:.8rem 0}
.plot-grid-4{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin:.8rem 0}
.plot-card{border:1px solid rgba(255,255,255,.08);border-radius:6px;overflow:hidden;
  transition:border-color .15s;background:rgba(255,255,255,.02)}
.plot-card:hover{border-color:#6366f1}
.plot-card a{display:block;padding:.5rem .7rem;text-decoration:none;
  color:#818cf8;font-weight:500;font-size:.82rem}
.plot-card a:hover{text-decoration:underline;background:rgba(99,102,241,.05)}
.plot-card .pl{display:block;font-size:.7rem;color:#9b97b0;font-weight:400;margin-top:.05rem}

/* ── Inline figure containers ── */
.fig-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:.8rem 0}
.fig-row>div{min-width:0;overflow:hidden}
.fig-row .plotly-graph-div{width:100%!important;min-height:350px}
.fig-row-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;margin:.8rem 0}
.fig-row-3>div{min-width:0;overflow:hidden}
.fig-row-3 .plotly-graph-div{width:100%!important;min-height:300px}
.fig-full{margin:.8rem 0}
.fig-full .plotly-graph-div{width:100%!important}
.fig-label{font-family:'Space Mono',monospace;font-size:.7rem;color:#9b97b0;
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem}

/* ── Equations ── */
.eq-block{background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.18);
  border-radius:4px;padding:.7rem 1rem;margin:.5rem 0;text-align:center}

/* ── Callouts ── */
.highlight{background:rgba(45,212,191,.08);border-left:2px solid #2dd4bf;
  padding:.6rem .9rem;font-size:.83rem;margin:.6rem 0;color:#c4f0ea}
.highlight strong{color:#2dd4bf}
.callout-result{background:rgba(99,102,241,.08);border-left:2px solid #6366f1;
  padding:.6rem .9rem;font-size:.83rem;margin:.6rem 0;color:#c7c4f0}

/* ── Architecture flow ── */
.arch-flow{display:flex;align-items:center;justify-content:center;gap:.7rem;
  margin:1rem 0;flex-wrap:wrap}
.arch-node{border:1.5px solid rgba(255,255,255,.2);border-radius:6px;
  padding:.55rem .9rem;text-align:center;font-size:.8rem;font-weight:600;
  min-width:105px;color:#e8e6f0;background:rgba(255,255,255,.03)}
.arch-node .detail{display:block;font-size:.66rem;font-weight:400;
  color:#9b97b0;margin-top:.15rem}
.arch-arrow{font-size:1.2rem;color:#4b4868}
.node-ff{background:rgba(99,102,241,.12);border-color:rgba(99,102,241,.4);color:#a5b4fc}
.node-res{background:rgba(45,212,191,.1);border-color:rgba(45,212,191,.35);color:#99f6e4}
.node-out{background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.3);color:#fde68a}

/* ── Links ── */
a{color:#818cf8;text-decoration:none}
a:hover{color:#a5b4fc;text-decoration:underline}

/* ── Lists ── */
ul{margin-left:1.2rem;font-size:.85rem;color:#c4c0d8}
li{margin-bottom:.4rem}
li strong{color:#e8e6f0}
.small{font-size:.78rem;color:#9b97b0}

/* ── Responsive ── */
@media(max-width:900px){
  .tab-bar{padding:.5rem 1rem}
  .tab-content{padding:1rem 1rem 3rem}
  .fig-row,.plot-grid-2,.plot-grid-4{grid-template-columns:1fr}
}
"""

DASH_JS = """
function showTab(id){
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  const tab=document.getElementById('tab-'+id);
  const btn=document.querySelector('.tab-btn[data-tab="'+id+'"]');
  if(tab)tab.classList.add('active');
  if(btn)btn.classList.add('active');
  window.scrollTo(0,0);
  history.replaceState(null,null,'#'+id);
  /* Trigger resize so Plotly redraws in newly visible container */
  requestAnimationFrame(()=>window.dispatchEvent(new Event('resize')));
}
document.querySelectorAll('.tab-btn').forEach(b=>{
  b.addEventListener('click',function(){showTab(this.dataset.tab)});
});
/* Arrow-key tab switching */
const tabIds=[...document.querySelectorAll('.tab-btn')].map(b=>b.dataset.tab);
document.addEventListener('keydown',function(e){
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
  const cur=tabIds.findIndex(id=>document.getElementById('tab-'+id).classList.contains('active'));
  if(e.key==='ArrowRight'&&cur<tabIds.length-1){e.preventDefault();showTab(tabIds[cur+1])}
  else if(e.key==='ArrowLeft'&&cur>0){e.preventDefault();showTab(tabIds[cur-1])}
});
/* Hash-based routing */
const hash=location.hash.slice(1);
if(hash&&document.getElementById('tab-'+hash)){showTab(hash)}
else{showTab('results')}
"""


def _field_grid(ka_slug, ka_label):
    """Build a plot-card grid for one ka value (link-based fallback)."""
    base = f"plots/prod_eval_{ka_slug}"
    plots = [
        ("Global Re(&phi;<sub>s</sub>)", f"{ka_slug}_global_real.html", "PINN vs Analytic"),
        ("Global |&phi;<sub>s</sub>|", f"{ka_slug}_global_mag.html", "Magnitude comparison"),
        ("Shadow boundary", f"{ka_slug}_shadow_boundary.html", "Downstream region"),
        ("Near surface", f"{ka_slug}_near_surface.html", "Illuminated side detail"),
        ("Overview + zoom", f"{ka_slug}_overview.html", "Zoom region annotations"),
    ]
    cards = ""
    for title, fname, desc in plots:
        cards += f"""<div class="plot-card"><a href="{base}/{fname}" target="_blank">
  {title}<span class="pl">{desc}</span></a></div>\n"""
    return f'<h3>ka = {ka_label}</h3>\n<div class="plot-grid">\n{cards}</div>'


def _embed_field(ka_slug, view, div_id, width="100%", height="420px"):
    """Embed a field comparison figure inline from outputs/."""
    path = os.path.join(OUTPUTS, f"prod_eval_{ka_slug}", f"{ka_slug}_{view}.html")
    return extract_figure_div(path, div_id, width=width, height=height)


def build_dashboard(prod, abl):
    # ── Dark-themed charts ──
    ch = {}
    ch["l2bar"] = _div(_darkify(chart_l2_bar()), "d-l2")
    ch["mxbar"] = _div(_darkify(chart_max_err_bar()), "d-mx")
    ch["mebar"] = _div(_darkify(chart_mean_err_bar()), "d-me")
    ch["errcomb"] = _div(_darkify(chart_errors_combined()), "d-errcomb")
    if prod:
        ch["ploss"] = _div(_darkify(chart_prod_loss(prod)), "d-ploss")
    if abl:
        ch["aloss"] = _div(_darkify(chart_abl_loss(abl)), "d-aloss")
        ab = chart_abl_bar(abl)
        if ab:
            ch["abar"] = _div(_darkify(ab), "d-abar")
        ch["acomp"] = _div(_darkify(chart_abl_components(abl)), "d-acomp")
        if "ff_2pi" in abl:
            ch["aloss_2pi"] = _div(_darkify(chart_abl_loss_2pi(abl)), "d-aloss2")
            ab2 = chart_abl_bar_2pi(abl)
            if ab2:
                ch["abar_2pi"] = _div(_darkify(ab2), "d-abar2")
            ch["acomp_2pi"] = _div(_darkify(chart_abl_components_2pi(abl)), "d-acomp2")

    # ── Generate gallery figures ──
    print("    Generating analytic gallery for dashboard...")
    ch["gallery"] = _div(_darkify(generate_analytic_gallery()), "d-gallery")

    print("    Generating error gallery for dashboard...")
    ch["errors"] = _div(_darkify(generate_error_gallery()), "d-errors")

    print("    Generating honeycomb residuals for dashboard...")
    hc_res_fig = generate_honeycomb_residuals()
    if hc_res_fig:
        ch["hc_resid"] = _div(_darkify(hc_res_fig), "d-hc-resid")

    # ── Generate 3-panel collocation figures ──
    print("    Generating collocation point figures...")
    colloc_figs = generate_collocation_3panel()
    ch["colloc_pde"] = _div(colloc_figs["pde"], "d-cpde")
    ch["colloc_bc"] = _div(colloc_figs["bc"], "d-cbc")
    ch["colloc_abc"] = _div(colloc_figs["abc"], "d-cabc")

    # ── Embed per-ka field comparisons inline ──
    print("    Embedding field comparison figures...")
    for i, p in enumerate(PROD):
        slug = p["slug"]
        ch[f"field_{slug}_real"] = _embed_field(slug, "global_real", f"d-fr-{i}", height="380px")
        ch[f"field_{slug}_mag"] = _embed_field(slug, "global_mag", f"d-fm-{i}", height="380px")
        ch[f"field_{slug}_shadow"] = _embed_field(slug, "shadow_boundary", f"d-fs-{i}", height="350px")
        ch[f"field_{slug}_near"] = _embed_field(slug, "near_surface", f"d-fn-{i}", height="350px")

    # ── Embed honeycomb field comparisons ──
    hc_plots = {
        "hc_real": extract_figure_div(
            os.path.join(OUTPUTS, "hc_summary", "hc_ka2_real_total.html"),
            "d-hc-real", height="400px"),
        "hc_mag": extract_figure_div(
            os.path.join(OUTPUTS, "hc_summary", "hc_ka2_mag_total.html"),
            "d-hc-mag", height="400px"),
        "hc_scat": extract_figure_div(
            os.path.join(OUTPUTS, "hc_summary", "hc_ka2_real_scattered.html"),
            "d-hc-scat", height="400px"),
        "an_real": extract_figure_div(
            os.path.join(OUTPUTS, "hc_summary", "analytic_ka2_real_total.html"),
            "d-an-real", height="400px"),
    }

    # ── Tab bar ──
    tabs = [
        ("takeaways", "Takeaways"),
        ("results", "Results"),
        ("fields", "Field Plots"),
        ("ablation", "Ablation & BCs"),
        ("honeycomb", "Honeycomb"),
        ("method", "Method"),
        ("background", "Background"),
    ]
    tab_btns = "\n".join(
        f'<button class="tab-btn" data-tab="{tid}">{lbl}</button>'
        for tid, lbl in tabs)

    # ════════════════════════════════════════════
    #  TAB 0: Takeaways
    # ════════════════════════════════════════════
    # Reuse ka=pi and ka=2pi field plots for visual interest
    tk_field_pi = ch.get("field_ka3.14_real", "")
    tk_field_2pi = ch.get("field_ka6.28_real", "")

    t_takeaways = f"""<div id="tab-takeaways" class="tab-content">
  <h2>Conclusion: PINNs as a Multi-Scale Probe</h2>
  <div style="background:rgba(45,212,191,.1);border:1.5px solid rgba(45,212,191,.35);
    border-radius:6px;padding:1rem 1.2rem;margin:.8rem 0">
    <p style="font-size:.9rem;color:#e8e6f0;margin:0;line-height:1.6">
      The goal of this project was to test whether a PINN can <strong>dynamically resolve
      physical fields at arbitrary length scales</strong> &mdash; effectively zooming in to
      reveal finer spatial structure. These results show that <strong style="color:#2dd4bf">
      yes, it can</strong> &mdash; but with a clear tradeoff:
      <strong style="color:#2dd4bf">resolving shorter wavelengths demands exponentially more
      compute.</strong></p>
  </div>

  <div class="fig-row" style="margin:1.2rem 0">
    <div><div class="fig-label">ka = &pi; &mdash; PINN vs Analytic &mdash; Re(&phi;<sub>s</sub>)</div>{tk_field_pi}</div>
    <div><div class="fig-label">ka = 2&pi; &mdash; PINN vs Analytic &mdash; Re(&phi;<sub>s</sub>)</div>{tk_field_2pi}</div>
  </div>

  <div class="fig-row">
    <div>
      <h3 style="color:#2dd4bf">Where PINNs Excel</h3>
      <p>For <strong>low-to-moderate spatial frequency</strong> (ka&nbsp;&le;&nbsp;&pi;), the PINN
      converges to 2&ndash;3% L2 error in 12&ndash;17 minutes. At these scales, PINNs offer
      genuine advantages over mesh-based solvers: continuous field access at any coordinate,
      no mesh generation, and physics enforced by construction. The network is a
      differentiable, resolution-independent surrogate &mdash; it can be queried anywhere
      without interpolation or remeshing.</p>
    </div>
    <div>
      <h3 style="color:#f87171">The Compute Wall</h3>
      <p>As ka grows, training cost scales steeply: ka=2&pi; needed 5&times; more epochs and a
      50% wider network (259 min vs 12 min). At ka=3&pi;, a 10+ hour run plateaued at 68% L2
      despite loss dropping 5 orders of magnitude. Zooming in to finer features is equivalent to
      probing higher spatial frequencies, and the network faces a fundamental
      resolution&ndash;compute tradeoff analogous to the Nyquist limit in signal processing.</p>
    </div>
  </div>

  <div class="metrics" style="margin-top:1.2rem">
    <div class="metric"><div class="label">Best L2 Error</div>
      <div class="value good">2.00%</div></div>
    <div class="metric"><div class="label">ka Range Validated</div>
      <div class="value">0.5 &ndash; 2&pi;</div></div>
    <div class="metric"><div class="label">Fastest Convergence</div>
      <div class="value">12 min</div></div>
    <div class="metric"><div class="label">Hardest Run</div>
      <div class="value">259 min</div></div>
  </div>

  <h3 style="margin-top:1.2rem">Key Results</h3>
  {results_table()}
  <div class="fig-row" style="margin-top:1rem">
    <div>{ch["l2bar"]}</div>
    <div>{ch["errcomb"]}</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:1.5rem">
    <div>
      <h3>1. Multi-scale accuracy</h3>
      <p>2&ndash;8% L2 across a 12&times; range of spatial frequency with a single architecture.</p>
    </div>
    <div>
      <h3>2. Physics-aware features</h3>
      <p>Fourier features with &sigma;=k embed the wavelength directly, defeating spectral bias.</p>
    </div>
    <div>
      <h3>3. Patience over engineering</h3>
      <p>The convergent ka=2&pi; run used default loss weights but 2.5&times; more epochs &mdash;
      49%&rarr;2%. Compute budget, not loss rebalancing, was the binding constraint.</p>
    </div>
  </div>
</div>"""

    # ════════════════════════════════════════════
    #  TAB 1: Results
    # ════════════════════════════════════════════
    conv_chart = ch.get("ploss", '<p class="small">Training curves unavailable (wandb data not loaded)</p>')
    t_results = f"""<div id="tab-results" class="tab-content">
  <h2>Production Results</h2>
  <p>All runs: circle boundary + BGT2 ABC, L&nbsp;=&nbsp;3.0. Trained on 4&times; NVIDIA RTX 4000 Ada.</p>
  <div class="metrics">
    <div class="metric"><div class="label">Best L2 Error</div>
      <div class="value good">2.00%</div></div>
    <div class="metric"><div class="label">ka Range</div>
      <div class="value">0.5 &ndash; 2&pi;</div></div>
    <div class="metric"><div class="label">Boundary</div>
      <div class="value">Circle + BGT2</div></div>
    <div class="metric"><div class="label">Optimizer</div>
      <div class="value">Adam &rarr; L-BFGS</div></div>
  </div>
  {results_table()}
  <div class="fig-row">
    <div>{ch["l2bar"]}</div>
    <div>{conv_chart}</div>
  </div>
  <div class="fig-row">
    <div>{ch["mebar"]}</div>
    <div>{ch["mxbar"]}</div>
  </div>
  <p>Mean error stays below 2% across all ka values. Max error peaks at 4.4% for ka=2&pi;,
  localized to the shadow boundary. All runs exhibit the Adam &rarr; L-BFGS transition (sharp loss drop).
  Higher ka requires progressively more compute: 12 min at ka=0.5 vs 259 min at ka=2&pi;.</p>
  <p>L2 relative error is highest at low ka because the scattered field amplitude is small
  (smaller normalization denominator) and these runs received 4&times; less compute (10K vs 50K epochs,
  smaller network). The low-ka errors could likely be reduced with comparable training budgets.</p>
  <h3>Scaling for ka = 2&pi;</h3>
  <table>
    <tr><th></th><th>ka &le; &pi;</th><th>ka = 2&pi;</th></tr>
    <tr><td>Neurons</td><td class="mono">256</td><td class="mono">384</td></tr>
    <tr><td>Layers</td><td class="mono">4</td><td class="mono">6</td></tr>
    <tr><td>Fourier feat.</td><td class="mono">64</td><td class="mono">96</td></tr>
    <tr><td>Adam epochs</td><td class="mono">10K</td><td class="mono">50K</td></tr>
    <tr><td>L-BFGS iters</td><td class="mono">200</td><td class="mono">300</td></tr>
    <tr><td>&lambda;<sub>pde</sub> / &lambda;<sub>bc</sub></td>
        <td class="mono">1.0 / 10</td><td class="mono">1.0 / 10</td></tr>
    <tr><td>Wall time</td><td class="mono">12&ndash;17 min</td><td class="mono">259 min</td></tr>
  </table>
  <p class="small">Default loss weights used for the convergent ka=2&pi; run. An earlier attempt
  with rebalanced weights (&lambda;<sub>pde</sub>=0.25, &lambda;<sub>bc</sub>=15) and only
  20K epochs stalled at 49% L2 error &mdash; sufficient training budget, not loss engineering,
  was the decisive factor.</p>

  <h2 style="margin-top:2rem">Conclusion: PINNs as a Multi-Scale Probe</h2>
  <div style="background:rgba(45,212,191,.1);border:1.5px solid rgba(45,212,191,.35);
    border-radius:6px;padding:1rem 1.2rem;margin:.8rem 0">
    <p style="font-size:.9rem;color:#e8e6f0;margin:0;line-height:1.6">
      The goal of this project was to test whether a PINN can <strong>dynamically resolve
      physical fields at arbitrary length scales</strong> &mdash; effectively zooming in to
      reveal finer spatial structure. These results show that it can, but with a clear
      tradeoff: <strong style="color:#2dd4bf">resolving shorter wavelengths demands
      exponentially more compute.</strong></p>
  </div>
  <div class="fig-row" style="margin-top:1rem">
    <div>
      <h3 style="color:#2dd4bf">Where PINNs Excel</h3>
      <p>For low-to-moderate spatial frequency (ka&nbsp;&le;&nbsp;&pi;), the PINN converges to
      2&ndash;3% L2 error in 12&ndash;17 minutes. At these scales, PINNs offer genuine advantages
      over mesh-based solvers: continuous field access at any point, no mesh generation, and
      physics enforced by construction. The network is a differentiable, resolution-independent
      surrogate for the field &mdash; it can be queried at arbitrary coordinates without
      interpolation or remeshing.</p>
    </div>
    <div>
      <h3 style="color:#f87171">The Compute Wall</h3>
      <p>As ka grows, training cost scales steeply: ka=2&pi; needed 5&times; more epochs and a
      50% wider network (259 min vs 12 min). At ka=3&pi;, a 10+ hour run plateaued at 68% L2
      despite loss dropping 5 orders of magnitude. Zooming in to finer features is equivalent to
      probing higher spatial frequencies, and the network faces a fundamental
      resolution&ndash;compute tradeoff analogous to the Nyquist limit in signal processing.
      At the frontier, <strong>compute budget &mdash; not architecture or loss design &mdash;
      is the binding constraint.</strong></p>
    </div>
  </div>
</div>"""

    # ════════════════════════════════════════════
    #  TAB 2: Fields
    # ════════════════════════════════════════════
    # Build per-ka embedded grids
    field_sections = ""
    for p in PROD:
        slug = p["slug"]
        lbl = p["label"]
        real_div = ch.get(f"field_{slug}_real", "")
        mag_div = ch.get(f"field_{slug}_mag", "")
        shadow_div = ch.get(f"field_{slug}_shadow", "")
        near_div = ch.get(f"field_{slug}_near", "")
        field_sections += f"""
  <h3>ka = {lbl} &mdash; L2 = {p["l2"]}%</h3>
  <div class="fig-row">
    <div><div class="fig-label">PINN vs Analytic &mdash; Re(&phi;<sub>s</sub>)</div>{real_div}</div>
    <div><div class="fig-label">PINN vs Analytic &mdash; |&phi;<sub>s</sub>|</div>{mag_div}</div>
  </div>
  <div class="fig-row">
    <div><div class="fig-label">Shadow boundary</div>{shadow_div}</div>
    <div><div class="fig-label">Near surface</div>{near_div}</div>
  </div>
"""

    t_fields = f"""<div id="tab-fields" class="tab-content">
  <h2>Field Comparisons</h2>
  <p>PINN predictions vs analytic Bessel/Hankel series. Each comparison shows PINN | Analytic | Error.</p>

  <h3>Analytic Scattered Field &mdash; All Wavenumbers</h3>
  <div class="fig-full">{ch["gallery"]}</div>

  <h3>Spatial Error Distribution &mdash; All Wavenumbers</h3>
  <div class="fig-full">{ch["errors"]}</div>

  {field_sections}
</div>"""

    # ════════════════════════════════════════════
    #  TAB 3: Ablation & BCs
    # ════════════════════════════════════════════
    abl_charts_pi = ""
    if "aloss" in ch:
        abl_charts_pi += f'<div>{ch["aloss"]}</div>'
    if "abar" in ch:
        abl_charts_pi += f'<div>{ch["abar"]}</div>'
    if not abl_charts_pi:
        abl_charts_pi = '<p class="small">ka=&pi; ablation charts unavailable (wandb data not loaded)</p>'

    abl_charts_2pi = ""
    if "aloss_2pi" in ch:
        abl_charts_2pi += f'<div>{ch["aloss_2pi"]}</div>'
    if "abar_2pi" in ch:
        abl_charts_2pi += f'<div>{ch["abar_2pi"]}</div>'

    abl_comp = ch.get("acomp", "")
    abl_comp_2pi = ch.get("acomp_2pi", "")

    t_ablation = f"""<div id="tab-ablation" class="tab-content">
  <h2>Fourier Feature Ablation</h2>
  <p>Parameter-matched comparison &mdash;
  FF-PINN (64 Fourier features, 256 hidden, ~296K params) vs
  plain MLP (272 hidden, ~298K params). Both used identical 10K Adam + 200 L-BFGS
  schedules &mdash; shorter than production &mdash; to isolate the architectural effect.
  Same loss weights (&lambda;<sub>pde</sub>=1, &lambda;<sub>bc</sub>=10).</p>

  <h3>ka = &pi; (low frequency)</h3>
  <div class="fig-row">{abl_charts_pi}</div>
  {f'<div class="fig-full">{abl_comp}</div>' if abl_comp else ''}
  <table style="margin:.8rem 0">
    <tr><th></th><th>FF-PINN</th><th>Plain MLP</th></tr>
    <tr><td>L2 relative</td><td class="mono">2.42%</td><td class="mono">2.42%</td></tr>
    <tr><td>Max error</td><td class="mono">2.97%</td><td class="mono">2.99%</td></tr>
  </table>
  <p>No meaningful difference &mdash; at low frequency, the standard MLP can represent
  the scattered field without Fourier features. Spectral bias is not a bottleneck here.</p>

  <h3 style="margin-top:1.5rem">ka = 2&pi; (high frequency)</h3>
  <div class="fig-row">{abl_charts_2pi if abl_charts_2pi else '<p class="small">ka=2&pi; ablation charts unavailable</p>'}</div>
  {f'<div class="fig-full">{abl_comp_2pi}</div>' if abl_comp_2pi else ''}
  <table style="margin:.8rem 0">
    <tr><th></th><th>FF-PINN</th><th>Plain MLP</th><th>Ratio</th></tr>
    <tr><td>L2 relative</td><td class="mono">10.1%</td><td class="mono">11.7%</td><td class="mono">1.15&times; worse</td></tr>
    <tr class="best"><td>Max error</td><td class="mono">22.7%</td><td class="mono">47.3%</td><td class="mono">2.1&times; worse</td></tr>
  </table>
  <p>The max error <strong>doubles</strong> without Fourier features. The plain MLP&rsquo;s L-BFGS phase actually
  <em>increased</em> its error (L2: 10.9%&rarr;11.7%), while the FF-PINN&rsquo;s L-BFGS phase dramatically
  <em>improved</em> it (19.6%&rarr;10.1%). This suggests Fourier features create a smoother loss landscape
  that second-order optimizers can exploit more effectively.</p>

  <div class="highlight" style="margin-top:1rem">
    <strong>Narrative:</strong> At low ka, spectral bias isn&rsquo;t a bottleneck and both architectures
    perform identically. As frequency doubles, the plain MLP degrades &mdash; particularly in max
    (pointwise) error, which is 2&times; worse. The Fourier feature projection maps raw coordinates
    into a bandwidth-matched frequency space (\\(\\sigma = k\\)), enabling the network to represent
    oscillatory structure the MLP alone struggles with.
  </div>

  <h2 style="margin-top:2rem">Absorbing Boundary Conditions</h2>
  <h3>First-Order ABC</h3>
  <div class="eq-block">\\[{r"\frac{\partial \phi_s}{\partial n} - ik\phi_s = 0"}\\]</div>
  <p>Assumes purely outgoing plane waves. Works on any boundary shape but ignores
  wavefront curvature &mdash; introducing O(1/r) reflection error.</p>
  <h3>BGT2 (Second-Order)</h3>
  <div class="eq-block">\\[{r"\frac{\partial \phi_s}{\partial r} - ik\phi_s + \frac{\phi_s}{2r} = 0"}\\]</div>
  <p>The curvature correction \\(\\phi_s/2r\\) accounts for the \\(1/\\sqrt{{r}}\\)
  amplitude decay of cylindrical waves, annihilating one more term in the
  scattered field series
  <span class="small">[Bayliss, Gunzburger &amp; Turkel, 1982]</span>.</p>
  <h3>Why Circular Domain?</h3>
  <ul>
    <li><strong>Symmetry match:</strong> circular boundary matches the cylindrical geometry</li>
    <li><strong>Uniform distance</strong> from scatterer to ABC boundary at all angles</li>
    <li><strong>No corner singularities:</strong> square boundaries have undefined normals at corners</li>
    <li><strong>BGT2 requires curvature:</strong> the \\(\\phi_s/2r\\) correction is derived for circular boundaries</li>
  </ul>
  <div class="highlight" style="margin-top:.8rem">
    <strong>Result:</strong> Circle + BGT2 gives the most physically consistent setup
    for this cylindrical scattering problem.
  </div>
</div>"""

    # ════════════════════════════════════════════
    #  TAB 4: Honeycomb
    # ════════════════════════════════════════════
    hc_resid_div = ch.get("hc_resid", '<p class="small">Residual diagnostics unavailable</p>')
    t_hc = f"""<div id="tab-honeycomb" class="tab-content">
  <h2>Extension: Honeycomb Acoustic Shield</h2>
  <p>19-circle hexagonal lattice (3 concentric rings), r<sub>s</sub>&nbsp;=&nbsp;0.15,
  d&nbsp;=&nbsp;0.4, same cluster radius as single cylinder. No analytic solution &mdash; pure PINN.</p>
  <table>
    <tr><th></th><th>Single Cylinder</th><th>Honeycomb Lattice</th></tr>
    <tr><td>Scatterers</td><td class="mono">1, r = 1.0</td>
        <td class="mono">19, r<sub>s</sub> = 0.15</td></tr>
    <tr><td>Validation</td><td class="mono">Analytic (Bessel/Hankel)</td>
        <td class="mono">PINN only</td></tr>
    <tr><td>ka</td><td class="mono">0.5 &ndash; 2&pi;</td>
        <td class="mono">2.0</td></tr>
  </table>

  <h3>Field Comparisons</h3>
  <div class="fig-row" style="margin-bottom:2rem">
    <div><div class="fig-label">Honeycomb &mdash; Re(&phi;<sub>total</sub>)</div>{hc_plots["hc_real"]}</div>
    <div><div class="fig-label">Honeycomb &mdash; |&phi;<sub>total</sub>|</div>{hc_plots["hc_mag"]}</div>
  </div>
  <div class="fig-row" style="margin-bottom:2rem">
    <div><div class="fig-label">Honeycomb &mdash; Re(&phi;<sub>s</sub>)</div>{hc_plots["hc_scat"]}</div>
    <div><div class="fig-label">Single Cylinder (Analytic) &mdash; Re(&phi;<sub>total</sub>)</div>{hc_plots["an_real"]}</div>
  </div>

  <h3>Residual Diagnostics</h3>
  <div class="fig-full" style="margin-bottom:2rem">{hc_resid_div}</div>

  <h3>Convergence</h3>
  <table>
    <tr><th>Loss Component</th><th>Value</th><th>Status</th></tr>
    <tr><td>PDE</td><td class="mono">4.92e-6</td>
        <td class="mono" style="color:#2dd4bf">Converged</td></tr>
    <tr><td>BC (19 surfaces)</td><td class="mono">2.00e-7</td>
        <td class="mono" style="color:#2dd4bf">Converged</td></tr>
    <tr><td>ABC</td><td class="mono">1.41e-7</td>
        <td class="mono" style="color:#2dd4bf">Converged</td></tr>
    <tr><td><strong>Total</strong></td><td class="mono"><strong>9.06e-6</strong></td>
        <td class="mono" style="color:#2dd4bf"><strong>Converged</strong></td></tr>
  </table>
  <div class="highlight">
    <strong>Acoustic shielding:</strong> |&phi;<sub>total</sub>|&nbsp;&lt;&nbsp;0.003
    inside the cluster &mdash; the scattered field almost perfectly cancels the incident wave.
  </div>
</div>"""

    # ════════════════════════════════════════════
    #  TAB 5: Method
    # ════════════════════════════════════════════
    t_method = f"""<div id="tab-method" class="tab-content">
  <h2>Methodology</h2>
  <h3>Loss Formulation</h3>
  <div class="eq-block">\\[{r"\mathcal{L} = \lambda_{\text{pde}}\,\mathcal{L}_{\text{pde}} + \lambda_{\text{bc}}\,\mathcal{L}_{\text{bc}} + \lambda_{\text{abc}}\,\mathcal{L}_{\text{abc}}"}\\]</div>
  <p>Each component is an MSE of the corresponding residual evaluated at
  sampled collocation points. The scattered-field formulation means the Neumann BC
  target is known analytically from the incident field.</p>
  <h3>Network Architecture</h3>
  {ARCH_HTML}
  <h3>Sampling</h3>
  <table>
    <tr><th>Region</th><th>Points</th><th>Method</th></tr>
    <tr><td>Interior</td><td class="mono">10K&ndash;20K</td>
        <td class="mono">Uniform rejection sampling</td></tr>
    <tr><td>Scatterer BC</td><td class="mono">200&ndash;400</td>
        <td class="mono">Uniform on circle</td></tr>
    <tr><td>Outer ABC</td><td class="mono">400&ndash;600</td>
        <td class="mono">Uniform on circle</td></tr>
  </table>
  <p>Collocation points are resampled every 2,000 epochs during Adam training
  with 50% blending (old + new) for stability.</p>
  <h3>Evaluation</h3>
  <p>Accuracy is measured against the analytic Bessel/Hankel series solution on a
  dense 200&times;200 grid. Metrics: L2 relative error and max pointwise error.
  The series uses \\(N = \\lceil ka + 20 \\rceil\\) terms for convergence.</p>

  <h2 style="margin-top:2rem">References</h2>
  <ol style="font-size:.82rem;line-height:1.7;color:#c4c0d8">
    <li>M. Raissi, P. Perdikaris, G.E. Karniadakis.
      <em>Physics-informed neural networks: A deep learning framework for solving
      forward and inverse problems involving nonlinear partial differential equations.</em>
      J. Comput. Phys. 378, 686&ndash;707 (2019).</li>
    <li>M. Tancik et al.
      <em>Fourier features let networks learn high frequency functions in low dimensional
      domains.</em> NeurIPS (2020).</li>
    <li>A. Bayliss, M. Gunzburger, E. Turkel.
      <em>Boundary conditions for the numerical solution of elliptic equations in exterior
      regions.</em> SIAM J. Appl. Math. 42(2), 430&ndash;451 (1982).</li>
    <li>L. Lu, X. Meng, Z. Mao, G.E. Karniadakis.
      <em>DeepXDE: A deep learning library for solving differential equations.</em>
      SIAM Rev. 63(1), 208&ndash;228 (2021).</li>
    <li>S. Wang, X. Yu, P. Perdikaris.
      <em>When and why PINNs fail to train: A neural tangent kernel perspective.</em>
      J. Comput. Phys. 449, 110768 (2022).</li>
    <li>P.M. Morse, K.U. Ingard.
      <em>Theoretical Acoustics.</em>
      Princeton University Press (1968). Ch.&nbsp;8: rigid cylinder scattering series solution.</li>
  </ol>
  <p style="margin-top:1.5rem;font-size:.78rem;color:#6b6880">
    Code: <a href="https://github.com/carlm451/helmholtzscatteringpinn">github.com/carlm451/helmholtzscatteringpinn</a>
  </p>
</div>"""

    # ════════════════════════════════════════════
    #  TAB 6: Background
    # ════════════════════════════════════════════
    t_bg = f"""<div id="tab-background" class="tab-content">
  <h2>Exact Analytic Solution</h2>
  <p>Acoustic scattering of a time-harmonic plane wave off a sound-hard (rigid)
  circular cylinder is one of the canonical problems in mathematical physics
  with a known exact solution. This makes it an ideal validation benchmark
  for PINN accuracy.</p>

  <h3>Jacobi-Anger Expansion</h3>
  <p>The incident plane wave \\(\\phi_{{\\text{{inc}}}} = e^{{ikx}}\\) is decomposed
  into cylindrical harmonics via the Jacobi-Anger identity:</p>
  <div class="eq-block">
    \\[\\phi_{{\\text{{inc}}}}(r,\\theta) = e^{{ikr\\cos\\theta}}
    = \\sum_{{n=-\\infty}}^{{\\infty}} i^n\\, J_n(kr)\\, e^{{in\\theta}}\\]
  </div>
  <p>where \\(J_n\\) is the Bessel function of the first kind of order \\(n\\),
  and \\((r,\\theta)\\) are polar coordinates centered on the cylinder.</p>

  <h3>Scattered Field</h3>
  <p>The scattered field is expanded in outgoing cylindrical waves using
  Hankel functions of the first kind \\(H_n^{{(1)}}\\), which satisfy the
  Sommerfeld radiation condition
  \\(\\lim_{{r\\to\\infty}} \\sqrt{{r}}\\,(\\partial\\phi_s/\\partial r - ik\\phi_s) = 0\\):</p>
  <div class="eq-block">
    \\[\\phi_s(r,\\theta) = \\sum_{{n=-\\infty}}^{{\\infty}} A_n\\, H_n^{{(1)}}(kr)\\, e^{{in\\theta}}\\]
  </div>

  <h3>Sound-Hard Boundary Condition</h3>
  <p>For a rigid cylinder the total field satisfies a Neumann (zero normal
  velocity) condition at the surface \\(r = a\\):</p>
  <div class="eq-block">
    \\[\\frac{{\\partial}}{{\\partial r}}\\bigl(\\phi_{{\\text{{inc}}}} + \\phi_s\\bigr)\\bigg|_{{r=a}} = 0\\]
  </div>
  <p>Substituting the series and solving term-by-term gives the exact coefficients:</p>
  <div class="eq-block">
    \\[A_n = -i^n\\,\\frac{{J_n'(ka)}}{{H_n^{{(1)\\prime}}(ka)}}\\]
  </div>
  <p>where primes denote derivatives with respect to the argument. The complete
  exact scattered field is therefore:</p>
  <div class="eq-block" style="background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.25)">
    \\[\\boxed{{\\phi_s(r,\\theta) = -\\sum_{{n=-\\infty}}^{{\\infty}} i^n\\,
    \\frac{{J_n'(ka)}}{{H_n^{{(1)\\prime}}(ka)}}\\, H_n^{{(1)}}(kr)\\, e^{{in\\theta}}}}\\]
  </div>

  <h3>Why This Matters for PINN Validation</h3>
  <ul>
    <li><strong>Pixel-level ground truth:</strong> the series can be evaluated at
    arbitrary \\((r,\\theta)\\), giving a dense reference field with no discretization
    error from a mesh or finite-difference grid.</li>
    <li><strong>Convergence control:</strong> truncating at
    \\(N \\approx \\lceil ka \\rceil + 20\\) terms gives machine-precision accuracy,
    so all reported PINN errors are true model errors.</li>
    <li><strong>Multi-scale test:</strong> varying \\(ka\\) from 0.5 to \\(2\\pi\\)
    sweeps from sub-wavelength (smooth, easy) to multi-wavelength (oscillatory, hard)
    regimes &mdash; stress-testing the PINN across a 12&times; range of spatial frequency.</li>
  </ul>

  <h2 style="margin-top:2.5rem">Training Point Types</h2>
  <p>The PINN is trained on three distinct sets of collocation points, each enforcing
  a different physical constraint. Points are resampled every 2,000 Adam epochs
  with 50% blending (old + new) for stability.</p>

  <div class="fig-row-3">
    <div>
      <div class="fig-label">PDE interior &mdash; 10K&ndash;20K pts</div>
      {ch["colloc_pde"]}
      <p class="small" style="margin-top:.4rem">
        Uniformly sampled inside the annular domain (between scatterer and ABC boundary).
        These points enforce the Helmholtz PDE: the scattered field must satisfy the
        wave equation everywhere in the fluid.</p>
    </div>
    <div>
      <div class="fig-label">Neumann BC &mdash; 200&ndash;400 pts</div>
      {ch["colloc_bc"]}
      <p class="small" style="margin-top:.4rem">
        Sampled on the scatterer surface (r&nbsp;=&nbsp;a).
        These enforce the sound-hard boundary condition: the normal derivative of the
        total field must vanish, meaning no energy passes through the cylinder wall.</p>
    </div>
    <div>
      <div class="fig-label">ABC outer boundary &mdash; 400&ndash;600 pts</div>
      {ch["colloc_abc"]}
      <p class="small" style="margin-top:.4rem">
        Sampled on the circular outer boundary (r&nbsp;=&nbsp;L).
        These enforce the absorbing boundary condition, which approximates a non-reflecting
        boundary so outgoing waves exit without artificial reflections.</p>
    </div>
  </div>

  <h2 style="margin-top:2.5rem">Loss Functions</h2>
  <p>Training minimizes a weighted sum of three residual losses, each corresponding to
  one of the point types above. All losses are mean squared residuals.</p>

  <div class="eq-block" style="margin:1rem 0">
    \\[\\mathcal{{L}} \\;=\\; \\lambda_{{\\text{{pde}}}}\\,\\mathcal{{L}}_{{\\text{{pde}}}}
    \\;+\\; \\lambda_{{\\text{{bc}}}}\\,\\mathcal{{L}}_{{\\text{{bc}}}}
    \\;+\\; \\lambda_{{\\text{{abc}}}}\\,\\mathcal{{L}}_{{\\text{{abc}}}}\\]
  </div>
  <p class="small">The weights \\(\\lambda\\) balance the three objectives.
  All production runs use default weights:
  \\(\\lambda_{{\\text{{pde}}}} = 1\\), \\(\\lambda_{{\\text{{bc}}}} = 10\\),
  \\(\\lambda_{{\\text{{abc}}}} = 1\\). The elevated \\(\\lambda_{{\\text{{bc}}}}\\) ensures
  the sound-hard boundary condition is enforced strongly.</p>

  <h3>PDE Residual Loss</h3>
  <div class="eq-block">
    \\[\\mathcal{{L}}_{{\\text{{pde}}}} = \\frac{{1}}{{N_{{\\text{{pde}}}}}}
    \\sum_{{i=1}}^{{N_{{\\text{{pde}}}}}}
    \\left| \\nabla^2 \\phi_s(\\mathbf{{x}}_i) + k^2 \\phi_s(\\mathbf{{x}}_i) \\right|^2\\]
  </div>
  <p>Measures how well the network output satisfies the Helmholtz wave equation at
  interior collocation points. A value near zero means the predicted field is a
  valid solution to the PDE. Computed via autograd second derivatives split into
  real and imaginary parts.</p>

  <h3>Neumann BC Loss</h3>
  <div class="eq-block">
    \\[\\mathcal{{L}}_{{\\text{{bc}}}} = \\frac{{1}}{{N_{{\\text{{bc}}}}}}
    \\sum_{{i=1}}^{{N_{{\\text{{bc}}}}}}
    \\left| \\frac{{\\partial \\phi_s}}{{\\partial n}}\\bigg|_{{r=a}}
    + \\frac{{\\partial \\phi_{{\\text{{inc}}}}}}{{\\partial n}}\\bigg|_{{r=a}} \\right|^2\\]
  </div>
  <p>Enforces the sound-hard (rigid) boundary on the cylinder surface: the total
  normal velocity must be zero. The network learns the scattered field whose
  normal derivative cancels that of the known incident wave
  \\(\\phi_{{\\text{{inc}}}} = e^{{ikx}}\\).</p>

  <h3>Absorbing BC Loss (BGT2)</h3>
  <div class="eq-block">
    \\[\\mathcal{{L}}_{{\\text{{abc}}}} = \\frac{{1}}{{N_{{\\text{{abc}}}}}}
    \\sum_{{i=1}}^{{N_{{\\text{{abc}}}}}}
    \\left| \\frac{{\\partial \\phi_s}}{{\\partial r}} - ik\\phi_s
    + \\frac{{\\phi_s}}{{2r}} \\right|^2\\]
  </div>
  <p>Penalizes spurious reflections from the truncated domain boundary. The BGT2
  operator assumes outgoing cylindrical waves with \\(1/\\sqrt{{r}}\\) amplitude decay,
  absorbing two leading terms of the scattered field expansion to minimize artificial
  reflections back into the domain.</p>

  <h2 style="margin-top:2.5rem">Evaluation Metrics</h2>
  <p>After training, the PINN is evaluated against the analytic Bessel/Hankel series
  solution on a dense 200&times;200 grid. The scattered field is complex-valued
  \\(\\phi_s = u + iv\\), so all errors use complex magnitude.</p>

  <h3>Pointwise Absolute Error</h3>
  <div class="eq-block">
    \\[e(\\mathbf{{x}}) = \\left| \\phi_s^{{\\text{{PINN}}}}(\\mathbf{{x}})
    - \\phi_s^{{\\text{{exact}}}}(\\mathbf{{x}}) \\right|
    = \\sqrt{{(u_{{\\text{{pred}}}} - u_{{\\text{{exact}}}})^2
    + (v_{{\\text{{pred}}}} - v_{{\\text{{exact}}}})^2}}\\]
  </div>
  <p>The magnitude of the complex error at each grid point. Shown as heatmaps in
  the error gallery plots. Reveals where the network struggles &mdash; typically near
  the scatterer surface and in the shadow region where field gradients are steepest.</p>

  <h3>L2 Relative Error</h3>
  <div class="eq-block">
    \\[\\epsilon_{{L2}} = \\frac{{\\| \\phi_s^{{\\text{{PINN}}}}
    - \\phi_s^{{\\text{{exact}}}} \\|_2}}{{\\| \\phi_s^{{\\text{{exact}}}} \\|_2}}
    = \\frac{{\\sqrt{{\\frac{{1}}{{N}} \\sum_i |\\phi_s^{{\\text{{PINN}}}}(\\mathbf{{x}}_i)
    - \\phi_s^{{\\text{{exact}}}}(\\mathbf{{x}}_i)|^2}}}}
    {{\\sqrt{{\\frac{{1}}{{N}} \\sum_i |\\phi_s^{{\\text{{exact}}}}(\\mathbf{{x}}_i)|^2}}}}\\]
  </div>
  <p>The primary accuracy metric, reported as a percentage. Measures overall
  solution quality normalized by the magnitude of the true field, so it is
  comparable across different ka values where field amplitudes vary.</p>

  <h3>Max Pointwise Error</h3>
  <div class="eq-block">
    \\[\\epsilon_{{\\max}} = \\max_i \\; e(\\mathbf{{x}}_i)\\]
  </div>
  <p>The worst-case error anywhere in the domain. Sensitive to localized failures
  &mdash; a network can have low L2 error but high max error if it fails in a small
  region (e.g., near the shadow boundary or Poisson bright spot).</p>

  <h3>Mean Pointwise Error</h3>
  <div class="eq-block">
    \\[\\bar{{e}} = \\frac{{1}}{{N}} \\sum_i e(\\mathbf{{x}}_i)\\]
  </div>
  <p>Average absolute error across all grid points. Less sensitive to outliers than
  max error but less normalized than L2 relative error. Useful for judging typical
  prediction quality at an arbitrary point in the domain.</p>
</div>"""

    body = "\n".join([t_takeaways, t_results, t_fields, t_ablation, t_hc, t_method, t_bg])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>HelmholtzPINN &mdash; Results Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Space+Grotesk:wght@500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="{PLOTLY_CDN}"></script>
<script>
MathJax = {{
  tex: {{inlineMath: [['\\\\(','\\\\)']],displayMath: [['\\\\[','\\\\]']]}},
  chtml: {{scale: 1.05}}
}};
</script>
<script src="{MATHJAX_CDN}" async></script>
<style>{DASH_CSS}</style>
</head>
<body>
<nav class="tab-bar">
  <div class="tab-bar-inner">
    <span class="brand">HelmholtzPINN</span>
    {tab_btns}
    <a href="slides.html" class="slides-link">SLIDES &rarr;</a>
  </div>
</nav>
{body}
<script>{DASH_JS}</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════
#  File Operations + Main
# ════════════════════════════════════════════════════════════════

def copy_plots():
    """Copy visualization HTML files from outputs/ to docs/plots/."""
    os.makedirs(PLOTS, exist_ok=True)

    # Production eval directories
    for d in ["prod_eval_ka0.50", "prod_eval_ka1.00",
              "prod_eval_ka3.14", "prod_eval_ka6.28"]:
        src = os.path.join(OUTPUTS, d)
        dst = os.path.join(PLOTS, d)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {src} -> {dst}")

    # Honeycomb summary
    src_hc = os.path.join(OUTPUTS, "hc_summary")
    dst_hc = os.path.join(PLOTS, "hc_summary")
    if os.path.isdir(src_hc):
        if os.path.exists(dst_hc):
            shutil.rmtree(dst_hc)
        shutil.copytree(src_hc, dst_hc)
        print(f"  Copied {src_hc} -> {dst_hc}")


def main():
    os.makedirs(DOCS, exist_ok=True)

    print("Collecting wandb data...")
    prod, abl = fetch_wandb()

    print("Copying visualization files...")
    copy_plots()

    print("Building slide deck...")
    slides_html = build_slides(prod, abl)
    slides_path = os.path.join(DOCS, "slides.html")
    with open(slides_path, "w") as f:
        f.write(slides_html)
    sz = os.path.getsize(slides_path)
    print(f"  {slides_path} ({sz / 1024:.0f} KB)")

    print("Building dashboard...")
    dash_html = build_dashboard(prod, abl)
    dash_path = os.path.join(DOCS, "index.html")
    with open(dash_path, "w") as f:
        f.write(dash_html)
    sz = os.path.getsize(dash_path)
    print(f"  {dash_path} ({sz / 1024:.0f} KB)")

    # Total size
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(DOCS)
        for f in fns
    )
    print(f"\nDone! Total docs/ size: {total / 1024 / 1024:.1f} MB")
    print(f"  Slides: {slides_path}")
    print(f"  Dashboard: {dash_path}")


if __name__ == "__main__":
    main()
