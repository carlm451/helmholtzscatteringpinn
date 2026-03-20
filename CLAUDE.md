# HelmholtzPINN -- Agent Reference

PINN solver for 2D acoustic scattering of a plane wave off a sound-hard circular cylinder. The network learns the **scattered field** $\phi_s$ (not the total field) by minimizing PDE, boundary condition, and absorbing boundary condition residuals simultaneously. The analytic Bessel/Hankel series solution is available for validation.

## Physics

**Governing equation:** $\nabla^2 \phi_s + k^2 \phi_s = 0$ (Helmholtz, scattered field only)

**Scattered-field formulation:** Network outputs $(u, v) = (\text{Re}(\phi_s), \text{Im}(\phi_s))$. Total field = incident + scattered, where $\phi_{inc} = e^{ikx}$ (plane wave along x-axis).

**Boundary conditions:**
- **Neumann BC** on cylinder surface ($r = a$): $\partial\phi_s/\partial n = -\partial\phi_{inc}/\partial n$
- **First-order ABC** on outer boundary: $\partial\phi_s/\partial n - ik\phi_s = 0$
- **BGT2 (second-order ABC)** on circular outer boundary: $\partial\phi_s/\partial r - ik\phi_s + \phi_s/(2r) = 0$

**Key parameter:** `ka` = wavenumber * cylinder radius. Higher ka = more oscillatory field = harder problem.

## Architecture

`FourierFeatureLayer` -> `ResidualBlock` MLP -> linear head -> `(u, v)`

- Fourier features with `sigma ~ k` to resolve wavelength-scale structure
- Residual blocks (pairs of layers with skip connections)
- Xavier init on all linear layers

## Key Config Knobs (`HelmholtzConfig`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ka` | pi | Wavenumber * radius (controls difficulty) |
| `a` | 1.0 | Cylinder radius |
| `L` | 3.0 | Domain half-size (larger L = weaker ABC error) |
| `outer_boundary` | "square" | Outer boundary shape: "square" or "circle" |
| `abc_order` | 1 | ABC order: 1 (first-order) or 2 (BGT2, requires circle) |
| `n_fourier_features` | 64 | Random Fourier feature count |
| `fourier_sigma` | k | Fourier feature bandwidth (auto-set to k) |
| `n_hidden_layers` | 4 | MLP depth (used as n_layers//2 residual blocks) |
| `n_hidden_neurons` | 256 | MLP width |
| `sampling_strategy` | "uniform" | "uniform", "radial_bias", or "rad" |
| `use_rad` | False | Enable RAD adaptive resampling |
| `lambda_pde/bc/abc` | 1.0/10.0/1.0 | Loss weights |

Factory methods: `HelmholtzConfig.ka_pi()`, `.ka_2pi()`, `.ka_3pi()` with tuned defaults.

## Repo Layout

```
main.py                  # CLI entry point (argparse -> HelmholtzConfig -> train/eval)
helmholtz/               # Core package
  config.py              # HelmholtzConfig dataclass
  network.py             # HelmholtzPINN model
  domain.py              # ScatteringDomain (point sampling)
  losses.py              # PDE, BC, ABC losses (all use autograd)
  train.py               # Adam + L-BFGS training loop + wandb
  evaluate.py            # L2/max error vs analytic
  analytic.py            # Bessel/Hankel series (numpy/scipy)
  visualize.py           # Plotly plots + zoom report
scripts/                 # Standalone tools
  eval_suite.py          # 9-step post-training eval with diagnostics
  plot_training.py       # Training curves from wandb history
  visualize_analytic_sweep.py  # Multi-ka analytic fields
```

## How to Run

```bash
# Always use the project venv
.venv/bin/python main.py [options]

# Source .env before any wandb commands (contains WANDB_API_KEY)
source .env

# Train (default ka=pi)
.venv/bin/python main.py

# Eval-only
.venv/bin/python main.py --eval-only checkpoints/helmholtz_ka3.14_lbfgs.pt

# Analytic-only (no network)
.venv/bin/python main.py --analytic-only

# Post-training eval suite
.venv/bin/python scripts/eval_suite.py checkpoints/helmholtz_ka3.14_lbfgs.pt

# Training curve plots
source .env && .venv/bin/python scripts/plot_training.py --run-path entity/project/run_id
```

## Current Best Config

Circle boundary + BGT2 ABC (`--outer-boundary circle --abc-order 2`) significantly outperforms square boundary + first-order ABC. The circular boundary matches the cylindrical symmetry and BGT2 adds a curvature correction term.

## Known Gotchas

- **autograd + no_grad conflict:** `evaluate.py` uses `@torch.no_grad()` but the model forward pass through Fourier features is fine. However, `check_bc_satisfaction` in `eval_suite.py` needs gradients and calls `model.train()` explicitly.
- **Circular domain masking:** When `outer_boundary="circle"`, evaluation must also mask points outside the circle (not just inside the scatterer). This is handled in `evaluate.py` line 22-24.
- **Fourier feature bandwidth:** `fourier_sigma` defaults to `k` in `__post_init__`. If you set `ka` after construction, `fourier_sigma` won't update. Use the constructor or factory methods.
- **L-BFGS on MPS:** Works but slower per-iteration than Adam. The strong_wolfe line search is the bottleneck.
- **wandb API key:** Must `source .env` before running `plot_training.py` or any script that calls the wandb API.
