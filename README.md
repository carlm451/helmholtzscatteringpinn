# HelmholtzPINN

Physics-informed neural network for 2D acoustic scattering off a rigid circular cylinder.

## Background

When a plane wave hits a solid cylinder, it produces a scattered wave field governed by the **Helmholtz equation**:

$$\nabla^2 \phi_s + k^2 \phi_s = 0$$

where $\phi_s$ is the scattered pressure field and $k = \omega/c$ is the wavenumber. The key dimensionless parameter is **ka** (wavenumber times cylinder radius) which controls the scattering regime: Rayleigh ($ka \ll 1$), resonance ($ka \sim 1$), and geometric optics ($ka \gg 1$).

**Boundary conditions:**
- **Sound-hard (Neumann) BC** on the cylinder surface: $\partial\phi_{total}/\partial n = 0$
- **Absorbing BC (ABC)** on the outer truncation boundary to approximate the Sommerfeld radiation condition

The analytic solution exists as a Bessel/Hankel series expansion, making this an ideal benchmark for validating numerical methods.

## Why a PINN?

A Physics-Informed Neural Network learns the scattered field by minimizing PDE residuals, boundary condition violations, and ABC residuals simultaneously. Advantages over traditional mesh-based methods:

- **Continuous field representation** -- query the solution at any point without interpolation
- **Zoom in** to fine features (shadow boundary, Poisson bright spot) without remeshing
- **Meshfree** -- no mesh generation or refinement needed

The network maps $(x, y) \to (u, v)$ where $u = \text{Re}(\phi_s)$ and $v = \text{Im}(\phi_s)$.

## Architecture

- **Random Fourier features** to overcome spectral bias at high wavenumbers
- **Residual MLP** with configurable depth/width
- **Scattered-field formulation** (learns $\phi_s$ only; total field = incident + scattered)
- **BGT2 absorbing boundary condition** on a circular truncation boundary (preferred) or first-order ABC on a square boundary

## Stack

| Component | Library |
|-----------|---------|
| Neural network | PyTorch |
| Analytic solution | scipy (Bessel/Hankel functions) |
| Visualization | Plotly (interactive HTML) |
| Experiment tracking | Weights & Biases |

## Repo Structure

```
main.py                             # CLI entry point
helmholtz/                          # Core library package
  __init__.py
  config.py                         # HelmholtzConfig dataclass
  network.py                        # FourierFeatureLayer + HelmholtzPINN
  domain.py                         # ScatteringDomain (sampling)
  losses.py                         # PDE, Neumann BC, ABC loss functions
  train.py                          # Adam + L-BFGS training loop
  evaluate.py                       # Evaluation metrics vs analytic
  analytic.py                       # Bessel/Hankel series solution
  visualize.py                      # Plotly comparison plots + zoom reports
scripts/                            # Standalone analysis tools
  eval_suite.py                     # Comprehensive post-training evaluation
  plot_training.py                  # Training curve plots from wandb
  visualize_analytic_sweep.py       # Multi-ka analytic field visualization
docs/                               # Working notes (gitignored)
run_sweep.sh                        # Hyperparameter sweep launcher
```

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install torch scipy plotly wandb tqdm

# Train with default settings (ka=pi, circle+BGT2)
python main.py

# Train with specific ka and options
python main.py --ka 6.28 --outer-boundary circle --abc-order 2 --no-wandb

# Eval-only from checkpoint
python main.py --eval-only checkpoints/helmholtz_ka3.14_lbfgs.pt

# Generate analytic solution plots only
python main.py --analytic-only --ka 3.14159

# Post-training evaluation suite
python scripts/eval_suite.py checkpoints/helmholtz_ka3.14_lbfgs.pt --run-id myrun

# Plot training curves from wandb
source .env && python scripts/plot_training.py --run-path entity/helmholtz-pinn/run_id

# Multi-ka analytic sweep visualization
python scripts/visualize_analytic_sweep.py
```

## Key Results

At $ka = \pi$ with circle boundary + BGT2 ABC:
- L2 relative error vs analytic: ~$10^{-2}$ range
- Captures shadow boundary, Poisson bright spot, and wake fringes
- Training: ~10k Adam epochs + ~200 L-BFGS epochs on Apple MPS

## References

- Raissi, Perdikaris, Karniadakis. "Physics-informed neural networks." *Journal of Computational Physics*, 2019.
- Mei, C.C. "Mathematical Analysis in Engineering." Cambridge University Press. (Helmholtz scattering derivation)
- Bayliss, Gunzburger, Turkel. "Boundary conditions for the numerical solution of elliptic equations in exterior regions." *SIAM J. Appl. Math.*, 1982. (BGT absorbing boundary conditions)
