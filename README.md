# HelmholtzPINN

Physics-informed neural network for 2D acoustic scattering off a rigid circular cylinder. Trained and validated across wavenumbers ka = 0.5 to 2π, achieving 2–8% L2 error against the exact Bessel/Hankel series solution.

**[Live Dashboard](https://carlm451.github.io/helmholtz/)** · **[Slides](https://carlm451.github.io/helmholtz/slides.html)** · **[PDF Report](https://carlm451.github.io/helmholtz/HelmholtzPINN_Slides.pdf)**

## Key Results

| ka | L2 Relative Error | Mean Error | Max Error | Network | Training | Wall Time |
|----|-------------------|------------|-----------|---------|----------|-----------|
| 0.5 | 8.23% | 1.90% | 4.04% | 256 / 4L / 64ff | 10K+200 | 12 min |
| 1.0 | 3.57% | 1.34% | 2.99% | 256 / 4L / 64ff | 10K+200 | 13 min |
| π | 2.41% | 1.09% | 2.96% | 256 / 4L / 64ff | 10K+200 | 17 min |
| 2π | **2.00%** | **0.93%** | 4.37% | 384 / 6L / 96ff | 50K+300 | 259 min |

All runs: circle boundary, BGT2 ABC, L = 3.0, trained on 4× NVIDIA RTX 4000 Ada.

## PINNs as a Multi-Scale Probe

This project tests whether a PINN can **dynamically resolve physical fields at arbitrary length scales** — effectively zooming in to reveal finer spatial structure.

**Where PINNs excel:** For low-to-moderate spatial frequency (ka ≤ π), the PINN converges to 2–3% L2 error in 12–17 minutes. At these scales, PINNs offer genuine advantages over mesh-based solvers: continuous field access at any coordinate, no mesh generation, and physics enforced by construction. The network is a differentiable, resolution-independent surrogate — it can be queried anywhere without interpolation or remeshing.

**The compute wall:** As ka grows, training cost scales steeply: ka=2π needed 5× more epochs and a 50% wider network (259 min vs 12 min). At ka=3π, a 10+ hour run plateaued at 68% L2 despite loss dropping 5 orders of magnitude. Zooming in to finer features is equivalent to probing higher spatial frequencies, and the network faces a fundamental resolution–compute tradeoff. At the frontier, **compute budget — not architecture or loss design — is the binding constraint** (the convergent 2π run used default loss weights but 2.5× more epochs: 49%→2%).

## Background

When a plane wave hits a solid cylinder, it produces a scattered wave field governed by the **Helmholtz equation**:

$$\nabla^2 \phi_s + k^2 \phi_s = 0$$

The key parameter **ka** (wavenumber × radius) controls the scattering regime. The analytic Bessel/Hankel series solution provides pixel-level validation.

**Boundary conditions:**
- **Neumann BC** on the cylinder: $\partial\phi_{total}/\partial n = 0$
- **BGT2 ABC** on the circular outer boundary: $\partial\phi_s/\partial r - ik\phi_s + \phi_s/2r = 0$

## Architecture

- **Random Fourier features** with σ = k to overcome spectral bias
- **Residual MLP** with skip connections (configurable depth/width)
- **Scattered-field formulation** — network learns $\phi_s$ only
- **Two-phase optimization** — Adam (cosine LR) → L-BFGS (strong Wolfe)

The network maps $(x, y) \to (u, v)$ where $u = \text{Re}(\phi_s)$ and $v = \text{Im}(\phi_s)$.

## Stack

| Component | Library |
|-----------|---------|
| Neural network | PyTorch |
| Analytic solution | scipy (Bessel/Hankel functions) |
| Visualization | Plotly (interactive HTML) |
| Experiment tracking | Weights & Biases |
| Dashboard/slides | Custom HTML generation |

## Repo Structure

```
main.py                             # CLI entry point
helmholtz/                          # Core library package
  config.py                         # HelmholtzConfig dataclass
  network.py                        # FourierFeatureLayer + HelmholtzPINN
  domain.py                         # ScatteringDomain (sampling)
  losses.py                         # PDE, Neumann BC, ABC loss functions
  train.py                          # Adam + L-BFGS training loop
  evaluate.py                       # Evaluation metrics vs analytic
  analytic.py                       # Bessel/Hankel series solution
  visualize.py                      # Plotly comparison plots + zoom reports
scripts/                            # Standalone analysis tools
  build_report.py                   # Dashboard + slides HTML generator
  slides_to_pdf.py                  # Playwright-based PDF export
  eval_suite.py                     # Post-training evaluation suite
  plot_training.py                  # Training curves from wandb
  run_ablation.sh                   # Ablation study launcher
docs/                               # Generated dashboard + slides
```

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install torch scipy plotly wandb tqdm

# Train with default settings (ka=pi, circle+BGT2)
python main.py

# Train with specific ka
python main.py --ka 6.28 --outer-boundary circle --abc-order 2

# Eval-only from checkpoint
python main.py --eval-only checkpoints/helmholtz_ka3.14_lbfgs.pt

# Build dashboard + slides
source .env && python scripts/build_report.py

# Export slides to PDF
python scripts/slides_to_pdf.py
```

## Extension: Honeycomb Acoustic Shield

A 19-circle hexagonal lattice scatterer at ka = 2, trained as a pure PINN with no analytic reference. Achieves total loss 9.06e-6 and demonstrates acoustic shielding: |φ_total| < 0.003 inside the cluster.

## References

- Raissi, Perdikaris, Karniadakis. "Physics-informed neural networks." *J. Comput. Phys.* 378, 686–707 (2019).
- Tancik et al. "Fourier features let networks learn high frequency functions in low dimensional domains." *NeurIPS* (2020).
- Bayliss, Gunzburger, Turkel. "Boundary conditions for the numerical solution of elliptic equations in exterior regions." *SIAM J. Appl. Math.* 42(2), 430–451 (1982).
- Wang, Yu, Perdikaris. "When and why PINNs fail to train: A neural tangent kernel perspective." *J. Comput. Phys.* 449, 110768 (2022).
- Morse, Ingard. *Theoretical Acoustics.* Princeton University Press (1968). Ch. 8.
