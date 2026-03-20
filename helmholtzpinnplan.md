# HELMHOLTZPINN — Project Specification

## 1. Overview & Motivation

This project implements a Physics-Informed Neural Network (PINN) to solve the 2D Helmholtz
scattering problem: a plane wave scattering off a rigid circular cylinder. The goal is to
demonstrate that a trained PINN learns a **continuous field representation** that can be
queried at arbitrary spatial resolution — enabling "zoom-in" to sub-regions of the domain
without re-meshing or re-solving, unlike traditional FEM approaches.

This is a take-home assessment. The evaluation rubric emphasizes:
- Domain expertise in physics / ML
- Resourcefulness in using pre-existing tools
- Creativity in framing the problem
- Attention to detail in analyzing results and experimentation
- Clarity in communicating results

The assignment prompt asks us to "train a PINN model to predict physical fields at arbitrary
length scales" and show that "given a fixed scene represented by a set of triangle meshes,
the model should accurately guess how a field changes as you zoom in to different areas."
We confirmed with the evaluator that **interpretation (A)** is correct: train a single PINN
over the full domain, then demonstrate zoom-in by querying at high resolution in sub-regions.

## 2. Physics Background

### 2.1 Governing Equation

The Helmholtz equation governs time-harmonic wave propagation:

    ∇²φ + k²φ = 0

where k = ω/c is the wavenumber, ω is angular frequency, c is wave speed.

### 2.2 Scattering Problem Setup

- **Incident wave**: A plane wave traveling in the +x direction:
  φ_inc(x,y) = exp(ikx)

- **Scatterer**: A rigid (sound-hard) circular cylinder of radius a centered at the origin.

- **Total field decomposition**:
  φ_total = φ_inc + φ_s
  where φ_s is the unknown scattered field.

- **The PINN solves for φ_s only.** The incident field is known analytically.

### 2.3 Boundary Conditions

**Sound-hard (Neumann) BC on scatterer surface** (r = a):
    ∂φ_total/∂n = 0  on r = a
    ⟹  ∂φ_s/∂n = −∂φ_inc/∂n  on r = a

In Cartesian coordinates, with the scatterer centered at (x₀, y₀):
    ∂φ/∂r = (x−x₀)/r · ∂φ/∂x + (y−y₀)/r · ∂φ/∂y

For φ_inc = exp(ikx):
    ∂φ_inc/∂r = (x−x₀)/r · ik · exp(ikx)

So the BC in Cartesian (at r = a) is:
    (x−x₀)/a · ∂φ_s/∂x + (y−y₀)/a · ∂φ_s/∂y = −(x−x₀)/a · ik · exp(ikx)

**Absorbing boundary condition (first-order Sommerfeld ABC) on outer box**:
    ∂φ_s/∂n − ik·φ_s = 0  on Γ_out

This approximates the Sommerfeld radiation condition for the truncated domain.
On each face of the box [-L,L]²:
- x = +L: ∂φ_s/∂x − ik·φ_s = 0
- x = −L: −∂φ_s/∂x − ik·φ_s = 0
- y = +L: ∂φ_s/∂y − ik·φ_s = 0
- y = −L: −∂φ_s/∂y − ik·φ_s = 0

### 2.4 Complex Field Handling

φ_s is complex-valued. We represent it as two real-valued outputs:
    φ_s(x,y) = u(x,y) + i·v(x,y)

The network maps (x,y) → (u, v).

All PDE and BC loss terms split into real and imaginary parts.

**PDE residual splits as:**
- Real: ∂²u/∂x² + ∂²u/∂y² + k²u = 0
- Imag: ∂²v/∂x² + ∂²v/∂y² + k²v = 0

**Neumann BC splits as (at r = a):**
- Real: n_x·∂u/∂x + n_y·∂u/∂y = −n_x·k·sin(kx)      [where n = (x/a, y/a)]
- Imag: n_x·∂v/∂x + n_y·∂v/∂y = +n_x·k·cos(kx)       [note: −ik·exp(ikx) = k·sin(kx) − ik·cos(kx)]

Wait — let's be precise. −ik·exp(ikx) = −ik·(cos(kx) + i·sin(kx)) = k·sin(kx) − ik·cos(kx).
So the RHS real part is k·sin(kx)·(x/a) and the RHS imaginary part is −k·cos(kx)·(x/a).

CORRECTION for general center (x₀,y₀): replace x with (x−x₀), and the incident field
evaluation point stays at x (not x−x₀) since the incident wave is global.

**ABC splits as:**
- Real: ∂u/∂n + k·v = 0
- Imag: ∂v/∂n − k·u = 0

### 2.5 Non-dimensionalization

It is convenient to non-dimensionalize with the cylinder radius a:
- Spatial coordinates: x̃ = x/a, ỹ = y/a
- Effective parameter: k̃ = ka (this is the key difficulty parameter)
- Domain becomes [-L/a, L/a]² \ B(0,1)

The quantity **ka** (often written kR in some texts) controls the problem difficulty:
- ka ~ 1: resonance regime, few angular modes, tractable
- ka ~ 5: distinct shadow/lit regions, several diffraction lobes
- ka > 10: many oscillations, very hard for PINNs

**Target range: ka = π to 3π** (1 to 3 wavelengths across diameter).

## 3. Analytic Reference Solution

### 3.1 Partial Wave Expansion

For a sound-hard cylinder of radius a at the origin with incident plane wave exp(ikx),
the scattered field in polar coordinates (r, θ) is:

    φ_s(r,θ) = − Σ_{n=−∞}^{∞} (i)^n · [J_n'(ka) / H_n^{(1)}'(ka)] · H_n^{(1)}(kr) · exp(inθ)

where:
- J_n is the Bessel function of the first kind, order n
- H_n^{(1)} is the Hankel function of the first kind, order n
- Prime (') denotes derivative with respect to the argument
- The derivative identity: J_n'(z) = J_{n−1}(z) − n/z · J_n(z)
  and similarly: H_n^{(1)}'(z) = H_{n−1}^{(1)}(z) − n/z · H_n^{(1)}(z)

**For sound-soft (Dirichlet) BC** (φ_total = 0 on r = a), the coefficient would instead be:
    − (i)^n · [J_n(ka) / H_n^{(1)}(ka)]

We implement **sound-hard** to follow the Mei notes and the DeepXDE reference.

### 3.2 Series Truncation

The number of terms needed for convergence scales as:
    N_terms ≈ 30 + (ka)^1.01  (heuristic from DeepXDE)

or more conservatively, sum from n = −N to N where N ≈ ka + 10.

### 3.3 Implementation Notes

The analytic solution should be implemented in a standalone module that:
1. Takes (x, y) points in Cartesian coordinates (arrays)
2. Converts to (r, θ) internally
3. Computes the series sum with sufficient terms
4. Returns complex φ_s, and optionally φ_total = exp(ikx) + φ_s
5. Can also return ∂φ_s/∂r for BC validation
6. Is vectorized (numpy/scipy, using scipy.special.jv and scipy.special.hankel1)

**Key reference implementation**: The DeepXDE example at
https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.sound.hard.abc.html
contains a working `sound_hard_circle_deepxde()` function. Use this as a starting point
but verify independently against the Mei notes formulas.

**Validation**: For ka → 0, the scattered field should approach the dipole pattern.
Plot the analytic solution at several ka values to sanity-check before training any PINN.

## 4. Network Architecture

### 4.1 Inputs and Outputs

- **Input**: (x, y) ∈ ℝ² — Cartesian coordinates (NOT polar)
  - This is deliberate: Cartesian inputs generalize trivially to multiple scatterers
    at arbitrary positions without any coordinate transform
- **Output**: (u, v) ∈ ℝ² — real and imaginary parts of φ_s

### 4.2 Fourier Feature Encoding

Vanilla MLPs suffer from **spectral bias** — they learn low-frequency components first
and struggle to represent the oscillatory Helmholtz solution. This is the central technical
challenge for this problem.

**Solution**: Fourier feature mapping on the input coordinates (Tancik et al., 2020).

    γ(x,y) = [sin(Bx), cos(Bx), sin(By), cos(By)]

where B is a set of frequency vectors. Options:
- **Fixed frequencies**: B sampled from N(0, σ²) where σ should be tuned to the
  wavenumber k. A reasonable starting point is σ ≈ k.
- **Learnable frequencies**: Let B be trainable parameters.
- **Deterministic grid**: Use integer multiples of k up to some maximum.

The Fourier features map (x,y) ∈ ℝ² to a higher-dimensional space ℝ^{2·n_features}
before feeding into the MLP.

### 4.3 MLP Architecture

    Input: Fourier features of (x,y) → dim = 2 * n_fourier_features * 2
    Hidden: 3–4 layers, 256–350 nodes each
    Activation: tanh (smooth, supports higher-order derivatives needed for Laplacian)
    Output: 2 nodes (u, v)
    Initialization: Glorot uniform

### 4.4 Design for Multiple Scatterers

The network architecture is **agnostic to the number of scatterers**. To add more circles:
1. Add their centers and radii to a configuration list
2. Modify the domain sampling to exclude all circles
3. Add boundary points on each new circle with the appropriate normal direction
4. The Neumann BC generalizes: n = (x−x₀, y−y₀)/r at each circle centered at (x₀,y₀)

The network itself does not change — it still maps (x,y) → (u,v).

## 5. Loss Function

### 5.1 Components

**Total loss = λ_PDE · L_PDE + λ_BC · L_BC + λ_ABC · L_ABC**

**L_PDE** — Helmholtz residual at interior collocation points:
    L_PDE = (1/N_int) Σ [(∂²u/∂x² + ∂²u/∂y² + k²u)² + (∂²v/∂x² + ∂²v/∂y² + k²v)²]

**L_BC** — Neumann BC on scatterer surface(s):
    For each scatterer centered at (x₀, y₀) with radius a:
    n_x = (x−x₀)/a,  n_y = (y−y₀)/a
    RHS_real = −n_x · k · sin(kx)     [from −Re(ik·exp(ikx))·n_x, note n_x only for x-component]

    ACTUALLY, let's be very careful:
    ∂φ_inc/∂n = n⃗ · ∇φ_inc = n_x · ik·exp(ikx) + n_y · 0
                = n_x · ik · (cos(kx) + i·sin(kx))
    So:
    Re(∂φ_inc/∂n) = −n_x · k · sin(kx)
    Im(∂φ_inc/∂n) =  n_x · k · cos(kx)

    The BC is ∂φ_s/∂n = −∂φ_inc/∂n, so:
    target_real = n_x · k · sin(kx)
    target_imag = −n_x · k · cos(kx)

    L_BC = (1/N_a) Σ [(n_x·∂u/∂x + n_y·∂u/∂y − target_real)²
                     + (n_x·∂v/∂x + n_y·∂v/∂y − target_imag)²]

**L_ABC** — Absorbing BC on outer box edges:
    On each face, with outward normal n⃗:
    L_ABC = (1/N_out) Σ [(∂u/∂n + k·v)² + (∂v/∂n − k·u)²]

### 5.2 Loss Weights

Starting values:
- λ_PDE = 1.0
- λ_BC = 10.0  (fewer points, critical to enforce accurately)
- λ_ABC = 1.0

Consider adaptive loss balancing (Wang et al., 2021 — "Understanding and Mitigating Gradient
Flow Pathologies in PINNs") if training stalls. The basic idea: scale each loss component's
weight inversely proportional to its gradient magnitude to equalize gradient contributions.

### 5.3 Point Sampling Strategy

**Interior collocation points**: Uniformly random in [-L,L]², reject points inside any
scatterer circle. Density: ~20 points per wavelength in each direction as a baseline,
so N_int ≈ (2L / (λ/20))² = (2L·k·20/(2π))² for the whole domain. For L=20a and ka=π,
this gives roughly 6400 points. Scale up as needed.

**Scatterer boundary points**: Uniformly spaced in angle θ ∈ [0, 2π) on each circle.
~8× the linear density used for interior points, or at minimum ~10 points per wavelength
along the circumference. For ka=π, circumference = 2πa ≈ 2λ, so ~40 points minimum.

**Outer boundary points**: Uniformly distributed along the 4 edges of the box.
Similar density to scatterer boundary: ~8× linear interior density.

**Resampling**: Consider resampling collocation points every N_resample iterations
(e.g., every 1000 steps) to improve coverage. Alternatively, use fixed points for
reproducibility during initial development, then add resampling as an enhancement.

## 6. Training Procedure

### 6.1 Optimizer

- **Phase 1**: Adam optimizer, lr = 1e-3, for ~10,000–20,000 iterations
  (fast initial convergence)
- **Phase 2**: L-BFGS optimizer for refinement (quasi-Newton, better for PINN fine-tuning)
  Many PINN implementations use this two-phase approach.

### 6.2 Learning Rate Schedule

For Adam phase: consider ReduceLROnPlateau or cosine annealing.
For L-BFGS: typically no schedule needed (it handles step sizes internally).

### 6.3 Monitoring

Track and log separately:
- L_PDE, L_BC, L_ABC (individual loss components)
- Total loss
- L2 error vs analytic solution (evaluated on a fixed test grid every N iterations)
- Max pointwise error vs analytic

### 6.4 Convergence Targets

- Relative L2 error < 1% against analytic solution on a dense test grid
- Pointwise error < 5% in most of the domain
- Higher errors are acceptable very near the scatterer surface and near box corners
  (where the first-order ABC is least accurate)

## 7. Implementation Plan

### Phase A: Analytic Solution Module (do this FIRST)

File: `analytic.py`

1. Implement `scattered_field(x, y, k, a, n_terms=None)` returning complex φ_s
2. Implement `total_field(x, y, k, a, n_terms=None)` returning φ_total
3. Implement `incident_field(x, y, k)` returning exp(ikx)
4. Implement helper `scattered_field_polar(r, theta, k, a, n_terms)` for internal use
5. Validate:
   - Plot |φ_total| for ka = π, 2π, 3π — should show clear scattering pattern
   - Check that φ_total satisfies Neumann BC: evaluate ∂φ_total/∂r at r=a numerically
     and verify it's ≈ 0
   - Check far-field: φ_s should decay as 1/√r

### Phase B: Domain and Sampling Module

File: `domain.py`

1. Define `ScatteringDomain` class:
   - Stores list of scatterer circles: [(center_x, center_y, radius), ...]
   - Stores box bounds: L
   - Methods:
     - `sample_interior(N)` → (x, y) tensor, rejection sampling
     - `sample_scatterer_boundary(N_per_circle)` → (x, y, n_x, n_y, circle_idx) tensors
     - `sample_outer_boundary(N)` → (x, y, n_x, n_y) tensors
     - `is_inside_scatterer(x, y)` → boolean mask

### Phase C: Network Module

File: `network.py`

1. `FourierFeatureLayer`: maps (x,y) → Fourier features
2. `HelmholtzPINN(nn.Module)`: Fourier features + MLP → (u, v)
3. Keep architecture configurable: n_layers, n_nodes, n_fourier_features, sigma

### Phase D: Loss and Training Module

File: `losses.py` and `train.py`

1. `helmholtz_residual(model, x, y, k)` → PDE loss
2. `neumann_bc_loss(model, x_bc, y_bc, nx, ny, k)` → BC loss on scatterer(s)
3. `abc_loss(model, x_abc, y_abc, nx_abc, ny_abc, k)` → ABC loss on outer box
4. Training loop with Adam + optional L-BFGS
5. Logging of all loss components + analytic error

### Phase E: Evaluation & Visualization

File: `evaluate.py` and `visualize.py`

1. Dense grid evaluation: compare PINN vs analytic on fine meshgrid
2. Error heatmaps
3. Zoom-in visualization (see Section 8)
4. Sweep over k values

## 8. Zoom-In Demonstration Plan

This is the core deliverable for the assignment. The zoom-in demo should show:

### 8.1 Global View (Level 0)

- Full domain [-L, L]² at moderate resolution (e.g., 200×200 grid)
- Plot Re(φ_total), |φ_total|, and pointwise error |φ_s^PINN − φ_s^analytic|
- This establishes the global scattering pattern

### 8.2 Zoom Level 1: Shadow Boundary Region

- **Where**: Behind the cylinder, around θ ≈ ±π/2 at r ≈ 2a to 5a
  Specifically: a rectangular patch roughly [a, 5a] × [−3a, 3a]
- **What to show**: The transition from the illuminated region to the geometric shadow.
  This transition is smooth (governed by diffraction), with spatial scale ~√(λR).
  On the global plot this looks like a blurry transition; zoomed in, you should see
  the structured Fresnel diffraction pattern.
- **Resolution**: 200×200 on this sub-region (much finer effective resolution than global)
- **Comparison**: PINN prediction vs analytic, side by side, plus error

### 8.3 Zoom Level 2: Near-Surface (Lit Side)

- **Where**: A patch on the front face of the cylinder, extending ~0.5λ off the surface
  Specifically: x ∈ [−a−λ/2, −a+0.1a], y ∈ [−a/2, a/2] (or use polar: r ∈ [a, a+λ/2])
- **What to show**: The total field should be exactly zero normal derivative at the surface
  and oscillatory just off the surface. This tests BC enforcement quality at fine scale.
- **Why interesting**: This is where the incident and scattered fields interact most strongly.

### 8.4 Zoom Level 3: Poisson/Arago Bright Spot (Shadow Axis)

- **Where**: Directly behind the cylinder on the shadow axis, y ≈ 0, x ∈ [a, 10a]
- **What to show**: 1D line plot of |φ_total| along the shadow axis.
  Diffraction creates a local intensity maximum right behind the cylinder (the Poisson
  bright spot). This is a subtle diffraction feature invisible on the global plot.
- **Why interesting**: A pure curve-fit or low-frequency approximation would miss this.
  If the PINN gets it right, it demonstrates genuine learning of wave physics.

### 8.5 Zoom Level 4: Wake Interference Fringes

- **Where**: A few radii behind the cylinder, x ∈ [3a, 8a], y ∈ [−4a, 4a]
- **What to show**: Diffracted waves from both sides of the cylinder interfere in the wake,
  creating a fringe pattern. Zoom in to resolve individual fringes.
- **Comparison**: Check fringe spacing against analytic — wrong spacing would indicate
  the PINN learned the wrong effective wavenumber.

### 8.6 Presentation Format

For each zoom level, produce a panel with:
1. PINN prediction (color plot of Re(φ_s) or |φ_total|)
2. Analytic reference (same color scale)
3. Absolute error heatmap
4. If applicable: 1D line cuts comparing PINN vs analytic

Also produce a "zoom overview" figure showing the full domain with rectangles indicating
where each zoom region is, with insets or arrows pointing to the zoomed panels.

### 8.7 The Punchline Narrative

Frame the zoom-in demo around this key point:

> "The PINN was trained once. Each zoom level is simply a forward pass at new (x,y)
> coordinates — no re-training, no re-meshing, no new solver run. The continuous
> learned representation gives us arbitrary-resolution access to the physics."

This directly answers the assignment prompt about "predicting fields at arbitrary length
scales" and "zooming in to different areas of the scene."

## 9. Multi-k Evaluation

### 9.1 Sweep Parameters

Train and evaluate at multiple ka values to show robustness:
- ka = π    (~1 wavelength across diameter — easiest)
- ka = 2π   (~2 wavelengths — moderate)
- ka = 3π   (~3 wavelengths — challenging)

For each, report:
- Relative L2 error on full domain
- Relative L2 error in each zoom sub-region
- Training time / iterations to convergence
- Visual comparison

### 9.2 Expected Behavior

Error should increase with ka due to:
- More oscillations to represent → harder for network
- Spectral bias more pronounced at higher frequency
- More collocation points needed

This is a legitimate finding to report — it shows understanding of PINN limitations.

### 9.3 Mitigation Strategies (if time permits)

- Increase Fourier feature bandwidth σ with k
- Increase network width/depth
- Adaptive collocation point resampling (concentrate near high-residual regions)
- Curriculum learning: train on low k first, then fine-tune at higher k

## 10. Extension: Two Scatterers (Stretch Goal)

If time allows after the single-cylinder results are solid:

1. Add a second circle at, say, (5a, 0) — spacing of a few wavelengths
2. No analytic solution available → use the PINN prediction alone, but validate
   that the PDE residual is small and the BCs are satisfied
3. The zoom-in target becomes the **gap between the two scatterers**, where
   multiple scattering creates interference fringes
4. Optionally: generate a FEM reference using FEniCS + Gmsh for quantitative comparison

This extension demonstrates the Cartesian-input design paying off — the network
architecture is unchanged, only the domain sampling and BC point generation change.

## 11. Key References

### Physics
- C.C. Mei, MIT lecture notes, "Diffraction by a circular cylinder, theory and simulation"
  https://web.mit.edu/fluids-modules/waves/www/material/chap-5.pdf
- Moiola, "Helmholtz equation and scattering" (Univ. Pavia lecture notes)
  https://mate.unipv.it/moiola/T/MNAPDE2022/MNAPDE2022.pdf
- KTH lecture notes, "Helmholtz Equation and High Frequency Approximations"
  https://www.csc.kth.se/utbildning/kth/kurser/DN2255/ndiff12/Lecture5.pdf

### PINNs
- Raissi, Perdikaris & Karniadakis (2019), "Physics-informed neural networks"
  https://arxiv.org/pdf/1711.10561
- Wang, Teng & Perdikaris (2021), "Understanding and Mitigating Gradient Flow
  Pathologies in Physics-Informed Neural Networks"
- Tancik et al. (2020), "Fourier Features Let Networks Learn High Frequency Functions"

### Code References
- DeepXDE Helmholtz sound-hard scattering example (complete working code):
  https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.sound.hard.abc.html

### Visualization
- Penn State acoustics demos (partial wave expansion, cylinder scattering animations):
  https://www.acs.psu.edu/drussell/Demos/PartialWaveExpansion/PlaneWaveExpansion.html
  https://www.acs.psu.edu/drussell/Demos/Scatter/Scatter.html

## 12. File Structure

```
helmholtz_pinn/
├── HELMHOLTZPINN.md      # This spec document
├── analytic.py           # Analytic Bessel/Hankel series solution
├── domain.py             # Domain geometry, point sampling
├── network.py            # Fourier features + MLP architecture
├── losses.py             # PDE, BC, ABC loss functions
├── train.py              # Training loop (Adam + L-BFGS)
├── evaluate.py           # Error metrics vs analytic
├── visualize.py          # Plotting: global, zoom panels, line cuts
├── config.py             # Hyperparameters, physical parameters
├── main.py               # Top-level entry point
└── notebooks/
    └── demo.ipynb        # Jupyter notebook for final presentation
```

## 13. Implementation Priority Order

1. **analytic.py** — Get the reference solution right first. Validate visually.
2. **domain.py** — Point sampling with scatterer exclusion.
3. **network.py** — Fourier features + MLP.
4. **losses.py** — All three loss terms with proper autodiff.
5. **train.py** — Basic Adam training loop, logging individual losses.
6. **evaluate.py** — L2 error vs analytic on test grid.
7. **Iterate on 3–6** — Tune hyperparameters, achieve < 1% L2 error for ka = π.
8. **visualize.py** — Global + zoom plots.
9. **Multi-k sweep** — Repeat for ka = 2π, 3π.
10. **Two-scatterer extension** (stretch goal).

## 14. Common Pitfalls to Watch For

- **Forgetting to exclude scatterer interior from collocation points**: Points inside
  the circle will have nonsense gradients. Always rejection-sample.
- **Wrong sign conventions on BCs**: The Neumann BC involves ∂φ_s/∂n = −∂φ_inc/∂n.
  Getting the sign wrong means the PINN converges to garbage. Validate by checking
  the analytic solution satisfies your BC expressions numerically.
- **Insufficient Fourier feature bandwidth**: If σ << k, the network can't represent
  the oscillations. If σ >> k, the network wastes capacity on frequencies not present.
  Start with σ ≈ k and tune.
- **Not enough points on scatterer boundary**: The BC is the hardest constraint.
  Under-sampling leads to the field "leaking" through the scatterer.
- **Using ReLU activation**: ReLU has discontinuous second derivative → Laplacian is
  zero almost everywhere. Use tanh or sin activations for Helmholtz.
- **First-order ABC reflection**: The first-order Sommerfeld ABC on the box is only
  approximate. Expect some spurious reflection from the outer boundary. This is a known
  limitation, not a bug. If it's problematic, increase L or use a second-order ABC.
- **Complex arithmetic errors**: Be meticulous about splitting real/imaginary parts.
  Write unit tests that verify loss terms against manually computed values at known points.