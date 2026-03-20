# L=3 A/B Comparison: Smart Sampling vs Better ABCs

**Date**: 2026-03-20
**wandb project**: [helmholtz-pinn](https://wandb.ai/cmerrigan-alynix/helmholtz-pinn)

## Motivation

Prior eval_suite diagnostics identified the first-order ABC as the dominant error source:
- Analytic ABC violation mean = 0.16 at L=3 (the exact solution doesn't satisfy the imposed ABC)
- The PINN satisfies the (wrong) ABC perfectly (residual 3.4e-04)
- Errors concentrate near the outer boundary (r=[2.5,3] has 2x the error of r=[1,1.25])

We tested three techniques to break this bottleneck, all at L=3 with 10K interior points, 5K Adam + 50 L-BFGS:

1. **Circle + BGT2 ABC** — circular outer boundary with second-order absorbing condition
2. **Radial-biased sampling** — r^{-alpha} density concentrating points near scatterer
3. **RAD adaptive resampling** — residual-proportional point redistribution (Lu et al. 2023)

## Results

| Run | Config | L2_rel | Max Error | vs Baseline |
|-----|--------|--------|-----------|-------------|
| Baseline | square, ABC1, uniform | 7.13% | 10.1% | -- |
| Radial bias | square, ABC1, radial (alpha=0.5) | 7.12% | 10.1% | ~same |
| RAD adaptive | square, ABC1, uniform+RAD | 7.09% | 10.1% | ~same |
| **Circle+BGT2** | **circle, ABC2, uniform** | **2.42%** | **2.96%** | **2.9x better** |
| **Combined** | **circle, ABC2, radial+RAD** | **2.43%** | **2.96%** | **2.9x better** |

## Key Findings

### 1. The ABC is the entire bottleneck at L=3

Neither radial-biased sampling nor RAD adaptive resampling produced any measurable improvement while using the first-order ABC on a square boundary. The 7.1% error floor is set entirely by the ABC approximation quality, not by point density or distribution.

### 2. Circle + BGT2 cuts error by 3x

Switching from a square boundary with first-order ABC to a circular boundary with the second-order BGT2 condition reduced L2_rel from 7.13% to 2.42% — a 2.9x improvement with no increase in computational cost.

The BGT2 condition on a circle of radius R:

```
d(phi_s)/dr - ik*phi_s + phi_s/(2r) = 0
```

captures the 1/sqrt(r) amplitude decay of 2D cylindrical waves. This has O(1/R^2) error vs O(1/R) for first-order — a factor of R improvement at the boundary.

### 3. Sampling tricks don't stack with ABC fix

The combined run (circle + BGT2 + radial bias + RAD) produced 2.43% — identical to circle + BGT2 alone. At L=3, once the ABC error is removed, the remaining ~2.4% error is the network/PDE residual floor that sampling improvements cannot address.

Sampling tricks may become relevant at L=5+ where point density drops, but at L=3 the domain is small enough that 10K uniform points provide adequate coverage.

## Bugs Found and Fixed

1. **Eval masking for circular domains**: The evaluation grid was square [-L,L]^2, including corner regions (r > L) outside the circular training domain. Network predictions in these unseen regions inflated error metrics. Fixed by masking out r > L points in `evaluate.py`.

2. **Radial sampling R for square domains**: The radial sampler used R=L, missing square corners at r > L. Fixed to use R = L*sqrt(2) for square domains.

3. **RAD `torch.no_grad()` conflict**: The `with torch.no_grad():` wrapper around RAD residual computation prevented the internal autograd calls needed for second derivatives. Fixed by removing the wrapper (the function already uses `create_graph=False`).

## Production Run Plan

Based on these results, the winning configuration for production runs is:

```
--outer-boundary circle --abc-order 2 --adam-epochs 10000 --lbfgs-epochs 200
```

Planned runs across 5 ka values to compare against analytic sweep:

| ka | k | Description | Expected L2_rel |
|----|---|-------------|-----------------|
| 0.5 | 0.5 | Sub-wavelength | < 1% |
| 1.0 | 1.0 | Transition regime | < 2% |
| pi | 3.14 | 1 wavelength/radius | < 2% |
| 2*pi | 6.28 | 2 wavelengths/radius | TBD |
| 3*pi | 9.42 | 3 wavelengths/radius | TBD |

## Implementation

New code in this round:
- `config.py`: 7 new fields (sampling_strategy, radial_alpha, use_rad, rad_k, rad_c, outer_boundary, abc_order)
- `main.py`: 5 new CLI args
- `domain.py`: `sample_circular_outer_boundary()`, `_sample_interior_radial()`, dispatch logic
- `losses.py`: `abc_loss_2nd_order()` (BGT2), `total_loss()` dispatch on abc_order
- `train.py`: `_compute_pointwise_pde_residual()`, RAD resampling branch
- `evaluate.py`: Circular domain masking in eval grid
- `run_sweep.sh`: Automated experiment runner
