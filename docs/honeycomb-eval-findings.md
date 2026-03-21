# Honeycomb Eval Findings & Next Steps

First converged honeycomb training run (ka=3.14, L=4.0, circle+BGT2). Eval run from wandb artifact `helmholtz-hc_ka3.14_L4.0_circ_bgt2-lbfgs:v0`, 12-plot diagnostic suite. All plots in `outputs/hc_eval/index.html`.

---

## Summary Metrics

| Metric | Value | Verdict |
|--------|-------|---------|
| PDE residual median | 7.66e-2 | Moderate |
| PDE residual max | 1.11e+2 | Hot spots near scatterer gaps |
| BC residual mean | 1.71e-3 | Good |
| BC residual max | 5.48e-3 | Good |
| ABC residual mean (BGT2) | 7.23e-4 | Good |
| ABC residual max | 2.10e-3 | Good |
| Energy flux CoV | 0.876 | Bad — should be near 0 |

## Key Finding: Incident Field Cancellation

The network learned phi_s ≈ -phi_inc over a broad region, not just inside the cluster. This makes the total field near zero everywhere within ~2 radii of the cluster.

| Region | Mean |phi_scattered| | Mean |phi_total| |
|--------|----------------------|-------------------|
| Inside cluster gaps (r < 0.8) | 1.000 | 0.0005 |
| Just outside cluster (1.2 < r < 2.0) | 0.995 | 0.036 |
| Far field (r > 2.5) | 0.453 | 0.656 |

**Inside the cluster**, phi_total ≈ 0 is partially physical: the gaps are sub-wavelength (gap/λ = 0.05), so the honeycomb blocks most of the wave. The `structuredscatterer.md` plan predicted this — sub-wavelength apertures act as effective barriers.

**Just outside the cluster**, phi_total ≈ 0 is **wrong**. There should be standing waves on the lit side and a partial shadow on the dark side. The scattered field magnitude should decay as ~1/√r, not stay pinned at |phi_inc| = 1.0 all the way out to r ≈ 2.

**Root cause**: The network found a low-loss shortcut — phi_s = -phi_inc approximately satisfies the Neumann BCs (hence BC residual is small) and roughly satisfies the Helmholtz equation near the cluster (phi_inc itself satisfies Helmholtz), but violates the PDE in the transition region and completely fails energy conservation. It's a local minimum, not the physical solution.

## What's Working

- **BC satisfaction on all 19 scatterers**: mean residual 1.7e-3, uniformly distributed. The boundary sampling and lambda_bc=20 are sufficient.
- **ABC (BGT2) at r=L**: mean 7.2e-4, max 2.1e-3. The circular boundary + second-order ABC is not the bottleneck.
- **Far-field pattern** shows qualitatively correct shape — forward scattering is dominant, and the pattern differs from the solid cylinder. But amplitudes are not trustworthy given the near-field issues.
- **Cluster zoom** shows smooth field in the gaps, no wild oscillations — the network is at least producing a smooth solution.

## What's Broken

- **PDE residual hot spots** at max=111, concentrated at the boundary of the "dead zone" where the field transitions from near-zero to physical values (~r = 2–2.5).
- **Energy flux** ramps from ~0 at r=1.5 to ~0.7 at r=3.5 instead of being constant. The network hasn't learned to propagate scattered energy outward from the cluster.
- **Radial decay** shows PINN mean |phi_s| ≈ 1.0 at r=1.5 (should be ~0.5 based on solid cylinder analytic), then drops to 0.45 at r=3.5. The near-field is completely wrong.

---

## Recommended Next Training Runs

### Run 1: Increase PDE weight + longer training

The current loss weights (lambda_pde=1, lambda_bc=20) let the network satisfy BCs at the expense of PDE accuracy. Rebalance so PDE compliance isn't sacrificed.

```bash
.venv/bin/python main.py --honeycomb --ka 3.14 \
    --lambda-pde 5.0 --lambda-bc 10.0 \
    --adam-epochs 40000 --lbfgs-epochs 500
```

**Rationale**: The BC is already well-satisfied at lambda_bc=10. Boosting lambda_pde forces the network to actually solve Helmholtz rather than just matching boundary conditions with a shortcut.

### Run 2: Curriculum learning — start from solid cylinder

Train on a single solid cylinder (analytic available), then fine-tune on the honeycomb. The solid cylinder solution is a reasonable starting point since the honeycomb scatters similarly in the far field.

```bash
# Phase 1: train on solid cylinder ka=3.14, circle+BGT2
.venv/bin/python main.py --ka 3.14 --outer-boundary circle --abc-order 2 \
    --adam-epochs 15000 --lbfgs-epochs 200

# Phase 2: fine-tune on honeycomb (load phase 1 checkpoint)
# Needs a --resume flag added to main.py
```

**Rationale**: The solid cylinder solution gives the network a physically correct starting point for the far-field structure. Fine-tuning then adjusts the near-field for the honeycomb gaps. This avoids the local minimum where phi_s = -phi_inc.

### Run 3: Tighter near-field sampling

The current `cluster_bias` strategy puts 60% of points within 1.5× cluster radius. Increase this and add explicit sampling in the transition zone r = 1.0–2.0 where the PDE residual is worst.

```bash
.venv/bin/python main.py --honeycomb --ka 3.14 \
    --n-interior 30000 --lambda-pde 5.0 \
    --adam-epochs 30000 --lbfgs-epochs 400
```

Could also modify `_sample_interior_cluster_bias` to use `near_fraction=0.75` and extend `R_near` to `2.0 * cluster_radius`.

### Run 4: Lower ka first

ka=3.14 means λ ≈ 2.0 and gap/λ = 0.05. The sub-wavelength gaps make this a very hard problem. Try ka=1.0 first (λ ≈ 6.28, gap/λ = 0.016) — at this frequency the honeycomb is deeply sub-wavelength and should scatter almost identically to a solid cylinder. This gives us a reference point where we can validate the PINN against the solid cylinder analytic.

```bash
.venv/bin/python main.py --honeycomb --ka 1.0 \
    --lambda-pde 5.0 --adam-epochs 20000 --lbfgs-epochs 300
```

Then sweep upward: ka = 1.0, 2.0, 3.14, looking for the frequency where the honeycomb starts to differ from the solid cylinder.

### Run 5: RAD adaptive resampling

Enable residual-based adaptive distribution (RAD) to automatically concentrate points where the PDE residual is high — which is exactly the transition zone the network is struggling with.

```bash
.venv/bin/python main.py --honeycomb --ka 3.14 \
    --use-rad --rad-k 2.0 --lambda-pde 5.0 \
    --adam-epochs 30000 --lbfgs-epochs 400
```

---

## Priority Order

1. **Run 4** (lower ka) — quick validation that the PINN can solve honeycomb scattering at all
2. **Run 1** (rebalanced weights + longer) — simplest fix, see if more PDE pressure breaks the shortcut
3. **Run 3** (more near-field points) — if Run 1 still has the dead zone
4. **Run 5** (RAD) — adaptive version of Run 3
5. **Run 2** (curriculum) — if the above don't work, pre-train on the known solution

## What to Look For

After each run, use the eval suite to check:
- Energy flux CoV < 0.1 (energy conservation)
- |phi_total| at r=1.5 should be O(1), not near zero
- Radial decay should track ~1/√r and be closer to solid cylinder at lower ka
- Honeycomb vs solid cylinder comparison should show meaningful differences at ka > 2
