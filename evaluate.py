"""Evaluation metrics: PINN predictions vs analytic solution."""

import torch
import numpy as np


@torch.no_grad()
def evaluate_against_analytic(model, domain, config, analytic_fn,
                              x_range=None, y_range=None, grid_size=None):
    """Evaluate model accuracy against analytic solution on a dense grid.

    Args:
        analytic_fn: callable(x_np, y_np, k, a, n_terms) -> complex ndarray
    Returns:
        dict with l2_rel, l2_abs, max_err, mean_err
    """
    model.eval()
    gs = grid_size or config.eval_grid_size
    x_flat, y_flat, valid = domain.sample_test_grid(gs, x_range, y_range)

    # PINN prediction
    u_pred, v_pred = model(x_flat, y_flat)
    u_pred = u_pred[valid].cpu().numpy()
    v_pred = v_pred[valid].cpu().numpy()

    # Analytic reference
    x_np = x_flat[valid].cpu().numpy()
    y_np = y_flat[valid].cpu().numpy()
    phi_analytic = analytic_fn(x_np, y_np, config.k, config.a, config.n_series_terms)

    u_exact = np.real(phi_analytic)
    v_exact = np.imag(phi_analytic)

    # Filter out any NaN from analytic solution
    valid_mask = ~(np.isnan(u_exact) | np.isnan(v_exact))
    u_pred = u_pred[valid_mask]
    v_pred = v_pred[valid_mask]
    u_exact = u_exact[valid_mask]
    v_exact = v_exact[valid_mask]

    # Error metrics
    err_u = u_pred - u_exact
    err_v = v_pred - v_exact
    err_magnitude = np.sqrt(err_u ** 2 + err_v ** 2)
    exact_magnitude = np.sqrt(u_exact ** 2 + v_exact ** 2)

    l2_abs = np.sqrt(np.mean(err_u ** 2 + err_v ** 2))
    l2_exact = np.sqrt(np.mean(u_exact ** 2 + v_exact ** 2))
    l2_rel = l2_abs / (l2_exact + 1e-16)

    max_err = np.max(err_magnitude)
    mean_err = np.mean(err_magnitude)

    model.train()
    return {
        "l2_rel": float(l2_rel),
        "l2_abs": float(l2_abs),
        "max_err": float(max_err),
        "mean_err": float(mean_err),
    }


@torch.no_grad()
def check_symmetry(model, config, n_points=500):
    """Check symmetry: u(x,y) ~ u(x,-y) and v(x,y) ~ -v(x,-y).

    For a plane wave incident along x-axis, the real part is symmetric
    and the imaginary part is antisymmetric in y.
    """
    model.eval()
    x = torch.linspace(-config.L, config.L, n_points, device=config.device)
    y = torch.rand(n_points, device=config.device) * config.L * 0.5 + 0.1

    u_pos, v_pos = model(x, y)
    u_neg, v_neg = model(x, -y)

    sym_u = torch.mean((u_pos - u_neg) ** 2).item()
    sym_v = torch.mean((v_pos + v_neg) ** 2).item()

    model.train()
    return {"symmetry_u_mse": sym_u, "symmetry_v_mse": sym_v}
