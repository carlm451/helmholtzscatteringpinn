"""Analytic solution for 2D Helmholtz scattering off a rigid circular cylinder.

Uses Bessel/Hankel series expansion. All functions use numpy/scipy only (no torch).
"""

import numpy as np
from scipy import special


def incident_field(x, y, k):
    """Plane wave incident field: exp(ikx)."""
    return np.exp(1j * k * x)


def scattered_field(x, y, k, a=1.0, n_terms=None, center=(0, 0)):
    """Scattered field via Bessel/Hankel series.

    phi_s = -sum_{n=-N}^{N} i^n * [J_n'(ka) / H_n^(1)'(ka)] * H_n^(1)(kr) * exp(in*theta)

    Points inside the scatterer (r < a) are set to NaN.
    """
    cx, cy = center
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    ka = k * a
    if n_terms is None:
        n_terms = int(30 + ka ** 1.01) + 5

    phi_s = np.zeros_like(r, dtype=complex)

    for n in range(-n_terms, n_terms + 1):
        # Derivatives of Bessel/Hankel functions at ka
        jn_prime = special.jvp(n, ka)
        # h1vp computes the derivative of the Hankel function of the first kind
        hn_prime = special.h1vp(n, ka)

        # Coefficient
        coeff = -(1j ** n) * (jn_prime / hn_prime)

        # H_n^(1)(kr) for all points
        hn_kr = special.hankel1(n, k * r)

        phi_s += coeff * hn_kr * np.exp(1j * n * theta)

    # Mask interior of scatterer
    phi_s[r < a] = np.nan + 1j * np.nan

    return phi_s


def total_field(x, y, k, a=1.0, n_terms=None, center=(0, 0)):
    """Total field = incident + scattered."""
    phi_inc = incident_field(x, y, k)
    phi_s = scattered_field(x, y, k, a=a, n_terms=n_terms, center=center)
    return phi_inc + phi_s


def scattered_field_gradient(x, y, k, a=1.0, n_terms=None, center=(0, 0)):
    """Gradient of scattered field (dphi_s/dx, dphi_s/dy) via finite differences."""
    eps = 1e-6
    phi_xp = scattered_field(x + eps, y, k, a=a, n_terms=n_terms, center=center)
    phi_xm = scattered_field(x - eps, y, k, a=a, n_terms=n_terms, center=center)
    phi_yp = scattered_field(x, y + eps, k, a=a, n_terms=n_terms, center=center)
    phi_ym = scattered_field(x, y - eps, k, a=a, n_terms=n_terms, center=center)

    dphi_dx = (phi_xp - phi_xm) / (2 * eps)
    dphi_dy = (phi_yp - phi_ym) / (2 * eps)
    return dphi_dx, dphi_dy


def verify_neumann_bc(k, a=1.0, n_terms=None, center=(0, 0), n_points=200):
    """Verify d(phi_total)/dr = 0 at r=a. Returns max absolute residual."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cx, cy = center
    x_surf = cx + a * np.cos(theta)
    y_surf = cy + a * np.sin(theta)
    nx = np.cos(theta)
    ny = np.sin(theta)

    # Forward finite differences (outward from surface to avoid r < a masking)
    eps = 1e-6
    phi_1 = total_field(x_surf + eps * nx, y_surf + eps * ny, k, a=a, n_terms=n_terms, center=center)
    phi_2 = total_field(x_surf + 2 * eps * nx, y_surf + 2 * eps * ny, k, a=a, n_terms=n_terms, center=center)
    phi_3 = total_field(x_surf + 3 * eps * nx, y_surf + 3 * eps * ny, k, a=a, n_terms=n_terms, center=center)
    # Second-order forward difference: f'(0) ~ (-3f(h) + 4f(2h) - f(3h)) / (2h)  -- shifted
    # Actually: standard 3-point forward: f'(x) ~ (-3f(x) + 4f(x+h) - f(x+2h)) / (2h)
    # We can't evaluate at x (r=a gives NaN), so use: f'(x) ~ (f(x+h) - f(x-h))/(2h)
    # workaround: two-point forward from r=a+eps
    dphi_dn = (-3 * phi_1 + 4 * phi_2 - phi_3) / (2 * eps)

    return np.max(np.abs(dphi_dn))


def verify_far_field_decay(k, a=1.0, n_terms=None, center=(0, 0)):
    """Verify |phi_s| decays as 1/sqrt(r) in the far field."""
    theta = 0.0
    cx, cy = center
    radii = np.array([5.0, 10.0, 20.0, 50.0])
    x = cx + radii * np.cos(theta)
    y = cy + radii * np.sin(theta)

    phi_s = scattered_field(x, y, k, a=a, n_terms=n_terms, center=center)
    amplitudes = np.abs(phi_s)

    # |phi_s| * sqrt(r) should be approximately constant for large r
    scaled = amplitudes * np.sqrt(radii)
    variation = np.std(scaled) / np.mean(scaled)
    return variation, scaled
