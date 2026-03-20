"""Physics-informed loss functions for Helmholtz scattering.

All losses use torch.autograd.grad for computing spatial derivatives.
"""

import torch


def compute_derivatives(model, x, y):
    """Compute all first and second derivatives of model output w.r.t. x, y.

    Returns: u, v, du_dx, du_dy, dv_dx, dv_dy, d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2
    """
    u, v = model(x, y)
    ones = torch.ones_like(u)

    # First derivatives
    du_dx = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, ones, create_graph=True, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, ones, create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, ones, create_graph=True, retain_graph=True)[0]

    # Second derivatives
    d2u_dx2 = torch.autograd.grad(du_dx, x, ones, create_graph=True, retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, ones, create_graph=True, retain_graph=True)[0]
    d2v_dx2 = torch.autograd.grad(dv_dx, x, ones, create_graph=True, retain_graph=True)[0]
    d2v_dy2 = torch.autograd.grad(dv_dy, y, ones, create_graph=True, retain_graph=True)[0]

    return u, v, du_dx, du_dy, dv_dx, dv_dy, d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2


def pde_residual_loss(model, x, y, k):
    """Helmholtz PDE residual: nabla^2(phi_s) + k^2 * phi_s = 0.

    Split into real/imag:
        d2u/dx2 + d2u/dy2 + k^2 * u = 0
        d2v/dx2 + d2v/dy2 + k^2 * v = 0
    """
    u, v, _, _, _, _, d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2 = compute_derivatives(model, x, y)

    res_u = d2u_dx2 + d2u_dy2 + k ** 2 * u
    res_v = d2v_dx2 + d2v_dy2 + k ** 2 * v

    return torch.mean(res_u ** 2 + res_v ** 2)


def neumann_bc_loss(model, x, y, nx, ny, k):
    """Neumann BC on scatterer surface: d(phi_s)/dn = -d(phi_inc)/dn.

    phi_inc = exp(ikx), so d(phi_inc)/dx = ik*exp(ikx), d(phi_inc)/dy = 0.
    d(phi_inc)/dn = nx * ik * exp(ikx)

    Target for scattered field normal derivative:
        Re(-d(phi_inc)/dn) = nx * k * sin(kx)
        Im(-d(phi_inc)/dn) = -nx * k * cos(kx)
    """
    u, v = model(x, y)
    ones = torch.ones_like(u)

    du_dx = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, ones, create_graph=True, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, ones, create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, ones, create_graph=True, retain_graph=True)[0]

    du_dn = nx * du_dx + ny * du_dy
    dv_dn = nx * dv_dx + ny * dv_dy

    # Target: -d(phi_inc)/dn split into real and imaginary parts
    kx = k * x
    target_real = nx * k * torch.sin(kx)
    target_imag = -nx * k * torch.cos(kx)

    return torch.mean((du_dn - target_real) ** 2 + (dv_dn - target_imag) ** 2)


def abc_loss(model, x, y, nx, ny, k):
    """First-order absorbing boundary condition: d(phi_s)/dn - ik*phi_s = 0.

    Split into real/imag:
        du/dn + k*v = 0
        dv/dn - k*u = 0
    """
    u, v = model(x, y)
    ones = torch.ones_like(u)

    du_dx = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, ones, create_graph=True, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, ones, create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, ones, create_graph=True, retain_graph=True)[0]

    du_dn = nx * du_dx + ny * du_dy
    dv_dn = nx * dv_dx + ny * dv_dy

    res_real = du_dn + k * v
    res_imag = dv_dn - k * u

    return torch.mean(res_real ** 2 + res_imag ** 2)


def total_loss(model, interior, boundary, outer, k, config):
    """Compute weighted total loss from all components."""
    loss_pde = pde_residual_loss(model, interior["x"], interior["y"], k)
    loss_bc = neumann_bc_loss(model, boundary["x"], boundary["y"],
                              boundary["nx"], boundary["ny"], k)
    loss_abc = abc_loss(model, outer["x"], outer["y"],
                        outer["nx"], outer["ny"], k)

    total = (config.lambda_pde * loss_pde +
             config.lambda_bc * loss_bc +
             config.lambda_abc * loss_abc)

    return total, loss_pde, loss_bc, loss_abc
