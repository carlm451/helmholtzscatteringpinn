"""Geometry and point sampling for the scattering domain."""

import torch
import math


class ScatteringDomain:
    def __init__(self, L, scatterers, device="cpu"):
        self.L = L
        self.scatterers = scatterers  # list of (cx, cy, r)
        self.device = device

    def is_inside_any_scatterer(self, x, y):
        """Boolean mask: True where (x,y) is inside any scatterer."""
        mask = torch.zeros_like(x, dtype=torch.bool)
        for cx, cy, r in self.scatterers:
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = mask | (dist < r)
        return mask

    def sample_interior(self, n):
        """Sample n points in [-L, L]^2, excluding scatterer interiors.

        Uses rejection sampling. Returns dict with x, y (requires_grad=True).
        """
        points_x = []
        points_y = []
        remaining = n

        while remaining > 0:
            # Oversample to account for rejection
            n_try = int(remaining * 1.5) + 100
            x = (2 * torch.rand(n_try, device=self.device) - 1) * self.L
            y = (2 * torch.rand(n_try, device=self.device) - 1) * self.L
            inside = self.is_inside_any_scatterer(x, y)
            valid = ~inside
            x_valid = x[valid]
            y_valid = y[valid]
            take = min(remaining, len(x_valid))
            points_x.append(x_valid[:take])
            points_y.append(y_valid[:take])
            remaining -= take

        x = torch.cat(points_x).requires_grad_(True)
        y = torch.cat(points_y).requires_grad_(True)
        return {"x": x, "y": y}

    def sample_scatterer_boundary(self, n_per_scatterer):
        """Sample points on scatterer surfaces with outward normals."""
        all_x, all_y, all_nx, all_ny = [], [], [], []

        for cx, cy, r in self.scatterers:
            theta = torch.linspace(0, 2 * math.pi, n_per_scatterer + 1,
                                   device=self.device)[:-1]
            x = cx + r * torch.cos(theta)
            y = cy + r * torch.sin(theta)
            # Outward normals (away from center)
            nx = torch.cos(theta)
            ny = torch.sin(theta)
            all_x.append(x)
            all_y.append(y)
            all_nx.append(nx)
            all_ny.append(ny)

        x = torch.cat(all_x).requires_grad_(True)
        y = torch.cat(all_y).requires_grad_(True)
        nx = torch.cat(all_nx)
        ny = torch.cat(all_ny)
        return {"x": x, "y": y, "nx": nx, "ny": ny}

    def sample_outer_boundary(self, n_total):
        """Sample points on the outer box boundary [-L,L]^2 with outward normals."""
        n_per_side = n_total // 4
        remainder = n_total - 4 * n_per_side

        sides_x, sides_y, sides_nx, sides_ny = [], [], [], []
        L = self.L

        # Bottom: y = -L, normal = (0, -1)
        n = n_per_side + (1 if remainder > 0 else 0)
        x = torch.linspace(-L, L, n, device=self.device)
        y = torch.full_like(x, -L)
        sides_x.append(x); sides_y.append(y)
        sides_nx.append(torch.zeros_like(x)); sides_ny.append(torch.full_like(x, -1.0))

        # Top: y = L, normal = (0, 1)
        n = n_per_side + (1 if remainder > 1 else 0)
        x = torch.linspace(-L, L, n, device=self.device)
        y = torch.full_like(x, L)
        sides_x.append(x); sides_y.append(y)
        sides_nx.append(torch.zeros_like(x)); sides_ny.append(torch.full_like(x, 1.0))

        # Left: x = -L, normal = (-1, 0)
        n = n_per_side + (1 if remainder > 2 else 0)
        y = torch.linspace(-L, L, n, device=self.device)
        x = torch.full_like(y, -L)
        sides_x.append(x); sides_y.append(y)
        sides_nx.append(torch.full_like(y, -1.0)); sides_ny.append(torch.zeros_like(y))

        # Right: x = L, normal = (1, 0)
        x_r = torch.full((n_per_side,), L, device=self.device)
        y_r = torch.linspace(-L, L, n_per_side, device=self.device)
        sides_x.append(x_r); sides_y.append(y_r)
        sides_nx.append(torch.full_like(x_r, 1.0)); sides_ny.append(torch.zeros_like(x_r))

        x = torch.cat(sides_x).requires_grad_(True)
        y = torch.cat(sides_y).requires_grad_(True)
        nx = torch.cat(sides_nx)
        ny = torch.cat(sides_ny)
        return {"x": x, "y": y, "nx": nx, "ny": ny}

    def sample_test_grid(self, grid_size, x_range=None, y_range=None):
        """Regular meshgrid for evaluation. Returns (x_flat, y_flat, valid_mask)."""
        if x_range is None:
            x_range = (-self.L, self.L)
        if y_range is None:
            y_range = (-self.L, self.L)

        x_lin = torch.linspace(x_range[0], x_range[1], grid_size, device=self.device)
        y_lin = torch.linspace(y_range[0], y_range[1], grid_size, device=self.device)
        xx, yy = torch.meshgrid(x_lin, y_lin, indexing="xy")
        x_flat = xx.reshape(-1)
        y_flat = yy.reshape(-1)

        valid_mask = ~self.is_inside_any_scatterer(x_flat, y_flat)
        return x_flat, y_flat, valid_mask
