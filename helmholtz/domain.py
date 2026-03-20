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

    def sample_interior(self, n, strategy="uniform", alpha=0.5, outer_boundary="square",
                         cluster_radius=None):
        """Sample n points in domain, excluding scatterer interiors.

        Args:
            strategy: "uniform", "radial_bias", or "cluster_bias"
            alpha: radial bias exponent (used when strategy="radial_bias")
            outer_boundary: "square" or "circle" — determines domain shape
            cluster_radius: radius of scatterer cluster (used for cluster_bias)
        """
        if strategy == "cluster_bias":
            return self._sample_interior_cluster_bias(n, outer_boundary, cluster_radius)
        if strategy == "radial_bias":
            return self._sample_interior_radial(n, alpha, outer_boundary)
        return self._sample_interior_uniform(n, outer_boundary)

    def _sample_interior_uniform(self, n, outer_boundary="square"):
        """Uniform sampling in square or circular domain."""
        points_x = []
        points_y = []
        remaining = n

        while remaining > 0:
            n_try = int(remaining * 1.5) + 100
            if outer_boundary == "circle":
                # Sample uniformly in disk of radius L
                r = self.L * torch.sqrt(torch.rand(n_try, device=self.device))
                theta = 2 * math.pi * torch.rand(n_try, device=self.device)
                x = r * torch.cos(theta)
                y = r * torch.sin(theta)
            else:
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

    def _sample_interior_radial(self, n, alpha=0.5, outer_boundary="square"):
        """Radial-biased sampling: p(r) ~ r^{1-alpha}, concentrating points near scatterer.

        Uses inverse CDF: r = (a^{2-alpha} + U * (R^{2-alpha} - a^{2-alpha}))^{1/(2-alpha)}
        """
        a = max(s[2] for s in self.scatterers)  # largest scatterer radius
        # For square domain, R must reach corners at L*sqrt(2)
        R = self.L if outer_boundary == "circle" else self.L * math.sqrt(2)
        exp = 2 - alpha

        points_x = []
        points_y = []
        remaining = n
        # For square domain, oversample more since we reject outside square
        oversample = 1.8 if outer_boundary == "square" else 1.2

        while remaining > 0:
            n_try = int(remaining * oversample) + 100
            U = torch.rand(n_try, device=self.device)
            r = (a ** exp + U * (R ** exp - a ** exp)) ** (1.0 / exp)
            theta = 2 * math.pi * torch.rand(n_try, device=self.device)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)

            # Reject points outside domain
            if outer_boundary == "square":
                in_domain = (x.abs() <= self.L) & (y.abs() <= self.L)
            else:
                in_domain = (x ** 2 + y ** 2) <= R ** 2

            inside_scat = self.is_inside_any_scatterer(x, y)
            valid = in_domain & ~inside_scat
            x_valid = x[valid]
            y_valid = y[valid]
            take = min(remaining, len(x_valid))
            points_x.append(x_valid[:take])
            points_y.append(y_valid[:take])
            remaining -= take

        x = torch.cat(points_x).requires_grad_(True)
        y = torch.cat(points_y).requires_grad_(True)
        return {"x": x, "y": y}

    def _sample_interior_cluster_bias(self, n, outer_boundary="circle",
                                        cluster_radius=None, near_fraction=0.6):
        """Two-zone sampling: dense near cluster, sparse in far field.

        Near-field (near_fraction): uniform in disk of radius 1.5*cluster_radius
        Far-field (1-near_fraction): uniform in remaining domain
        """
        R_near = 1.5 * (cluster_radius or 1.0)
        n_near = int(n * near_fraction)
        n_far = n - n_near

        near = self._sample_disk_excluding_scatterers(n_near, R_near)
        far = self._sample_annulus_or_outer(n_far, R_near, outer_boundary)

        x = torch.cat([near["x"], far["x"]]).requires_grad_(True)
        y = torch.cat([near["y"], far["y"]]).requires_grad_(True)
        return {"x": x, "y": y}

    def _sample_disk_excluding_scatterers(self, n, R):
        """Rejection-sample uniformly in disk of radius R, excluding scatterer interiors."""
        points_x = []
        points_y = []
        remaining = n

        while remaining > 0:
            n_try = int(remaining * 2.0) + 100
            r = R * torch.sqrt(torch.rand(n_try, device=self.device))
            theta = 2 * math.pi * torch.rand(n_try, device=self.device)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            valid = ~self.is_inside_any_scatterer(x, y)
            x_valid = x[valid]
            y_valid = y[valid]
            take = min(remaining, len(x_valid))
            points_x.append(x_valid[:take])
            points_y.append(y_valid[:take])
            remaining -= take

        return {"x": torch.cat(points_x), "y": torch.cat(points_y)}

    def _sample_annulus_or_outer(self, n, R_inner, outer_boundary):
        """Sample uniformly outside disk of radius R_inner but within domain boundary."""
        points_x = []
        points_y = []
        remaining = n

        while remaining > 0:
            n_try = int(remaining * 2.0) + 100
            if outer_boundary == "circle":
                # Uniform in disk of radius L
                r = self.L * torch.sqrt(torch.rand(n_try, device=self.device))
                theta = 2 * math.pi * torch.rand(n_try, device=self.device)
                x = r * torch.cos(theta)
                y = r * torch.sin(theta)
            else:
                x = (2 * torch.rand(n_try, device=self.device) - 1) * self.L
                y = (2 * torch.rand(n_try, device=self.device) - 1) * self.L

            r2 = x**2 + y**2
            outside_near = r2 >= R_inner**2
            not_in_scat = ~self.is_inside_any_scatterer(x, y)
            valid = outside_near & not_in_scat
            x_valid = x[valid]
            y_valid = y[valid]
            take = min(remaining, len(x_valid))
            points_x.append(x_valid[:take])
            points_y.append(y_valid[:take])
            remaining -= take

        return {"x": torch.cat(points_x), "y": torch.cat(points_y)}

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

    def sample_circular_outer_boundary(self, n_total, R=None):
        """Sample points on circle of radius R with radial outward normals."""
        if R is None:
            R = self.L
        theta = torch.linspace(0, 2 * math.pi, n_total + 1, device=self.device)[:-1]
        x = R * torch.cos(theta)
        y = R * torch.sin(theta)
        nx = torch.cos(theta)  # radial outward normal
        ny = torch.sin(theta)
        return {
            "x": x.requires_grad_(True),
            "y": y.requires_grad_(True),
            "nx": nx,
            "ny": ny,
        }

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
