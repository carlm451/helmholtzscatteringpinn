"""Centralized configuration for HelmholtzPINN."""

import math
import torch
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HelmholtzConfig:
    # Physics
    ka: float = math.pi
    a: float = 1.0
    L: float = 3.0
    scatterers: List[Tuple[float, float, float]] = field(default_factory=lambda: [(0.0, 0.0, 1.0)])

    # Network
    n_fourier_features: int = 64
    fourier_sigma: float = None  # set in __post_init__
    n_hidden_layers: int = 4
    n_hidden_neurons: int = 256
    activation: str = "tanh"
    use_residual: bool = True

    # Sampling
    n_interior: int = 10000
    n_boundary: int = 200
    n_outer: int = 400
    resample_every: int = 2000
    sampling_strategy: str = "uniform"      # "uniform", "radial_bias", "rad"
    radial_alpha: float = 0.5               # bias exponent for radial sampling
    use_rad: bool = False                   # enable RAD adaptive resampling
    rad_k: float = 1.0                      # RAD concentration parameter
    rad_c: float = 0.0                      # RAD exploration parameter

    # Outer boundary
    outer_boundary: str = "square"          # "square" or "circle"
    abc_order: int = 1                      # 1 or 2 (BGT2)

    # Adam
    adam_lr: float = 1e-3
    adam_epochs: int = 10000
    scheduler: str = "cosine"
    grad_clip: float = 1.0

    # L-BFGS
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 50
    lbfgs_epochs: int = 200
    lbfgs_history_size: int = 50

    # Loss weights
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    lambda_abc: float = 1.0

    # Evaluation
    eval_every: int = 500
    eval_grid_size: int = 200

    # wandb
    wandb_project: str = "helmholtz-pinn"
    use_wandb: bool = True

    # Device
    device: str = None  # auto-detect in __post_init__

    # Analytic
    n_series_terms: int = None  # set in __post_init__

    def __post_init__(self):
        self.k = self.ka / self.a
        if self.fourier_sigma is None:
            self.fourier_sigma = self.k
        if self.n_series_terms is None:
            self.n_series_terms = int(self.ka + 20)
        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    @classmethod
    def ka_pi(cls, **kwargs):
        return cls(ka=math.pi, **kwargs)

    @classmethod
    def ka_2pi(cls, **kwargs):
        defaults = dict(
            n_interior=15000,
            n_boundary=300,
            n_outer=500,
            n_hidden_neurons=256,
            n_hidden_layers=6,
            adam_epochs=20000,
            lbfgs_epochs=300,
        )
        defaults.update(kwargs)
        return cls(ka=2 * math.pi, **defaults)

    @classmethod
    def ka_3pi(cls, **kwargs):
        defaults = dict(
            n_interior=20000,
            n_boundary=400,
            n_outer=600,
            n_hidden_neurons=512,
            n_hidden_layers=6,
            n_fourier_features=128,
            adam_epochs=25000,
            lbfgs_epochs=400,
        )
        defaults.update(kwargs)
        return cls(ka=3 * math.pi, **defaults)

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                d[k] = [list(t) if isinstance(t, tuple) else t for t in v]
            else:
                d[k] = v
        return d
