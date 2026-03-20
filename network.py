"""Neural network architecture for HelmholtzPINN."""

import torch
import torch.nn as nn
import math


class FourierFeatureLayer(nn.Module):
    """Random Fourier feature mapping to overcome spectral bias."""

    def __init__(self, n_features=64, sigma=1.0, learnable=False):
        super().__init__()
        B = torch.randn(n_features, 2) * sigma
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, coords):
        # coords: (N, 2)
        proj = coords @ self.B.T  # (N, n_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class ResidualBlock(nn.Module):
    def __init__(self, width, activation_fn):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation = activation_fn

    def forward(self, x):
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return self.activation(out + x)


def _get_activation(name):
    if name == "tanh":
        return nn.Tanh()
    elif name == "sin":
        return SinActivation()
    elif name == "relu":
        return nn.ReLU()
    else:
        raise ValueError(f"Unknown activation: {name}")


class HelmholtzPINN(nn.Module):
    """Physics-Informed Neural Network for Helmholtz scattering.

    Maps (x, y) -> (u, v) where u = Re(phi_s), v = Im(phi_s).
    """

    def __init__(self, config):
        super().__init__()
        self.fourier = FourierFeatureLayer(
            n_features=config.n_fourier_features,
            sigma=config.fourier_sigma,
        )
        input_dim = 2 * config.n_fourier_features
        hidden = config.n_hidden_neurons
        activation_fn = _get_activation(config.activation)

        layers = [nn.Linear(input_dim, hidden), activation_fn]

        if config.use_residual:
            n_blocks = config.n_hidden_layers // 2
            for _ in range(n_blocks):
                layers.append(ResidualBlock(hidden, _get_activation(config.activation)))
        else:
            for _ in range(config.n_hidden_layers - 1):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(_get_activation(config.activation))

        layers.append(nn.Linear(hidden, 2))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y):
        """Forward pass. x, y: (N,) tensors -> returns (u, v) each (N,)."""
        coords = torch.stack([x, y], dim=-1)  # (N, 2)
        out = self.fourier(coords)
        out = self.net(out)  # (N, 2)
        return out[:, 0], out[:, 1]
