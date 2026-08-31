"""Extensible regression-probe registry."""

from __future__ import annotations

from typing import Callable, Dict

from torch import nn

from .config import ProbeConfig

ProbeFactory = Callable[[int, ProbeConfig], nn.Module]


class LinearProbe(nn.Module):
    def __init__(self, embedding_dim: int, config: ProbeConfig):
        super().__init__()
        self.network = (
            nn.Sequential(nn.Linear(embedding_dim, config.linear_width), nn.Linear(config.linear_width, 1))
            if config.linear_width else nn.Linear(embedding_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, embedding_dim: int, config: ProbeConfig):
        super().__init__()
        widths = config.hidden_layers or (config.hidden_dim, max(config.hidden_dim // 2, 8))
        activation_names = config.activations
        activation_types = {
            "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "identity": nn.Identity,
        }
        layers = []
        incoming = embedding_dim
        for index, width in enumerate(widths):
            activation = activation_names[min(index, len(activation_names) - 1)]
            layers.extend((nn.Linear(incoming, width), activation_types[activation]()))
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            incoming = width
        layers.append(nn.Linear(incoming, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ProbeRegistry:
    def __init__(self):
        self._factories: Dict[str, ProbeFactory] = {}

    def register(self, name: str, factory: ProbeFactory) -> None:
        if not name:
            raise ValueError("Probe name cannot be empty.")
        self._factories[name] = factory

    def create(self, name: str, embedding_dim: int, config: ProbeConfig) -> nn.Module:
        if name not in self._factories:
            raise ValueError(f"Unknown probe {name!r}; available: {sorted(self._factories)}")
        return self._factories[name](embedding_dim, config)

    @property
    def names(self):
        return tuple(sorted(self._factories))


def default_probe_registry() -> ProbeRegistry:
    registry = ProbeRegistry()
    registry.register("linear", lambda dimension, config: LinearProbe(dimension, config))
    registry.register("mlp", lambda dimension, config: MLPProbe(dimension, config))
    return registry
