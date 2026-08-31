"""Configuration objects for downstream experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ProbeConfig:
    hidden_dim: int = 128
    linear_width: int = 0
    hidden_layers: Tuple[int, ...] = ()
    activations: Tuple[str, ...] = ("gelu",)
    dropout: float = 0.1


@dataclass(frozen=True)
class BenchmarkConfig:
    probes: Tuple[str, ...] = ("linear",)
    train_fractions: Tuple[float, ...] = (0.5,)
    evaluation_epochs: Tuple[int, ...] = (1500,)
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    log_every: int = 100
    save_predictions: bool = False
    probe: ProbeConfig = ProbeConfig()

    def validate(self) -> None:
        if not self.probes:
            raise ValueError("At least one probe is required.")
        if not self.train_fractions:
            raise ValueError("At least one train fraction is required.")
        if any(not 0.0 < value < 1.0 for value in self.train_fractions):
            raise ValueError("Train fractions must lie strictly between 0 and 1.")
        if not self.evaluation_epochs or any(value < 1 for value in self.evaluation_epochs):
            raise ValueError("Evaluation epochs must be positive integers.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        layers = self.probe.hidden_layers or (self.probe.hidden_dim, max(self.probe.hidden_dim // 2, 8))
        if any(value < 1 for value in layers):
            raise ValueError("MLP hidden-layer sizes must be positive.")
        if self.probe.linear_width < 0:
            raise ValueError("linear_width cannot be negative.")
        allowed_activations = {"relu", "gelu", "silu", "tanh", "leaky_relu", "identity"}
        if not self.probe.activations or not set(self.probe.activations) <= allowed_activations:
            raise ValueError(f"Activations must be selected from {sorted(allowed_activations)}.")
        if not 0.0 <= self.probe.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
