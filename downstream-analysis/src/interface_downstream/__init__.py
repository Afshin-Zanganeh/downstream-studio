"""Public API for interface downstream analysis."""

from .config import BenchmarkConfig, ProbeConfig
from .data import EmbeddingSet, TargetSet
from .models import ProbeRegistry, default_probe_registry
from .results import BenchmarkResult
from .runner import run_benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "EmbeddingSet",
    "ProbeConfig",
    "ProbeRegistry",
    "TargetSet",
    "default_probe_registry",
    "run_benchmark",
]

