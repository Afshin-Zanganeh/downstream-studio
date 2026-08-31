"""Data-agnostic benchmark engine."""

from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import BenchmarkConfig
from .data import EmbeddingSet, TargetSet, align_inputs
from .metrics import regression_metrics
from .models import ProbeRegistry, default_probe_registry
from .results import BenchmarkResult


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise ValueError("device must be auto, cpu, cuda, or mps.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split(n: int, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 3:
        raise ValueError("At least three finite target values are required.")
    order = np.random.default_rng(seed).permutation(n)
    n_train = max(1, min(n - 1, int(round(fraction * n))))
    return order[:n_train], order[n_train:]


def _split_seed(seed: int, target: str, fraction: float) -> int:
    digest = hashlib.sha256(f"{target}|{fraction:.8f}".encode()).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def _predict(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(values).float()), batch_size=batch_size)
    output: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            output.append(model(batch.to(device)).detach().cpu().numpy())
    return np.concatenate(output)


def run_benchmark(
    embedding_sets: Mapping[str, EmbeddingSet],
    target_set: TargetSet,
    config: BenchmarkConfig = BenchmarkConfig(),
    registry: Optional[ProbeRegistry] = None,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> BenchmarkResult:
    """Run separate regression probes for every target and embedding model."""
    config.validate()
    registry = registry or default_probe_registry()
    unknown = sorted(set(config.probes) - set(registry.names))
    if unknown:
        raise ValueError(f"Unknown probes {unknown}; available: {registry.names}")
    aligned = align_inputs(embedding_sets, target_set)
    device = _device(config.device)
    _seed_everything(config.seed)
    metrics_rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    checkpoints = sorted(set(config.evaluation_epochs))

    for target_name, target_all in aligned.targets.items():
        valid = np.isfinite(target_all)
        ids = aligned.ids[valid]
        y = target_all[valid].astype(np.float32)
        if len(y) < 3 or float(np.std(y)) == 0:
            continue
        for fraction in config.train_fractions:
            train_index, test_index = _split(len(y), fraction, _split_seed(config.seed, target_name, fraction))
            y_train_raw, y_test = y[train_index], y[test_index]
            y_mean, y_std = float(y_train_raw.mean()), float(y_train_raw.std())
            if y_std == 0:
                continue
            y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)

            for embedding_name, embedding_all in aligned.embeddings.items():
                matrix = embedding_all[valid]
                x_train_raw, x_test_raw = matrix[train_index], matrix[test_index]
                x_mean = x_train_raw.mean(axis=0, keepdims=True)
                x_std = x_train_raw.std(axis=0, keepdims=True)
                x_std[x_std < 1e-8] = 1.0
                x_train = ((x_train_raw - x_mean) / x_std).astype(np.float32)
                x_test = ((x_test_raw - x_mean) / x_std).astype(np.float32)

                for probe_name in config.probes:
                    _seed_everything(config.seed)
                    model = registry.create(probe_name, x_train.shape[1], config.probe).to(device)
                    train_x_tensor = torch.from_numpy(x_train)
                    train_y_tensor = torch.from_numpy(y_train)
                    # Keeping training data on an accelerator avoids copying every
                    # mini-batch from host memory on every epoch.
                    if device.type in {"cuda", "mps"}:
                        train_x_tensor = train_x_tensor.to(device)
                        train_y_tensor = train_y_tensor.to(device)
                    loader = None
                    if device.type == "cpu":
                        loader = DataLoader(
                            TensorDataset(train_x_tensor, train_y_tensor),
                            batch_size=min(config.batch_size, len(x_train)),
                            shuffle=True,
                            num_workers=config.num_workers,
                            generator=torch.Generator().manual_seed(config.seed),
                        )
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
                    )
                    loss_function = nn.MSELoss()
                    started = time.perf_counter()
                    final_loss = math.nan
                    for epoch in range(1, checkpoints[-1] + 1):
                        model.train()
                        loss_sum = 0.0
                        seen = 0
                        if loader is not None:
                            batches = loader
                        else:
                            order = torch.randperm(len(train_x_tensor), device=device)
                            batches = (
                                (train_x_tensor[index], train_y_tensor[index])
                                for start in range(0, len(order), config.batch_size)
                                for index in (order[start:start + config.batch_size],)
                            )
                        for batch_x, batch_y in batches:
                            if device.type == "cpu":
                                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                            optimizer.zero_grad(set_to_none=True)
                            loss = loss_function(model(batch_x), batch_y)
                            loss.backward()
                            optimizer.step()
                            loss_sum += float(loss.detach().cpu()) * len(batch_x)
                            seen += len(batch_x)
                        final_loss = loss_sum / max(seen, 1)
                        context = {
                            "embedding_model": embedding_name,
                            "target": target_name,
                            "probe": probe_name,
                            "train_fraction": fraction,
                        }
                        if epoch == 1 or epoch % config.log_every == 0 or epoch in checkpoints:
                            history_rows.append({**context, "epoch": epoch, "train_mse_normalized": final_loss})
                        if epoch not in checkpoints:
                            continue

                        prediction = _predict(model, x_test, device, config.batch_size) * y_std + y_mean
                        row = {
                            **context,
                            "epochs": epoch,
                            "seed": config.seed,
                            "n_total": len(y),
                            "n_train": len(train_index),
                            "n_test": len(test_index),
                            "embedding_dim": x_train.shape[1],
                            "train_seconds": time.perf_counter() - started,
                            "final_train_mse_normalized": final_loss,
                            **regression_metrics(y_test, prediction),
                        }
                        metrics_rows.append(row)
                        if progress_callback:
                            progress_callback({
                                **context, "epoch": epoch, "max_epoch": checkpoints[-1],
                                "r2": row["r2"], "mae": row["mae"],
                                "elapsed_seconds": row["train_seconds"],
                            })
                        # Persist individual predictions only for the final
                        # checkpoint. This keeps large multi-target runs from
                        # accumulating millions of redundant rows in memory.
                        if config.save_predictions and epoch == checkpoints[-1]:
                            for interface_id, truth, predicted in zip(ids[test_index], y_test, prediction):
                                prediction_rows.append({
                                    **context,
                                    "epochs": epoch,
                                    "id": interface_id,
                                    "y_true": float(truth),
                                    "y_pred": float(predicted),
                                    "residual": float(truth - predicted),
                                })

    if not metrics_rows:
        raise RuntimeError("No experiments completed; check target sample counts and variance.")
    return BenchmarkResult(
        metrics=metrics_rows,
        history=history_rows,
        predictions=prediction_rows,
        metadata={
            "common_ids": len(aligned.ids),
            "embedding_models": list(aligned.embeddings),
            "targets": list(aligned.targets),
            "config": {
                "probes": list(config.probes),
                "train_fractions": list(config.train_fractions),
                "evaluation_epochs": list(config.evaluation_epochs),
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "seed": config.seed,
                "device": str(device),
                "probe": {
                    "linear_width": config.probe.linear_width,
                    "hidden_layers": list(config.probe.hidden_layers),
                    "activations": list(config.probe.activations),
                    "dropout": config.probe.dropout,
                },
            },
        },
    )
