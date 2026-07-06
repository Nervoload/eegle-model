"""Minimal PyTorch classifier training loop."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from eegworkbench.config import TrainingConfig, to_plain_data


def fit_torch_classifier(
    model: Any,
    train_data: tuple[np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray],
    *,
    training: TrainingConfig,
    run_dir: Path,
) -> list[dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _seed_everything(training.seed)
    device = _resolve_device(training.device)
    model.to(device)

    train_loader = DataLoader(
        _tensor_dataset(train_data),
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
    )
    valid_loader = DataLoader(
        _tensor_dataset(valid_data),
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(train_data[1], training, device))
    use_amp = bool(training.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, training.epochs + 1):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        valid_loss, valid_acc = _evaluate_loss(model, valid_loader, criterion, device=device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_acc,
        }
        history.append(row)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if training.patience > 0 and stale_epochs >= training.patience:
                break

    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "training": to_plain_data(training),
            "history": history,
        },
        run_dir / "model.pt",
    )
    return history


def predict_torch_classifier(
    model: Any,
    X: np.ndarray,
    *,
    training: TrainingConfig,
) -> np.ndarray:
    probabilities = predict_torch_probabilities(model, X, training=training)
    return probabilities.argmax(axis=1)


def predict_torch_probabilities(
    model: Any,
    X: np.ndarray,
    *,
    training: TrainingConfig,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = _resolve_device(training.device)
    model.to(device)
    model.eval()
    dataset = TensorDataset(torch.from_numpy(np.asarray(X, dtype="float32")))
    loader = DataLoader(dataset, batch_size=training.batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = _as_logits(model(xb.to(device)))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def _run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    *,
    device: Any,
    optimizer: Any,
    scaler: Any,
    use_amp: bool,
) -> tuple[float, float]:
    import torch

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = _as_logits(model(xb))
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(yb.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
        total_items += batch_size
    return total_loss / total_items, total_correct / total_items


def _evaluate_loss(
    model: Any,
    loader: Any,
    criterion: Any,
    *,
    device: Any,
) -> tuple[float, float]:
    import torch

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = _as_logits(model(xb))
            loss = criterion(logits, yb)
            batch_size = int(yb.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
            total_items += batch_size
    return total_loss / total_items, total_correct / total_items


def _tensor_dataset(data: tuple[np.ndarray, np.ndarray]) -> Any:
    import torch
    from torch.utils.data import TensorDataset

    X, y = data
    return TensorDataset(
        torch.from_numpy(np.asarray(X, dtype="float32")),
        torch.from_numpy(np.asarray(y, dtype="int64")),
    )


def _as_logits(output: Any) -> Any:
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.ndim == 2:
        return output
    if output.ndim > 2:
        while output.ndim > 2 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        if output.ndim > 2:
            output = output.flatten(start_dim=1)
    return output


def _resolve_device(requested: str) -> Any:
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _class_weights(y: np.ndarray, training: TrainingConfig, device: Any) -> Any:
    import torch

    if training.class_weight is None:
        return None
    if isinstance(training.class_weight, list):
        return torch.tensor(training.class_weight, dtype=torch.float32, device=device)
    if str(training.class_weight).lower() != "balanced":
        raise ValueError(f"Unsupported class_weight value: {training.class_weight!r}")

    labels, counts = np.unique(y, return_counts=True)
    n_classes = int(labels.max()) + 1
    weights = np.ones(n_classes, dtype="float32")
    total = float(counts.sum())
    for label, count in zip(labels, counts):
        weights[int(label)] = total / (n_classes * float(count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
