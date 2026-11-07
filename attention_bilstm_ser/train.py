"""Train comparable LSTM, BiLSTM, and attention-BiLSTM models using actor holdout."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FEATURE_CACHE, ARTIFACT_DIR, LABELS, SAMPLE_RATE, SEED
from src.models import EmotionRNN, HAS_TORCH

if HAS_TORCH:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
else:
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def actor_disjoint_split(
    X: np.ndarray, y: np.ndarray, actors: np.ndarray, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return actor-disjoint (train_idx, val_idx, test_idx) indices (~70/15/15 split)."""
    unique_actors = np.unique(actors)
    if len(unique_actors) < 3:
        raise ValueError(f"Actor-disjoint split requires at least 3 distinct actors, got {len(unique_actors)}.")
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(split1.split(X, y, groups=actors))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed + 1)
    tr_rel, va_rel = next(split2.split(X[trainval_idx], y[trainval_idx], groups=actors[trainval_idx]))
    train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]
    return train_idx, val_idx, test_idx


def compute_balanced_weights(y_train: np.ndarray, num_classes: int = len(LABELS)) -> np.ndarray:
    """Compute class-balanced loss weights safely even if a subset of classes is present."""
    weights = np.ones(num_classes, dtype=np.float32)
    present = np.unique(y_train)
    if len(present) > 0:
        present_weights = compute_class_weight(class_weight="balanced", classes=present, y=y_train)
        for cls_idx, w in zip(present, present_weights):
            if 0 <= int(cls_idx) < num_classes:
                weights[int(cls_idx)] = float(w)
    return weights


def infer_lengths_from_features(X: np.ndarray) -> np.ndarray:
    """Infer active frame lengths from [N, T, D] standardized feature arrays."""
    n_samples, t_steps, _ = X.shape
    lengths = np.full(n_samples, t_steps, dtype=np.int64)
    for i in range(n_samples):
        nonzero = np.where(np.linalg.norm(X[i], axis=-1) > 1e-4)[0]
        if len(nonzero) > 0:
            lengths[i] = min(t_steps, max(1, int(nonzero[-1]) + 1))
    return lengths


def apply_spec_augment(xb: "torch.Tensor", time_mask_max: int = 16, freq_mask_max: int = 12) -> "torch.Tensor":
    """Apply SpecAugment time and feature-channel masking on training batches to reduce speaker overfitting."""
    xb_aug = xb.clone()
    _, t_steps, f_dim = xb_aug.shape
    if t_steps > time_mask_max and time_mask_max > 0:
        t_len = int(torch.randint(1, time_mask_max + 1, (1,)).item())
        t0 = int(torch.randint(0, t_steps - t_len, (1,)).item())
        xb_aug[:, t0 : t0 + t_len, :] = 0.0
    if f_dim > freq_mask_max and freq_mask_max > 0:
        f_len = int(torch.randint(1, freq_mask_max + 1, (1,)).item())
        f0 = int(torch.randint(0, f_dim - f_len, (1,)).item())
        xb_aug[:, :, f0 : f0 + f_len] = 0.0
    return xb_aug


def train_one(
    kind: str,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    device,
):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for model training. Install dependencies from requirements.txt.")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    len_tensor = torch.tensor(lengths, dtype=torch.long)

    tr = DataLoader(
        TensorDataset(X_tensor[train_idx], y_tensor[train_idx], len_tensor[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    va = DataLoader(
        TensorDataset(X_tensor[val_idx], y_tensor[val_idx], len_tensor[val_idx]),
        batch_size=args.batch_size,
    )
    te = DataLoader(
        TensorDataset(X_tensor[test_idx], y_tensor[test_idx], len_tensor[test_idx]),
        batch_size=args.batch_size,
    )

    model = EmotionRNN(input_dim=int(X.shape[-1]), hidden=args.hidden, num_classes=len(LABELS), kind=kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Class-balanced weights counter the 1:2 RAVDESS neutral class imbalance.
    cls_weights = compute_balanced_weights(y[train_idx], num_classes=len(LABELS))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cls_weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )

    best, best_state, wait = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb, lb in tr:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            xb_in = apply_spec_augment(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb_in, lengths=lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, lb in va:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                val_losses.append(criterion(model(xb, lengths=lb), yb).item())
        tl, vl = float(np.mean(losses)), float(np.mean(val_losses))
        scheduler.step(vl)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        print(f"{kind} epoch {epoch+1}/{args.epochs}: train={tl:.4f} val={vl:.4f}")
        if vl < best:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, _, lb in te:
            p = torch.softmax(model(xb.to(device), lengths=lb.to(device)), dim=1).cpu().numpy()
            probs.extend(p)
            preds.extend(p.argmax(1))
    truth = y[test_idx]
    metrics = {
        "accuracy": float(accuracy_score(truth, preds)),
        "macro_f1": float(f1_score(truth, preds, average="macro", zero_division=0)),
        "classification_report": classification_report(
            truth, preds, labels=np.arange(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0
        ),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "kind": kind,
            "input_dim": int(X.shape[-1]),
            "hidden": int(args.hidden),
            "labels": LABELS,
            "sample_rate": SAMPLE_RATE,
        },
        ARTIFACT_DIR / f"{kind}.pt",
    )
    np.savez_compressed(ARTIFACT_DIR / f"{kind}_test.npz", y=truth, pred=np.asarray(preds), probs=np.asarray(probs))
    with open(ARTIFACT_DIR / f"{kind}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        confusion_matrix(truth, preds, labels=np.arange(len(LABELS))),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"{kind}: test confusion matrix")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / f"{kind}_confusion.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set(xlabel="Epoch", ylabel="Loss", title=kind)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / f"{kind}_training.png", dpi=160)
    plt.close(fig)
    return {"model": kind, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(FEATURE_CACHE))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not Path(args.cache).is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {args.cache}. First download/extract RAVDESS and run "
            "`python prepare_data.py`; training cannot run on the synthetic demo clips."
        )
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to run train.py. Install requirements.txt first.")
    seed_all(args.seed)
    data = np.load(args.cache)
    X, y, actors = data["X"], data["y"], data["actors"]
    lengths = data["lengths"] if "lengths" in data.files else infer_lengths_from_features(X)
    train_idx, val_idx, test_idx = actor_disjoint_split(X, y, actors, seed=args.seed)
    print(
        f"Utterances: {len(y)}; actors train/val/test: "
        f"{len(set(actors[train_idx]))}/{len(set(actors[val_idx]))}/{len(set(actors[test_idx]))}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for kind in ("lstm", "bilstm", "attention_bilstm"):
        results.append(train_one(kind, X, y, lengths, train_idx, val_idx, test_idx, args, device))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
