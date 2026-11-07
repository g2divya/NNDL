"""Train comparable LSTM, BiLSTM, and attention-BiLSTM models using actor holdout."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FEATURE_CACHE, ARTIFACT_DIR, LABELS, SEED
from src.models import EmotionRNN


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(kind, X, y, train_idx, val_idx, test_idx, args, device):
    tr = DataLoader(TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx])), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx])), batch_size=args.batch_size)
    te = DataLoader(TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx])), batch_size=args.batch_size)
    model = EmotionRNN(input_dim=X.shape[-1], hidden=args.hidden, num_classes=len(LABELS), kind=kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Class-balanced weights counter the 1:2 RAVDESS neutral class imbalance
    classes = np.arange(len(LABELS))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y[train_idx])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )

    best, best_state, wait = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
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
    model.load_state_dict(best_state)
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, _ in te:
            p = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            probs.extend(p)
            preds.extend(p.argmax(1))
    truth = y[test_idx]
    metrics = {
        "accuracy": float(accuracy_score(truth, preds)),
        "macro_f1": float(f1_score(truth, preds, average="macro")),
        "classification_report": classification_report(
            truth, preds, labels=np.arange(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0
        ),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "kind": kind,
            "input_dim": X.shape[-1],
            "hidden": args.hidden,
            "labels": LABELS,
            "sample_rate": 22050,
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
    seed_all(args.seed)
    data = np.load(args.cache)
    X, y, actors = data["X"], data["y"], data["actors"]
    # Actor-disjoint 70/15/15 split prevents speaker identity leakage across evaluation sets.
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    trainval_idx, test_idx = next(split1.split(X, y, groups=actors))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=args.seed + 1)
    tr_rel, va_rel = next(split2.split(X[trainval_idx], y[trainval_idx], groups=actors[trainval_idx]))
    train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]
    print(f"Utterances: {len(y)}; actors train/val/test: {len(set(actors[train_idx]))}/{len(set(actors[val_idx]))}/{len(set(actors[test_idx]))}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for kind in ("lstm", "bilstm", "attention_bilstm"):
        results.append(train_one(kind, X, y, train_idx, val_idx, test_idx, args, device))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
