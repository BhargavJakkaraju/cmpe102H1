"""
Logistic Regression with L1 Regularization (Sparse Feature Selection)
=======================================================================
Algorithm: Binary Logistic Regression + L1 penalty via PyTorch autograd
Dataset:   Synthetic high-dimensional data (200 samples, 50 features)
           with only 5 truly predictive features (sparse ground truth)

Math
----
Sigmoid:
    σ(z) = 1 / (1 + e^{-z})

Binary Cross-Entropy (log-loss):
    BCE(ŷ, y) = -(1/N) Σ [ y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i) ]

L1-Regularized Objective (Lasso Logistic):
    J(θ) = BCE(σ(Xθ), y) + λ * ||θ||_1
         = BCE + λ * Σ_j |θ_j|

The L1 penalty drives irrelevant weights exactly to zero, automatically
performing feature selection. Gradient of L1 is sign(θ) (sub-gradient at 0).

Comparison: L2 (Ridge) shrinks all weights; L1 (Lasso) selects a sparse subset.
"""

import sys
import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_task_metadata() -> dict:
    return {
        "series": "Logistic Regression",
        "level": "new_3",
        "id": "logreg_new1_l1_sparse",
        "algorithm": "Logistic Regression (L1 / Lasso — Sparse Feature Selection)",
        "description": (
            "Binary logistic regression with L1 regularization on high-dimensional "
            "synthetic data (50 features, only 5 informative). Demonstrates automatic "
            "feature selection by comparing L1 vs L2 weight sparsity and recovery of "
            "the true informative features."
        ),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": "Synthetic: 200 samples, 50 features, 5 truly informative; 80/20 split.",
            "implementation": "nn.Linear + sigmoid; BCE + L1 penalty; Adam optimizer; compare L1 vs L2.",
            "evaluation": "Accuracy, Precision, Recall, F1, confusion matrix; weight sparsity; feature recovery rate.",
            "validation": "L1 accuracy > 0.85; L1 sparsity > L2 sparsity; L1 recovers ≥3/5 true features in top-5 weights.",
        },
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    n_samples: int = 200,
    n_features: int = 50,
    n_informative: int = 5,
    batch_size: int = 32,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    X               = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)
    true_w          = np.zeros(n_features, dtype=np.float32)
    true_w[informative_idx] = rng.choice([-2.0, -1.5, 1.5, 2.0], size=n_informative)

    logits = X @ true_w
    probs  = 1.0 / (1.0 + np.exp(-logits))
    y      = (probs > 0.5).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_val,   dtype=torch.float32)
    yv = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=len(Xv))

    return train_loader, val_loader, (Xv, yv), informative_idx.tolist()


def build_model(in_features: int = 50, device: torch.device = None) -> nn.Module:
    if device is None:
        device = get_device()
    model = nn.Linear(in_features, 1).to(device)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model


def reg_loss(
    y_pred_logit: torch.Tensor,
    y_true: torch.Tensor,
    model: nn.Module,
    reg_type: str,
    lam: float,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(y_pred_logit, y_true)
    if lam == 0.0:
        return bce
    w = model.weight
    penalty = w.abs().sum() if reg_type == "l1" else (w ** 2).sum()
    return bce + lam * penalty


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    reg_type  = cfg.get("reg_type", "l1")
    lam       = cfg.get("lambda", 1e-2)
    epochs    = cfg.get("epochs", 500)

    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        e_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = reg_loss(model(X_b), y_b, model, reg_type, lam)
            loss.backward()
            optimizer.step()
            e_losses.append(loss.item())
        train_losses.append(float(np.mean(e_losses)))

        model.eval()
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(device), y_v.to(device)
                val_bce  = nn.functional.binary_cross_entropy_with_logits(model(X_v), y_v).item()
        val_losses.append(val_bce)

    return {"train_loss_history": train_losses, "val_loss_history": val_losses}


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    with torch.no_grad():
        X, y   = X.to(device), y.to(device)
        logits = model(X)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= threshold).float()

    y_np    = y.cpu().numpy().flatten().astype(int)
    pred_np = preds.cpu().numpy().flatten().astype(int)

    acc      = float((preds == y).float().mean().item())
    precision = float(precision_score(y_np, pred_np, zero_division=0))
    recall   = float(recall_score(y_np, pred_np, zero_division=0))
    f1       = float(f1_score(y_np, pred_np, zero_division=0))
    cm       = confusion_matrix(y_np, pred_np).tolist()
    weights  = model.weight.detach().cpu().numpy().flatten()
    sparsity = float((np.abs(weights) < 0.01).mean())

    return {
        "accuracy": acc, "precision": precision,
        "recall": recall, "f1": f1,
        "confusion_matrix": cm, "sparsity": sparsity,
    }


def predict(
    model: nn.Module, X: torch.Tensor, device: torch.device, threshold: float = 0.5
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        return (torch.sigmoid(logits) >= threshold).cpu().numpy().astype(int)


def save_artifacts(
    histories: dict,
    all_metrics: dict,
    w_l1: np.ndarray,
    w_l2: np.ndarray,
    informative_idx: list,
    output_dir: str = "outputs",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for name, hist in histories.items():
        axes[0].plot(hist["val_loss_history"], label=name)
    axes[0].set_title("Val BCE Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    x = np.arange(len(w_l1))
    axes[1].bar(x, np.abs(w_l1), alpha=0.6, label="L1", color="tab:blue")
    axes[1].bar(x, np.abs(w_l2), alpha=0.6, label="L2", color="tab:orange")
    for idx in informative_idx:
        axes[1].axvline(idx, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].set_title("Weight Magnitudes (red = true features)")
    axes[1].set_xlabel("Feature index"); axes[1].legend(); axes[1].grid(alpha=0.3)

    top10 = np.argsort(np.abs(w_l1))[::-1][:10]
    axes[2].barh(range(10), np.abs(w_l1[top10]), color="tab:blue")
    axes[2].set_yticks(range(10)); axes[2].set_yticklabels([f"f{i}" for i in top10])
    axes[2].set_title("Top-10 L1 Weights"); axes[2].grid(alpha=0.3)

    fig.suptitle("Logistic Regression: L1 vs L2 Regularization (Sparse Feature Selection)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "logreg_new1_weights.png"), dpi=100)
    plt.close(fig)

    with open(os.path.join(output_dir, "logreg_new1_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Task: {get_task_metadata()['id']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    N_FEATURES    = 50
    N_INFORMATIVE = 5
    cfg_base = {"lr": 1e-3, "epochs": 600, "lambda": 5e-3}

    train_loader, val_loader, (X_val, y_val), informative_idx = make_dataloaders(
        n_features=N_FEATURES, n_informative=N_INFORMATIVE
    )
    print(f"True informative feature indices: {sorted(informative_idx)}\n")

    all_metrics   = {}
    all_histories = {}
    models        = {}

    for reg_type in ("l1", "l2"):
        set_seed(42)
        model   = build_model(in_features=N_FEATURES, device=device)
        cfg     = {**cfg_base, "reg_type": reg_type}
        history = train(model, train_loader, val_loader, cfg, device)
        metrics = evaluate(model, X_val, y_val, device)
        all_metrics[reg_type]   = metrics
        all_histories[reg_type] = history
        models[reg_type]        = model

        w   = model.weight.detach().cpu().numpy().flatten()
        top = set(np.argsort(np.abs(w))[::-1][:N_INFORMATIVE].tolist())
        rec = len(top & set(informative_idx))

        print(
            f"[{reg_type.upper()}]  "
            f"Acc={metrics['accuracy']:.3f}  F1={metrics['f1']:.3f}  "
            f"Sparsity={metrics['sparsity']:.2%}  "
            f"Feature recovery={rec}/{N_INFORMATIVE}"
        )

    w_l1 = models["l1"].weight.detach().cpu().numpy().flatten()
    w_l2 = models["l2"].weight.detach().cpu().numpy().flatten()

    top_l1      = set(np.argsort(np.abs(w_l1))[::-1][:N_INFORMATIVE].tolist())
    recovery_l1 = len(top_l1 & set(informative_idx))

    save_artifacts(all_histories, all_metrics, w_l1, w_l2, informative_idx)

    print("\n--- Assertions ---")

    acc_l1 = all_metrics["l1"]["accuracy"]
    assert acc_l1 > 0.85, f"FAIL: L1 accuracy = {acc_l1:.3f}, expected > 0.85"
    print(f"[PASS] L1 accuracy > 0.85: {acc_l1:.3f}")

    sp_l1 = all_metrics["l1"]["sparsity"]
    sp_l2 = all_metrics["l2"]["sparsity"]
    assert sp_l1 > sp_l2, f"FAIL: L1 sparsity ({sp_l1:.2%}) not > L2 sparsity ({sp_l2:.2%})"
    print(f"[PASS] L1 sparsity ({sp_l1:.2%}) > L2 sparsity ({sp_l2:.2%})")

    assert recovery_l1 >= 3, (
        f"FAIL: L1 recovered only {recovery_l1}/{N_INFORMATIVE} true features in top-5"
    )
    print(f"[PASS] L1 recovered {recovery_l1}/{N_INFORMATIVE} true features in top-5 weights")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
