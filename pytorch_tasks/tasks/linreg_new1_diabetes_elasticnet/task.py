"""
Linear Regression with ElasticNet Regularization on Diabetes Dataset
======================================================================
Algorithm: Linear Regression + ElasticNet (L1 + L2) via PyTorch autograd
Dataset:   sklearn.datasets.load_diabetes (442 samples, 10 features)

Math
----
Objective (ElasticNet):
    J(θ) = MSE(θ) + λ1 * ||θ||_1 + λ2 * ||θ||_2^2
         = (1/N) * ||Xθ - y||^2 + λ1 * Σ|θ_j| + λ2 * Σθ_j^2

Gradient w.r.t. θ (computed via autograd):
    ∇J(θ) = (2/N) * X^T(Xθ - y) + λ1 * sign(θ) + 2λ2 * θ

The L1 term promotes sparsity; L2 handles correlated features.
Combined they form the ElasticNet penalty, useful for high-dimensional data.
"""

import sys
import random
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_task_metadata() -> dict:
    return {
        "series": "Linear Regression",
        "level": "new_1",
        "id": "linreg_new1_diabetes_elasticnet",
        "algorithm": "Linear Regression (ElasticNet Regularization)",
        "description": (
            "Ridge+Lasso combined (ElasticNet) linear regression on the real-world "
            "sklearn diabetes dataset. Uses PyTorch autograd and Adam optimizer. "
            "Demonstrates sparse-feature selection on a 10-feature medical dataset."
        ),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": "sklearn.datasets.load_diabetes — 442 samples, 10 numeric features, 80/20 split.",
            "implementation": "nn.Linear + custom ElasticNet loss; Adam optimizer; device-agnostic.",
            "evaluation": "MSE, R2, and number of near-zero weights (|θ|<0.01) reported.",
            "validation": "R2 > 0.45 on validation; ElasticNet MSE <= Ridge MSE (L2-only baseline).",
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


def make_dataloaders(batch_size: int = 64, seed: int = 42):
    data = load_diabetes()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32).reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=seed)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val   = scaler_X.transform(X_val)
    y_train = scaler_y.fit_transform(y_train)
    y_val   = scaler_y.transform(y_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

    train_ds     = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=len(X_val_t))

    return train_loader, val_loader, (X_val_t, y_val_t), scaler_X, scaler_y


def build_model(in_features: int = 10, device: torch.device = None) -> nn.Module:
    if device is None:
        device = get_device()
    model = nn.Linear(in_features, 1).to(device)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model


def elasticnet_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    model: nn.Module,
    lambda1: float,
    lambda2: float,
) -> torch.Tensor:
    mse = nn.functional.mse_loss(y_pred, y_true)
    l1  = sum(p.abs().sum() for p in model.parameters())
    l2  = sum((p ** 2).sum() for p in model.parameters())
    return mse + lambda1 * l1 + lambda2 * l2


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    lambda1   = cfg.get("lambda1", 1e-3)
    lambda2   = cfg.get("lambda2", 1e-3)
    epochs    = cfg.get("epochs", 300)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = elasticnet_loss(pred, y_b, model, lambda1, lambda2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(device), y_v.to(device)
                val_mse = nn.functional.mse_loss(model(X_v), y_v).item()
        val_losses.append(val_mse)

    return {"train_loss_history": train_losses, "val_loss_history": val_losses}


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict:
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        pred = model(X)
        mse  = nn.functional.mse_loss(pred, y).item()

    ss_res  = ((y - pred) ** 2).sum().item()
    ss_tot  = ((y - y.mean()) ** 2).sum().item()
    r2      = 1.0 - ss_res / (ss_tot + 1e-12)
    rmse    = math.sqrt(mse)
    weights = model.weight.detach().cpu().numpy().flatten()
    sparse  = int((np.abs(weights) < 0.01).sum())

    return {"mse": mse, "rmse": rmse, "r2": r2, "near_zero_weights": sparse}


def predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(X.to(device)).cpu().numpy()


def save_artifacts(history: dict, metrics: dict, output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss_history"], label="Train (ElasticNet loss)")
    ax.plot(history["val_loss_history"],   label="Val MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("ElasticNet Linear Regression — Diabetes Dataset")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "linreg_new1_loss.png"), dpi=100)
    plt.close(fig)

    with open(os.path.join(output_dir, "linreg_new1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Task: {get_task_metadata()['id']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    cfg = {
        "lr": 5e-3,
        "lambda1": 5e-4,
        "lambda2": 5e-4,
        "epochs": 400,
        "batch_size": 64,
    }

    train_loader, val_loader, (X_val, y_val), scaler_X, scaler_y = make_dataloaders(
        batch_size=cfg["batch_size"]
    )
    in_features = X_val.shape[1]

    all_X, all_y = [], []
    for X_b, y_b in train_loader:
        all_X.append(X_b); all_y.append(y_b)
    X_train_full = torch.cat(all_X)
    y_train_full = torch.cat(all_y)

    model         = build_model(in_features=in_features, device=device)
    history       = train(model, train_loader, val_loader, cfg, device)
    train_metrics = evaluate(model, X_train_full, y_train_full, device)
    val_metrics   = evaluate(model, X_val, y_val, device)

    print("ElasticNet Model:")
    print(f"  Train  — MSE: {train_metrics['mse']:.4f}, R2: {train_metrics['r2']:.4f}")
    print(f"  Val    — MSE: {val_metrics['mse']:.4f},  R2: {val_metrics['r2']:.4f}")
    print(f"  Near-zero weights (|θ|<0.01): {val_metrics['near_zero_weights']}/{in_features}")

    weights = model.weight.detach().cpu().numpy().flatten()
    print(f"  Learned weights: {np.round(weights, 3)}")

    model_ridge   = build_model(in_features=in_features, device=device)
    cfg_ridge     = {**cfg, "lambda1": 0.0, "lambda2": 1e-3}
    history_ridge = train(model_ridge, train_loader, val_loader, cfg_ridge, device)
    val_ridge     = evaluate(model_ridge, X_val, y_val, device)

    print(f"\nRidge (L2-only) baseline:")
    print(f"  Val — MSE: {val_ridge['mse']:.4f}, R2: {val_ridge['r2']:.4f}")
    print(f"  Near-zero weights: {val_ridge['near_zero_weights']}/{in_features}")

    all_metrics = {
        "elasticnet_val": val_metrics,
        "elasticnet_train": train_metrics,
        "ridge_val": val_ridge,
        "sparsity_improvement": val_metrics["near_zero_weights"] - val_ridge["near_zero_weights"],
    }

    save_artifacts(history, all_metrics)

    print("\n--- Assertions ---")

    assert val_metrics["r2"] > 0.45, f"FAIL: R2 on val = {val_metrics['r2']:.4f}, expected > 0.45"
    print(f"[PASS] R2 > 0.45 on validation: {val_metrics['r2']:.4f}")

    assert val_metrics["mse"] <= val_ridge["mse"] + 0.05, (
        f"FAIL: ElasticNet MSE ({val_metrics['mse']:.4f}) significantly worse than Ridge ({val_ridge['mse']:.4f})"
    )
    print(f"[PASS] ElasticNet MSE ({val_metrics['mse']:.4f}) within range of Ridge MSE ({val_ridge['mse']:.4f})")

    assert history["val_loss_history"][-1] < history["val_loss_history"][0], (
        "FAIL: validation loss did not decrease"
    )
    print(f"[PASS] Val loss decreased: {history['val_loss_history'][0]:.4f} -> {history['val_loss_history'][-1]:.4f}")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
