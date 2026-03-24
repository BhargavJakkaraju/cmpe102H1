"""
Linear Regression with Mini-Batch SGD on Wine Quality Dataset
==============================================================
Algorithm: Linear Regression using PyTorch nn.Module + SGD with momentum
Dataset:   Synthetic 12-feature wine-like multivariate regression data

Math
----
Model:   ŷ = Xθ + b

MSE loss: J(θ) = (1/N) Σ (ŷ_i - y_i)²

SGD with momentum update:
    v_t  = μ * v_{t-1} - lr * ∇J(θ_t)
    θ_{t+1} = θ_t + v_t

where μ is the momentum coefficient (e.g., 0.9).

Mini-batch gradient approximates full-batch gradient at fraction of cost:
    ∇J(θ) ≈ (2/B) X_B^T (X_B θ - y_B)  for batch B ⊂ {1..N}
"""

import sys
import os
import random
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_task_metadata() -> dict:
    return {
        "series": "Linear Regression",
        "level": "new_2",
        "id": "linreg_new2_minibatch_wine",
        "algorithm": "Linear Regression (Mini-Batch SGD with Momentum)",
        "description": (
            "Regression on a multi-feature wine-like dataset using mini-batch SGD "
            "with momentum. Compares three batch sizes (full-batch, mini-batch 32, "
            "mini-batch 8) to illustrate convergence behaviour. Device-agnostic PyTorch."
        ),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": "Synthetic 12-feature wine-like dataset (N=600): y = linear combination + noise; 80/20 split.",
            "implementation": "nn.Linear + optim.SGD(momentum=0.9); three batch-size variants; loss curves compared.",
            "evaluation": "MSE, RMSE, R2 on val split for each variant.",
            "validation": "Mini-batch-32 R2 > 0.85; loss monotonically trending down.",
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


def _make_synthetic_wine(n_samples: int = 600, n_features: int = 12, seed: int = 42):
    rng    = np.random.default_rng(seed)
    X      = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    true_w = rng.standard_normal(n_features).astype(np.float32)
    y      = X @ true_w + 0.5 * rng.standard_normal(n_samples).astype(np.float32)
    return X, y, true_w


def make_dataloaders(batch_size: int = 32, seed: int = 42):
    X, y, true_w = _make_synthetic_wine(seed=seed)
    y = y.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=seed)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train  = scaler_X.fit_transform(X_train)
    X_val    = scaler_X.transform(X_val)
    y_train  = scaler_y.fit_transform(y_train)
    y_val    = scaler_y.transform(y_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

    train_ds   = TensorDataset(X_train_t, y_train_t)
    val_ds     = TensorDataset(X_val_t,   y_val_t)

    loaders = {
        "full_batch":    DataLoader(train_ds, batch_size=len(train_ds), shuffle=True),
        "mini_batch_32": DataLoader(train_ds, batch_size=32, shuffle=True),
        "mini_batch_8":  DataLoader(train_ds, batch_size=8,  shuffle=True),
    }
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))

    return loaders, val_loader, (X_val_t, y_val_t), true_w


def build_model(in_features: int = 12, device: torch.device = None) -> nn.Module:
    if device is None:
        device = get_device()
    model = nn.Linear(in_features, 1).to(device)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.get("lr", 1e-2),
        momentum=cfg.get("momentum", 0.9),
    )
    criterion = nn.MSELoss()
    epochs    = cfg.get("epochs", 300)

    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        batch_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(device), y_v.to(device)
                val_losses.append(nn.functional.mse_loss(model(X_v), y_v).item())

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

    ss_res = ((y - pred) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    r2     = 1.0 - ss_res / (ss_tot + 1e-12)

    return {"mse": mse, "rmse": math.sqrt(mse), "r2": r2}


def predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(X.to(device)).cpu().numpy()


def save_artifacts(histories: dict, all_metrics: dict, output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"full_batch": "tab:blue", "mini_batch_32": "tab:orange", "mini_batch_8": "tab:green"}
    for name, hist in histories.items():
        axes[0].plot(hist["train_loss_history"], label=name, color=colors[name], alpha=0.8)
        axes[1].plot(hist["val_loss_history"],   label=name, color=colors[name], alpha=0.8)

    axes[0].set_title("Train MSE"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Val MSE");   axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("Mini-Batch SGD — Batch Size Comparison (Wine Regression)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "linreg_new2_loss.png"), dpi=100)
    plt.close(fig)

    with open(os.path.join(output_dir, "linreg_new2_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Task: {get_task_metadata()['id']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    cfg_base = {"lr": 1e-2, "momentum": 0.9, "epochs": 300}

    train_loaders, val_loader, (X_val, y_val), true_w = make_dataloaders()
    in_features = X_val.shape[1]

    all_metrics   = {}
    all_histories = {}

    for variant_name, t_loader in train_loaders.items():
        set_seed(42)
        model   = build_model(in_features=in_features, device=device)
        history = train(model, t_loader, val_loader, cfg_base, device)
        metrics = evaluate(model, X_val, y_val, device)
        all_metrics[variant_name]   = metrics
        all_histories[variant_name] = history

        print(
            f"[{variant_name:>15}]  "
            f"MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}"
        )

    save_artifacts(all_histories, all_metrics)

    print("\n--- Assertions ---")

    mb32_r2 = all_metrics["mini_batch_32"]["r2"]
    assert mb32_r2 > 0.85, f"FAIL: mini_batch_32 R2 = {mb32_r2:.4f}, expected > 0.85"
    print(f"[PASS] mini_batch_32 R2 > 0.85: {mb32_r2:.4f}")

    for name, hist in all_histories.items():
        first_half  = np.mean(hist["val_loss_history"][:50])
        second_half = np.mean(hist["val_loss_history"][-50:])
        assert second_half < first_half, (
            f"FAIL: [{name}] val loss not trending down ({first_half:.4f} -> {second_half:.4f})"
        )
        print(f"[PASS] [{name}] val loss trending down: {first_half:.4f} -> {second_half:.4f}")

    mse_mb = all_metrics["mini_batch_32"]["mse"]
    mse_fb = all_metrics["full_batch"]["mse"]
    assert mse_mb <= mse_fb * 1.05, (
        f"FAIL: mini_batch_32 MSE ({mse_mb:.4f}) > full_batch MSE ({mse_fb:.4f}) by >5%"
    )
    print(f"[PASS] mini_batch_32 MSE ({mse_mb:.4f}) within 5% of full_batch MSE ({mse_fb:.4f})")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
