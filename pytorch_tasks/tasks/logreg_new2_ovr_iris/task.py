"""
Logistic Regression: One-vs-Rest (OvR) Multiclass — Iris Dataset
=================================================================
Algorithm: Binary Logistic Regression extended to multiclass via OvR strategy
Dataset:   sklearn.datasets.load_iris (150 samples, 4 features, 3 classes)

Math
----
Sigmoid (binary logistic unit):
    σ(z) = 1 / (1 + e^{-z}),   z = Xθ_k

For K classes, train K independent binary classifiers:
    Classifier k: "class k vs. all others"
    y_k^{(i)} = 1  if  label^{(i)} == k,  else  0

Binary Cross-Entropy for classifier k:
    L_k(θ_k) = -(1/N) Σ_i [ y_k^i log σ(z_k^i) + (1 - y_k^i) log(1 - σ(z_k^i)) ]

Prediction (argmax of K confidence scores):
    ŷ = argmax_k σ(x^T θ_k + b_k)

Compared to softmax regression (joint K-way):
  - OvR trains K models independently; parameters do not interact.
  - Softmax sums to 1 across classes; OvR scores are not calibrated probabilities.
  - OvR is simpler to implement and scale to many classes.
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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression as SklearnLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_task_metadata() -> dict:
    return {
        "series": "Logistic Regression",
        "level": "new_4",
        "id": "logreg_new2_ovr_iris",
        "algorithm": "Logistic Regression (One-vs-Rest Multiclass, Iris)",
        "description": (
            "Multiclass classification on the Iris dataset using a custom OvR "
            "wrapper around K binary PyTorch logistic regressors. Each binary model "
            "is trained independently with BCE loss. Final prediction = argmax of "
            "K sigmoid scores. Results compared to sklearn LogisticRegression."
        ),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": "sklearn.datasets.load_iris — 150 samples, 4 features, 3 classes; 80/20 split.",
            "implementation": "K binary nn.Linear classifiers; custom OvR predict(); Adam optimizer; no CrossEntropyLoss.",
            "evaluation": "Per-class accuracy, macro-F1, confusion matrix on val split; comparison with sklearn OvR LR.",
            "validation": "Macro-F1 > 0.88; PyTorch OvR within 5% accuracy of sklearn OvR.",
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


def make_dataloaders(batch_size: int = 16, seed: int = 42):
    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=len(X_val_t))

    return (
        train_loader, val_loader,
        (X_val_t, y_val_t),
        (X_val, y_val),
        int(y.max()) + 1,
        scaler,
    )


class OvRLogisticRegression(nn.Module):
    """
    One-vs-Rest Logistic Regression:
    K binary classifiers (nn.Linear(in, 1)) each trained on a binary label.
    Prediction = argmax of K sigmoid outputs.
    """

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.n_classes   = n_classes
        self.classifiers = nn.ModuleList(
            [nn.Linear(in_features, 1) for _ in range(n_classes)]
        )
        for clf in self.classifiers:
            nn.init.xavier_uniform_(clf.weight)
            nn.init.zeros_(clf.bias)

    def forward_logits(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cat([clf(X) for clf in self.classifiers], dim=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(X))

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(X).argmax(dim=1)


def build_model(in_features: int = 4, n_classes: int = 3, device: torch.device = None):
    if device is None:
        device = get_device()
    return OvRLogisticRegression(in_features, n_classes).to(device)


def train(
    model: OvRLogisticRegression,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-2))
    epochs    = cfg.get("epochs", 500)
    K         = model.n_classes

    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for X_b, y_b in train_loader:
            X_b   = X_b.to(device)
            y_bin = torch.zeros(len(y_b), K, device=device)
            for k in range(K):
                y_bin[:, k] = (y_b == k).float()

            optimizer.zero_grad()
            logits = model.forward_logits(X_b)
            loss   = nn.functional.binary_cross_entropy_with_logits(logits, y_bin)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v   = X_v.to(device)
                y_bin_v = torch.zeros(len(y_v), K, device=device)
                for k in range(K):
                    y_bin_v[:, k] = (y_v == k).float()
                logits_v = model.forward_logits(X_v)
                val_bce  = nn.functional.binary_cross_entropy_with_logits(logits_v, y_bin_v).item()
        val_losses.append(val_bce)

    return {"train_loss_history": train_losses, "val_loss_history": val_losses}


def evaluate(
    model: OvRLogisticRegression,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict:
    model.eval()
    X     = X.to(device)
    preds = model.predict(X).cpu().numpy()
    y_np  = y.numpy()

    acc      = float((preds == y_np).mean())
    macro_f1 = float(f1_score(y_np, preds, average="macro", zero_division=0))
    cm       = confusion_matrix(y_np, preds).tolist()

    per_class_acc = {}
    for k in range(model.n_classes):
        mask = y_np == k
        per_class_acc[f"class_{k}"] = float((preds[mask] == k).mean()) if mask.any() else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
    }


def predict(model: OvRLogisticRegression, X: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    return model.predict(X.to(device)).cpu().numpy()


def save_artifacts(
    history: dict,
    metrics_pytorch: dict,
    metrics_sklearn: dict,
    output_dir: str = "outputs",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss_history"], label="Train BCE (OvR sum)")
    axes[0].plot(history["val_loss_history"],   label="Val BCE (OvR sum)")
    axes[0].set_title("OvR Training Loss (Iris)"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    cm = np.array(metrics_pytorch["confusion_matrix"])
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1, 2]); axes[1].set_yticks([0, 1, 2])
    axes[1].set_xticklabels(["setosa", "versicolor", "virginica"], rotation=30)
    axes[1].set_yticklabels(["setosa", "versicolor", "virginica"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix — PyTorch OvR")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    fig.colorbar(im, ax=axes[1])

    fig.suptitle("Logistic Regression OvR — Iris Dataset")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "logreg_new2_ovr_iris.png"), dpi=100)
    plt.close(fig)

    combined = {"pytorch_ovr": metrics_pytorch, "sklearn_ovr": metrics_sklearn}
    with open(os.path.join(output_dir, "logreg_new2_metrics.json"), "w") as f:
        json.dump(combined, f, indent=2)


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Task: {get_task_metadata()['id']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    cfg = {"lr": 1e-2, "epochs": 1000}

    (
        train_loader, val_loader,
        (X_val_t, y_val_t),
        (X_val_np, y_val_np),
        n_classes, scaler,
    ) = make_dataloaders()

    in_features = X_val_t.shape[1]

    model      = build_model(in_features=in_features, n_classes=n_classes, device=device)
    history    = train(model, train_loader, val_loader, cfg, device)
    metrics_pt = evaluate(model, X_val_t, y_val_t, device)

    print("PyTorch OvR:")
    print(f"  Accuracy  : {metrics_pt['accuracy']:.4f}")
    print(f"  Macro-F1  : {metrics_pt['macro_f1']:.4f}")
    print(f"  Per-class : {metrics_pt['per_class_accuracy']}")

    all_X, all_y = [], []
    for X_b, y_b in train_loader:
        all_X.append(X_b.numpy()); all_y.append(y_b.numpy())
    X_train_np = np.concatenate(all_X)
    y_train_np = np.concatenate(all_y)

    sk_clf   = SklearnLR(max_iter=1000, random_state=42)
    sk_clf.fit(X_train_np, y_train_np)
    sk_preds = sk_clf.predict(X_val_np)

    sk_acc    = float((sk_preds == y_val_np).mean())
    sk_f1     = float(f1_score(y_val_np, sk_preds, average="macro", zero_division=0))
    sk_cm     = confusion_matrix(y_val_np, sk_preds).tolist()
    metrics_sk = {"accuracy": sk_acc, "macro_f1": sk_f1, "confusion_matrix": sk_cm}

    print(f"\nsklearn OvR LR:")
    print(f"  Accuracy  : {sk_acc:.4f}")
    print(f"  Macro-F1  : {sk_f1:.4f}")
    print(f"\nAccuracy gap (sklearn - pytorch): {sk_acc - metrics_pt['accuracy']:.4f}")

    save_artifacts(history, metrics_pt, metrics_sk)

    print("\n--- Assertions ---")

    macro_f1_pt = metrics_pt["macro_f1"]
    assert macro_f1_pt > 0.88, f"FAIL: PyTorch OvR macro-F1 = {macro_f1_pt:.4f}, expected > 0.88"
    print(f"[PASS] Macro-F1 > 0.88: {macro_f1_pt:.4f}")

    acc_gap = abs(metrics_pt["accuracy"] - sk_acc)
    assert acc_gap <= 0.05, (
        f"FAIL: Accuracy gap vs sklearn = {acc_gap:.4f}, expected <= 0.05"
    )
    print(f"[PASS] Accuracy gap vs sklearn within 5%: PyTorch={metrics_pt['accuracy']:.4f}, sklearn={sk_acc:.4f}")

    assert history["val_loss_history"][-1] < history["val_loss_history"][10], (
        "FAIL: validation loss did not decrease from epoch 10 to end"
    )
    print(f"[PASS] Val loss decreased: {history['val_loss_history'][10]:.4f} -> {history['val_loss_history'][-1]:.4f}")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
