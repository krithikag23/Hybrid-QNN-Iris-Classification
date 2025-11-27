# src/visualize_boundary.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from .data import load_iris_petal
from .hybrid_qnn import HybridQNN


def plot_qnn_decision_boundary(model_path="hybrid_qnn_iris.pt", save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test, scaler, target_names = load_iris_petal()

    # We will plot in original (scaled) 2D feature space
    all_X = np.vstack([X_train, X_test])
    x_min, x_max = all_X[:, 0].min() - 0.5, all_X[:, 0].max() + 0.5
    y_min, y_max = all_X[:, 1].min() - 0.5, all_X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Load model
    device = torch.device("cpu")
    model = HybridQNN(n_features=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
        logits = model(grid_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    Z = (probs > 0.5).astype(float).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.1, 0.5, 1.1], cmap="RdYlBu")

    # Training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", edgecolor="k", label="Train")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", marker="x", label="Test")

    plt.xlabel("Petal length (scaled)")
    plt.ylabel("Petal width (scaled)")
    plt.title("Hybrid QNN Decision Boundary on Iris (Petal Features)")
    plt.legend()

    out_path = os.path.join(save_dir, "decision_boundary_qnn.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved decision boundary plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_qnn_decision_boundary()
