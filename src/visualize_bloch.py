# src/visualize_bloch.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from .data import load_iris_petal
from .hybrid_qnn import HybridQNN


def prob_to_bloch_coords(prob: float):
    """
    Map probability (0..1) to a point on the Bloch sphere.

    Interpret prob ~ 'confidence for class 1'.
    theta in [0, pi], phi = 0 for simplicity.
    """
    theta = prob * np.pi
    phi = 0.0

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def plot_bloch_points(points, labels, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.1)

    for (x, y, z), lbl in zip(points, labels):
        color = "tab:green" if lbl == 1 else "tab:blue"
        ax.scatter(x, y, z, color=color, s=40, alpha=0.9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Hybrid QNN Iris Classification – Bloch Sphere View")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="tab:green", label="Class 1 (non-setosa)"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="tab:blue", label="Class 0 (setosa)"),
    ]
    ax.legend(handles=handles, loc="upper left")

    out_path = os.path.join(save_dir, "bloch_sphere_qnn.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved Bloch sphere plot to {out_path}")
    plt.show()


def visualize_bloch(num_samples=30, model_path="hybrid_qnn_iris.pt"):
    device = torch.device("cpu")

    X_train, X_test, y_train, y_test, _, _ = load_iris_petal()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_np = np.array(y_test)

    model = HybridQNN(n_features=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Random subset of test points
    if num_samples > len(X_test_t):
        num_samples = len(X_test_t)
    idx = np.random.choice(len(X_test_t), size=num_samples, replace=False)

    X_sample = X_test_t[idx]
    y_sample = y_test_np[idx]

    with torch.no_grad():
        logits = model(X_sample)
        probs = torch.sigmoid(logits).cpu().numpy()

    points = [prob_to_bloch_coords(p) for p in probs]
    plot_bloch_points(points, y_sample)


if __name__ == "__main__":
    visualize_bloch()
