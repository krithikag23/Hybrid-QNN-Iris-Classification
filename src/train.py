# src/train.py

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from .data import load_iris_petal
from .hybrid_qnn import HybridQNN


def train_model(epochs=20, lr=0.01, batch_size=16):
    device = torch.device("cpu")

    X_train, X_test, y_train, y_test, _, _ = load_iris_petal()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = HybridQNN(n_features=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test))
            preds = (probs > 0.5).float()
            acc = accuracy_score(y_test.cpu(), preds.cpu())
            print(f"Epoch {epoch+1:02d}: Test Acc = {acc:.3f}")

    torch.save(model.state_dict(), "hybrid_qnn_iris.pt")
    print("Model saved → hybrid_qnn_iris.pt")


if __name__ == "__main__":
    train_model()
