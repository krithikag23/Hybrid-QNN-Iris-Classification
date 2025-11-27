# src/hybrid_qnn.py

import pennylane as qml
import torch
from torch import nn

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    # Encode features
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Entangle
    qml.CNOT(wires=[0, 1])

    # Variational layer
    for i in range(n_qubits):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)

    return qml.expval(qml.PauliZ(0))


class HybridQNN(nn.Module):
    def __init__(self, n_features=2):
        super().__init__()
        self.fc_in = nn.Linear(n_features, n_qubits)

        weight_shapes = {"weights": (n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.fc_out = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.tanh(self.fc_in(x))
        x = self.q_layer(x)
        x = self.fc_out(x)
        return x.squeeze(-1)
