# Hybrid QNN Iris Classification
This project implements a **Hybrid Quantum Neural Network (QNN)** using **PennyLane + PyTorch** to classify Iris flowers using **petal length and petal width**, and visualizes the model in two ways:

- A **2D decision boundary** of the quantum classifier
- A **Bloch sphere view** that maps prediction confidence to quantum states

## 🔍 Problem Overview

We use the classic **Iris dataset** (from `sklearn.datasets`) and reduce the task to:
- **Binary classification:**
  - Class `0` → Setosa 
  - Class `1` → Versicolor + Virginica (combined)

Input features:

- Petal length  
- Petal width  

These are scaled and fed into a **hybrid classical–quantum model**.

---

## ⚛ Model Architecture

```text
Iris Petal Features (2D)
        ↓  (StandardScaler)
Linear Layer (2 → 2)
        ↓  (tanh)
2-Qubit Variational Quantum Circuit (PennyLane)
        ↓   expval(Z)
Linear Layer (1 → 1)
        ↓
Logit → Sigmoid → Binary class (0 / 1)
