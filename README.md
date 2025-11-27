# Hybrid QNN Iris Classification
This project implements a **Hybrid Quantum Neural Network (QNN)** using **PennyLane + PyTorch** to classify Iris flowers using **petal length and petal width**, and visualizes the model in two ways:

- A **2D decision boundary** of the quantum classifier
- A **Bloch sphere view** that maps prediction confidence to quantum states

## 🔍 Problem Overview

We use the classic **Iris dataset** (from `sklearn.datasets`) and reduce the task to:
- **Binary classification:**
