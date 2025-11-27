# src/data.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_iris_petal(test_size=0.2, random_state=42):
    iris = load_iris()
    X = iris.data[:, 2:4]  # Petal Length & Petal Width
    y = iris.target

    # Reduce to binary classification: class 0 vs others
    y = np.where(y == 0, 0, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, iris.target_names
