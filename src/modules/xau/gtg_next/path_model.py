from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class KNNPathModel:
    """Path analogue model built on scikit-learn, not a custom KNN engine."""

    def __init__(self, n_neighbors: int = 64) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self._targets: np.ndarray | None = None

    def fit(self, x: np.ndarray, y_path: np.ndarray) -> "KNNPathModel":
        x = np.asarray(x, dtype=float)
        y_path = np.asarray(y_path, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be [samples, features]")
        if y_path.ndim != 2 or y_path.shape[1] != 2:
            raise ValueError("y_path must be [samples, 2] => MFE_ATR, MAE_ATR")
        if len(x) != len(y_path):
            raise ValueError("x and y_path sample counts differ")
        if len(x) < self.n_neighbors:
            raise ValueError("not enough samples for configured n_neighbors")

        z = self.scaler.fit_transform(x)
        self.nn.fit(z)
        self._targets = y_path.copy()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._targets is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z = self.scaler.transform(x)
        _, indices = self.nn.kneighbors(z, return_distance=True)
        # Robust analogue estimate: median path of nearest historical states.
        return np.median(self._targets[indices], axis=1)
