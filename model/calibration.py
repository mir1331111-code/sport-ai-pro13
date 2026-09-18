"""model/calibration.py — Platt с hold-out и early stopping."""
from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np


class PlattCalibrator:
    """Платт по 3 исходам. Обучается на train, останавливается по val NLL."""

    def __init__(self, val_frac: float = 0.2, max_iter: int = 500,
                 lr: float = 0.01, reg: float = 0.01, patience: int = 20):
        self.a = np.ones(3)
        self.b = np.zeros(3)
        self.val_frac = val_frac
        self.max_iter = max_iter
        self.lr = lr
        self.reg = reg
        self.patience = patience
        self.trained = False
        self.val_nll: Optional[float] = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    @staticmethod
    def _nll(p: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def fit(self, logits: Sequence[float], outcomes: Sequence[float]) -> None:
        L = np.asarray(logits, dtype=float)
        O = np.asarray(outcomes, dtype=float)
        if L.size < 240 or L.size != O.size:
            return
        n_matches = L.size // 3
        L = L[:n_matches * 3].reshape(n_matches, 3)
        O = O[:n_matches * 3].reshape(n_matches, 3)
        n_val = max(20, int(n_matches * self.val_frac))
        tr = slice(0, n_matches - n_val)
        va = slice(n_matches - n_val, n_matches)

        for c in range(3):
            z, y = L[:, c], O[:, c]
            z_mu = z[tr].mean()
            z_sd = max(float(z[tr].std()), 1e-6)
            z_n = (z - z_mu) / z_sd
            a, b = 1.0, 0.0
            best_val = float("inf")
            best_a, best_b = a, b
            patience = self.patience
            for _ in range(self.max_iter):
                p_tr = self._sigmoid(a * z_n[tr] + b)
                err = p_tr - y[tr]
                ga = float((err * z_n[tr]).mean() + self.reg * a)
                gb = float(err.mean())
                a -= self.lr * ga
                b -= self.lr * gb
                val = self._nll(self._sigmoid(a * z_n[va] + b), y[va])
                if val < best_val - 1e-5:
                    best_val, best_a, best_b = val, a, b
                    patience = self.patience
                else:
                    patience -= 1
                    if patience <= 0:
                        break
            self.a[c] = best_a / z_sd
            self.b[c] = best_b - best_a * z_mu / z_sd
        self.trained = True
        self.val_nll = best_val

    def calibrate(self, p: float, ci: int = 0) -> float:
        if not self.trained:
            return p
        p = min(max(p, 1e-6), 1 - 1e-6)
        z = math.log(p / (1 - p))
        return float(self._sigmoid(np.array([self.a[ci] * z + self.b[ci]]))[0])
