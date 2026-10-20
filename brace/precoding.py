"""RZF and uncertainty aware RZF (Eq. 6) precoders and the sum rate (Eq. 3)."""
import numpy as np
from .config import M, K


def rzf(Hh, rho, extra=0.0):
    """extra = sum_k sigma_hat_k^2 gives UA-RZF; extra = 0 gives standard RZF."""
    A = Hh.conj().T @ Hh + (K / rho + extra) * np.eye(M)
    W = np.linalg.solve(A, Hh.conj().T)
    return W * np.sqrt(rho / np.linalg.norm(W) ** 2)


def sum_rate(Htrue, W):
    G = np.abs(Htrue @ W) ** 2
    s = np.diag(G); i = G.sum(1) - s
    return float(np.sum(np.log2(1 + s / (i + 1.0))))
