"""AR baseline (Sec. 5.3): 4th-order AR per user, coefficients shared across
antennas and fitted by regularized least squares, iterated tau steps."""
import numpy as np


def ar_predict(hist, tau, p=4, lam=1e-3):
    Xr, yr = [], []
    for t in range(p, hist.shape[0]):
        Xr.append(hist[t - p:t][::-1].T)
        yr.append(hist[t])
    Xr = np.concatenate(Xr, 0); yr = np.concatenate(yr, 0)
    a = np.linalg.solve(Xr.conj().T @ Xr + lam * np.eye(p), Xr.conj().T @ yr)
    buf = list(hist[-p:])
    for _ in range(tau):
        buf.append(sum(a[i] * buf[-1 - i] for i in range(p)))
    return buf[-1]
