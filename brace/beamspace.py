"""Beam selection and BRACE channel/error reconstruction (Secs. 4.1 and 4.3)."""
import numpy as np
from .config import F, M, B, S2
from .gru import predict_series


def top_beams(hist, b=B):
    """Fixed budget: the b strongest beams of the noisy history (NH x M)."""
    bs = hist @ F.conj()
    e = np.mean(np.abs(bs) ** 2, axis=0)
    return np.argsort(e)[::-1][:b], bs


def top_beams_adaptive(hist, eta, bmin=4, bmax=24):
    """BRACE-A: smallest beam set holding a fraction eta of the history energy."""
    bs = hist @ F.conj()
    e = np.mean(np.abs(bs) ** 2, 0)
    o = np.argsort(e)[::-1]; c = np.cumsum(e[o]) / e.sum()
    nb = int(np.clip(np.searchsorted(c, eta) + 1, bmin, bmax))
    return o[:nb], bs


def brace_predict(model, hist, selector=top_beams):
    """Returns predicted antenna-domain channel h_hat (M,) and per-entry
    error variance sigma_hat^2 (Eq. 5)."""
    idx, bs = selector(hist)
    y, v = predict_series(model, bs[:, idx].T, True)
    b = np.zeros(M, complex); b[idx] = y
    unsel = np.setdiff1d(np.arange(M), idx)
    leak = np.maximum(np.mean(np.abs(bs[:, unsel]) ** 2, 0) - S2, 0).sum()
    return b @ F.T, (np.maximum(v - S2, 0).sum() + leak) / M
