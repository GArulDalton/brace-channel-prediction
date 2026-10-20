"""NMSE and paired sum-rate statistics over stored drops."""
import numpy as np
from .precoding import rzf, sum_rate


def nmse_db(drops, m):
    num = sum(np.sum(np.abs(P[m] - H) ** 2) for H, P, V in drops)
    den = sum(np.sum(np.abs(H) ** 2) for H, P, V in drops)
    return float(10 * np.log10(num / den))


def rates(drops, snr_db=20, perfect=True):
    """Per-drop sum rates for every prediction (RZF) and every variance (UA-RZF)."""
    rho = 10 ** (snr_db / 10); r = {}
    for H, P, V in drops:
        if perfect:
            r.setdefault('perfect', []).append(sum_rate(H, rzf(H, rho)))
        for m in P:
            r.setdefault(m, []).append(sum_rate(H, rzf(P[m], rho)))
        for m in V:
            r.setdefault(m + '_ua', []).append(sum_rate(H, rzf(P[m], rho, V[m].sum())))
    return {k: np.array(v) for k, v in r.items()}


def paired(a, b):
    d = a - b
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))
