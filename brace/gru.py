"""Tiny GRU predictor implemented with autograd (Sec. 4.2).

Heteroscedastic head: outputs (Re mu, Im mu, log variance) trained with the
complex Gaussian NLL of Eq. (4). Point head: (Re mu, Im mu) with MSE.
"""
import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.misc.optimizers import adam
from .config import HIDDEN as H, TRAIN_ITERS, TRAIN_LR, TRAIN_BATCH


def init_params(rng, nin=2, nout=3):
    s = 0.2
    p = {}
    for g in ['z', 'r', 'h']:
        p['W' + g] = rng.standard_normal((nin, H)) * s
        p['U' + g] = rng.standard_normal((H, H)) * s / np.sqrt(H) * 2
        p['b' + g] = np.zeros(H)
    p['Wo'] = rng.standard_normal((H, nout)) * 0.1
    p['bo'] = np.zeros(nout)
    return p


def _sig(x):
    return 0.5 * (anp.tanh(0.5 * x) + 1)


def gru_forward(p, X):
    """X: (batch, steps, 2) -> (batch, nout)"""
    h = anp.zeros((X.shape[0], H))
    for t in range(X.shape[1]):
        x = X[:, t, :]
        z = _sig(anp.dot(x, p['Wz']) + anp.dot(h, p['Uz']) + p['bz'])
        r = _sig(anp.dot(x, p['Wr']) + anp.dot(h, p['Ur']) + p['br'])
        hh = anp.tanh(anp.dot(x, p['Wh']) + anp.dot(r * h, p['Uh']) + p['bh'])
        h = (1 - z) * h + z * hh
    return anp.dot(h, p['Wo']) + p['bo']


def to_feats(series):
    """Normalize complex series (N, steps): RMS scaling and phase rotation
    so that the newest sample is real and positive (Sec. 4.1)."""
    sc = np.sqrt(np.mean(np.abs(series) ** 2, axis=1, keepdims=True)) + 1e-9
    s = series / sc
    ph = np.exp(-1j * np.angle(s[:, -1:]))
    s = s * ph
    return np.stack([s.real, s.imag], axis=-1), sc[:, 0], ph[:, 0]


def train_gru(Xs, Ys, hetero, rng, iters=TRAIN_ITERS, bs=TRAIN_BATCH, lr=TRAIN_LR):
    p = init_params(rng, 2, 3 if hetero else 2)
    N = Xs.shape[0]
    idx_all = [rng.integers(0, N, bs) for _ in range(iters)]

    def loss(p, it):
        idx = idx_all[it]
        out = gru_forward(p, Xs[idx]); y = Ys[idx]
        e2 = (out[:, 0] - y[:, 0]) ** 2 + (out[:, 1] - y[:, 1]) ** 2
        if hetero:
            lv = out[:, 2]
            return anp.mean(e2 * anp.exp(-lv) + lv)
        return anp.mean(e2)
    return adam(grad(loss), p, step_size=lr, num_iters=iters)


def predict_series(p, series, hetero):
    """series: (N, steps) complex -> predicted value (N,), variance (N,) or None"""
    X, sc, ph = to_feats(series)
    out = np.asarray(gru_forward(p, X))
    y = (out[:, 0] + 1j * out[:, 1]) / ph * sc
    var = np.exp(out[:, 2]) * sc ** 2 if hetero else None
    return y, var


def n_params(p):
    return int(sum(v.size for v in p.values()))
