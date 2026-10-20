"""Training data: noisy pilot histories and noisy future pilots as labels."""
import numpy as np
from .config import NH, F
from .beamspace import top_beams
from .gru import to_feats


def make_train(nreal, tau, rng, domain, gen):
    Xs, Ys = [], []
    T = NH + tau
    for _ in range(nreal):
        v = rng.uniform(30, 350)
        h, hn = gen(v, T, rng)
        hist = hn[:NH]; lab = hn[NH - 1 + tau]
        if domain == 'beam':
            idx, bs = top_beams(hist)
            ser = bs[:, idx].T
            tgt = (lab @ F.conj())[idx]
        else:
            ser = hist.T; tgt = lab
        X, sc, ph = to_feats(ser)
        y = tgt * ph / sc
        Xs.append(X); Ys.append(np.stack([y.real, y.imag], -1))
    return np.concatenate(Xs), np.concatenate(Ys)
