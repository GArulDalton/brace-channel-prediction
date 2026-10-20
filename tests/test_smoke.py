"""Fast end-to-end check (about 30 s): python -m pytest tests  or  python tests/test_smoke.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from brace.config import M, K, NH, NAR
from brace.channels import gen_user_geometric, gen_user_uma
from brace.data import make_train
from brace.gru import train_gru, n_params
from brace.beamspace import brace_predict, top_beams_adaptive
from brace.baselines import ar_predict
from brace.precoding import rzf, sum_rate


def test_pipeline():
    rng = np.random.default_rng(0)
    for gen in (gen_user_geometric, gen_user_uma):
        h, hn = gen(120, 20, rng)
        assert h.shape == (20, M) and abs(np.mean(np.abs(h) ** 2) - 1) < 0.6
    X, Y = make_train(50, 4, rng, 'beam', gen_user_geometric)
    p = train_gru(X, Y, True, rng, iters=20)
    assert n_params(p) == 3459
    H, Hh, V = [], [], []
    for k in range(K):
        h, hn = gen_user_geometric(120, NAR + 5, rng)
        hist = hn[NAR - NH:NAR]
        hh, v = brace_predict(p, hist)
        hh2, v2 = brace_predict(p, hist, lambda x: top_beams_adaptive(x, 0.95))
        assert np.isfinite(ar_predict(hn[:NAR], 4)).all() and v >= 0 and v2 >= 0
        H.append(h[NAR - 1 + 4]); Hh.append(hh); V.append(v)
    W = rzf(np.array(Hh), 100.0, sum(V))
    assert abs(np.linalg.norm(W) ** 2 - 100.0) < 1e-6
    assert sum_rate(np.array(H), W) > 0


if __name__ == '__main__':
    test_pipeline(); print('smoke test passed')
