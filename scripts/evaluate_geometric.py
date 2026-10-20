"""Speed and horizon sweeps on the geometric channel (Sec. 6.1 to 6.6).

  python scripts/evaluate_geometric.py --job speed     # 6 speeds, tau = 4, 150 drops
  python scripts/evaluate_geometric.py --job horizon   # 120 km/h, tau = 1, 2, 4, 8
"""
import argparse, os
import numpy as np
from _common import load_model, save, RESULTS
from brace.config import K, NAR, NH
from brace.channels import gen_user_geometric
from brace.baselines import ar_predict
from brace.gru import predict_series
from brace.beamspace import brace_predict


def drops(v, tau, n, seed):
    gru_ant = load_model('geometric', 'ant', tau); brace_m = load_model('geometric', 'beam', tau)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        H, P, V = [], {}, {'brace': []}
        for k in range(K):
            h, hn = gen_user_geometric(v, NAR + tau + 1, rng); n0 = NAR - 1
            hist = hn[n0 - NH + 1:n0 + 1]
            o = {'hold': hn[n0],
                 'ar64': ar_predict(hn[n0 - NAR + 1:n0 + 1], tau),
                 'ar16': ar_predict(hist, tau),
                 'gru_ant': predict_series(gru_ant, hist.T, False)[0]}
            o['brace'], var = brace_predict(brace_m, hist)
            H.append(h[n0 + tau]); V['brace'].append(var)
            for m, x in o.items():
                P.setdefault(m, []).append(x)
        out.append((np.array(H), {m: np.array(x) for m, x in P.items()},
                    {m: np.array(x) for m, x in V.items()}))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', choices=['speed', 'horizon'], required=True)
    ap.add_argument('--drops', type=int, default=150)
    ap.add_argument('--speeds', type=int, nargs='*', default=[30, 60, 120, 200, 280, 350])
    a = ap.parse_args(); res = {}
    if a.job == 'speed':
        for v in a.speeds:
            res[v] = drops(v, 4, a.drops, v); print('speed', v, 'done')
    else:
        for tau in [1, 2, 4, 8]:
            res[tau] = drops(120, tau, a.drops, 500 + tau); print('tau', tau, 'done')
    save(res, os.path.join(RESULTS, f'geometric_{a.job}.pkl'))
