"""3GPP TR 38.901 style UMa validation (Sec. 6.7).

  python scripts/evaluate_uma.py --job LOS
  python scripts/evaluate_uma.py --job NLOS
  python scripts/evaluate_uma.py --job bsweep            # fixed B at NLOS, 120 km/h
  python scripts/evaluate_uma.py --job adaptive --eta 0.95
  python scripts/evaluate_uma.py --job adaptive --eta 0.99
"""
import argparse, os
import numpy as np
from _common import load_model, save, RESULTS
from brace.config import K, NAR, NH
from brace.channels import gen_user_uma
from brace.baselines import ar_predict
from brace.gru import predict_series
from brace.beamspace import brace_predict, top_beams, top_beams_adaptive

TAU = 4


def drops(v, scen, n, seed, selector=top_beams, only_brace=False):
    m_uma = load_model('uma', 'beam', TAU)
    if not only_brace:
        m_geo = load_model('geometric', 'beam', TAU); m_ant = load_model('uma', 'ant', TAU)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        H, P, V = [], {}, {}
        for k in range(K):
            h, hn = gen_user_uma(v, NAR + TAU + 1, rng, scen); n0 = NAR - 1
            hist = hn[n0 - NH + 1:n0 + 1]; o = {}; vv = {}
            if not only_brace:
                o['hold'] = hn[n0]
                o['ar64'] = ar_predict(hn[n0 - NAR + 1:n0 + 1], TAU)
                o['gru_ant'] = predict_series(m_ant, hist.T, False)[0]
                o['brace_geo'], vv['brace_geo'] = brace_predict(m_geo, hist)
            o['brace'], vv['brace'] = brace_predict(m_uma, hist, selector)
            H.append(h[n0 + TAU])
            for m, x in o.items(): P.setdefault(m, []).append(x)
            for m, x in vv.items(): V.setdefault(m, []).append(x)
        out.append((np.array(H), {m: np.array(x) for m, x in P.items()},
                    {m: np.array(x) for m, x in V.items()}))
    return out


def seed_of(scen, v):
    return (1 if scen == 'LOS' else 2) * 1000 + v


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', choices=['LOS', 'NLOS', 'bsweep', 'adaptive'], required=True)
    ap.add_argument('--eta', type=float, default=0.95)
    ap.add_argument('--drops', type=int, default=100)
    a = ap.parse_args(); res = {}
    if a.job in ('LOS', 'NLOS'):
        for v in [30, 120, 200, 350]:
            res[v] = drops(v, a.job, a.drops, seed_of(a.job, v)); print(a.job, v, 'done')
        name = f'uma_{a.job}.pkl'
    elif a.job == 'bsweep':
        for b in [4, 6, 10, 16, 24]:
            res[b] = drops(120, 'NLOS', a.drops, 777, lambda h, b=b: top_beams(h, b), True); print('B', b, 'done')
        name = 'uma_bsweep.pkl'
    else:
        for scen in ['LOS', 'NLOS']:
            for v in [30, 120, 350]:
                res[(scen, v)] = drops(v, scen, a.drops, seed_of(scen, v),
                                       lambda h: top_beams_adaptive(h, a.eta), True)
                print(scen, v, 'done')
        name = f'uma_adaptive_{a.eta}.pkl'
    save(res, os.path.join(RESULTS, name))
