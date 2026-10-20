"""Train one predictor.

  python scripts/train.py --channel geometric --domain beam --tau 4
  python scripts/train.py --channel uma --domain ant --tau 4

domain 'beam' = BRACE (heteroscedastic GRU on beamspace series)
domain 'ant'  = antenna-domain GRU baseline (MSE)
Seeds match the paper: geometric 1000+tau, uma 3800.
"""
import argparse, time
import numpy as np
from _common import save, model_path
from brace.channels import GENERATORS
from brace.data import make_train
from brace.gru import train_gru, n_params
from brace.config import TRAIN_ITERS

ap = argparse.ArgumentParser()
ap.add_argument('--channel', choices=['geometric', 'uma'], default='geometric')
ap.add_argument('--domain', choices=['beam', 'ant'], default='beam')
ap.add_argument('--tau', type=int, default=4, help='horizon in slots (0.25 ms each)')
ap.add_argument('--iters', type=int, default=TRAIN_ITERS)
a = ap.parse_args()

seed = 1000 + a.tau if a.channel == 'geometric' else 3800
rng = np.random.default_rng(seed)
nreal = 6000 if a.domain == 'beam' else 1875          # 60,000 series either way
X, Y = make_train(nreal, a.tau, rng, a.domain, GENERATORS[a.channel])
t0 = time.time()
p = train_gru(X, Y, a.domain == 'beam', rng, iters=a.iters)
save(p, model_path(a.channel, a.domain, a.tau))
print(f'{a.channel} {a.domain} tau={a.tau}: {X.shape[0]} series, '
      f'{n_params(p)} parameters, {time.time() - t0:.0f} s')
