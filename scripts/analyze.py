"""Compute every number reported in the paper from the stored drops and
write results/results.json.

  python scripts/analyze.py
"""
import os, json, glob
import numpy as np
from _common import load, RESULTS
from brace.metrics import nmse_db, rates, paired
from brace.precoding import rzf, sum_rate

out = {}
f = lambda n: os.path.join(RESULTS, n)
r2 = lambda x: round(float(x), 2)

if os.path.exists(f('geometric_speed.pkl')):
    S = load(f('geometric_speed.pkl')); meths = ['hold', 'ar16', 'ar64', 'gru_ant', 'brace']
    out['nmse_vs_speed_db'] = {v: {m: r2(nmse_db(S[v], m)) for m in meths} for v in S}
    out['rate_vs_speed_20dB'] = {v: {k: r2(x.mean()) for k, x in rates(S[v]).items()} for v in S}
    if 120 in S:
        out['rate_vs_snr_120kmh'] = {s: {k: r2(x.mean()) for k, x in rates(S[120], s).items()}
                                     for s in [0, 5, 10, 15, 20, 25, 30]}
    cv = np.mean([V['brace'].sum() for v in S for H, P, V in S[v]])
    out['ablation_20dB'] = {}
    for v in S:
        r = rates(S[v]); rc = []
        for H, P, V in S[v]:
            rc.append(sum_rate(H, rzf(P['brace'], 100.0, cv)))
        rc = np.array(rc)
        out['ablation_20dB'][v] = {
            'gru_ant': r2(r['gru_ant'].mean()), 'ar64': r2(r['ar64'].mean()),
            'brace_rzf': r2(r['brace'].mean()), 'brace_ua_const': r2(rc.mean()),
            'brace_ua': r2(r['brace_ua'].mean()),
            'ua_gain_mean_se': [round(x, 3) for x in paired(r['brace_ua'], r['brace'])],
            'adaptive_vs_const_mean_se': [round(x, 3) for x in paired(r['brace_ua'], rc)],
            'ua_win_fraction': r2((r['brace_ua'] > r['brace']).mean())}
    cal = {}
    for v in S:
        vp = np.concatenate([V['brace'] for H, P, V in S[v]])
        vt = np.concatenate([np.mean(np.abs(P['brace'] - H) ** 2, 1) for H, P, V in S[v]])
        rk = lambda x: np.argsort(np.argsort(x))
        cal[v] = {'pred': round(vp.mean(), 3), 'true': round(vt.mean(), 3),
                  'ratio': r2(vt.mean() / vp.mean()), 'spearman': r2(np.corrcoef(rk(vp), rk(vt))[0, 1])}
    out['calibration'] = cal

if os.path.exists(f('geometric_horizon.pkl')):
    Hz = load(f('geometric_horizon.pkl'))
    out['nmse_vs_horizon_db_120kmh'] = {t * 0.25: {m: r2(nmse_db(Hz[t], m)) for m in
                                        ['hold', 'ar16', 'ar64', 'gru_ant', 'brace']} for t in Hz}

for scen in ['LOS', 'NLOS']:
    if os.path.exists(f(f'uma_{scen}.pkl')):
        R = load(f(f'uma_{scen}.pkl')); out[f'uma_{scen}'] = {}
        for v, dl in R.items():
            r = rates(dl)
            out[f'uma_{scen}'][v] = {'nmse_db': {m: r2(nmse_db(dl, m)) for m in dl[0][1]},
                                     'rate': {k: r2(x.mean()) for k, x in r.items()},
                                     'ua_gain_mean_se': [round(x, 3) for x in paired(r['brace_ua'], r['brace'])]}

if os.path.exists(f('uma_bsweep.pkl')):
    Bs = load(f('uma_bsweep.pkl'))
    out['uma_bsweep_nlos_120kmh'] = {b: {'nmse_db': r2(nmse_db(dl, 'brace')),
                                         'rate_ua': r2(rates(dl, perfect=False)['brace_ua'].mean())} for b, dl in Bs.items()}

for p in sorted(glob.glob(f('uma_adaptive_*.pkl'))):
    eta = p.split('_')[-1][:-4]; A = load(p); key = f'uma_adaptive_eta{eta}'; out[key] = {}
    for (scen, v), dl in A.items():
        r = rates(dl, perfect=False); e = {'nmse_db': r2(nmse_db(dl, 'brace')), 'rate_ua': r2(r['brace_ua'].mean())}
        base = f(f'uma_{scen}.pkl')
        if os.path.exists(base):
            ref = rates(load(base)[v], perfect=False)
            for m in ['ar64', 'gru_ant']:
                e[f'vs_{m}_mean_se'] = [round(x, 2) for x in paired(r['brace_ua'], ref[m])]
        out[key][f'{scen}_{v}'] = e

with open(f('results.json'), 'w') as fh:
    json.dump(out, fh, indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:3000])
