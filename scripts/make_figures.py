"""Draw Figs. 1 to 6 (600 dpi PNG and TIFF, sans-serif, no in-figure titles).

  python scripts/make_figures.py                 # from results/paper_results.json
  python scripts/make_figures.py --source run    # from results/results.json (your run)
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _common import RESULTS, ROOT

ap = argparse.ArgumentParser(); ap.add_argument('--source', choices=['paper', 'run'], default='paper')
a = ap.parse_args()
R = json.load(open(os.path.join(RESULTS, 'paper_results.json' if a.source == 'paper' else 'results.json')))
OUT = os.path.join(ROOT, 'figures'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': 0.8, 'lines.linewidth': 1.3})
DPI = 600
NAME = {'perfect': 'Perfect CSI', 'hold': 'Sample-and-hold', 'ar16': 'AR-16', 'ar64': 'AR-64', 'gru_ant': 'Antenna GRU',
        'brace': 'BRACE', 'brace_ua': 'BRACE + UA-RZF'}
ST = {'perfect': ('', '-', 'black'), 'hold': ('s', ':', '#7f7f7f'), 'ar16': ('v', '--', '#8c6d1f'), 'ar64': ('^', '--', '#2ca02c'),
      'gru_ant': ('D', '-.', '#9467bd'), 'brace': ('o', '-', '#1f77b4'), 'brace_ua': ('*', '-', '#d62728')}


def save(fig, n):
    fig.savefig(os.path.join(OUT, f'Fig{n}.png'), dpi=DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, f'Fig{n}.tif'), dpi=DPI, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def line(d, keys, xl, yl, n, rate=False):
    xs = sorted(d, key=float); fig, ax = plt.subplots(figsize=(4.6, 3.7))
    for m in keys:
        if m not in d[xs[0]]: continue
        mk, ls, c = ST[m]
        lab = 'BRACE + RZF' if (rate and m == 'brace') else NAME[m]
        ax.plot([float(x) for x in xs], [d[x][m] for x in xs], marker=mk, ls=ls, color=c, ms=5, label=lab,
                mfc='white' if m in ('ar16', 'hold') else c)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3, lw=.5)
    ax.legend(fontsize=7.5, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); save(fig, n)


# Fig. 1 block diagram
labs = ["Noisy SRS\npilots, last\n$N_h$ slots", "DFT to\nbeamspace,\nkeep top $B$", "Shared GRU\nper beam:\nmean, log var.",
        "Rebuild\n$\\hat{\\mathbf{h}}_k$ and\n$\\hat{\\sigma}_k^2$", "UA-RZF\nprecoder", "Downlink\nat $n+\\tau$"]
fig, ax = plt.subplots(figsize=(6.85, 1.4)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0.3, 1.7)
for i, l in enumerate(labs):
    x = 0.15 + i * 2.0
    ax.add_patch(FancyBboxPatch((x, 0.45), 1.7, 1.1, boxstyle="round,pad=0.05",
                                fc='#e6e6e6' if i in (2, 4) else 'white', ec='black', lw=0.9))
    ax.text(x + 0.85, 1.0, l, ha='center', va='center', fontsize=7)
    if i < 5:
        ax.annotate('', xy=(x + 2.0, 1.0), xytext=(x + 1.75, 1.0), arrowprops=dict(arrowstyle='->', lw=1.0))
save(fig, 1)

m5 = ['hold', 'ar16', 'ar64', 'gru_ant', 'brace']; m6 = ['perfect', 'hold', 'ar64', 'gru_ant', 'brace', 'brace_ua']
if 'nmse_vs_speed_db' in R: line(R['nmse_vs_speed_db'], m5, 'User speed (km/h)', 'NMSE (dB)', 2)
if 'nmse_vs_horizon_db_120kmh' in R: line(R['nmse_vs_horizon_db_120kmh'], m5, 'Prediction horizon (ms)', 'NMSE (dB)', 3)
if 'rate_vs_snr_120kmh' in R: line(R['rate_vs_snr_120kmh'], m6, 'Transmit SNR (dB)', 'Sum rate (bit/s/Hz)', 4, True)
if 'rate_vs_speed_20dB' in R: line(R['rate_vs_speed_20dB'], m6, 'User speed (km/h)', 'Sum rate (bit/s/Hz)', 5, True)

# Fig. 6 UMa bars
need = ['uma_LOS', 'uma_NLOS', 'uma_adaptive_eta0.95', 'uma_adaptive_eta0.99']
if all(k in R for k in need):
    meth = ['AR-64', 'Antenna GRU', 'BRACE, B = 10', 'BRACE-A, η = 0.95', 'BRACE-A, η = 0.99']
    cols = ['#2ca02c', '#9467bd', '#1f77b4', '#ff7f0e', '#d62728']; hat = ['', '//', '..', 'xx', '\\\\']
    fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.9), sharey=True)
    for ax, s, lab in zip(axs, ['LOS', 'NLOS'], ['(a) UMa LOS', '(b) UMa NLOS']):
        vs = ['30', '120', '350']; x = np.arange(3); w = 0.16
        vals = [[R[f'uma_{s}'][v]['rate'][m] for v in vs] for m in ['ar64', 'gru_ant', 'brace_ua']]
        vals += [[R[f'uma_adaptive_eta{e}'][f'{s}_{v}']['rate_ua'] for v in vs] for e in ['0.95', '0.99']]
        for j in range(5):
            ax.bar(x + (j - 2) * w, vals[j], w, label=meth[j], color=cols[j], hatch=hat[j], edgecolor='black', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels([v + ' km/h' for v in vs]); ax.grid(axis='y', alpha=.3, lw=.5); ax.set_xlabel(lab)
    axs[0].set_ylabel('Sum rate (bit/s/Hz)'); axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); save(fig, 6)
print('figures written to', OUT)
