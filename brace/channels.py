"""Channel generators.

gen_user_geometric : L-path geometric model with per-path Doppler (Sec. 3, Table 1)
gen_user_uma       : narrowband 3GPP TR 38.901 style UMa LOS/NLOS model (Sec. 6.7)

Both return (h, h_noisy): true channel and noisy SRS estimates, shape (T, M).
"""
import numpy as np
from .config import M, L, AS_DEG, TS, LAM, S2


def steer(theta):
    m = np.arange(M)[:, None]
    return np.exp(1j * np.pi * m * np.sin(theta)[None, :])


def _add_pilot_noise(h, rng):
    n = np.sqrt(S2 / 2) * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    return h, h + n


def gen_user_geometric(v_kmh, T, rng):
    fd = (v_kmh / 3.6) / LAM
    c = rng.uniform(-np.pi / 3, np.pi / 3)
    th = c + np.deg2rad(rng.laplace(0, AS_DEG / np.sqrt(2), L))
    phi = rng.uniform(0, 2 * np.pi, L)
    f = fd * np.cos(phi)
    g = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2 * L)
    t = np.arange(T) * TS
    coef = g[None, :] * np.exp(1j * 2 * np.pi * f[None, :] * t[:, None])
    h = coef @ steer(th).T
    return _add_pilot_noise(h, rng)


# ---- TR 38.901 style UMa (parameter values should be checked against the spec) ----
ALPHA = np.array([0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195, 2.1551])
ALPHA = np.concatenate([ALPHA, -ALPHA])          # 20 intra-cluster ray offsets
FCG = 3.5
UMA = {
    'NLOS': dict(N=20, rtau=2.3, lgDS=(-6.28 - 0.204 * np.log10(FCG), 0.39),
                 lgASD=(1.5 - 0.1144 * np.log10(1 + FCG), 0.28),
                 cASD=2.0, cASA=15.0, zeta=3.0, Cphi=1.289),
    'LOS': dict(N=12, rtau=2.5, lgDS=(-6.955 - 0.0963 * np.log10(FCG), 0.66),
                lgASD=(1.06 + 0.1114 * np.log10(FCG), 0.28),
                cASD=5.0, cASA=11.0, zeta=3.0, Cphi=1.146, K=(9.0, 3.5)),
}


def gen_user_uma(v_kmh, T, rng, scen=None):
    """scen in {'LOS','NLOS'}; None draws LOS/NLOS with probability 0.5 (training mix)."""
    if scen is None:
        scen = 'LOS' if rng.random() < 0.5 else 'NLOS'
    p = UMA[scen]; N = p['N']
    DS = 10 ** rng.normal(*p['lgDS']); ASD = min(10 ** rng.normal(*p['lgASD']), 104.0)
    tau = np.sort(-p['rtau'] * DS * np.log(rng.random(N))); tau -= tau[0]
    Pn = np.exp(-tau * (p['rtau'] - 1) / (p['rtau'] * DS)) * 10 ** (-rng.normal(0, p['zeta'], N) / 10)
    Pn /= Pn.sum()
    Cphi = p['Cphi']; KR = 0.0
    if scen == 'LOS':
        KdB = rng.normal(*p['K']); KR = 10 ** (KdB / 10)
        Cphi *= 1.1035 - 0.028 * KdB - 0.002 * KdB ** 2 + 0.0001 * KdB ** 3
    phi0 = rng.uniform(-np.pi / 3, np.pi / 3)
    phin = 2 * (ASD / 1.4) * np.sqrt(-np.log(Pn / Pn.max())) / Cphi
    Xn = rng.choice([-1, 1], N); Yn = rng.normal(0, ASD / 7, N)
    phin = Xn * phin + Yn
    if scen == 'LOS':
        phin = phin - phin[0]
    aod_c = phi0 + np.deg2rad(phin)
    keep = 10 * np.log10(Pn / Pn.max()) > -25
    aod_c, Pn = aod_c[keep], Pn[keep]; Nk = keep.sum()
    aod = (aod_c[:, None] + np.deg2rad(p['cASD']) * ALPHA[None, rng.permutation(20)]).ravel()
    aoa_c = rng.uniform(0, 2 * np.pi, Nk)
    aoa = (aoa_c[:, None] + np.deg2rad(p['cASA']) * ALPHA[None, rng.permutation(20)]).ravel()
    pw = np.repeat(Pn / 20, 20)
    if scen == 'LOS':
        pw = pw / (KR + 1)
        aod = np.append(aod, phi0); aoa = np.append(aoa, rng.uniform(0, 2 * np.pi)); pw = np.append(pw, KR / (KR + 1))
    fd = (v_kmh / 3.6) / LAM; thv = rng.uniform(0, 2 * np.pi)
    f = fd * np.cos(aoa - thv)
    g = np.sqrt(pw) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(pw)))
    t = np.arange(T) * TS
    h = (g[None, :] * np.exp(1j * 2 * np.pi * f[None, :] * t[:, None])) @ steer(aod).T
    return _add_pilot_noise(h, rng)


GENERATORS = {'geometric': gen_user_geometric, 'uma': gen_user_uma}
