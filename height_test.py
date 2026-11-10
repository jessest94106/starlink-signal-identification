"""Satellite-free height test: which orbit heights can make this Doppler curve?

Model: circular orbit of radius r = R + h over a spherical Earth. Central angle between receiver and satellite:
cos(gamma) = cos(beta) cos(w (t - tca)), beta = cross-track angle at closest approach. Range rho, range rate
rho' = r R cos(beta) w sin(w (t - tca)) / rho, Doppler f = c - fc/c_light * rho'.
w = Kepler rate at h + Omega_earth * u, u in [-1, 1] (Earth rotation, unknown inclination). Free: tca, beta, u,
offset c. The satellite must stay above the horizon over the track. For each h on a grid: best rms (Hz).

Usage: height_test.py TRACKS.csv CARRIER_HZ TRACK [TRACK ...]
Prints per track: best h, rms, and the heights whose rms <= best + max(3 Hz, 0.5 x best).
"""
import sys

import numpy as np
from scipy.optimize import least_squares

MU, R, OMEGA, CL = 3.986004418e14, 6.366e6, 7.2921159e-5, 299792458.0   # R: geocentric radius at 47.65 deg N
H = np.arange(300e3, 2001e3, 25e3)
U = np.linspace(-1, 1, 5)
NPTS = 60                                                               # points used in the coarse grid


def doppler(t, h, beta, tca, w, c, fc):
    r = R + h
    g = w * (t - tca)
    rho = np.sqrt(r * r + R * R - 2 * r * R * np.cos(beta) * np.cos(g))
    return c - fc / CL * r * R * np.cos(beta) * w * np.sin(g) / rho


def visible(t, h, beta, tca, w):
    return np.all(np.cos(beta) * np.cos(w * (t - tca)) >= R / (R + h), axis=-1)


def profile(t, z, fc):
    """Best rms (Hz) and parameters for each height in H."""
    sub = np.linspace(0, len(t) - 1, min(NPTS, len(t))).astype(int)
    ts, zs = t[sub], z[sub]
    out = []
    for h in H:
        wk = np.sqrt(MU / (R + h) ** 3)
        bmax = np.arccos(R / (R + h))
        tcas = np.arange(t[0] - 600, t[-1] + 600, 3.0)[:, None, None]
        betas = np.linspace(0, bmax, 60)[None, :, None]
        best = (np.inf, None)
        for u in U:
            w = wk + OMEGA * u
            m = doppler(ts, h, betas, tcas, w, 0.0, fc)
            c = (zs - m).mean(-1, keepdims=True)
            e = np.sqrt(((zs - m - c) ** 2).mean(-1))
            e[~visible(ts, h, betas, tcas, w)] = np.inf
            i, j = np.unravel_index(np.argmin(e), e.shape)
            if e[i, j] < best[0]:
                best = (e[i, j], (betas[0, j, 0], tcas[i, 0, 0], u, c[i, j, 0]))
        if not np.isfinite(best[0]):
            out.append((np.inf, None)); continue
        b0, tc0, u0, c0 = best[1]
        fun = lambda p: doppler(t, h, p[0], p[1], wk + OMEGA * p[2], p[3], fc) - z
        lo, hi = [0, t[0] - 900, -1, -np.inf], [bmax, t[-1] + 900, 1, np.inf]
        r = least_squares(fun, [b0, tc0, u0, c0], bounds=(lo, hi), x_scale=[0.01, 10, 0.5, 1e3])
        ok = visible(t, h, r.x[0], r.x[1], wk + OMEGA * r.x[2])
        out.append((np.sqrt(np.mean(r.fun ** 2)) if ok else np.inf, r.x))
    return np.array([e for e, _ in out]), [p for _, p in out]


def consistent(rms):
    lim = rms.min() + max(3.0, 0.5 * rms.min())
    return H[rms <= lim]


if __name__ == '__main__':
    tracks_csv, fc = sys.argv[1], float(sys.argv[2])
    d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
    for i in map(int, sys.argv[3:]):
        p = d[d[:, 0] == i]
        rms, _ = profile(p[:, 1], p[:, 2], fc)
        ok = consistent(rms)
        print(f'T{i}: {p[-1, 1] - p[0, 1]:.0f} s, best h {H[rms.argmin()] / 1e3:.0f} km (rms {rms.min():.1f} Hz), '
              f'consistent {ok.min() / 1e3:.0f}-{ok.max() / 1e3:.0f} km; rms at 550 km {rms[H == 550e3][0]:.1f} Hz, '
              f'at 1200 km {rms[H == 1200e3][0]:.1f} Hz')
