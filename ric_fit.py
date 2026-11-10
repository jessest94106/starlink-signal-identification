"""Does a TLE position error explain a track? Fit orbit shift (along-track, s) plus constant radial and cross-track
position offsets (km) plus frequency offset, for the best few satellites of a plain fit.

Plain fit = shift +-6 s and offset only (as the pipeline). Satellites: nearest TLE, above the horizon.
Usage: ric_fit.py TRACKS.csv CARRIER_HZ START_UTC TLE_FILE TRACK [TRACK ...]
"""
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from norad_id import C, SIEG, _geometry, load_tles

SHORT, DT_MAX, D_MAX = 6, 6.0, 20.0                  # shortlist size; shift (s) and offset (km) bounds
OFF_LO, OFF_HI = -125e3, -55e3


def dop(sat, t, fc, dt=0.0, d_rc=(0.0, 0.0)):
    return -_geometry([sat], t + dt, SIEG, d_rc)[1][0] * fc / C


def shortlist(t, z, fc, sats):
    ids = list(sats)
    el = np.vstack([_geometry([sats[n] for n in ids[k:k + 2000]], t[[0, -1]], SIEG)[2] for k in range(0, len(ids), 2000)])
    up = [n for n, e in zip(ids, el) if np.nanmin(e) > 0]
    tg = np.arange(t[0] - DT_MAX - 2, t[-1] + DT_MAX + 2, 1.0)
    D = np.vstack([-_geometry([sats[n] for n in up[k:k + 2000]], tg, SIEG)[1] * fc / C for k in range(0, len(up), 2000)])
    dts = np.arange(-DT_MAX, DT_MAX + 1e-9, 0.05)
    res = []
    for n, row in zip(up, D):
        R = z[None] - CubicSpline(tg, row)(t[None] + dts[:, None])
        c = R.mean(1)
        e = np.sqrt(((R - c[:, None]) ** 2).mean(1))
        e[(c < OFF_LO) | (c > OFF_HI)] = np.inf
        k = e.argmin()
        res.append((e[k], n, dts[k], c[k]))
    return sorted(res)[:SHORT]


def ric(t, z, fc, sat, dt0, c0):
    fun = lambda p: dop(sat, t, fc, p[0], (p[1], p[2])) + p[3] - z
    r = least_squares(fun, [dt0, 0, 0, c0], bounds=([-DT_MAX, -D_MAX, -D_MAX, OFF_LO], [DT_MAX, D_MAX, D_MAX, OFF_HI]),
                      x_scale=[0.5, 1, 1, 100])
    return np.sqrt(np.mean(r.fun ** 2)), r.x


if __name__ == '__main__':
    tracks_csv, fc, start, tle = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    t0 = np.datetime64(start.removesuffix('Z'), 'ns').astype('int64') / 1e9
    sats = load_tles(tle, at_unix=t0)
    d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
    for tr in map(int, sys.argv[5:]):
        p = d[d[:, 0] == tr]
        t, z = t0 + p[:, 1], p[:, 2]
        print(f'T{tr} ({p[-1, 1] - p[0, 1]:.0f} s):')
        rows = []
        for e, n, dt, c in shortlist(t, z, fc, sats):
            rr, x = ric(t, z, fc, sats[n], dt, c)
            rows.append((rr, n, e, x))
        for rr, n, e, x in sorted(rows):
            print(f'   SAT-{n:<6} plain {e:7.1f} Hz -> with position offsets {rr:6.1f} Hz: shift {x[0]:+5.2f} s, '
                  f'radial {x[1]:+6.2f} km, cross-track {x[2]:+6.2f} km, offset {x[3] / 1e3:+7.2f} kHz')
