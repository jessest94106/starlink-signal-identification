"""Wide catalog search for tracks no satellite explains: every TLE epoch of every object above the horizon,
orbit shift up to +-60 s, offset within the Starlink tone range (receiver ~ -90 kHz, satellite +-35 kHz).

Catches satellites whose nearest TLE is off (manoeuvre: a later or earlier TLE fits) and objects not named
STARLINK (pass several TLE files). Coarse: 0.5-s shifts on <= 40 points; then the best 30 TLEs refined (0.01 s).

Usage: wide_search.py TRACKS.csv CARRIER_HZ START_UTC TLE_FILE[,TLE_FILE...] TRACK [TRACK ...]
Prints per track the best 5 objects: rms (Hz), shift (s), offset (kHz), TLE epoch, and how many of its TLEs fit.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from sgp4.api import Satrec

from norad_id import C, SIEG, _geometry

DT_MAX, DT_COARSE, NPTS, TOP = 60.0, 0.5, 40, 30
OFF_LO, OFF_HI = -125e3, -55e3


def load_all(paths):
    """Every TLE in 3LE files -> list of (norad, name, epoch_unix, Satrec); duplicates dropped."""
    out, seen = [], set()
    for path in paths:
        lines = [s.rstrip() for s in Path(path).read_text().splitlines()]
        for k in range(len(lines) - 1):
            a, b = lines[k], lines[k + 1]
            if a.startswith('1 ') and b.startswith('2 ') and (a, b) not in seen:
                seen.add((a, b))
                s = Satrec.twoline2rv(a, b)
                name = lines[k - 1][2:].strip() if k and lines[k - 1].startswith('0 ') else str(s.satnum)
                out.append((s.satnum, name, (s.jdsatepoch - 2440587.5 + s.jdsatepochF) * 86400, s))
    return out


def search(t, z, fc, tles):
    """t: unix s, z: Hz. Returns [(rms, dt, offset, tle_index)] best first, one entry per TLE refined."""
    tm = np.array([t[0], 0.5 * (t[0] + t[-1]), t[-1]])
    up = []
    for k in range(0, len(tles), 2000):
        _, _, el = _geometry([x[3] for x in tles[k:k + 2000]], tm, SIEG)
        up += list(k + np.where(np.nanmax(el, 1) > 0)[0])
    tg = np.arange(t[0] - DT_MAX - 2, t[-1] + DT_MAX + 2, 0.5)
    dop = np.vstack([-_geometry([tles[i][3] for i in up[k:k + 2000]], tg, SIEG)[1] * fc / C for k in range(0, len(up), 2000)])
    sub = np.linspace(0, len(t) - 1, min(NPTS, len(t))).astype(int)
    dts = np.arange(-DT_MAX, DT_MAX + 1e-9, DT_COARSE)
    ts = t[sub][None] + dts[:, None]
    coarse = []
    for j, i in enumerate(up):
        if not np.isfinite(dop[j]).all():
            continue
        R = z[sub][None] - np.interp(ts, tg, dop[j])
        c = R.mean(1)
        e = np.sqrt(((R - c[:, None]) ** 2).mean(1))
        e[(c < OFF_LO) | (c > OFF_HI)] = np.inf
        k = e.argmin()
        coarse.append((e[k], dts[k], j, i))
    coarse.sort()
    fine = []
    for _, d0, j, i in coarse[:TOP]:
        sp = CubicSpline(tg, dop[j])
        dd = np.arange(d0 - 0.6, d0 + 0.6, 0.01)
        R = z[None] - sp(t[None] + dd[:, None])
        c = R.mean(1)
        e = np.sqrt(((R - c[:, None]) ** 2).mean(1))
        e[(c < OFF_LO) | (c > OFF_HI)] = np.inf
        k = e.argmin()
        fine.append((e[k], dd[k], c[k], i))
    return sorted(fine), coarse


if __name__ == '__main__':
    tracks_csv, fc, start, files = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4].split(',')
    t0 = np.datetime64(start.removesuffix('Z'), 'ns').astype('int64') / 1e9
    tles = load_all(files)
    d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
    for tr in map(int, sys.argv[5:]):
        p = d[d[:, 0] == tr]
        fine, coarse = search(t0 + p[:, 1], p[:, 2], fc, tles)
        best = {}
        for e, dt, c, i in fine:
            best.setdefault(tles[i][0], (e, dt, c, i))
        rows = sorted(best.values())[:5]
        print(f'T{tr} ({p[-1, 1] - p[0, 1]:.0f} s):')
        for e, dt, c, i in rows:
            n, name, ep, _ = tles[i]
            nfit = sum(1 for ce, _, _, ci in coarse if tles[ci][0] == n and ce <= 2 * max(e, 5))
            age = (t0 + p[0, 1] - ep) / 3600
            print(f'   {name:<22} {n:6d} rms {e:7.1f} Hz  shift {dt:+6.2f} s  offset {c / 1e3:+7.2f} kHz  '
                  f'TLE {age:+6.1f} h old  ({nfit} of its TLEs fit)')
