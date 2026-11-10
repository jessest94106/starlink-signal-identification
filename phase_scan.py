"""Is the track a catalogued satellite at the wrong place along its orbit (stale TLE, manoeuvre)?

Each satellite's orbit plane is kept fixed in space; the satellite is slid along it by up to +-10 min and the
Earth is rotated to the true time, so the pass geometry over the receiver changes as a real one would.
Coarse: 2-s slides on <= 40 points, free offset (Starlink range); best 30 satellites refined (0.05 s, all points).

Usage: phase_scan.py TRACKS.csv CARRIER_HZ START_UTC TLE_FILE[,TLE_FILE...] TRACK [TRACK ...] [-x NORAD]
  -x NORAD: leave one satellite out (chance-match control on confirmed tracks).
Prints the best 5 satellites: rms (Hz), slide (s), offset (kHz).
"""
import sys

import numpy as np

from norad_id import C, SIEG, _geometry, load_tles

SLIDE, STEP, NPTS, TOP = 600.0, 2.0, 40, 30
OFF_LO, OFF_HI = -125e3, -55e3


def rms_fit(z, D):
    """D: [..., points] Doppler; returns rms with best offset in the allowed range (inf outside)."""
    c = (z - D).mean(-1)
    e = np.sqrt(((z - D - c[..., None]) ** 2).mean(-1))
    return np.where((c >= OFF_LO) & (c <= OFF_HI), e, np.inf), c


def scan(t, z, fc, sats, slides):
    """Best (rms, slide, offset) per satellite over slides (s)."""
    ids = list(sats)
    T = np.broadcast_to(t[None], (len(slides), len(t))).ravel()
    TO = (t[None] + slides[:, None]).ravel()
    out = {}
    for k in range(0, len(ids), 300):
        chunk = ids[k:k + 300]
        _, rd, el = _geometry([sats[n] for n in chunk], T, SIEG, t_orbit=TO)
        D = (-rd * fc / C).reshape(len(chunk), len(slides), len(t))
        vis = (el.reshape(len(chunk), len(slides), len(t)) > 0).all(-1)
        e, c = rms_fit(z, D)
        e[~vis | ~np.isfinite(e)] = np.inf
        for j, n in enumerate(chunk):
            i = np.argmin(e[j])
            if np.isfinite(e[j, i]):
                out[n] = (e[j, i], slides[i], c[j, i])
    return out


def search(t, z, fc, sats):
    sub = np.linspace(0, len(t) - 1, min(NPTS, len(t))).astype(int)
    tm = np.array([t[0], t[-1]])
    coarse_sl = np.arange(-SLIDE, SLIDE + 1e-9, 60.0)                 # who can be overhead at all
    ids = list(sats)
    T = np.broadcast_to(tm[None], (len(coarse_sl), 2)).ravel()
    TO = (tm[None] + coarse_sl[:, None]).ravel()
    near = [n for k in range(0, len(ids), 2000) for n, e in zip(ids[k:k + 2000],
            _geometry([sats[n] for n in ids[k:k + 2000]], T, SIEG, t_orbit=TO)[2]) if np.nanmax(e) > -10]
    first = scan(t[sub], z[sub], fc, {n: sats[n] for n in near}, np.arange(-SLIDE, SLIDE + 1e-9, STEP))
    best = sorted(first.items(), key=lambda kv: kv[1][0])[:TOP]
    final = []
    for n, (_, s0, _) in best:
        r = scan(t, z, fc, {n: sats[n]}, np.arange(s0 - STEP, s0 + STEP + 1e-9, 0.05))
        if n in r:
            final.append((*r[n], n))
    return sorted(final)


if __name__ == '__main__':
    args = sys.argv[1:]
    skip = args[args.index('-x') + 1] if '-x' in args else None
    args = [a for i, a in enumerate(args) if a != '-x' and (i == 0 or args[i - 1] != '-x')]
    tracks_csv, fc, start, files = args[0], float(args[1]), args[2], args[3].split(',')
    t0 = np.datetime64(start.removesuffix('Z'), 'ns').astype('int64') / 1e9
    sats = {}
    for f in files:
        sats.update(load_tles(f, at_unix=t0))
    sats.pop(skip, None)
    d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
    for tr in map(int, args[4:]):
        p = d[d[:, 0] == tr]
        res = search(t0 + p[:, 1], p[:, 2], fc, sats)
        print(f'T{tr} ({p[-1, 1] - p[0, 1]:.0f} s)' + (f' without {skip}' if skip else '') + ':')
        for e, s, c, n in res[:5]:
            print(f'   {n:>6} rms {e:7.1f} Hz  slide {s:+7.2f} s  offset {c / 1e3:+7.2f} kHz')
