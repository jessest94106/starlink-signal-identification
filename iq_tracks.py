"""Tone tracks from a SigMF cf32 capture.

1. Detect: incoherent waterfall (122 Hz bins, 0.26 s rows), fixed spurs and row level removed, peaks > THR_DB.
2. Link peaks into tracks by extrapolating each track's recent slope.
3. Refine every point on the raw I/Q: remove the track's fitted chirp over the row (0.26 s, coherent),
   FFT (3.8 Hz bins), interpolate the peak. Points whose refined SNR is low are dropped.

4. Clean: fit a smooth curve to each track's own points (no satellite involved) and drop points far from it
   (stray peaks at track ends). The kept points are the measured ones, not smoothed values.
5. Extend and join (coherent follow, no satellite involved): step past each track end in 1-s blocks; remove the
   track's own extrapolated curve (quadratic over its last 6 s), FFT (1 Hz bins) with a +-10 Hz/s rate search,
   keep the peak if > 13 dB within +-50 Hz; stop after 5 misses. Tracks whose extended points run on top of each
   other (median gap <= 10 Hz over >= 3 points) are joined into one.

Usage: iq_tracks.py DATA.sigmf-data OUT.csv
       iq_tracks.py clean IN.csv OUT.csv        (step 4 only, on an existing track file; track numbers kept)
       iq_tracks.py join DATA.sigmf-data IN.csv OUT.csv   (step 5 only; prints old -> new track numbers)
Output CSV: track, t_rel_s (from the first sample), freq_hz (baseband), snr_db.
"""
import json
import sys
from pathlib import Path

import numpy as np

NFFT, AVG = 4096, 32          # detection: 122 Hz bins, 32 FFTs averaged per row
THR_DB, DC_HZ = 3.0, 3000.0   # peak threshold over the noise floor; ignore LO leakage near DC
MIN_POINTS, MIN_SPAN_S, MAX_MISS = 15, 6.0, 6
REFINE_SNR_DB = 10.0          # in the 3.8 Hz refined bins
CLEAN_KNOT_S, CLEAN_SIG, CLEAN_FLOOR_HZ = 5.0, 5.0, 8.0   # smooth-curve knots; outlier cut = max(5 x robust spread, 8 Hz)
EXT_DEC, EXT_FIT_S, EXT_SPAN_HZ, EXT_SNR_DB, EXT_MISS = 50, 6.0, 50.0, 13.0, 5
EXT_RATES = np.arange(-10, 10.5, 1.0)
JOIN_HZ, JOIN_PTS, DUP_S = 10.0, 3, 0.5


def waterfall(x, fs):
    rows = x.size // (NFFT * AVG)
    win = np.hanning(NFFT).astype(np.float32)
    W = np.empty((rows, NFFT), np.float32)
    for r in range(rows):
        blk = np.asarray(x[r * NFFT * AVG:(r + 1) * NFFT * AVG]).reshape(AVG, NFFT)
        W[r] = (np.abs(np.fft.fftshift(np.fft.fft(blk * win, axis=1), axes=1)) ** 2).mean(0)
    W = 10 * np.log10(W + 1e-30)
    W -= np.median(W, 0)                       # fixed spurs and passband shape
    W -= np.median(W, 1, keepdims=True)        # gain changes
    return W, np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / fs)), (np.arange(rows) + 0.5) * NFFT * AVG / fs


def link(W, f, t):
    df, dt = f[1] - f[0], t[1] - t[0]
    ok = (np.abs(f) > DC_HZ)
    active, done = [], []
    for r in range(W.shape[0]):
        row = W[r]
        pk = np.where(ok[1:-1] & (row[1:-1] > THR_DB) & (row[1:-1] >= row[:-2]) & (row[1:-1] >= row[2:]))[0] + 1
        free = set(pk.tolist())
        for tr in active:
            rr, ff = np.array(tr['r']), np.array(tr['f'])
            slope = np.polyfit(rr[-8:], ff[-8:], 1)[0] if len(rr) >= 3 else 0.0
            pred, gate = ff[-1] + slope * (r - rr[-1]), (3 if len(rr) >= 3 else 12) * df
            cand = [k for k in free if abs(f[k] - pred) <= gate]
            if cand:
                k = min(cand, key=lambda k: abs(f[k] - pred))
                free.discard(k)
                tr['r'].append(r)
                tr['f'].append(f[k])
        for k in free:
            active.append(dict(r=[r], f=[f[k]]))
        still = []
        for tr in active:
            (still if r - tr['r'][-1] <= MAX_MISS else done).append(tr)
        active = still
    keep = [tr for tr in done + active if len(tr['r']) >= MIN_POINTS and (tr['r'][-1] - tr['r'][0]) * dt >= MIN_SPAN_S]
    return [(t[np.array(tr['r'])], np.array(tr['f'])) for tr in keep]


def refine(x, fs, tt, ff):
    """Re-measure each point with the track's chirp removed over the row; returns (t, f, snr_db) of good points."""
    n = NFFT * AVG
    poly = np.polyfit(tt, ff, min(3, len(tt) - 1))
    nz = 1 << (n - 1).bit_length()
    fr = np.fft.fftfreq(nz, 1 / fs)
    win = np.hanning(n)
    out = []
    for tc in tt:
        s0 = int(round(tc * fs - n / 2))
        if s0 < 0 or s0 + n > x.size:
            continue
        ts = (s0 + np.arange(n)) / fs
        fp = np.polyval(poly, ts)
        y = np.asarray(x[s0:s0 + n]) * np.exp(-2j * np.pi * np.cumsum(fp) / fs) * win
        P = np.abs(np.fft.fft(y, nz)) ** 2
        near = np.where(np.abs(fr) < 60)[0]   # residual must be small after the chirp fit
        k = near[np.argmax(P[near])]
        a, b, c = np.log(P[[k - 1, k, (k + 1) % nz]])
        d = 0.5 * (a - c) / (a - 2 * b + c)
        snr = 10 * np.log10(P[k] / np.median(P))
        if snr >= REFINE_SNR_DB:
            out.append((tc, np.polyval(poly, tc) + fr[k] + d * (fs / nz), snr))
    return np.array(out).reshape(-1, 3)


def clean(p):
    """Drop points far from a smooth curve through the track's own points; returns the kept rows."""
    from scipy.interpolate import LSQUnivariateSpline
    t, f = p[:, 0], p[:, 1]
    keep = np.ones(len(t), bool)
    for _ in range(3):
        tk = t[keep]
        knots = np.arange(tk[0] + CLEAN_KNOT_S, tk[-1] - CLEAN_KNOT_S / 2, CLEAN_KNOT_S)
        try:
            fit = LSQUnivariateSpline(tk, f[keep], knots, k=3)(t)
        except ValueError:                                   # too few points between knots: plain cubic
            fit = np.polyval(np.polyfit(tk, f[keep], min(3, keep.sum() - 1)), t)
        r = f - fit
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) <= max(CLEAN_SIG * mad, CLEAN_FLOOR_HZ)
    return p[keep]


def extend(x, fs, p):
    """Follow a track (rows t, f, snr) past both ends with 1-s coherent blocks; returns new rows (t, f, snr)."""
    n = int(fs)
    m, fd = n // EXT_DEC, fs / EXT_DEC
    tb = (np.arange(m) + 0.5) / fd - 0.5
    fr = np.fft.fftshift(np.fft.fftfreq(m, 1 / fd))
    near = np.where(np.abs(fr) <= EXT_SPAN_HZ)[0]
    win = np.hanning(m)
    chirp = np.exp(-1j * np.pi * EXT_RATES[:, None] * tb ** 2) * win
    new = []
    for sgn in (1, -1):
        pts = [tuple(r[:2]) for r in p]
        tc, miss = (p[-1, 0] if sgn > 0 else p[0, 0]), 0
        while miss < EXT_MISS:
            tc += sgn
            s0 = int(round((tc - 0.5) * fs))
            if s0 < 0 or s0 + n > x.size:
                break
            q = np.array(pts)
            q = q[q[:, 0] >= q[:, 0].max() - EXT_FIT_S] if sgn > 0 else q[q[:, 0] <= q[:, 0].min() + EXT_FIT_S]
            poly = np.polyfit(q[:, 0] - tc, q[:, 1], min(2, len(q) - 1))
            f0 = np.polyval(poly, 0.0)
            if abs(f0) < DC_HZ or abs(f0) > fs / 2 - fd:
                break
            ts = (s0 + np.arange(n)) / fs - tc
            y = (np.asarray(x[s0:s0 + n]) * np.exp(-2j * np.pi * np.cumsum(np.polyval(poly, ts)) / fs)).reshape(-1, EXT_DEC).mean(1)
            P = np.abs(np.fft.fftshift(np.fft.fft(y * chirp, axis=1), axes=1)) ** 2
            ri, k = np.unravel_index(np.argmax(P[:, near]), (len(EXT_RATES), len(near)))
            k = near[k]
            snr = 10 * np.log10(P[ri, k] / np.median(P[ri]))
            if not snr >= EXT_SNR_DB:                     # also catches all-zero blocks
                miss += 1
                continue
            a, b, c = np.log(P[ri, k - 1:k + 2])
            row = (tc, f0 + fr[k] + 0.5 * (a - c) / (a - 2 * b + c) * (fd / m), snr)
            pts.append(row[:2]); new.append(row)
            miss = 0
    return np.array(new).reshape(-1, 3)


def join(x, fs, ps):
    """Extend every track, join tracks whose points run on top of each other; returns (tracks, old-number groups, seconds added)."""
    ext = [extend(x, fs, p) for p in ps]
    full = [np.vstack([np.c_[p[:, :3], np.zeros(len(p))], np.c_[e, np.ones(len(e))]]) for p, e in zip(ps, ext)]
    full = [a[np.argsort(a[:, 0])] for a in full]
    root = list(range(len(ps)))
    find = lambda i: i if root[i] == i else find(root[i])
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            a, b = full[i], full[j]
            inb = (b[:, 0] >= a[0, 0]) & (b[:, 0] <= a[-1, 0])
            if inb.sum() >= JOIN_PTS and np.median(np.abs(np.interp(b[inb, 0], a[:, 0], a[:, 1]) - b[inb, 1])) <= JOIN_HZ:
                root[find(j)] = find(i)
    groups = {}
    for i in range(len(ps)):
        groups.setdefault(find(i), []).append(i)
    out, added = [], []
    for g in groups.values():
        a = np.vstack([full[i] for i in g])
        orig, e = a[a[:, 3] == 0], a[a[:, 3] == 1]
        kept = list(orig[:, 0])
        keep_e = []
        for r in e[np.argsort(-e[:, 2])]:                  # extension points: drop duplicates, strongest first
            if not kept or np.min(np.abs(np.array(kept) - r[0])) >= DUP_S:
                kept.append(r[0]); keep_e.append(r)
        a = np.vstack([orig] + keep_e)[:, :3]
        out.append(a[np.argsort(a[:, 0])])
        added.append(len(keep_e))
    return out, list(groups.values()), added


def tracks(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.sigmf-meta').read_text())
    fs = float(meta['global']['core:sample_rate'])
    x = np.memmap(path, dtype=np.complex64, mode='r')
    W, f, t = waterfall(x, fs)
    ps = [clean(p) for p in (refine(x, fs, tt, ff) for tt, ff in link(W, f, t)) if len(p) >= MIN_POINTS]
    return join(x, fs, ps)[0]


if __name__ == '__main__':
    if sys.argv[1] == 'clean':
        d = np.loadtxt(sys.argv[2], delimiter=',', skiprows=1)
        ids = np.unique(d[:, 0]).astype(int)
        res = [clean(d[d[:, 0] == i, 1:]) for i in ids]
        print('\n'.join(f'track {i}: kept {len(r)} of {(d[:, 0] == i).sum()} points' for i, r in zip(ids, res)
                        if len(r) < (d[:, 0] == i).sum()))
        sys.argv[2] = sys.argv[3]
    elif sys.argv[1] == 'join':
        meta = json.loads(Path(sys.argv[2]).with_suffix('.sigmf-meta').read_text())
        x = np.memmap(sys.argv[2], dtype=np.complex64, mode='r')
        d = np.loadtxt(sys.argv[3], delimiter=',', skiprows=1)
        ids = np.unique(d[:, 0]).astype(int)
        res, groups, added = join(x, float(meta['global']['core:sample_rate']), [d[d[:, 0] == i, 1:] for i in ids])
        print('\n'.join(f'new {k} <- old {"+".join(str(ids[i]) for i in g)}, {a} 1-s points added' for k, (g, a) in enumerate(zip(groups, added))))
        sys.argv[2] = sys.argv[4]
    else:
        res = tracks(sys.argv[1])
    with open(sys.argv[2], 'w') as fh:
        fh.write('track,t_rel_s,freq_hz,snr_db\n')
        for i, p in enumerate(res):
            fh.writelines(f'{i},{a:.6f},{b:.3f},{c:.1f}\n' for a, b, c in p)
    for i, p in enumerate(res):
        print(f'track {i}: {p[0, 0]:6.1f}-{p[-1, 0]:6.1f} s, {p[0, 1] / 1e3:8.2f} -> {p[-1, 1] / 1e3:8.2f} kHz, '
              f'{len(p)} pts, median SNR {np.median(p[:, 2]):.1f} dB')
