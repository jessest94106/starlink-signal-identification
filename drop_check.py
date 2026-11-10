"""Find dropped samples in a SigMF capture from phase jumps of one continuous reference line.

A steady line keeps a smooth phase. If L samples are lost, the recorded line jumps ahead by
2*pi*f*L/fs (mod 2*pi), where f is the line's baseband frequency. The line is followed in 50-ms blocks
after removing a smooth fit of its frequency (from tracker points); jumps above 5 sigma are reported
with the packet-multiple m (L = m * SPP) that best explains each jump.

Usage: drop_check.py DATA.sigmf-data TRACKS.csv TRACK_IDS(comma) OUT.png
"""
import sys

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

FS, BLK, SPP = 500e3, 25_000, 2040          # sample rate; 50-ms blocks; B210 samples per USB packet (sc16)
NSIG, MAX_M = 5.0, 200


def follow(x, t_pts, f_pts):
    knots = np.arange(t_pts[0] + 20, t_pts[-1] - 20, 20.0)
    fit = LSQUnivariateSpline(t_pts, f_pts, knots, k=3)
    k0, k1 = int(t_pts[0] * FS) // BLK, int(t_pts[-1] * FS) // BLK
    z, tb, fb, ph = [], [], [], 0.0
    for k in range(k0, k1):
        t = (k * BLK + np.arange(BLK)) / FS
        p = ph + 2 * np.pi * np.cumsum(fit(t)) / FS
        ph = p[-1]
        z.append(np.mean(np.asarray(x[k * BLK:(k + 1) * BLK]) * np.exp(-1j * p)))
        tb.append(t[-1]); fb.append(float(fit(t[-1])))
    return np.array(z), np.array(tb), np.array(fb)


def jumps(z, fb):
    dphi = np.angle(z[1:] * np.conj(z[:-1]))
    med = np.array([np.median(dphi[max(0, i - 10):i + 11]) for i in range(len(dphi))])
    r = np.angle(np.exp(1j * (dphi - med)))
    amp = np.abs(z)
    ok = (amp[1:] > 0.5 * np.median(amp)) & (amp[:-1] > 0.5 * np.median(amp))   # line present on both sides
    sig = 1.4826 * np.median(np.abs(r[ok]))
    hit = np.where(ok & (np.abs(r) > NSIG * sig))[0]
    out = []
    for i in hit:
        m = np.arange(1, MAX_M + 1)
        pred = np.angle(np.exp(2j * np.pi * fb[i] * m * SPP / FS))
        err = np.abs(np.angle(np.exp(1j * (r[i] - pred))))
        out.append((i, r[i], m[np.argmin(err)], err.min()))
    return out, sig, ok


if __name__ == '__main__':
    data, tracks_csv, ids, out_png = sys.argv[1:5]
    x = np.memmap(data, dtype=np.complex64, mode='r')
    d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
    sel = np.isin(d[:, 0], [int(i) for i in ids.split(',')])
    p = d[sel][np.argsort(d[sel, 1])]
    z, tb, fb = follow(x, p[:, 1], p[:, 2])
    J, sig, ok = jumps(z, fb)
    print(f'followed {tb[0]:.0f}-{tb[-1]:.0f} s ({ok.mean():.0%} of block pairs usable), phase noise {sig:.2f} rad, '
          f'{len(J)} jumps > {NSIG:.0f} sigma')
    for i, r, m, e in J:
        print(f'  {tb[i]:7.2f} s  jump {r:+.2f} rad  line at {fb[i] / 1e3:6.2f} kHz  best fit {m:3d} packets ({m * SPP / FS * 1e3:6.1f} ms, misfit {e:.2f} rad)')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    dphi = np.angle(z[1:] * np.conj(z[:-1]))
    ax[0].plot(tb[1:][ok], dphi[ok], '.', ms=1)
    for i, *_ in J:
        ax[0].axvline(tb[i], color='r', lw=0.5)
    ax[0].set(ylabel='phase change per 50 ms (rad)', title='reference line: phase steps (red = detected jumps)')
    ax[1].plot(tb, 20 * np.log10(np.abs(z) + 1e-12), lw=0.5)
    ax[1].set(xlabel='s into capture', ylabel='line amplitude (dB)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=80)
