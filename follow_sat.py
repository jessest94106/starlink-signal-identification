"""Coherent follow of one candidate satellite through a whole capture (E01/G03 predict-strip-detect).

Removes the candidate's predicted Doppler (fitted offset + orbit time shift) from the raw I/Q, then per 1-s block:
FFT (1 Hz bins) after an extra residual-rate search of +-10 Hz/s in 1 Hz/s steps.
The right satellite's tone becomes a line near 0 Hz wherever the tone is on, including gaps between tracks.

Usage: follow_sat.py DATA.sigmf-data CURVES_DIR CARRIER_HZ START_UTC NORAD OFFSET_HZ DT_S OUT.png
Prints per block: time, SNR (dB), residual frequency (Hz), residual rate (Hz/s).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from norad_id import load_sim_csvs

DEC, SPAN_HZ, RATES = 50, 500, np.arange(-10, 10.5, 1.0)


def follow(data, curves_dir, fc, start, norad, offset_hz, dt_s):
    fs = float(json.loads(Path(data).with_suffix('.sigmf-meta').read_text())['global']['core:sample_rate'])
    x = np.memmap(data, dtype=np.complex64, mode='r')
    tp, fp = load_sim_csvs([Path(curves_dir) / f'{norad}.csv'], fc)[norad]
    sp = CubicSpline(tp, fp)
    t0 = np.datetime64(start.removesuffix('Z'), 'ns').astype('int64') / 1e9
    n, fd = int(fs), fs / DEC
    tb = (np.arange(n // DEC) + 0.5) / fd - 0.5                       # block-centred time after decimation
    fr = np.fft.fftshift(np.fft.fftfreq(n // DEC, 1 / fd))
    keep = np.abs(fr) <= SPAN_HZ
    win = np.hanning(n // DEC)
    phase, rows, stats = 0.0, [], []
    for k in range(x.size // n):
        ta = t0 + dt_s + (k * n + np.arange(n)) / fs
        if ta[0] < tp[0] or ta[-1] > tp[-1]:
            rows.append(np.full(keep.sum(), np.nan))
            stats.append((k + 0.5, np.nan, np.nan, np.nan))
            continue
        ph = phase + 2 * np.pi * np.cumsum(sp(ta) + offset_hz) / fs
        phase = ph[-1] % (2 * np.pi)
        y = (np.asarray(x[k * n:(k + 1) * n]) * np.exp(-1j * ph)).reshape(-1, DEC).mean(1)
        P = np.array([np.abs(np.fft.fftshift(np.fft.fft(y * np.exp(-1j * np.pi * r * tb ** 2) * win))) ** 2 for r in RATES])[:, keep]
        ri, fi = np.unravel_index(np.argmax(P), P.shape)
        rows.append(10 * np.log10(P[RATES == 0][0] / np.median(P[RATES == 0][0])))
        stats.append((k + 0.5, 10 * np.log10(P[ri, fi] / np.median(P[ri])), fr[keep][fi], RATES[ri]))
    return np.array(rows), fr[keep], np.array(stats)


if __name__ == '__main__':
    data, curves_dir, fc, start, norad, off, dt, out = sys.argv[1:9]
    img, f, st = follow(data, curves_dir, float(fc), start, norad, float(off), float(dt))
    for s in st:
        print(f'{s[0]:6.1f} s  SNR {s[1]:5.1f} dB  f {s[2]:+7.1f} Hz  rate {s[3]:+5.1f} Hz/s')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True, gridspec_kw=dict(width_ratios=[3, 1, 1]))
    ax[0].imshow(img, aspect='auto', origin='lower', extent=[f[0], f[-1], 0, len(img)], cmap='viridis', vmin=0, vmax=20)
    ax[0].set(xlabel='residual frequency after removing predicted Doppler (Hz)', ylabel='s into capture',
              title=f'{norad}: 1-s coherent FFT, rate 0')
    good = st[:, 1] > 13
    ax[1].plot(st[:, 1], st[:, 0], '.-', ms=3)
    ax[1].axvline(13, color='r', lw=0.8)
    ax[1].set(xlabel='peak SNR (dB), best rate', title='detection')
    ax[2].plot(st[good, 3], st[good, 0], 'o', ms=3)
    ax[2].set(xlabel='residual rate (Hz/s)', xlim=(-10.5, 10.5), title='rate search (SNR > 13 dB)')
    plt.tight_layout()
    plt.savefig(out, dpi=80)
