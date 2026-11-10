"""Waterfall of one capture with identified tracks labelled, plus residuals against the top-3 candidates per track.

Usage: plot_ids.py DATA.sigmf-data TRACKS.csv CURVES_DIR CARRIER_HZ START_UTC OUT.png   (also writes OUT_residuals.png)
START_UTC: time of the first sample (e.g. from match_capture.py).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

from iq_tracks import waterfall
from match_capture import SIGMA_DT_S, SIGMA_HZ
from norad_id import identify, load_sim_csvs

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

data, tracks_csv, curves_dir, fc, start, out = sys.argv[1:7]
fs = float(json.loads(Path(data).with_suffix('.sigmf-meta').read_text())['global']['core:sample_rate'])
W, f, t = waterfall(np.memmap(data, dtype=np.complex64, mode='r'), fs)
d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
rel = [(d[d[:, 0] == i, 1], d[d[:, 0] == i, 2]) for i in np.unique(d[:, 0])]
preds = load_sim_csvs(sorted(Path(curves_dir).glob('*.csv')), float(fc))
t0 = np.datetime64(start.removesuffix('Z'), 'ns').astype('int64') / 1e9
res, _ = identify([(t0 + tt, ff) for tt, ff in rel], preds, sigma_hz=SIGMA_HZ, sigma_dt_s=SIGMA_DT_S, top=3)


def fitted(i, norad, tt):
    """Candidate curve with its own fitted offset and orbit-time shift, at track-relative times tt."""
    r = identify([(t0 + rel[i][0], rel[i][1])], {norad: preds[norad]}, sigma_hz=SIGMA_HZ, sigma_dt_s=SIGMA_DT_S)[0][0]
    tp, fp = preds[norad]
    return np.interp(t0 + tt + r['dt_s'], tp, fp) + r['offset_hz']


fig, ax = plt.subplots(figsize=(13, max(9, (t[-1] - t[0]) / 35)))
ax.imshow(W, aspect='auto', origin='lower', extent=[f[0] / 1e3, f[-1] / 1e3, t[0], t[-1]], cmap='viridis', vmin=0, vmax=8)
for i, ((tt, ff), r) in enumerate(zip(rel, res)):
    ok = r['status'] == 'accepted'
    if r['norad']:
        span = np.linspace(tt[0] - 15, tt[-1] + 15, 200)
        ax.plot(fitted(i, r['norad'], span) / 1e3, span, '--', color='w' if ok else 'orange', lw=1)
    ax.text(ff[0] / 1e3 + 4, tt[0], f'T{i} {r["norad"]}\n{r["status"]}, {r["rms_hz"]:.1f} Hz, x{r["ratio"]:.0f}',
            color='w' if ok else 'orange', fontsize=8, va='top')
ax.set(xlabel='baseband frequency (kHz)', ylabel=f's after {start}', xlim=(f[0] / 1e3, f[-1] / 1e3),
       title='waterfall (dB over noise), dashed = matched SGP4 curve (white accepted, orange not)')
fig.tight_layout()
fig.savefig(out, dpi=80)

cols = 5
fig, axs = plt.subplots(-(-len(rel) // cols), cols, figsize=(4 * cols, 3 * -(-len(rel) // cols)), squeeze=False)
for a in axs.flat[len(rel):]:
    a.axis('off')
for i, ((tt, ff), r) in enumerate(zip(rel, res)):
    a = axs.flat[i]
    for k, (norad, rms) in enumerate(c for c in r['candidates'] if np.isfinite(c[1])):
        a.plot(tt, ff - fitted(i, norad, tt), '.' if k == 0 else '-', ms=3, lw=1, label=f'{norad} {rms:.0f} Hz')
    a.set_title(f'T{i}: {r["status"]}', fontsize=9)
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
fig.suptitle('residual (Hz) of each track vs its top-3 candidates; flat at 0 = match')
fig.tight_layout()
fig.savefig(out.replace('.png', '_residuals.png'), dpi=70)
for i, r in enumerate(res):
    print(f'T{i}: {r["status"]:<9} ' + ', '.join(f'{n} {x:.1f} Hz' for n, x in r['candidates']))
