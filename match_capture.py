"""Identify the tracks of one capture: scan the unknown start time, run norad_id.identify, plot the overlay.

Usage: match_capture.py TRACKS.csv CURVES_DIR CARRIER_HZ START_UTC TAU_MAX_S OUT.png
START_UTC: earliest possible first-sample time; the true start is searched in [START, START + TAU_MAX_S].
"""
import sys
from pathlib import Path

import numpy as np

from norad_id import identify, load_sim_csvs

# ~2 Hz tracks + model error; GP TLEs. Per-track offsets: on the 2026-05-04 data each satellite's tone sat a few kHz
# from the others (not multiples of 43.945 kHz), so one shared LO offset does not fit real tones.
SIGMA_HZ, SIGMA_DT_S = 3.0, 0.4


def run(tracks_csv, curves_dir, carrier_hz, start_utc, tau_max_s):
    d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
    rel = [(d[d[:, 0] == i, 1], d[d[:, 0] == i, 2]) for i in np.unique(d[:, 0])]
    preds = load_sim_csvs(sorted(Path(curves_dir).glob('*.csv')), carrier_hz)
    t0 = np.datetime64(start_utc.removesuffix('Z'), 'ns').astype('int64') / 1e9
    def score(tau):
        res, b = identify([(t0 + tau + t, f) for t, f in rel], preds, sigma_hz=SIGMA_HZ, sigma_dt_s=SIGMA_DT_S,
                          common_lo=False)
        return np.median([r['rms_hz'] for r in res]), tau, res, b

    scan = [score(tau) for tau in np.arange(0, tau_max_s + 1e-9, 1.0)]          # coarse 1 s, then 0.25 s around the best
    best = min(scan, key=lambda s: s[0])[1]
    scan += [score(tau) for tau in np.arange(best - 0.75, best + 0.76, 0.25) if tau % 1 and 0 <= tau <= tau_max_s]
    return rel, preds, t0, sorted(scan, key=lambda s: s[1])


if __name__ == '__main__':
    tracks_csv, curves_dir, fc, start, tau_max, out_png = sys.argv[1:7]
    rel, preds, t0, scan = run(tracks_csv, curves_dir, float(fc), start, float(tau_max))
    cost, tau, res, b = min(scan, key=lambda s: s[0])
    print('start offset scan (s: median RMS Hz):', ' '.join(f'{s[1]:.2f}:{s[0]:.0f}' for s in scan[::4]))
    print(f'best start = {start} + {tau:.2f} s')
    for i, r in enumerate(res):
        print(f'track {i}: {r["status"]:<9} {str(r["norad"]):<10} rms {r["rms_hz"]:7.1f} Hz  ratio {r["ratio"]:7.1f}  '
              f'offset {r["offset_hz"] / 1e3:8.2f} kHz  dt {r["dt_s"]:+.2f} s  next: {r["candidates"][1]}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot([s[1] for s in scan], [s[0] for s in scan], '.-')
    ax[0].set(xlabel='start offset (s)', ylabel='median best RMS over tracks (Hz)', yscale='log', title='start-time scan')
    for i, ((t, f), r) in enumerate(zip(rel, res)):
        ax[1].plot(t, f / 1e3, 'k.', ms=3)
        if r['norad']:
            tp, fp = preds[r['norad']]
            m = (tp >= t0 + tau + t[0] - 20) & (tp <= t0 + tau + t[-1] + 20)
            ax[1].plot(tp[m] - t0 - tau, (fp[m] + r['offset_hz'] + r['dt_s'] * np.gradient(fp[m], tp[m])) / 1e3,
                       '-', lw=1, label=f'{i}: {r["norad"]} ({r["status"]})')
    ax[1].set(xlabel=f's after {start} + {tau:.2f} s', ylabel='baseband kHz', title='tracks (dots) vs matched SGP4 curves')
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
