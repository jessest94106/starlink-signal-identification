"""One raw capture end to end, no manual steps: tone tracks -> start time -> staged ID.

1. Tracks: iq_tracks.tracks (detect, link, refine, clean, coherent extend-and-join).
2. Predicted curves: SGP4 for every satellite that rises above 10 deg (receiver at SIEG), above-horizon part only.
3. Start time, fingerprint: scan the start +-SCAN_S around the file-name time (PC clock), 1 s then 0.25 s; keep the start
   with the lowest median per-track mismatch (as match_capture.py).
4. Start time, 15-s tone switches: follow each accepted satellite's tone (0.1-s blocks), find on/off steps > 6 dB; the
   switches sit on the 15-s GPS grid, so shift the start by the median phase. Fewer than 3 switches: use the fingerprint.
5. staged_id.staged with both starts.

Usage: run_capture.py DATA.sigmf-data TLE_FILE OUT_DIR [START_GUESS_UTC]
Writes OUT_DIR/tracks.csv, staged.txt, summary.json. The file name must hold _f<GHz>GHz_<YYYYMMDDTHHMMSS>; that time
is the start guess unless START_GUESS_UTC is given (some recorders name the file at the end of the recording).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from iq_tracks import tracks
from match_capture import SIGMA_DT_S, SIGMA_HZ
from norad_id import identify, load_tles, predict
from staged_id import staged

SCAN_S, EDGE_BLK, EDGE_DB, MIN_EDGES = 20.0, 50_000, 6.0, 3


def curves(sats, t0, dur, fc):
    t = t0 + np.arange(-SCAN_S - 90, dur + SCAN_S + 90, 1.0)
    preds, elev = predict(sats, t, fc)
    return {f'SAT-{n}': (t[elev[n] >= 0], f[elev[n] >= 0]) for n, (_, f) in preds.items() if (elev[n] >= 0).sum() > 10}


def fingerprint(raw, preds, t_file):
    score = lambda tau: np.median([r['rms_hz'] for r in identify([(t_file + tau + t, f) for t, f in raw.values()], preds,
                                                                  sigma_hz=SIGMA_HZ, sigma_dt_s=SIGMA_DT_S)[0]])
    coarse = {tau: score(tau) for tau in np.arange(-SCAN_S, SCAN_S + 1e-9, 1.0)}
    b = min(coarse, key=coarse.get)
    fine = {tau: score(tau) for tau in np.arange(b - 0.75, b + 0.76, 0.25)}
    fine.update(coarse)
    return t_file + min(fine, key=fine.get)


def switches(x, fs, t0, acc, preds, traces=None):
    """Tone on/off times (s into capture) of accepted satellites {norad: (offset_hz, dt_s)}.
    traces: optional dict, filled with {norad: (block times s, tone power dB, edge times s)} for plotting."""
    fr = np.fft.fftfreq(EDGE_BLK, 1 / fs)
    near = np.abs(fr) < 400
    out = []
    for norad, (off, dt) in acc.items():
        tp, fp = preds[norad]
        sp, ph, P, T = CubicSpline(tp, fp), 0.0, [], []
        for k in range(x.size // EDGE_BLK):
            ta = t0 + dt + (k * EDGE_BLK + np.arange(EDGE_BLK)) / fs
            if ta[0] < tp[0] or ta[-1] > tp[-1]:
                ph = 0.0; continue
            p = ph + 2 * np.pi * np.cumsum(sp(ta) + off) / fs
            ph = p[-1] % (2 * np.pi)
            Y = np.abs(np.fft.fft(np.asarray(x[k * EDGE_BLK:(k + 1) * EDGE_BLK]) * np.exp(-1j * p))) ** 2
            P.append(10 * np.log10(Y[near].max() / np.median(Y))); T.append(k * EDGE_BLK / fs)
        P, T = np.array(P), np.array(T)
        step = np.array([P[k:k + 10].mean() - P[max(0, k - 10):k].mean() if 3 <= k <= len(P) - 3 else 0 for k in range(len(P))])
        got = []
        for k in np.argsort(-np.abs(step)):
            if abs(step[k]) < EDGE_DB:
                break
            if all(abs(T[k] - g) >= 2 for g in got):
                got.append(T[k])
        out += got
        if traces is not None:
            traces[norad] = (T, P, np.array(got))
    return np.array(out)


def main(data, tle, out_dir, start_guess=None):
    data, out = Path(data), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = re.search(r'_f([\d.]+)GHz_(\d{8}T\d{6})', data.name)
    fc = float(m[1]) * 1e9
    s = start_guess or f'{m[2][:4]}-{m[2][4:6]}-{m[2][6:8]}T{m[2][9:11]}:{m[2][11:13]}:{m[2][13:15]}'
    t_file = np.datetime64(s.removesuffix('Z'), 'ns').astype('int64') / 1e9
    csv = out / 'tracks.csv'
    if data.exists():
        fs = float(json.loads(data.with_suffix('.sigmf-meta').read_text())['global']['core:sample_rate'])
        x = np.memmap(data, dtype=np.complex64, mode='r')
        dur = x.size / fs
    else:                                   # raw gone, tracks kept: no tone-switch start (fingerprint start only)
        x, d = None, np.loadtxt(csv, delimiter=',', skiprows=1, ndmin=2)
        dur = d[:, 1].max() + 1
    if not csv.exists():
        with open(csv, 'w') as fh:
            fh.write('track,t_rel_s,freq_hz,snr_db\n')
            for i, p in enumerate(tracks(data)):
                fh.writelines(f'{i},{a:.6f},{b:.3f},{c:.1f}\n' for a, b, c in p)
    d = np.loadtxt(csv, delimiter=',', skiprows=1, ndmin=2)
    raw = {i: (d[d[:, 0] == i, 1], d[d[:, 0] == i, 2]) for i in np.unique(d[:, 0]).astype(int)}
    summary = dict(file=data.name, fc=fc, duration_s=dur, tracks=len(raw))
    if raw:
        sats = load_tles(tle, at_unix=t_file)
        preds = curves(sats, t_file, dur, fc)
        t_fp = fingerprint(raw, preds, t_file)
        res = identify([(t_fp + t, f) for t, f in raw.values()], preds, sigma_hz=SIGMA_HZ, sigma_dt_s=SIGMA_DT_S)[0]
        acc = {r['norad']: (r['offset_hz'], r['dt_s']) for r in res if r['status'] == 'accepted'}
        E = switches(x, fs, t_fp, acc, preds) if x is not None else np.array([])
        ph = ((t_fp + E + 18 + 7.5) % 15) - 7.5                      # GPS = UTC + 18 s; phase in (-7.5, 7.5]
        t_sw = t_fp - np.median(ph) if len(E) >= MIN_EDGES else t_fp
        lines = staged(raw, preds, t_sw, t_fp, sats)
        (out / 'staged.txt').write_text('\n'.join(lines) + '\n')
        iso = lambda t: str(np.datetime64(int(t * 1e3), 'ms'))
        summary.update(start_file=iso(t_file), start_fingerprint=iso(t_fp), start_switches=iso(t_sw), switches=len(E),
                       switch_phase_spread_s=float(np.std(ph)) if len(E) else None, result=lines[-1])
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main(*sys.argv[1:5])
