"""Baseline on every Drive capture: the team's Drive matchers on the same tone tracks and SGP4 curves as run_capture.py.

1. Drive Code/sat_identification_least_squares.py score_candidates (unchanged; its default 0.01-s time slide over the
   whole predicted pass + free offset). It only ranks; our rule labels it: best <= 15 Hz and runner-up >= 2x best =>
   determined; best <= 15 Hz otherwise => ambiguous; else unknown.
2. Drive web app processing/satellite_matcher.py rule: absolute RMSE < 1500 Hz at the file-name (PC-clock) times.

Writes data/drive/<date>/<setup>/<capture>/baseline.txt (one line per track). Usage: baseline_all.py [WORKERS]
"""
import importlib.util
import json
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from norad_id import load_tles, predict, write_sim_csvs
from run_capture import SCAN_S

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('drive_ls', HERE / 'drive_code/sat_identification_least_squares.py')
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)
LIMIT_HZ, RATIO = 15.0, 2.0
ts = lambda s: np.datetime64(s, 'ms').astype('int64') / 1e3


def one(cap):
    s = json.loads((cap / 'summary.json').read_text())
    d = np.loadtxt(cap / 'tracks.csv', delimiter=',', skiprows=1, ndmin=2)
    if not len(d):
        return
    t0, fc = ts(s['start_switches']), s['fc']                 # same curves as run_capture.curves (start only sets the window)
    ls.BIN_HZ, ls.BIN_ZERO_HZ, ls.RECEIVER_HZ = 1.0, 0.0, fc   # bins in Hz; EPOCH_S stays at the Drive default 0.01 s
    t = t0 + np.arange(-SCAN_S - 90, s['duration_s'] + SCAN_S + 90, 1.0)
    preds, elev = predict(load_tles(HERE / f'tles/{cap.parts[-3]}.tle', at_unix=t0), t, fc)
    t_log = ts(s['start_file'])
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        write_sim_csvs(preds, elev, fc, tmp)
        csvs = sorted(Path(tmp).glob('*.csv'))
        for i in np.unique(d[:, 0]).astype(int):
            tr, f = d[d[:, 0] == i, 1], d[d[:, 0] == i, 2]
            k = np.round((tr - tr[0]) / ls.EPOCH_S).astype(int)
            bins = np.full(k[-1] + 1, np.nan); bins[k] = f
            r = sorted(ls.score_candidates(bins, csvs), key=lambda x: x['rmse_bins'])
            b1, b2 = r[0]['rmse_bins'], r[1]['rmse_bins']
            lab = 'unknown' if not b1 <= LIMIT_HZ else 'determined' if b2 >= RATIO * b1 else 'ambiguous'
            e = {n: np.sqrt(np.mean((f - np.interp(t_log + tr, pt[elev[n] >= 0], pf[elev[n] >= 0])) ** 2))
                 for n, (pt, pf) in preds.items()}
            wn = min(e, key=e.get)
            out.append(f'track {i}: LS {lab:<10} {r[0]["satellite"]:<10} rmse {b1:7.1f} 2nd {r[1]["satellite"]:<10} {b2:7.1f} | '
                       f'web {"match" if e[wn] < 1500 else "none":<5} SAT-{wn} {e[wn]:8.0f} Hz')
    (cap / 'baseline.txt').write_text('\n'.join(out) + '\n')
    return str(cap)


if __name__ == '__main__':
    caps = sorted(p.parent for p in (HERE / 'data/drive').glob('*/*/*/staged.txt'))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        for c in ex.map(one, caps):
            print(c, flush=True)
