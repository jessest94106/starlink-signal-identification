"""Baseline: the team's Drive matchers on our tone tracks and SGP4 curve CSVs.

1. Code/sat_identification_least_squares.py (score_candidates, unchanged): free time slide over the whole predicted
   pass + free offset. It has no accept rule; we apply ours: best RMSE <= 15 Hz and runner-up >= 2x best.
2. Web app processing/satellite_matcher.py rule: absolute RMSE < 1500 Hz on the logged (PC-clock) time stamps.

Usage: baseline_drive.py CAPTURE BAND CARRIER_HZ LOG_START_UTC
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

from norad_id import load_sim_csvs

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('drive_ls', HERE / 'drive_code/sat_identification_least_squares.py')
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)
ROW_S = 4096 * 32 / 500e3                                       # iq_tracks.py point spacing (0.262 s)

cap, band, fc, log = sys.argv[1:5]
ls.EPOCH_S, ls.BIN_HZ, ls.BIN_ZERO_HZ, ls.RECEIVER_HZ = ROW_S, 1.0, 0.0, float(fc)   # bins in Hz, one per track row
d = np.loadtxt(HERE / f'data/{cap}/tracks_{band}.csv', delimiter=',', skiprows=1)
csvs = sorted((HERE / f'curves/{cap}_{band}').glob('*.csv'))
preds = load_sim_csvs(csvs, float(fc))
t_log = np.datetime64(log.removesuffix('Z'), 'ns').astype('int64') / 1e9
ours = {int(m.group(1)): m.group(2) for l in (HERE / f'data/{cap}/joint_{band}.txt').read_text().splitlines()
        for m in [re.match(r'track (\d+): accepted\s+(\S+)', l)] if m}
for i in np.unique(d[:, 0]).astype(int):
    t, f = d[d[:, 0] == i, 1], d[d[:, 0] == i, 2]
    k = np.round((t - t[0]) / ROW_S).astype(int)
    bins = np.full(k[-1] + 1, np.nan); bins[k] = f
    r = ls.score_candidates(bins, csvs)
    best, second = r[0], r[1]
    ok = best['rmse_bins'] <= 15 and second['rmse_bins'] >= 2 * best['rmse_bins']
    e = {n: np.sqrt(np.mean((f - np.interp(t_log + t, *p)) ** 2)) for n, p in preds.items() if p[0][0] <= t_log + t[0] and t_log + t[-1] <= p[0][-1]}
    wn = min(e, key=e.get) if e else None
    print(f'{cap} {band} T{i}: LS {"accepted" if ok else "not":<8} {best["satellite"]:<10} rmse {best["rmse_bins"]:6.1f} '
          f'2nd {second["rmse_bins"]:6.1f} | web {"match" if wn and e[wn] < 1500 else "none":<5} {wn} {e[wn] if wn else np.nan:8.0f} Hz '
          f'| ours {ours.get(i)}')
