"""Residual vs predicted elevation: partial-match unknowns (fit on good half) and confirmed tracks (fit on whole track)."""
import sys, pickle, re, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from norad_id import load_tles, predict

rows = pickle.load(open(sys.argv[1], 'rb'))
ROOT = Path(__file__).resolve().parents[1]
ts = lambda s: np.datetime64(s, 'ms').astype('int64') / 1e3
SATS = {}
def sats_of(date, t0):
    if date not in SATS: SATS[date] = load_tles(f'{ROOT.parents[1]}/tles/{date}.tle', at_unix=t0)
    return SATS[date]
def fit(t, z, n, sats, fc, good):
    tg = np.arange(t[0] - 20, t[-1] + 20, 0.5)
    pr, el = predict({n: sats[n]}, tg, fc, min_elev_deg=-90)
    sp, se = CubicSpline(*pr[n]), CubicSpline(tg, el[n])
    dt = minimize_scalar(lambda dt: np.var(z[good] - sp(t[good] + dt)), bounds=(-5, 5), method='bounded').x
    off = np.mean(z[good] - sp(t[good] + dt))
    return z - sp(t + dt) - off, se(t + dt), sp(t + dt, 1)
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
# unknowns with a good half
for r in rows:
    if not (min(e for _, e in r['half']) < 25 and len({n for n, _ in r['half']}) == 1): continue
    d = np.loadtxt(ROOT / r['cap'] / 'tracks.csv', delimiter=',', skiprows=1, ndmin=2); p = d[d[:, 0] == r['track']]
    t, z = r['tstart'] + p[:, 1], p[:, 2]; m = len(t) // 2
    good = slice(0, m) if r['half'][0][1] < r['half'][1][1] else slice(m, None)
    res, el, g = fit(t, z, r['best'].removeprefix('SAT-'), sats_of(r['date'], r['tstart']), r['fc'], good)
    ax[0].plot(el, np.abs(res), 'r-', lw=0.7, alpha=0.6)
    ax[2].plot(np.abs(g), np.abs(res), 'r-', lw=0.7, alpha=0.6)
# confirmed tracks >= 20 s
nconf = []
for cap in sorted(p.parent for p in ROOT.glob('*/*/*/staged.txt') if '2026-01-20' not in str(p)):
    s = json.loads((cap / 'summary.json').read_text()); t0 = ts(s['start_switches'])
    d = np.loadtxt(cap / 'tracks.csv', delimiter=',', skiprows=1, ndmin=2)
    for i, n in re.findall(r'^track (\d+): confirmed SAT-(\d+)', (cap / 'staged.txt').read_text(), re.M):
        p = d[d[:, 0] == int(i)]
        if p[-1, 1] - p[0, 1] < 20: continue
        t, z = t0 + p[:, 1], p[:, 2]
        res, el, g = fit(t, z, n, sats_of(cap.parts[-3], t0), s['fc'], slice(None))
        ax[1].plot(el, np.abs(res), 'b-', lw=0.6, alpha=0.5)
        ax[2].plot(np.abs(g), np.abs(res), 'b-', lw=0.4, alpha=0.3)
        nconf.append((el.max(), np.sqrt(np.mean(res ** 2))))
nconf = np.array(nconf)
for lo in (0, 50, 60, 70, 80):
    s = nconf[(nconf[:, 0] >= lo) & (nconf[:, 0] < lo + (50 if lo == 0 else 10))]
    print(f'confirmed >=20 s, max elev {lo}-{lo + (50 if lo == 0 else 10)}: {len(s)} tracks, median rms {np.median(s[:, 1]) if len(s) else np.nan:.1f} Hz')
for a, ttl in zip(ax, ['unknowns: |residual| vs elevation (fit on good half)', 'confirmed tracks >= 20 s: |residual| vs elevation', '|residual| vs Doppler steepness (red unknown, blue confirmed)']):
    a.set_ylim(0, 600); a.set_title(ttl); a.set_ylabel('|residual| Hz'); a.grid(alpha=0.3)
ax[0].set_xlabel('elevation deg'); ax[1].set_xlabel('elevation deg'); ax[2].set_xlabel('|Doppler rate| Hz/s')
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=70)
