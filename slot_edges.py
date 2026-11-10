"""Usage: slot_edges.py CAPTURE_INDEX(0-3) > data/slot_edges_<i>.txt
Power steps of every accepted satellite's tone (0.1-s blocks after removing its predicted Doppler); phase within the 15-s GPS cycle."""
import sys, re, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.interpolate import CubicSpline
from norad_id import load_sim_csvs
R = Path(__file__).resolve().parent
caps = [('c204624', '11.575GHz', 11.575e9, '2026-05-04T20:46:22.75'), ('c213016', '12.325GHz', 12.325e9, '2026-05-04T21:30:15.50'),
        ('c224056', '11.575GHz', 11.575e9, '2026-05-04T22:40:54.37'), ('c224056', '12.325GHz', 12.325e9, '2026-05-04T22:46:09.75')]
cap, band, fc, start = caps[int(sys.argv[1])]
fs, n = 500e3, 50_000
data = next((R / 'data' / cap).glob(f'*_f{band}_*.sigmf-data'))
x = np.memmap(data, dtype=np.complex64, mode='r')
t0 = np.datetime64(start, 'ns').astype('int64') / 1e9
sats = {}
for l in (R / 'data' / cap / f'match_{band}.txt').read_text().splitlines():
    m = re.match(r'track \d+: accepted\s+(SAT-\d+).*offset\s+(\S+) kHz\s+dt\s+(\S+) s', l)
    if m and m.group(1) not in sats and abs(float(m.group(2))) < 250:
        sats[m.group(1)] = (float(m.group(2)) * 1e3, float(m.group(3)))
fr = np.fft.fftfreq(n, 1 / fs); band_ = np.abs(fr) < 400
edges = {}
for norad, (off, dt) in sats.items():
    tp, fp = load_sim_csvs([R / 'curves' / f'{cap}_{band}' / f'{norad}.csv'], fc)[norad]; sp = CubicSpline(tp, fp)
    ph, P, T = 0.0, [], []
    for k in range(x.size // n):
        ta = t0 + dt + (k * n + np.arange(n)) / fs
        if ta[0] < tp[0] or ta[-1] > tp[-1]: ph = 0.0; continue
        p = ph + 2 * np.pi * np.cumsum(sp(ta) + off) / fs; ph = p[-1] % (2 * np.pi)
        Y = np.abs(np.fft.fft(np.asarray(x[k * n:(k + 1) * n]) * np.exp(-1j * p))) ** 2
        P.append(10 * np.log10(Y[band_].max() / np.median(Y))); T.append(k * n / fs)
    P, T = np.array(P), np.array(T)
    L = np.array([P[max(0, k - 10):k].mean() if k >= 3 else np.nan for k in range(len(P))])
    Rr = np.array([P[k:k + 10].mean() if k <= len(P) - 3 else np.nan for k in range(len(P))])
    step = Rr - L
    for k in np.argsort(-np.abs(np.nan_to_num(step))):
        if abs(step[k]) < 6: break
        if np.any(np.abs(T[k] - np.array([e[0] for e in edges.get(norad, [])] or [1e9])) < 2): continue
        edges.setdefault(norad, []).append((T[k], step[k]))
    for te, s in sorted(edges.get(norad, [])):
        utc = t0 + te
        print(f'{cap} {band} {norad} edge {te:7.1f} s step {s:+5.1f} dB  GPS mod 15 = {(utc + 18) % 15:5.2f}')
