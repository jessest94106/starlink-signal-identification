"""Same satellite, several tracks (confirmed + unknown): can ONE orbit correction (along-track dt, radial, cross-track km)
fit all of them (offset free per track)? Compare: confirmed-only fit, then unknown with that orbit, then all jointly."""
import sys, pickle, re, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from norad_id import load_tles, _geometry, SIEG, C
ROOT = Path(__file__).resolve().parents[1]
ts = lambda s: np.datetime64(s, 'ms').astype('int64') / 1e3
rows = pickle.load(open(sys.argv[1], 'rb'))
def dop(sat, t, fc, q):
    _, rd, el = _geometry([sat], t, SIEG, d_rc=(q[1], q[2]), t_orbit=t + q[0])
    return -rd[0] * fc / C, el[0]
def resid(q, sat, trks, fc):
    out = []
    for t, z in trks:
        D, _ = dop(sat, t, fc, q)
        r = z - D; out.append(r - np.median(r))
    return np.concatenate(out)
def rms_each(q, sat, trks, fc):
    out = []
    for t, z in trks:
        D, el = dop(sat, t, fc, q); r = z - D; r -= np.mean(r)
        out.append((np.sqrt(np.mean(r ** 2)), el.max()))
    return out
for r in rows:
    if not r['used'] or abs(r['off'] - r['used'][0][3]) > 2e3: continue
    cap = ROOT / r['cap']
    s = json.loads((cap / 'summary.json').read_text()); t0 = ts(s['start_switches'])
    d = np.loadtxt(cap / 'tracks.csv', delimiter=',', skiprows=1, ndmin=2)
    n = r['best'].removeprefix('SAT-'); sat = load_tles(f'{ROOT.parents[1]}/tles/{r["date"]}.tle', at_unix=t0)[n]
    tr = lambda i: (t0 + d[d[:, 0] == i, 1], d[d[:, 0] == i, 2])
    conf = [tr(j) for j, *_ in r['used']]; unk = tr(r['track'])
    def fit(trks):      # absolute finite-difference steps: t ~ 1.8e9 s, a relative step at dt = 0 vanishes
        f = lambda q: resid(q, sat, trks, s['fc'])
        J = lambda q: np.stack([(f(q + h) - f(q - h)) / (2 * h.sum()) for h in np.diag([1e-3, 1e-2, 1e-2])], 1)
        best = min((least_squares(f, [d0, 0, 0], jac=J, loss='soft_l1', f_scale=10, bounds=([-5, -30, -30], [5, 30, 30]))
                    for d0 in (-2, 0, 2)), key=lambda r: r.cost)
        return best.x
    qc, qa = fit(conf), fit(conf + [unk])
    f = lambda q, trks: ' '.join(f'{e:5.1f}' for e, _ in rms_each(q, sat, trks, s['fc']))
    print(f'{r["cap"][-42:]} {r["best"]} unknown T{r["track"]} ({unk[0][-1] - unk[0][0]:.0f} s, max el {rms_each([0,0,0], sat, [unk], s["fc"])[0][1]:.0f}) '
          f'+ confirmed {[f"T{j}" for j, *_ in r["used"]]}\n'
          f'   TLE as is:          confirmed rms {f([0, 0, 0], conf)} | unknown {f([0, 0, 0], [unk])}\n'
          f'   orbit fit to confirmed (dt {qc[0]:+.2f} s, R {qc[1]:+5.1f} C {qc[2]:+5.1f} km): unknown {f(qc, [unk])}\n'
          f'   orbit fit to all     (dt {qa[0]:+.2f} s, R {qa[1]:+5.1f} C {qa[2]:+5.1f} km): confirmed {f(qa, conf)} | unknown {f(qa, [unk])}', flush=True)
