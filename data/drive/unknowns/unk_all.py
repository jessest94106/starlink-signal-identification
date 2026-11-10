"""All long unknowns (>=30 s) across Drive captures: shape, best free fit, best-fit satellite already used elsewhere?"""
import sys, json, re, pickle
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from norad_id import identify, load_tles
from run_capture import curves

ROOT = Path(__file__).resolve().parents[1]
ts = lambda s: np.datetime64(s, 'ms').astype('int64') / 1e3

def one(cap):
    s = json.loads((cap / 'summary.json').read_text())
    d = np.loadtxt(cap / 'tracks.csv', delimiter=',', skiprows=1, ndmin=2)
    st = dict((int(i), r) for i, r in re.findall(r'^track (\d+): (.*)$', (cap / 'staged.txt').read_text(), re.M))
    raw = {i: (d[d[:, 0] == i, 1], d[d[:, 0] == i, 2], d[d[:, 0] == i, 3]) for i in st}
    kind = {i: r.split()[0].split('/')[0] for i, r in st.items()}
    conf = {i: re.match(r'confirmed (SAT-\d+)', r)[1] for i, r in st.items() if r.startswith('confirmed')}
    longu = [i for i in st if kind[i] not in ('confirmed', 'not-LEO', 'look-alike', 'ambiguous') and raw[i][0][-1] - raw[i][0][0] >= 30]
    if not longu:
        return []
    t0 = ts(s['start_switches'])
    sats = load_tles(f'{ROOT.parents[1]}/tles/{cap.parts[-3]}.tle', at_unix=t0)
    preds = curves(sats, t0, s['duration_s'], s['fc'])
    ids = longu + list(conf)
    trk = [(t0 + raw[i][0], raw[i][1]) for i in ids]
    res = identify(trk, preds, sigma_hz=3, sigma_dt_s=1.0, max_rms_hz=1e9, top=3)[0]
    # halves
    half = []
    for i in longu:
        t, f, _ = raw[i]; m = len(t) // 2
        r = identify([(t0 + t[:m], f[:m]), (t0 + t[m:], f[m:])], preds, sigma_hz=3, sigma_dt_s=1.0, max_rms_hz=1e9, top=1)[0]
        half.append([(x['norad'], x['rms_hz']) for x in r])
    out = []
    confoff = {i: res[len(longu) + k]['offset_hz'] for k, i in enumerate(conf)}
    for k, i in enumerate(longu):
        t, f, snr = raw[i]
        p = np.polyfit(t - t.mean(), f, 2)
        r = res[k]
        used = [(j, raw[j][0][0], raw[j][0][-1], confoff[j]) for j, n in conf.items() if n == r['norad']]
        out.append(dict(cap='/'.join(cap.parts[-3:]), date=cap.parts[-3], track=i, dur=t[-1] - t[0], t0=t[0], t1=t[-1],
                        f0=f[0], f1=f[-1], rate=p[1], curv=2 * p[0], snr=np.median(snr), npts=len(t),
                        best=r['norad'], rms=r['rms_hz'], off=r['offset_hz'], dt=r['dt_s'], cands=r['candidates'],
                        used=used, half=half[k], fc=s['fc'], tstart=t0, status=st[i],
                        conf_rates=[np.polyfit(raw[j][0], raw[j][1], 1)[0] for j in conf],
                        conf_offs=list(confoff.values())))
    return out

if __name__ == '__main__':
    caps = sorted(p.parent for p in ROOT.glob('*/*/*/staged.txt') if '2026-01-20' not in str(p))
    with ProcessPoolExecutor(8) as ex:
        rows = [r for rs in ex.map(one, caps) for r in rs]
    pickle.dump(rows, open(sys.argv[1], 'wb'))
    print(len(rows))
