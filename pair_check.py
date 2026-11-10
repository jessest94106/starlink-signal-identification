"""Two receivers, same time: find tones seen by both (same Doppler shape over >= 5 s, constant frequency difference =
the two receivers' offset difference; a relative time shift of up to +-3 s is allowed, since each receiver's start
is estimated on its own) and compare their labels.

Usage: pair_check.py   (reads data/drive/*/*_rx1|_rx2/*)
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / 'data/drive'
ts = lambda s: np.datetime64(s, 'ms').astype('int64') / 1e3


def load(cap):
    s = json.loads((cap / 'summary.json').read_text())
    if 'start_switches' not in s:
        return []
    d = np.loadtxt(cap / 'tracks.csv', delimiter=',', skiprows=1, ndmin=2)
    lab = {int(i): r for i, r in re.findall(r'^track (\d+): (.*)$', (cap / 'staged.txt').read_text(), re.M)}
    t0 = ts(s['start_switches'])
    return [(t0 + d[d[:, 0] == i, 1], d[d[:, 0] == i, 2], lab[i]) for i in lab]


tot = dict(agree=0, disagree=0, one=0, neither=0)
for c1 in sorted(ROOT.glob('*/*_rx1/*/summary.json')):
    c1 = c1.parent
    for c2 in sorted(ROOT.glob(f'{c1.parts[-3]}/{c1.parts[-2][:-1]}2/*/summary.json')):
        c2 = c2.parent
        if c2.name.split('_')[1] != c1.name.split('_')[1] or abs(ts(json.loads((c1 / 'summary.json').read_text()).get('start_file', '1970-01-01')) - ts(json.loads((c2 / 'summary.json').read_text()).get('start_file', '1970-01-01'))) > 120:
            continue
        A, B = load(c1), load(c2)
        for ta, fa, la in A:
            for tb, fb, lb in B:
                best = None
                for sh in np.arange(-3, 3.001, 0.05):
                    m = (tb + sh >= ta[0]) & (tb + sh <= ta[-1])
                    if m.sum() < 15 or tb[m][-1] - tb[m][0] < 5:
                        continue
                    df = np.interp(tb[m] + sh, ta, fa) - fb[m]
                    if best is None or np.std(df) < best[0]:
                        best = (np.std(df), df)
                if best is None or best[0] > 20:
                    continue
                df = best[1]
                ca, cb = [re.match(r'confirmed (SAT-\d+)', l) for l in (la, lb)]
                k = 'agree' if ca and cb and ca[1] == cb[1] else 'disagree' if ca and cb else 'one' if ca or cb else 'neither'
                tot[k] += 1
                if True:
                    print(f'{c1.parts[-2][:-4]} {c1.name}: offset diff {np.median(df) / 1e3:+.2f} kHz  rx1 [{la[:40]}]  rx2 [{lb[:40]}]')
print(tot)
