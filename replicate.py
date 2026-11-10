"""Re-run the identification of every Drive capture from the files in this repo (no raw I/Q needed) and compare it with
the stored result.

Per capture data/drive/<date>/<setup>/<capture>/ it uses tracks.csv (tone tracks from the raw I/Q), summary.json
(file-name time, duration, and the two start times found from the raw I/Q) and tles/<date>.tle. Predicted curves are
rebuilt as in run_capture.py, then staged_id.staged runs with the stored start times. Nothing is written.

Usage: replicate.py [WORKERS]
Prints one line per capture (same / DIFF), then the totals. A capture is "same" when every track gets the same status
and NORAD ID as in staged.txt; "text" says whether the whole report (incl. rms, dt, offset) is identical.
"""
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from norad_id import load_tles
from run_capture import curves
from staged_id import staged

HERE = Path(__file__).resolve().parent
ts = lambda s: np.datetime64(s, 'ms').astype('int64') / 1e3
LABEL = lambda lines: {int(i): ' '.join(r.split()[:2]) for i, r in re.findall(r'^track (\d+): (.*)$', '\n'.join(lines), re.M)}


def one(cap):
    s = json.loads((cap / 'summary.json').read_text())
    if 'start_switches' not in s:
        return cap, None, None, 0
    d = np.loadtxt(cap / 'tracks.csv', delimiter=',', skiprows=1, ndmin=2)
    raw = {i: (d[d[:, 0] == i, 1], d[d[:, 0] == i, 2]) for i in np.unique(d[:, 0]).astype(int)}
    t_file = ts(s['start_file'])
    sats = load_tles(str(HERE / 'tles' / f'{cap.parts[-3]}.tle'), at_unix=t_file)
    new = staged(raw, curves(sats, t_file, s['duration_s'], s['fc']), ts(s['start_switches']), ts(s['start_fingerprint']), sats)
    old = (cap / 'staged.txt').read_text().splitlines()
    a, b = LABEL(old), LABEL(new)
    return cap, a == b, old == new, sum(a[k] != b.get(k) for k in a)


if __name__ == '__main__':
    caps = sorted(p.parent for p in (HERE / 'data/drive').glob('*/*/*/summary.json'))
    n_same = n_text = n_run = n_diff_tracks = 0
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        for cap, same, text, k in ex.map(one, caps):
            name = '/'.join(cap.parts[-3:])
            if same is None:
                print(f'{name:<66} skip (no tracks)', flush=True)
                continue
            n_run += 1; n_same += same; n_text += text; n_diff_tracks += k
            print(f'{name:<66} {"same" if same else f"DIFF ({k} tracks)"}{"" if text else "  (text differs)"}', flush=True)
    print(f'{n_same} of {n_run} captures give the same IDs ({n_diff_tracks} tracks differ); {n_text} identical reports')
