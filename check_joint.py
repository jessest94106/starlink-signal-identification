"""Missing-satellite test for joint_id: for every accepted track, remove its satellite from the candidates and rerun.
Pass = the track is no longer accepted (unknown or ambiguous); fail = it is confidently given another satellite.

Usage: check_joint.py JOINT.txt TRACKS.csv CURVES_DIR CARRIER_HZ START_UTC TLE_FILE
"""
import re
import sys
from pathlib import Path

import numpy as np

from joint_id import joint
from norad_id import load_sim_csvs, load_tles

joint_txt, tracks_csv, curves_dir, fc, start, tle = sys.argv[1:7]
acc = [(int(m.group(1)), m.group(2)) for l in Path(joint_txt).read_text().splitlines()
       for m in [re.match(r'track (\d+): accepted\s+(\S+)', l)] if m]
d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
t0 = np.datetime64(start.removesuffix('Z'), 'ns').astype('int64') / 1e9
tracks = [(t0 + d[d[:, 0] == i, 1], d[d[:, 0] == i, 2]) for i in np.unique(d[:, 0])]
preds = load_sim_csvs(sorted(Path(curves_dir).glob('*.csv')), float(fc))
sats = load_tles(tle, at_unix=t0)
fails = 0
for i, n in acc:
    r = joint(tracks, {k: v for k, v in preds.items() if k != n}, t0, sats)[0][i]
    fails += r['status'] == 'accepted'
    print(f'T{i} {n} removed -> {r["status"]} {r["norad"]}' + ('   <-- FAIL' if r['status'] == 'accepted' else ''))
print(f'{fails} of {len(acc)} accepted tracks confidently re-labelled when their satellite was removed')
