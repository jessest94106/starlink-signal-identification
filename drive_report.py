"""Summarize run_drive.py results: per capture and per date, confirmed rate, unknowns (long = >= 30 s), start time;
and, for two-receiver setups, whether simultaneous rx1/rx2 recordings confirm the same satellites.

Usage: drive_report.py > data/drive/report.txt
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / 'data/drive'
LONG_S = 30.0


def load(cap):
    s = json.loads((cap / 'summary.json').read_text())
    st = {}
    if (cap / 'staged.txt').exists():
        d = np.loadtxt(cap / 'tracks.csv', delimiter=',', skiprows=1, ndmin=2)
        for i, rest in re.findall(r'^track (\d+): (.*)$', (cap / 'staged.txt').read_text(), re.M):
            t = d[d[:, 0] == int(i), 1]
            m = re.match(r'confirmed (SAT-\d+)', rest)
            st[int(i)] = (rest.split()[0].split('/')[0], m[1] if m else None, t[0], t[-1])
    return s, st


caps = sorted(p.parent for p in ROOT.glob('*/*/*/summary.json'))
rows, by_date, sats = [], defaultdict(lambda: defaultdict(int)), {}
print(f'{"capture":<62} {"min":>4} {"trk":>4} {"conf":>4} {"line":>4} {"look":>4} {"amb":>4} {"unk":>4} {"unkL":>4}  start(file-switch) s')
for cap in caps:
    s, st = load(cap)
    c = defaultdict(int)
    for k, (status, n, t0, t1) in st.items():
        c[{'confirmed': 'conf', 'not-LEO': 'line', 'look-alike': 'look', 'ambiguous': 'amb'}.get(status, 'unk')] += 1
        if status not in ('confirmed', 'not-LEO', 'look-alike', 'ambiguous') and t1 - t0 >= LONG_S:
            c['unkL'] += 1
    off = ''
    if 'start_switches' in s:
        ts = lambda x: np.datetime64(x, 'ms').astype('int64') / 1e3
        off = f'{ts(s["start_file"]) - ts(s["start_switches"]):+6.2f} ({s["switches"]} switches)'
    name = '/'.join(cap.parts[-3:])
    print(f'{name:<62} {s["duration_s"] / 60:4.1f} {s["tracks"]:4d} {c["conf"]:4d} {c["line"]:4d} {c["look"]:4d} {c["amb"]:4d} '
          f'{c["unk"]:4d} {c["unkL"]:4d}  {off}')
    date = cap.parts[-3]
    for k in ('conf', 'line', 'look', 'amb', 'unk', 'unkL'):
        by_date[date][k] += c[k]
    by_date[date]['trk'] += s['tracks']
    by_date[date]['files'] += 1
    sats[cap] = {n for status, n, _, _ in st.values() if n}

print('\nper date:')
tot = defaultdict(int)
for date, c in sorted(by_date.items()):
    for k, v in c.items():
        tot[k] += v
for date, c in sorted(by_date.items()) + [('all', tot)]:
    n = max(c['trk'], 1)
    print(f'{date:<10} {c["files"]:3d} files {c["trk"]:4d} tracks: confirmed {c["conf"] + c["line"]:4d} ({(c["conf"] + c["line"]) / n:4.0%}; '
          f'{c["conf"]} IDs + {c["line"]} not-LEO), look-alike {c["look"]}, ambiguous {c["amb"]}, unknown {c["unk"]} '
          f'(long {c["unkL"]})')

print('\ntwo receivers, same time (rx1 vs rx2 confirmed satellites):')
for cap in caps:
    if not cap.parts[-2].endswith('_rx1'):
        continue
    t1 = cap.name.split('_')[-1]
    for other in caps:
        if other.parts[-2] == cap.parts[-2][:-1] + '2' and other.name.split('_')[1] == cap.name.split('_')[1]:
            dt = abs((np.datetime64(f'{t1[:4]}-{t1[4:6]}-{t1[6:8]}T{t1[9:11]}:{t1[11:13]}:{t1[13:]}') -
                      np.datetime64(f'{(o := other.name.split("_")[-1])[:4]}-{o[4:6]}-{o[6:8]}T{o[9:11]}:{o[11:13]}:{o[13:]}')).astype(int))
            if dt <= 120:
                a, b = sats[cap], sats[other]
                print(f'{"/".join(cap.parts[-2:]):<55} both {len(a & b):3d}  rx1 only {len(a - b):3d}  rx2 only {len(b - a):3d}')

# Baseline (baseline_all.py) vs latest (staged) on the same tracks. Latest: ambiguous = ambiguous, look-alike, or a
# different answer from the two start times; unknown = no satellite within the limits in either run.
print('\nbaseline vs latest, same tracks (per date; baseline determined split: agrees / differs / latest has no ID):')
KIND = lambda r: ('determined' if r.startswith('confirmed') else 'not-sat' if r.startswith('not-LEO')
                  else 'unknown' if r.split()[0] == 'unknown' else 'ambiguous')
tab = defaultdict(lambda: defaultdict(int))
for cap in caps:
    if not (cap / 'baseline.txt').exists():
        continue
    st = dict((int(i), r) for i, r in re.findall(r'^track (\d+): (.*)$', (cap / 'staged.txt').read_text(), re.M))
    for i, r in re.findall(r'^track (\d+): (.*)$', (cap / 'baseline.txt').read_text(), re.M):
        lab, sat = re.match(r'LS (\S+)\s+(SAT-\d+)', r).groups()
        web, wsat = re.search(r'web (\S+)\s+(SAT-\d+)', r).groups()
        k = KIND(st[int(i)]); ours = re.match(r'confirmed (SAT-\d+)', st[int(i)])
        for dte in (cap.parts[-3], 'all'):
            c = tab[dte]
            c['tracks'] += 1; c['L ' + k] += 1; c['B ' + lab] += 1
            if lab == 'determined':
                c['B agree' if ours and ours[1] == sat else 'B differ' if ours else 'B noref'] += 1
            if web == 'match':
                c['W match'] += 1; c['W agree'] += bool(ours and ours[1] == wsat)
print(f'{"date":<11}{"tracks":>7} | latest: {"det":>4}{"notsat":>7}{"amb":>5}{"unk":>5} | baseline: {"det":>4} (agree/differ/noref) {"amb":>5}{"unk":>5} | web match (agree)')
for dte, c in sorted(tab.items(), key=lambda kv: (kv[0] == 'all', kv[0])):
    print(f'{dte:<11}{c["tracks"]:7d} | latest: {c["L determined"]:4d}{c["L not-sat"]:7d}{c["L ambiguous"]:5d}{c["L unknown"]:5d} | '
          f'baseline: {c["B determined"]:4d} ({c["B agree"]}/{c["B differ"]}/{c["B noref"]}) {c["B ambiguous"]:13d}{c["B unknown"]:5d} | '
          f'web {c["W match"]} ({c["W agree"]})')
