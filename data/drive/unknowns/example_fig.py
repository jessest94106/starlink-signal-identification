"""Baseline vs latest on one capture: waterfall + tracks colored by outcome, labeled with NORAD IDs."""
import sys, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cap, spec, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
S = np.load(spec); S -= np.median(S, 0); S -= np.median(S, 1, keepdims=True)
d = np.loadtxt(cap / 'tracks.csv', delimiter=',', skiprows=1, ndmin=2)
staged = dict((int(i), r) for i, r in re.findall(r'^track (\d+): (.*)$', (cap / 'staged.txt').read_text(), re.M))
base = dict((int(i), r) for i, r in re.findall(r'^track (\d+): (.*)$', (cap / 'baseline.txt').read_text(), re.M))
C = dict(determined='#1a9e4a', notsat='#2f6fd6', ambiguous='#f08c00', unknown='#d62f2f', wrong='#9b30d9')
def latest(i):
    r = staged[i]; m = re.match(r'confirmed SAT-(\d+)', r)
    if m: return 'determined', m[1]
    if r.startswith('not-LEO'): return 'notsat', 'line'
    if r.split()[0] == 'unknown': return 'unknown', '?'
    return 'ambiguous', '?'
def baseline(i):
    lab, sat = re.match(r'LS (\S+)\s+SAT-(\d+)', base[i]).groups()
    if lab != 'determined': return lab, '?'
    k, n = latest(i)
    return ('determined' if (k == 'determined' and n == sat) else 'wrong'), sat
fig, axs = plt.subplots(1, 2, figsize=(20, 8.5), sharey=True)
ext = [0, S.shape[0] * 0.25, -250, 250]
for ax, fn, ttl in [(axs[0], baseline, 'Baseline: Drive matcher (free time slide) + accept rule'), (axs[1], latest, 'Latest: staged joint ID (fixed start, both starts, look-alike check)')]:
    ax.imshow(S.T, origin='lower', aspect='auto', extent=ext, cmap='Greys', vmin=1.5, vmax=7)
    cnt = {}
    for i in sorted(staged):
        p = d[d[:, 0] == i]
        k, n = fn(i); cnt[k] = cnt.get(k, 0) + 1
        ax.plot(p[:, 1], p[:, 2] / 1e3, '-', color=C[k], lw=2.4, solid_capstyle='round', alpha=0.95)
        j = len(p) // 2
        ax.annotate(f'T{i} {n}', (p[j, 1], p[j, 2] / 1e3), xytext=(6, 6 if i % 2 else -14), textcoords='offset points', fontsize=9,
                    color=C[k], fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.set_title(ttl + '\n' + ', '.join(f'{v} {dict(determined="determined", notsat="not-satellite", ambiguous="ambiguous", unknown="unknown", wrong="differ from latest")[k]}' for k, v in sorted(cnt.items(), key=lambda kv: list(C).index(kv[0]))), fontsize=12)
    ax.set_xlabel('time from first sample (s)', fontsize=11)
axs[0].set_ylabel('baseband frequency (kHz)', fontsize=11); axs[0].set_ylim(-250, 250)
for k, lab in [('determined', 'determined (agrees with latest)'), ('wrong', 'determined, disagrees with latest'), ('notsat', 'not a satellite (constant line)'), ('ambiguous', 'ambiguous'), ('unknown', 'unknown')]:
    axs[1].plot([], [], color=C[k], lw=4, label=lab)
fig.legend(loc='lower center', ncol=5, fontsize=11, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out, dpi=110)
