"""Staged identification of one capture: strict limits first, then loosen step by step. Confirmed tracks stay confirmed.

Per stage (mismatch limit, orbit-shift allowance), only tracks not yet confirmed can be confirmed:
  - joint_id.joint is run with the start from the 15-s tone switches and with the start from the orbit fingerprint;
  - confirmed = accepted in both runs with the same satellite, and the look-alike check passes: with that satellite
    hidden from this track, neither run accepts another satellite for it;
  - confirmed tracks are frozen (label kept; they still shape the shared receiver offset and per-satellite numbers).
Constant lines (|rate| < 50 Hz/s over > 5 s) are labelled not-LEO up front: a LEO satellite cannot hold its Doppler.

Usage: staged_id.py TRACKS.csv CURVES_DIR CARRIER_HZ START_SWITCHES START_FINGERPRINT TLE_FILE OUT.txt
"""
import sys
from pathlib import Path

import numpy as np

from joint_id import joint
from norad_id import load_sim_csvs, load_tles

STAGES = [(15, 0.5), (20, 0.5), (30, 1.0), (50, 2.0)]     # (mismatch limit Hz, orbit-shift allowance s; clip at 3x)
LINE_RATE, LINE_DUR = 50.0, 5.0


def staged(raw, preds, t_sw, t_fp, sats):
    """raw {track: (t_rel, f)}; t_sw, t_fp: unix starts. Returns the report lines."""
    ids = sorted(raw)
    lines = [i for i, (t, f) in raw.items() if abs(np.polyfit(t, f, 1)[0]) < LINE_RATE and t[-1] - t[0] > LINE_DUR]
    sat_ids = [i for i in ids if i not in lines]
    runs = {k: (t0, [(t0 + raw[i][0], raw[i][1]) for i in sat_ids]) for k, t0 in (('sw', t_sw), ('fp', t_fp))}
    J = lambda k, **kw: joint(runs[k][1], preds, runs[k][0], sats, **kw)[0]

    frozen, info, look, last = {}, {}, {}, {}
    log = [f'{len(ids)} tracks: {len(lines)} constant lines (not-LEO), {len(sat_ids)} satellite-like']
    for L, dtf in STAGES if sat_ids else []:
        kw = dict(gamma_rms=L, dt_floor=dtf, frozen=dict(frozen))
        rs, rf = J('sw', **kw), J('fp', **kw)
        last = {i: (rs[i], rf[i]) for i in range(len(sat_ids))}
        new = [i for i in range(len(sat_ids)) if i not in frozen and rs[i]['status'] == rf[i]['status'] == 'accepted'
               and rs[i]['norad'] == rf[i]['norad']]
        got = 0
        for i in new:
            n = rs[i]['norad']
            a, b = J('sw', exclude={i: n}, **kw)[i], J('fp', exclude={i: n}, **kw)[i]
            if a['status'] != 'accepted' and b['status'] != 'accepted':
                frozen[i], got = n, got + 1
                info[i] = (L, dtf, rs[i]['rms_hz'], rs[i]['dt_s'], rs[i]['c_hz'])
                look.pop(i, None)
            else:
                look[i] = (n, a['norad'] if a['status'] == 'accepted' else b['norad'])
        log.append(f'stage limit {L} Hz, orbit allowance {dtf} s (clip {3 * dtf:.1f} s): {len(new)} candidates, {got} confirmed '
                   f'(total {len(frozen)})')

    out = log[:]
    for i in ids:
        if i in lines:
            out.append(f'track {i}: not-LEO   constant line, rate {np.polyfit(*raw[i], 1)[0]:+.1f} Hz/s'); continue
        j = sat_ids.index(i)
        if j in frozen:
            L, dtf, rms, dt, c = info[j]
            out.append(f'track {i}: confirmed {frozen[j]:<10} at stage {L} Hz / {dtf} s  rms {rms:5.1f} Hz  dt {dt:+.2f} s  sat offset {c / 1e3:+.2f} kHz')
        elif j in look:
            out.append(f'track {i}: look-alike {look[j][0]} / {look[j][1]}')
        else:
            a, b = last[j]
            st = a['status'] if a['status'] == b['status'] else f'{a["status"]}/{b["status"]}'
            out.append(f'track {i}: {st:<9} {a["norad"]} / {b["norad"]}  rms {a["rms_hz"]:.1f} Hz')
    n_conf, n_line = len(frozen), len(lines)
    out.append(f'confirmed: {n_conf} IDs + {n_line} not-LEO = {n_conf + n_line} of {len(ids)} ({(n_conf + n_line) / max(len(ids), 1):.0%}); '
               f'IDs {n_conf} of {len(sat_ids)} satellite-like ({n_conf / max(len(sat_ids), 1):.0%})')
    return out


if __name__ == '__main__':
    tracks_csv, curves_dir, fc, start_sw, start_fp, tle, out_txt = sys.argv[1:8]
    d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
    raw = {i: (d[d[:, 0] == i, 1], d[d[:, 0] == i, 2]) for i in np.unique(d[:, 0]).astype(int)}
    preds = load_sim_csvs(sorted(Path(curves_dir).glob('*.csv')), float(fc))
    ts = lambda s: np.datetime64(s.removesuffix('Z'), 'ns').astype('int64') / 1e9
    out = staged(raw, preds, ts(start_sw), ts(start_fp), load_tles(tle, at_unix=ts(start_sw)))
    Path(out_txt).write_text('\n'.join(out) + '\n')
    print('\n'.join(out))
