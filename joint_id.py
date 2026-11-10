"""Joint NORAD identification of all tone tracks of one capture, steps 6-9 (report section 3a).

Model for track i labelled satellite j:
    z_i(t) = f_j(t + tau + dt_j) + b_w(t) + c_j + noise
shared by the capture: b_w (receiver offset per 2-min window, neighbours kept close; a constant per window cannot
fake a start-time error inside a track), tau (start-time correction; fixed at 0 by default: start from 15-s edges);
per satellite: c_j (tone offset, spread SIGMA_C) and dt_j (orbit shift, size from TLE age and generation).

Loop until labels stop changing:
  1. cost of every (track, candidate) with the shared numbers fixed; a candidate that already explains other
     tracks brings its offset/shift from those tracks as a tight prior;
  2. choose all labels at once (integer program): one satellite cannot make two different lines at the same
     time; "none of these" costs GAMMA_RMS^2 per point;
  3. refit shared numbers + per-satellite numbers from all labelled tracks.
Ambiguity: forbid each track's label, re-solve; total cost rises by < max(9, 3 x own cost) => ambiguous.

Usage: joint_id.py TRACKS.csv CURVES_DIR CARRIER_HZ START_UTC TLE_FILE OUT.txt
START_UTC: first-sample time from match_capture.py.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import LinearConstraint, milp

from norad_id import identify, load_sim_csvs, load_tles

FS = 500e3                      # recording bandwidth: offsets wrap every FS (band-edge alias)
SIGMA = 4.0                     # Hz per point: track noise + model error
CAP, MAX_OUT = 5.0, 0.1         # robust: a point costs at most CAP^2; more than 10 % such points => candidate invalid
GAMMA_RMS = 15.0                # "none of these" = 15 Hz RMS per point
SIGMA_C, SIGMA_C_TIGHT = 9e3, 2e3         # tone offset spread; same satellite in another track (repeats agree to 0.1-2 kHz)
SIGMA_DT_TIGHT, SIGMA_TAU = 0.4, 2.0      # repeats of one satellite differ in orbit shift by up to 0.8 s
WINDOW_S, SIGMA_STEP, SIGMA_BABS = 120.0, 1500.0, 50e3     # receiver-offset windows; neighbour step; weak absolute prior
TOP_K, OVERLAP_S, SAME_LINE_HZ = 10, 1.0, 1e3
# 2026-05-04 GP TLEs: fitted per-satellite shifts median 0.34 s, 90% 0.85 s (model gave ~0.05-0.1 s) => floor 0.5 s
DT_FLOOR = 0.5
wrap = lambda x: (x + FS / 2) % FS - FS / 2


def sigma_dt(sat, t0, floor=DT_FLOOR):
    """Orbit time-shift allowance from TLE age and generation, at least `floor`."""
    age_h = abs((sat.jdsatepoch - 2440587.5 + sat.jdsatepochF) * 86400 - t0) / 3600
    A, k = (0.049, 1.44) if int(sat.intldesg[:2]) >= 23 else (0.23, 0.75)      # v2-mini from 2023, else v1.x
    return max(floor, A * max(age_h, 0.1) ** k / 7.6)


def fit_one(t, z, sp, tau, base, mu_c, s_c, mu_dt, s_dt, cap_hz=CAP * SIGMA):
    """Best (c, dt) of one track against one curve with Gaussian priors; returns cost, c, dt, rms."""
    r0 = wrap(z - sp(t + tau + mu_dt) - base)
    g = sp(t + tau + mu_dt, 1)
    w = np.ones_like(r0)
    for _ in range(2):
        A = np.array([[w.sum() / SIGMA**2 + 1 / s_c**2, (w * g).sum() / SIGMA**2],
                      [(w * g).sum() / SIGMA**2, (w * g * g).sum() / SIGMA**2 + 1 / s_dt**2]])
        c, d = np.linalg.solve(A, [(w * r0).sum() / SIGMA**2 + mu_c / s_c**2, (w * g * r0).sum() / SIGMA**2])
        d = np.clip(d, -3 * s_dt, 3 * s_dt)                     # linear model only valid for small shifts
        c = ((w * (r0 - g * d)).sum() / SIGMA**2 + mu_c / s_c**2) / (w.sum() / SIGMA**2 + 1 / s_c**2)
        res = r0 - c - g * d
        w = (np.abs(res) <= cap_hz).astype(float)
    inl = np.abs(res) <= cap_hz
    cost = ((res[inl] / SIGMA) ** 2).sum() + (cap_hz / SIGMA) ** 2 * (~inl).sum() + ((c - mu_c) / s_c) ** 2 + (d / s_dt) ** 2
    if abs(c - mu_c) > 3 * s_c or (~inl).mean() > MAX_OUT:     # impossible tone offset, or curve misses the track
        cost = np.inf
    return cost, c, mu_dt + d, np.sqrt(np.mean(res[inl] ** 2)) if inl.any() else np.inf


def solve_labels(C, gamma, cands, conflicts, forbid=None):
    """Integer program: one label (or none) per track; conflicting tracks cannot share a satellite."""
    var = [(i, j) for i in range(len(cands)) for j in cands[i]] + [(i, None) for i in range(len(cands))]
    idx = {v: n for n, v in enumerate(var)}
    cost = np.array([C[i][j] if j else gamma[i] for i, j in var])
    rows, lo, hi = [], [], []
    for i in range(len(cands)):
        a = np.zeros(len(var)); a[[idx[(i, j)] for j in cands[i]] + [idx[(i, None)]]] = 1
        rows.append(a); lo.append(1); hi.append(1)
    for i, k in conflicts:
        for j in set(cands[i]) & set(cands[k]):
            a = np.zeros(len(var)); a[[idx[(i, j)], idx[(k, j)]]] = 1
            rows.append(a); lo.append(0); hi.append(1)
    ub = np.ones(len(var))
    if forbid:
        ub[idx[forbid]] = 0
    r = milp(cost, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(len(var)),
             bounds=(np.zeros(len(var)), ub))
    lab = [None] * len(cands)
    for n, v in enumerate(var):
        if r.x[n] > 0.5 and v[1]:
            lab[v[0]] = v[1]
    return lab, r.fun


def joint(tracks, preds, t0, sats, fit_tau=False, max_iter=8, gamma_rms=GAMMA_RMS, dt_floor=DT_FLOOR, frozen=None, exclude=None):
    """tracks: list of (t_unix, f_hz). Returns per-track results, shared numbers, per-satellite numbers.

    gamma_rms: mismatch limit (Hz); dt_floor: orbit-shift allowance (clip at 3x). frozen {track: norad}: confirmed
    labels kept fixed (they still shape the shared numbers). exclude {track: norad}: hide one candidate from one
    track (look-alike check).
    """
    frozen, exclude = frozen or {}, exclude or {}
    cap_hz = max(CAP * SIGMA, 2 * gamma_rms)          # bad-point cut scales with the limit
    sp = {n: CubicSpline(*map(np.asarray, p)) for n, p in preds.items()}
    first, _ = identify(tracks, preds, sigma_hz=3, sigma_dt_s=max(1.0, dt_floor), top=TOP_K)   # wide shortlist
    cands = [[frozen[i]] if i in frozen else [n for n, x in r['candidates'] if np.isfinite(x) and n != exclude.get(i)]
             for i, r in enumerate(first)]
    sdt = {n: sigma_dt(sats[n.removeprefix('SAT-')], t0, dt_floor) for c in cands for n in c}
    T0 = min(t[0] for t, _ in tracks)
    nW = int((max(t[-1] for t, _ in tracks) - T0) // WINDOW_S) + 1
    win = lambda t: np.minimum(((t - T0) // WINDOW_S).astype(int), nW - 1)
    gamma = [1e12 if i in frozen else len(t) * (gamma_rms / SIGMA) ** 2 for i, (t, _) in enumerate(tracks)]
    conflicts = []                                   # different lines at the same time
    for i, (ti, zi) in enumerate(tracks):
        for k in range(i + 1, len(tracks)):
            tk, zk = tracks[k]
            lo, hi = max(ti[0], tk[0]), min(ti[-1], tk[-1])
            if hi - lo > OVERLAP_S:
                m = (ti >= lo) & (ti <= hi)
                if np.median(np.abs(wrap(zi[m] - np.interp(ti[m], tk, zk)))) > SAME_LINE_HZ:
                    conflicts.append((i, k))
    b_init = np.median([r['offset_hz'] for r in first if r['status'] == 'accepted'] or [0])
    tau, bw = 0.0, np.full(nW, b_init)
    D = np.diff(np.eye(nW), axis=0)                  # neighbouring windows
    lab, per, seen = [None] * len(tracks), {}, []
    for it in range(max_iter):
        C, fits = [], []
        for i, (t, z) in enumerate(tracks):
            Ci, Fi = {}, {}
            for n in cands[i]:
                others = [per[(k, n)] for k in range(len(tracks)) if k != i and lab[k] == n and (k, n) in per]
                if others:
                    mu_c, s_c = np.mean([o[1] for o in others]), SIGMA_C_TIGHT
                    mu_dt, s_dt = np.mean([o[2] for o in others]), SIGMA_DT_TIGHT
                else:
                    mu_c, s_c, mu_dt, s_dt = 0.0, SIGMA_C, 0.0, sdt[n]
                Fi[n] = fit_one(t, z, sp[n], tau, bw[win(t)], mu_c, s_c, mu_dt, s_dt, cap_hz)
                Ci[n] = Fi[n][0]
            C.append(Ci); fits.append(Fi)
        allowed = [[n for n in cands[i] if np.isfinite(C[i][n])] for i in range(len(tracks))]
        new, total = solve_labels(C, gamma, allowed, conflicts)
        per = {(i, n): fits[i][n] for i, n in enumerate(new) if n}
        # refit shared numbers + per-satellite (c, dt) from all labelled tracks (Gauss-Newton with priors)
        S = sorted({n for n in new if n}); nb = 1 + nW; si = {n: nb + 2 * k for k, n in enumerate(S)}
        th = np.zeros(nb + 2 * len(S)); th[0], th[1:nb] = tau, bw
        for n in S:
            th[si[n]] = np.mean([per[(i, m)][1] for i, m in enumerate(new) if m == n])
            th[si[n] + 1] = np.mean([per[(i, m)][2] for i, m in enumerate(new) if m == n])
        prec = np.r_[1 / SIGMA_TAU**2 if fit_tau else 1e12, np.full(nW, 1 / SIGMA_BABS**2),
                     np.ravel([[1 / SIGMA_C**2, 1 / sdt[n]**2] for n in S])]
        mu = np.r_[0.0, np.full(nW, b_init), np.zeros(2 * len(S))]
        Ps = np.zeros((len(th), len(th))); Ps[1:nb, 1:nb] = D.T @ D / SIGMA_STEP**2
        for _ in range(4):
            J, R = [], []
            for i, n in enumerate(new):
                if not n:
                    continue
                t, z = tracks[i]; c0, d0 = si[n], si[n] + 1
                g = sp[n](t + th[0] + th[d0], 1)
                R.append(wrap(z - sp[n](t + th[0] + th[d0]) - th[1:nb][win(t)] - th[c0]))
                Ji = np.zeros((len(t), len(th))); Ji[:, 0] = Ji[:, d0] = -g; Ji[:, c0] = -1
                Ji[np.arange(len(t)), 1 + win(t)] = -1
                J.append(Ji)
            if not J:
                break
            J, R = np.vstack(J), np.concatenate(R)
            w = (np.abs(R) <= CAP * SIGMA) / SIGMA**2
            H = J.T @ (J * w[:, None]) + np.diag(prec) + Ps
            th -= np.linalg.solve(H, J.T @ (w * R) + prec * (th - mu) + Ps @ th)
            for n in S:
                th[si[n] + 1] = np.clip(th[si[n] + 1], -3 * sdt[n], 3 * sdt[n])
        tau, bw = th[0], th[1:nb].copy()
        if new == lab or new in seen:          # converged, or cycling between labelings
            lab = new
            break
        seen.append(new)
        lab = new
    out = []
    for i, n in enumerate(lab):
        if not n:
            out.append(dict(norad=None, status='unknown', rms_hz=min((fits[i][m][3] for m in cands[i] if np.isfinite(C[i][m])), default=np.inf),
                            cost=gamma[i], alt=None, margin=np.nan, c_hz=np.nan, dt_s=np.nan))
            continue
        if i in frozen:
            out.append(dict(norad=n, status='frozen', rms_hz=fits[i][n][3], cost=C[i][n], alt=None, margin=np.nan,
                            c_hz=fits[i][n][1], dt_s=fits[i][n][2]))
            continue
        alt_lab, alt_total = solve_labels(C, gamma, allowed, conflicts, forbid=(i, n))
        margin = alt_total - total
        # ambiguous only if another satellite can take this track at nearly the same total cost
        status = 'ambiguous' if alt_lab[i] and margin < max(9.0, 3 * C[i][n]) else 'accepted'
        out.append(dict(norad=n, status=status, rms_hz=fits[i][n][3], cost=C[i][n], alt=alt_lab[i], margin=margin,
                        c_hz=fits[i][n][1], dt_s=fits[i][n][2]))
    sat = {n: (th[si[n]], th[si[n] + 1], sdt[n]) for n in S}
    return out, dict(tau=tau, b_windows=bw, iterations=it + 1), sat


if __name__ == '__main__':
    tracks_csv, curves_dir, fc, start, tle, out_txt = sys.argv[1:7]
    d = np.loadtxt(tracks_csv, delimiter=',', skiprows=1)
    t0 = np.datetime64(start.removesuffix('Z'), 'ns').astype('int64') / 1e9
    tracks = [(t0 + d[d[:, 0] == i, 1], d[d[:, 0] == i, 2]) for i in np.unique(d[:, 0])]
    preds = load_sim_csvs(sorted(Path(curves_dir).glob('*.csv')), float(fc))
    res, sh, sat = joint(tracks, preds, t0, load_tles(tle, at_unix=t0))
    lines = [f'start {start} + tau {sh["tau"]:+.2f} s; receiver offset per {WINDOW_S:.0f} s window (kHz): '
             + ' '.join(f'{b / 1e3:.2f}' for b in sh['b_windows']) + f'; {sh["iterations"]} iterations']
    for i, r in enumerate(res):
        lines.append(f'track {i}: {r["status"]:<9} {str(r["norad"]):<10} rms {r["rms_hz"]:6.1f} Hz  margin {r["margin"]:8.1f}  '
                     f'sat offset {r["c_hz"] / 1e3:+7.2f} kHz  dt {r["dt_s"]:+.2f} s  alt {r["alt"]}')
    for n, (c, dt, s) in sat.items():
        lines.append(f'{n}: offset {c / 1e3:+7.2f} kHz from receiver, orbit shift {dt:+.2f} s (allowance {s:.2f} s)')
    Path(out_txt).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
