"""Assign Doppler tracks to NORAD IDs by a joint fit against predicted S-curves.

Extends Drive Code/sat_identification_least_squares.py. That scorer slides each track over the
whole predicted pass with a free time lag and a free offset, so many satellites fit. Here each track
is compared on its own UTC timestamps:

    z_i(t) - f_j(t) ~= b + k*comb_hz + fdot_j(t)*dt_j

b: frequency offset, fitted per track by default. common_lo=True shares one b across tracks (one LNB + SDR chain);
   on the 2026-05-04 recording each satellite's tone sat a few kHz apart, so per-track is the default.
k: tone index in the 9-tone comb (only with comb_hz). dt_j: TLE along-track error as a time shift,
with a Gaussian prior (sigma_dt_s) and clipped at 3 sigma.

Each track gets 'accepted' (runner-up RMS >= ratio_min x best), 'ambiguous', or 'unknown'
(best RMS > max_rms_hz: satellite not in the catalog, manoeuvre, or not a clean tone).

Predicted curves come from TLEs + SGP4 (predict) or from Drive sim CSVs (load_sim_csvs).

    python norad_id.py                                   self-checks
    python norad_id.py tles 2026-05-04 may04.tle         Space-Track GP history (env SPACETRACK_USER, SPACETRACK_PASS)
    python norad_id.py curves may04.tle 2026-05-04T22:40:56Z 330 11.575e9 curves_224056/
                                                         Drive-format CSVs, one per satellite above 10 deg
"""
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from sgp4.api import Satrec, SatrecArray

C = 299_792_458.0
WGS84_A, WGS84_E2, OMEGA_E = 6378137.0, 6.69437999014e-3, 7.292115e-5
# UW Sieg Building roof: lat, lon (OpenStreetMap); WGS84 height ~ 50 m ground + ~20 m roof - 23 m geoid.
# ponytail: roof height estimated, not surveyed; 10 m error moves Doppler by at most ~5 Hz.
SIEG = (47.654894, -122.306486, 47.0)


def load_tles(path, at_unix=None):
    """2LE/3LE text -> {norad: Satrec}. With several TLEs per satellite keep the epoch nearest at_unix (else the latest)."""
    lines = [s.strip() for s in Path(path).read_text().splitlines()]
    best = {}
    for a, b in zip(lines, lines[1:]):
        if a.startswith('1 ') and b.startswith('2 '):
            s = Satrec.twoline2rv(a, b)
            epoch = (s.jdsatepoch - 2440587.5 + s.jdsatepochF) * 86400
            key = epoch if at_unix is None else -abs(epoch - at_unix)
            if str(s.satnum) not in best or key > best[str(s.satnum)][0]:
                best[str(s.satnum)] = (key, s)
    return {n: s for n, (_, s) in best.items()}


def _geometry(sats, t, site, d_rc=(0.0, 0.0), t_orbit=None):
    """Range [m], range rate [m/s], elevation [deg] from site, Earth-fixed frame. Arrays [satellite, time].

    d_rc: constant (radial, cross-track) position offset in km, for TLE error tests.
    t_orbit: times at which to take the satellite's position in space (default t); the Earth is rotated to t.
             t_orbit = t + shift slides the satellite along its orbit (plane fixed in space).
    """
    to = t if t_orbit is None else t_orbit
    day = np.floor(to / 86400)
    err, r, v = SatrecArray(sats).sgp4(2440587.5 + day, to / 86400 - day)   # TEME, km and km/s
    if any(d_rc):
        h = np.cross(r, v)
        r = r + d_rc[0] * r / np.linalg.norm(r, axis=-1, keepdims=True) + d_rc[1] * h / np.linalg.norm(h, axis=-1, keepdims=True)
    day = np.floor(t / 86400)
    jd, fr = 2440587.5 + day, t / 86400 - day
    # GMST (IAU-82, as sgp4's gstime) from the split date; ponytail: no polar motion / UT1-UTC (sub-Hz here).
    tu = ((jd - 2451545.0) + fr) / 36525.0
    g = np.radians((67310.54841 + (876600 * 3600 + 8640184.812866) * tu + 0.093104 * tu ** 2 - 6.2e-6 * tu ** 3) / 240.0)
    rot = lambda x: np.stack([np.cos(g) * x[..., 0] + np.sin(g) * x[..., 1],
                              -np.sin(g) * x[..., 0] + np.cos(g) * x[..., 1], x[..., 2]], -1)
    r_ef = rot(r) * 1e3
    v_ef = rot(v) * 1e3 - np.cross([0.0, 0.0, OMEGA_E], r_ef)  # Earth rotation; missing it costs up to ~12 kHz
    lat, lon = np.radians(site[:2])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    nn = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    d = r_ef - up * [nn + site[2], nn + site[2], nn * (1 - WGS84_E2) + site[2]]
    rng = np.linalg.norm(d, axis=-1)
    rdot = np.einsum('stk,stk->st', d, v_ef) / rng
    el = np.degrees(np.arcsin(d @ up / rng))
    rng[err != 0] = rdot[err != 0] = el[err != 0] = np.nan
    return rng, rdot, el


def predict(sats, t_unix, carrier_hz, site=SIEG, min_elev_deg=10.0):
    """SGP4 Doppler at a fixed receiver -> ({norad: (t_unix, doppler_hz)}, {norad: elevation_deg}).

    Keeps satellites that rise above min_elev_deg within t_unix.
    ponytail: no light-time correction (~3 ms shift, absorbed by identify's dt_j).
    """
    t = np.asarray(t_unix, float)
    ids, preds, elev = list(sats), {}, {}
    for k in range(0, len(ids), 500):                          # chunks keep memory ~100 MB for 10k satellites
        chunk = ids[k:k + 500]
        _, rdot, el = _geometry([sats[n] for n in chunk], t, site)
        for n, rd, e in zip(chunk, rdot, el):
            if np.isfinite(rd).all() and e.max() >= min_elev_deg:
                preds[n], elev[n] = (t, -rd * carrier_hz / C), e
    return preds, elev


def write_sim_csvs(preds, elev, carrier_hz, folder):
    """Drive sim CSV format (SAT-<norad>.csv: timestamp, relative_velocity_kms receding +), rows above the horizon."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for n, (t, f) in preds.items():
        up = elev[n] >= 0
        stamps = (t[up] * 1e9).astype('int64').astype('datetime64[ns]')
        rows = (f'{s}Z,{-x * C / carrier_hz / 1e3:.9f},{e:.3f}\n' for s, x, e in zip(stamps, f[up], elev[n][up]))
        (folder / f'SAT-{n}.csv').write_text('timestamp,relative_velocity_kms,elevation_deg\n' + ''.join(rows))


def fetch_spacetrack(day, out_path, user, password, filt='OBJECT_NAME/%5ESTARLINK'):
    """TLE history (3LE) of objects matching `filt` (Space-Track predicates, e.g. 'OBJECT_NAME/%5EONEWEB') from Space-Track for day-1 .. day+2 (UTC 'YYYY-MM-DD'). Needs a free account."""
    import http.cookiejar
    import urllib.parse
    import urllib.request
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    login = op.open('https://www.space-track.org/ajaxauth/login',
                    urllib.parse.urlencode({'identity': user, 'password': password}).encode()).read()
    if b'Failed' in login:
        raise RuntimeError('Space-Track login failed')
    d = np.datetime64(day, 'D')
    text = op.open('https://www.space-track.org/basicspacedata/query/class/gp_history/' + filt +
                   f'/EPOCH/{d - 1}--{d + 2}/orderby/NORAD_CAT_ID,EPOCH/format/3le').read()
    Path(out_path).write_bytes(text)
    return text.count(b'\n1 ')


def load_sim_csvs(paths, carrier_hz):
    """Drive sim CSVs (timestamp UTC, relative_velocity_kms receding +) -> {stem: (unix_s, doppler_hz)}."""
    preds = {}
    for path in map(Path, paths):
        with path.open(newline='') as fh:
            rows = list(csv.DictReader(fh))
        t = np.array([np.datetime64(r['timestamp'].removesuffix('Z').removesuffix('+00:00'), 'ns') for r in rows])
        v = np.array([float(r['relative_velocity_kms']) for r in rows])
        preds[path.stem] = (t.astype('int64') / 1e9, -v * 1e3 / C * carrier_hz)
    return preds


def _cost(b, mr, mrr, mg, mgg, mrg, lam, dt_max):
    """Mean-square residual of r - b - g*dt at the best clipped dt, plus the dt prior."""
    ug, uu = mrg - b * mg, mrr - 2 * b * mr + b * b
    dt = np.clip(ug / (mgg + lam), -dt_max, dt_max)
    return np.maximum(uu - 2 * dt * ug + dt * dt * (mgg + lam), 0.0), dt


def identify(tracks, preds, sigma_hz=5.0, sigma_dt_s=0.4, common_lo=False, comb_hz=None,
             ratio_min=2.0, max_rms_hz=None, top=3):
    """tracks: list of (unix_s, freq_hz). preds: {norad: (unix_s, doppler_hz)} in the same frequency reference.

    sigma_dt_s: TLE along-track error as time (SupGP ~0.05 s, GP ~0.4 s).
    comb_hz: 43945.0 for pilot tones with unknown tone index; None when the index is known.
    Returns (one result dict per track, common LO offset in Hz or None).
    """
    max_rms_hz = 5 * sigma_hz if max_rms_hz is None else max_rms_hz
    lam = sigma_hz ** 2 / sigma_dt_s ** 2   # ponytail: per-sample prior weight, as in archive/sims/tone_joint.py
    dt_max = 3 * sigma_dt_s
    ids = list(preds)
    splines = [CubicSpline(np.asarray(t, float), np.asarray(f, float)) for t, f in preds.values()]
    S = np.full((5, len(tracks), len(ids)), np.nan)   # means of r, r*r, g, g*g, r*g
    for i, (t, z) in enumerate(tracks):
        t, z = np.asarray(t, float), np.asarray(z, float)
        for j, sp in enumerate(splines):
            # ponytail: O(tracks x candidates) Python loop; prefilter by elevation/beam if the catalog is >10k
            if sp.x[0] <= t[0] and t[-1] <= sp.x[-1]:
                r, g = z - sp(t), sp(t, 1)
                S[:, i, j] = r.mean(), (r * r).mean(), g.mean(), (g * g).mean(), (r * g).mean()
    mr, mrr, mg, mgg, mrg = S
    wrap = (lambda m, b: b + comb_hz * np.round((m - b) / comb_hz)) if comb_hz else (lambda m, b: b + 0 * m)

    # Per-track fit (b free per track) = the same cost on centred statistics.
    c, dt = _cost(0.0, 0.0, mrr - mr ** 2, 0.0, mgg - mg ** 2, mrg - mr * mg, lam, dt_max)
    c = np.where(np.isnan(c), np.inf, c)
    off = mr - mg * dt
    b_lo = None
    if common_lo:
        # Hypotheses for b: fitted offsets of each track's 5 best candidates. Keep the b with the lowest total cost.
        hyp = np.concatenate([off[i, np.argsort(c[i])[:5]] for i in range(len(tracks))])
        hyp = hyp[np.isfinite(hyp)][:, None, None]
        bb = wrap(mr, hyp)
        cj, dtj = _cost(bb, mr, mrr, mg, mgg, mrg, lam, dt_max)
        cj = np.where(np.isnan(cj), np.inf, cj)
        best = np.argmin(np.where(np.isfinite(cj), cj, 1e30).min(2).sum(1))
        # Refine b over the inlier tracks (closed form, dt unclipped), then re-score.
        i, j = np.arange(len(tracks)), np.argmin(cj[best], 1)
        inl = cj[best][i, j] <= max_rms_hz ** 2
        i, j = (i[inl], j[inl]) if inl.any() else (i, j)
        kf = bb[best][i, j] - hyp[best, 0, 0]
        G, g = mgg[i, j] + lam, mg[i, j]
        b_lo = float(np.sum(mr[i, j] - kf - g * (mrg[i, j] - kf * g) / G) / np.sum(1 - g * g / G))
        off = wrap(mr, b_lo)
        c, dt = _cost(off, mr, mrr, mg, mgg, mrg, lam, dt_max)
        c = np.where(np.isnan(c), np.inf, c)

    rms = np.sqrt(c)
    out = []
    for i in range(len(tracks)):
        o = np.argsort(rms[i])
        r1, r2 = rms[i, o[0]], (rms[i, o[1]] if len(o) > 1 else np.inf)
        ratio = r2 / max(r1, 1e-9)
        status = 'unknown' if not r1 <= max_rms_hz else 'ambiguous' if ratio < ratio_min else 'accepted'
        j = o[0]
        out.append(dict(
            norad=ids[j] if np.isfinite(r1) else None, status=status, rms_hz=float(r1), ratio=float(ratio),
            offset_hz=float(off[i, j]), dt_s=float(dt[i, j]),
            tone_index=int(round((off[i, j] - b_lo) / comb_hz)) if comb_hz and b_lo is not None else None,
            candidates=[(ids[k], float(rms[i, k])) for k in o[:top]]))
    return out, b_lo


def _self_check():
    # SGP4 geometry: range rate must equal d(range)/dt in the Earth-fixed frame (fails without the Earth-rotation term).
    from sgp4.api import WGS72
    t0 = np.datetime64('2026-05-04T22:40:56', 's').astype('int64').astype(float)
    s = Satrec()
    s.sgp4init(WGS72, 'i', 1, (t0 / 86400 + 2440587.5) - 2433281.5, 1e-4, 0.0, 0.0, 1e-4, 0.0,
               np.radians(53.0), np.radians(10.0), 2 * np.pi / 95.6, np.radians(200.0))
    t = t0 + np.arange(0, 3600, 0.1)
    rng_m, rdot, el = _geometry([s], t, SIEG)
    err = np.abs(np.gradient(rng_m[0], t)[1:-1] - rdot[0, 1:-1]).max()
    assert err < 0.01, err
    print(f'SGP4 range-rate vs d(range)/dt: max diff {err * 1e3:.2f} mm/s, max elevation {np.nanmax(el):.1f} deg')

    # Matcher on straight-line passes: 3 tracks share an unknown LO offset; the satellite of a 4th track
    # is missing from the catalog and must not be accepted.
    rng = np.random.default_rng(0)
    fc, v = 11.575e9, 7.6e3
    tc, d = rng.uniform(-300, 300, 80), rng.uniform(550e3, 1500e3, 80)
    curve = lambda j, t: -fc / C * v * v * (t - tc[j]) / np.hypot(d[j], v * (t - tc[j]))
    tp = np.arange(-400.0, 400.0)
    ts = np.arange(0, 10, 0.1)
    truth = [3, 17, 42, 55]
    b = 12_345.0
    tracks = [(ts, curve(j, ts) + b + rng.normal(0, 5, ts.size)) for j in truth]
    preds = {f'SAT-{j}': (tp, curve(j, tp + rng.normal(0, 0.05))) for j in range(80) if j != 55}
    res, b_hat = identify(tracks, preds, sigma_dt_s=0.1, common_lo=True)
    for j, r in zip(truth, res):
        print(f'SAT-{j}: {r["status"]:<9} {r["norad"]:<7} rms {r["rms_hz"]:8.1f} Hz  ratio {r["ratio"]:6.1f}')
    assert all(r['status'] == 'accepted' and r['norad'] == f'SAT-{j}' for j, r in zip(truth[:3], res[:3]))
    assert res[3]['status'] != 'accepted' and abs(b_hat - b) < 20
    print(f'LO offset {b_hat:.1f} Hz (true {b}); self-check passed')


if __name__ == '__main__':
    import os
    import sys
    cmd = sys.argv[1:2]
    if cmd == ['tles']:
        n = fetch_spacetrack(sys.argv[2], sys.argv[3], os.environ['SPACETRACK_USER'], os.environ['SPACETRACK_PASS'], *sys.argv[4:5])
        print(f'{n} TLEs -> {sys.argv[3]}')
    elif cmd == ['curves']:
        tle, start, dur, fc, out = sys.argv[2:7]
        t0 = np.datetime64(start.removesuffix('Z'), 'ns').astype('int64') / 1e9
        t = t0 + np.arange(-60.0, float(dur) + 61)              # 1-s spacing, 60-s margin
        preds, elev = predict(load_tles(tle, at_unix=t0), t, float(fc))
        write_sim_csvs(preds, elev, float(fc), out)
        print(f'{len(preds)} satellites above 10 deg -> {out}')
    else:
        _self_check()
