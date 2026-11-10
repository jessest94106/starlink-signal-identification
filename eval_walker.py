"""Compare the Drive matchers with norad_id.identify on the archive's Walker sim (~10k sats, Seattle).

Setup per trial: known receiver position, 4 simultaneous tone tracks (sats above 40 deg), unknown LO offset
(+-20 kHz) shared by all tracks, unknown tone index k, 5 Hz noise, 10 Hz track rate. Predictions for every
satellite above 10 deg, 1-s spacing over +-300 s (like the Drive sim CSVs), with a TLE along-track error.
'missing' re-runs with track 0's satellite removed from the catalog: any accepted ID there is false.
"""
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

from norad_id import identify

HERE = Path(__file__).resolve().parent
SIMS = HERE / 'archive/sims'
exec((SIMS / 'overlap.py').read_text().split('rows=[]')[0])
exec((SIMS / 'assoc.py').read_text().split('\n', 3)[3].split('res={}')[0])
fc = 11.575e9                 # Drive RECEIVER_HZ
lam = c / fc

spec = importlib.util.spec_from_file_location('drive_ls', HERE / 'drive_code/sat_identification_least_squares.py')
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)
ls.EPOCH_S, ls.BIN_HZ, ls.BIN_ZERO_HZ, ls.RECEIVER_HZ = 0.1, 1.0, 0.0, fc   # bins in Hz, one per 0.1 s

dF, K, sig_n = 43945.0, 4, 5.0
base = np.datetime64('2026-02-06T21:00:00', 'ns')
rng = np.random.default_rng(5)


def write_csvs(folder, tg, Fp, names):
    stamps = [str(base + np.timedelta64(int(round(s * 1e9)), 'ns')) + 'Z' for s in tg]
    for j, name in enumerate(names):
        vel = -Fp[:, j] * c / fc / 1e3
        (folder / f'{name}.csv').write_text('timestamp,relative_velocity_kms\n' +
                                            ''.join(f'{s},{v:.9f}\n' for s, v in zip(stamps, vel)))
    return [folder / f'{n}.csv' for n in names]


def run(sig_s, T, trials):
    res = {m: [] for m in ('Drive LS (offset + free time slide)', 'Web matcher (abs RMSE < 1.5 kHz)',
                           'New, per-track offset', 'New, joint LO + tone comb')}
    miss = {m: [] for m in res}
    for _ in range(trials):
        t0 = rng.uniform(0, 86400)
        el = elev(t0, lat, lon)
        cand, hi = np.where(el > 10)[0], np.where(el > 40)[0]
        if hi.size < K:
            continue
        truth = rng.choice(hi, K, replace=False)
        ts, tg = np.arange(0, T, 0.1), np.arange(-300.0, T + 300)
        Fp = dop(t0 + tg, lat, lon, rng.normal(0, sig_s, N) / 7.6e3)[:, cand]
        meas = dop(t0 + ts, lat, lon, np.zeros(N))[:, truth] + rng.uniform(-20e3, 20e3) \
            + rng.integers(-4, 5, K) * dF + rng.normal(0, sig_n, (ts.size, K))
        names = [str(j) for j in cand]
        tracks = [(t0 + ts, meas[:, k]) for k in range(K)]
        preds = {n: (t0 + tg, Fp[:, j]) for j, n in enumerate(names)}
        gone = str(truth[0])

        def score(method, k, top, accepted, missing):
            res[method].append((top == str(truth[k]), accepted, accepted and top != str(truth[k])))
            if missing is not None:
                miss[method].append(missing)

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_csvs(Path(tmp), tg, Fp, names)
            for k in range(K):
                r = ls.score_candidates(meas[:, k], paths)
                acc = lambda r: r[1]['rmse_bins'] >= 2 * r[0]['rmse_bins']
                rm = [x for x in r if x['satellite'] != gone]
                score('Drive LS (offset + free time slide)', k, r[0]['satellite'], acc(r), acc(rm) if k == 0 else None)
        for k in range(K):
            e = np.sqrt(((meas[:, [k]] - np.stack([np.interp(t0 + ts, t0 + tg, Fp[:, j]) for j in range(len(cand))], 1)) ** 2).mean(0))
            o = np.argsort(e)
            o_m = [j for j in o if names[j] != gone]
            score('Web matcher (abs RMSE < 1.5 kHz)', k, names[o[0]], e[o[0]] < 1500, e[o_m[0]] < 1500 if k == 0 else None)
        for method, kw in (('New, per-track offset', dict(common_lo=False)),
                           ('New, joint LO + tone comb', dict(common_lo=True, comb_hz=dF))):
            out, _ = identify(tracks, preds, sigma_hz=sig_n, sigma_dt_s=sig_s / 7.6e3, **kw)
            out_m, _ = identify(tracks, {n: p for n, p in preds.items() if n != gone}, sigma_hz=sig_n,
                                sigma_dt_s=sig_s / 7.6e3, **kw)
            for k, r in enumerate(out):
                score(method, k, r['norad'], r['status'] == 'accepted', out_m[0]['status'] == 'accepted' if k == 0 else None)
    return res, miss


if __name__ == '__main__':
    print('fraction of tracks: correct top-1 | accepted | accepted but wrong | accepted when true sat missing')
    for sig_s, name in ((400, 'SupGP-like 0.4 km'), (3000, 'GP-like 3 km')):
        for T in (3.0, 10.0):
            res, miss = run(sig_s, T, trials=40)
            n_tr = len(next(iter(res.values())))
            print(f'\n{name}, {T:.0f}-s tracks ({n_tr} tracks)')
            for m, v in res.items():
                v = np.array(v)
                print(f'  {m:<38} {v[:, 0].mean():.2f} | {v[:, 1].mean():.2f} | {v[:, 2].mean():.2f} | {np.mean(miss[m]):.2f}')
