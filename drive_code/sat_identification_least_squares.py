"""CFAR-based satellite identification algorithm.
score_candidates(cfar_bins, simulation_csvs) -> list[dict]

Input:
  cfar_bins: 1D bin locations, one per epoch. Preserve missing
    epochs as NaN. Provide one representative bin per epoch of a single track.
  simulation_csvs: iterable of CSV paths, one satellite per file. Required
    columns: timestamp (UTC ISO-8601), relative_velocity_kms (receding positive).

Output: candidates sorted by RMSE, lowest first. offset_bins is the
        simulated-minus-observed offset. Subtract it to overlay the simulation.
        Unsupported candidates have infinite RMSE and None alignment fields.

Dependencies: NumPy
"""
import csv
from pathlib import Path
import numpy as np

# Update them for a different receiver.
EPOCH_S = 0.01
BIN_HZ = 100.0
BIN_ZERO_HZ = -250_000.0
RECEIVER_HZ = 11.575e9
MAX_SIM_GAP_S = 2.0


def _correlate(signal, template):
    """FFT-based valid correlation; avoids allocating a large sliding matrix."""
    size = 1 << (len(signal) + len(template) - 2).bit_length()
    full = np.fft.irfft(np.fft.rfft(signal, size) * np.fft.rfft(template[::-1], size), size)
    return full[len(template) - 1:len(signal)]


def score_candidates(cfar_bins, simulation_csvs):
    """Return one best offset-adjusted least-squares score per satellite CSV."""
    track = np.asarray(cfar_bins, dtype=float)
    if track.ndim != 1 or np.isinf(track).any() or np.isfinite(track).sum() < 3:
        raise ValueError('CFAR bins must be 1D with at least three finite observations; use NaN for gaps.')
    observed = np.isfinite(track).astype(float)
    count, mean = observed.sum(), np.nanmean(track)
    centered = np.where(observed, track - mean, 0.0)
    results = []
    for filename in simulation_csvs:
        path = Path(filename)
        result = dict(satellite=path.stem, rmse_bins=float('inf'), start_utc=None, offset_bins=None)
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f'{path}: empty simulation CSV.')
        if any(not r['timestamp'].endswith(('Z', '+00:00')) for r in rows):
            raise ValueError(f'{path}: timestamps must explicitly use UTC.')
        utc = np.array([np.datetime64(r['timestamp'].removesuffix('Z').removesuffix('+00:00'), 'ns') for r in rows])
        times = (utc - utc[0]) / np.timedelta64(1, 's')
        velocity = np.array([float(r['relative_velocity_kms']) for r in rows])
        if not np.isfinite(times).all() or not np.isfinite(velocity).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f'{path}: timestamps must increase and velocities must be finite.')
        grid = np.arange(int(np.floor(times[-1] / EPOCH_S)) + 1) * EPOCH_S
        if len(grid) < len(track):
            results.append(result)
            continue
        bins = (-velocity * 1000 / 299_792_458.0 * RECEIVER_HZ - BIN_ZERO_HZ) / BIN_HZ
        prediction = np.interp(grid, times, bins)
        right = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
        left = np.maximum(right - 1, 0)
        supported = ((times[right] - times[left] <= MAX_SIM_GAP_S)
                     | np.isclose(grid, times[right], rtol=0, atol=1e-8))
        # Center first to reduce cancellation in the accumulated error calculation.
        baseline = prediction[supported].mean()
        values = np.where(supported, prediction - baseline, 0.0)
        sums = _correlate(values, observed)
        errors = (_correlate(values * values, observed) - sums * sums / count
                  + centered @ centered - 2 * _correlate(values, centered))
        errors[np.rint(_correlate(supported.astype(float), observed)) < count] = np.inf
        best = int(np.argmin(errors))
        if np.isfinite(errors[best]):
            result.update(rmse_bins=float(np.sqrt(max(0.0, errors[best]) / count)),
                          start_utc=str(utc[0] + np.timedelta64(round(best * EPOCH_S * 1e9), 'ns')) + 'Z',
                          offset_bins=float(sums[best] / count + baseline - mean))
        results.append(result)
    return sorted(results, key=lambda r: (r['rmse_bins'], r['satellite']))
