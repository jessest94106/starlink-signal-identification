"""Check iq_tracks.join on a synthetic weak chirp: two track pieces with a 3-s gap must be extended and joined."""
import numpy as np

from iq_tracks import join

fs, T = 500e3, 40
rng = np.random.default_rng(0)
t = np.arange(int(fs * T)) / fs
f = 20e3 + 150 * (t - 20) + 2 * (t - 20) ** 2                     # Hz
s = 0.05 * np.exp(2j * np.pi * np.cumsum(f) / fs)                    # -26 dB per sample; ~ +31 dB in a 1-s FFT
s[int(18 * fs):int(21 * fs)] = 0                                    # tone off for 3 s
x = (s + (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size)) / np.sqrt(2)).astype(np.complex64)
piece = lambda a, b: np.c_[np.arange(a, b, 0.26), 20e3 + 150 * (np.arange(a, b, 0.26) - 20) + 2 * (np.arange(a, b, 0.26) - 20) ** 2, np.full(len(np.arange(a, b, 0.26)), 20.0)]
out, groups, added = join(x, fs, [piece(10, 16), piece(23, 29)])
assert groups == [[0, 1]], groups
a = out[0]
assert a[0, 0] < 3 and a[-1, 0] > 37, (a[0, 0], a[-1, 0])          # followed to near both file ends
fit = 20e3 + 150 * (a[:, 0] - 20) + 2 * (a[:, 0] - 20) ** 2
assert np.max(np.abs(a[:, 1] - fit)) < 2, np.max(np.abs(a[:, 1] - fit))
print('ok', groups, added, f'{a[0, 0]:.1f}-{a[-1, 0]:.1f} s')
