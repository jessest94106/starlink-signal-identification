# Starlink signal identification

Identify which Starlink satellite (NORAD ID) produced each Ku-band downlink tone in a ground recording, or label the tone ambiguous or unknown. Recordings: USRP + LNB (LO 9.75 GHz), 500 kS/s complex, on the UW Sieg Building roof (47.654894° N, 122.306486° W, ~47 m).

## Method

1. **Tone tracks** (`iq_tracks.py`): waterfall (122 Hz × 0.26 s) → peak linking → re-measured on the raw I/Q with the chirp removed (1–2 Hz noise) → coherent extend-and-join of pieces of the same line.
2. **Predicted Doppler** (`norad_id.py`): SGP4 from Space-Track TLEs, range rate at the receiver (Earth rotation included), for every satellite above the horizon.
3. **Start time** (`run_capture.py`): the capture PC clock is off by up to ~2.5 s, so the start is found twice: (a) orbit fingerprint, the common time shift that best fits all tracks; (b) tone on/off switches, which fall on the GPS 15-s grid.
4. **Matching** (`staged_id.py`, `joint_id.py`): per track and candidate, fit a free frequency offset and a small along-track orbit shift; score = RMS mismatch. Stages loosen the limits: 15 Hz / ±1.5 s → 20 Hz → 30 Hz / ±3 s → 50 Hz / ±6 s.
5. **Decision**: *confirmed* = accepted with both start times, same satellite, and the look-alike test passes (remove the chosen satellite: the track must not be confidently given another one). Nearly constant lines (|rate| < 50 Hz/s over > 5 s) are *not-LEO*. Everything else is *look-alike*, *ambiguous* or *unknown*.

## Results (all 60 Drive captures, `data/drive/report.txt`)

| Date | Files | Tracks | Confirmed | Confirmed IDs | Not-LEO | Unknown |
|---|---|---|---|---|---|---|
| 2025-12-17 | 2 | 0 | – | – | – | – |
| 2026-01-20 | 3 | 20 | 2 (10%) | 0 | 2 | 18 |
| 2026-02-06 | 10 | 66 | 34 (52%) | 25 | 9 | 32 |
| 2026-03-03 | 4 | 10 | 0 | 0 | 0 | 10 |
| 2026-03-05 | 28 | 372 | 228 (61%) | 205 | 23 | 120 |
| 2026-03-09 | 9 | 76 | 30 (39%) | 23 | 7 | 44 |
| 2026-05-04 | 4 | 69 | 54 (78%) | 52 | 2 | 14 |
| **All** | **60** | **613** | **348 (57%)** | **305** | **43** | **238** |

On the same tracks and curves, the team's earlier least-squares matcher (`drive_code/sat_identification_least_squares.py`) determines 21 tracks, and the web-app matcher agrees with none of ours.

## Setup

Python 3.12:

```bash
pip install -r requirements.txt
```

## Replicate the results

Steps 1–4 use only files in this repo.

```bash
python norad_id.py          # 1. self-check: SGP4 Doppler and matcher on synthetic tracks -> "self-check passed"
python test_join.py         #    coherent extend-and-join on a synthetic weak chirp -> "ok [[0, 1]] ..."

python replicate.py 8       # 2. re-identify every capture from its stored tracks and start times (~16 CPU-minutes)
                            #    expected last line:
                            #    "57 of 57 captures give the same IDs (0 tracks differ); 57 identical reports"

python drive_report.py > data/drive/report.txt   # 3. rebuild the results table above
python baseline_all.py 8                         # 4. re-run the baseline matchers (writes baseline.txt per capture; ~5 CPU-hours)
git status --short data/                         #    nothing listed = identical to the stored results
```

### From raw I/Q

The raw recordings (`*.sigmf-data`, 0.5–3.6 GB each) are not in this repo; their `.sigmf-meta` files are. With a recording:

```bash
python run_capture.py DATA.sigmf-data tles/<date>.tle OUT_DIR [START_GUESS_UTC]
```

- Writes `OUT_DIR/tracks.csv`, `staged.txt` (one line per track: status, NORAD ID, stage, mismatch, orbit shift, offset) and `summary.json` (start times, counts). About 1 minute per 5-minute recording.
- The file name must contain `_f<GHz>GHz_<YYYYMMDDTHHMMSS>`; that time is taken as the start. The 2025-12-17 and 2026-01-20 files are named at the **end** of the recording: pass the start, e.g. `2026-01-20T23:01:06` for `r001_f11.575GHz_20260120T230606`.
- `.sigmf-meta` must sit next to the data file (sample rate is read from it).
- Check: the output should equal the stored `data/drive/<date>/<setup>/<capture>/`.
- `run_drive.py` runs all captures; it needs `data/drive/manifest.tsv` (not included; tab-separated: date, setup, file name, Google Drive file ID, bytes, start guess).

### TLEs

Space-Track TLE history for each capture date is in `tles/<date>.tle`. To fetch again (Space-Track account):

```bash
SPACETRACK_USER=... SPACETRACK_PASS=... python norad_id.py tles 2026-05-04 tles/2026-05-04.tle
```

### Another receiver site

The receiver position is `SIEG` in `norad_id.py`.

## Layout

| Path | Contents |
|---|---|
| `iq_tracks.py`, `norad_id.py`, `match_capture.py`, `joint_id.py`, `staged_id.py`, `run_capture.py` | Pipeline |
| `replicate.py`, `drive_report.py`, `baseline_all.py`, `run_drive.py`, `pair_check.py` | Batch runs, replication, summary, baselines, two-receiver comparison |
| `check_joint.py`, `test_join.py` | Checks (missing-satellite test, join test) |
| other top-level `*.py` | Studies (tone switch timing, orbit-shift scans, height test, wide TLE search) |
| `data/drive/<date>/<setup>/<capture>/` | Per capture: `tracks.csv`, `staged.txt`, `summary.json`, `baseline.txt`, `run.log` |
| `data/c*/` | 2026-05-04 captures step by step: tracks, matches, waterfalls with IDs, coherent-follow plots |
| `data/drive/unknowns/` | Study of the long unknown tracks |
| `tles/`, `may04*.tle`, `curves/` | Space-Track TLEs and predicted Doppler curves (Drive CSV format) |
| `drive_code/`, `drive_2025/` | Team code and preprocessing from Drive (baselines) |
| `archive/sims/` | Walker-constellation simulations (used by `eval_walker.py`) |
