from datetime import datetime
import os
import glob
import re
from matplotlib import pyplot as plt
import numpy as np
import sigmf
from scipy import constants


MIN_DB = -150
MAX_DB = -90

# Smaller values use less RAM but may be slower.
# This controls plotting downsampling only; the saved .npy waterfall is still full length.
MAX_PLOT_TIME_BINS = 5000


def compress_all_sigmf_in_cwd(k, NFFT, NOVERLAP):
    """
    Find all .sigmf-data files in the current working directory and compress each one.

    Memory-light version:
    - does NOT load the whole .sigmf-data file into RAM
    - does NOT build the full STFT matrix
    - writes compressed rows directly to the output .txt file
    """
    cwd = os.getcwd()
    data_files = sorted(glob.glob(os.path.join(cwd, "*.sigmf-data")))

    if not data_files:
        print("[INFO] No .sigmf-data files found")
        return

    for data_path in data_files:
        base = data_path.replace(".sigmf-data", "")
        meta_path = base + ".sigmf-meta"

        if not os.path.exists(meta_path):
            print(f"[WARN] Missing meta file for {data_path}, skipping")
            continue

        print(f"[INFO] Compressing {os.path.basename(base)}")

        try:
            compressSigmf(
                filename=base,
                k=k,
                NFFT=NFFT,
                NOVERLAP=NOVERLAP,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {data_path}: {e}")

    print("[INFO] Compression complete")


def _read_sigmf_metadata(filename):
    """Read SigMF metadata and return sample rate, center frequency, and datatype string."""
    datafile = sigmf.fromfile(filename)
    global_metadata = datafile.get_global_info()
    captures = datafile.get_captures()

    sample_rate = float(global_metadata.get("core:sample_rate"))
    center_freq = float(captures[0]["core:frequency"])
    dtype_str = global_metadata.get("core:datatype", "cf32_le")

    return sample_rate, center_freq, dtype_str


def _open_sigmf_memmap(data_path, dtype_str):
    """
    Open raw SigMF data as a memory map.

    Supported here:
    - cf32_le: complex64 IQ samples
    - ci16_le: interleaved int16 IQ samples, converted frame-by-frame
    """
    if dtype_str == "cf32_le":
        raw = np.memmap(data_path, dtype=np.complex64, mode="r")
        return raw, "complex64", raw.shape[0]

    if dtype_str == "ci16_le":
        raw_iq = np.memmap(data_path, dtype=np.int16, mode="r")
        if raw_iq.shape[0] % 2 != 0:
            print("[WARN] Odd number of int16 values in ci16 file; dropping final value")
        sample_count = raw_iq.shape[0] // 2
        return raw_iq, "ci16", sample_count

    raise ValueError(f"Unsupported SigMF datatype: {dtype_str}")


def _read_frame(raw, storage_kind, start, NFFT, sample_count):
    """
    Read one complex frame. Pads the final partial frame with zeros.
    This is intentionally small: only one NFFT frame is in RAM at a time.
    """
    end = min(start + NFFT, sample_count)
    valid_len = max(0, end - start)

    frame = np.zeros(NFFT, dtype=np.complex64)

    if valid_len == 0:
        return frame

    if storage_kind == "complex64":
        frame[:valid_len] = raw[start:end]
    elif storage_kind == "ci16":
        iq = raw[2 * start : 2 * end].astype(np.float32)
        frame[:valid_len] = iq[0::2] + 1j * iq[1::2]
    else:
        raise ValueError(f"Unsupported storage kind: {storage_kind}")

    return frame


def _format_complex_row(row):
    """
    Format complex values without parentheses so each line can be parsed quickly later.
    Example token: 1.23456789e+03-2.5e-02j
    """
    tokens = []
    for z in row:
        z = complex(z)
        tokens.append(f"{z.real:.10g}{z.imag:+.10g}j")
    return " ".join(tokens) + "\n"


def compressSigmf(filename, k, NFFT, NOVERLAP):
    """
    Compress a .sigmf file pair into a relative-velocity sparse representation in txt format.

    This replaces the old full-file STFT call with frame-by-frame FFT processing.
    It keeps only one NFFT frame and one FFT spectrum in memory at a time.
    """
    try:
        sample_rate, center_freq, dtype_str = _read_sigmf_metadata(filename)
    except FileNotFoundError:
        print("Error: The file " + filename + ".sigmf-meta" + " was not found.")
        return None, None, None

    data_path = filename + ".sigmf-data"
    raw, storage_kind, sample_count = _open_sigmf_memmap(data_path, dtype_str)

    if sample_count == 0:
        print(f"[WARN] No samples in {data_path}")
        return sample_rate, None, None

    step = NFFT - NOVERLAP
    if step <= 0:
        raise ValueError("NOVERLAP must be smaller than NFFT")

    # Include a final zero-padded frame if the data length is not an exact multiple.
    num_frames = int(np.ceil(max(sample_count - NFFT, 0) / step)) + 1

    # Same frequency axis idea as scipy.signal.stft(..., return_onesided=False) + fftshift.
    f = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0 / sample_rate))
    t = (np.arange(num_frames) * step + NFFT / 2) / sample_rate

    window = np.hanning(NFFT).astype(np.float32)
    window_scale = np.sum(window)
    if window_scale == 0:
        window_scale = 1.0

    DCWidth = 5000
    kill_inds = np.where(np.abs(f) <= DCWidth)[0]
    c = constants.c

    k_eff = min(k, NFFT)
    out_path = filename + ".txt"

    with open(out_path, "w") as out:
        for i in range(num_frames):
            start = i * step
            frame = _read_frame(raw, storage_kind, start, NFFT, sample_count)

            spectrum = np.fft.fftshift(np.fft.fft(frame * window)) / window_scale
            working_row = np.abs(spectrum)
            working_row[kill_inds] = 0

            large_inds = np.argpartition(working_row, -k_eff)[-k_eff:]
            large_inds = large_inds[np.argsort(f[large_inds])]

            large_vals = spectrum[large_inds]
            rel_vels = c * f[large_inds] / center_freq

            # Keep same layout as the original code: first k columns are relative velocities,
            # next k columns are complex STFT values. Store velocities as complex for compatibility.
            row = np.concatenate((rel_vels.astype(np.complex128), large_vals.astype(np.complex128)))
            out.write(_format_complex_row(row))

            if (i + 1) % 10000 == 0:
                print(f"  compressed {i + 1}/{num_frames} frames")

    print(f"  Saved compressed file: {out_path}")
    return sample_rate, f, t


def get_num_samples(txt_path: str) -> int:
    """
    Count number of non-empty rows in a compressed txt file.
    Each row corresponds to one STFT time frame.
    """
    count = 0
    with open(txt_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _parse_complex_line(line):
    """
    Parse one compressed txt row.
    Handles both the new format and older np.savetxt complex format with parentheses.
    """
    clean = line.replace("(", "").replace(")", "")
    return np.fromstring(clean, sep=" ", dtype=np.complex128)


def _iter_compressed_rows(txt_path):
    """Yield one compressed row at a time instead of loading the whole txt file."""
    with open(txt_path, "r") as f:
        for line in f:
            if line.strip():
                row = _parse_complex_line(line)
                if row.size:
                    yield row


def _downsample_for_plot(Sxx_db_T, max_cols=MAX_PLOT_TIME_BINS):
    """
    Downsample along the time direction for plotting only.
    Input shape is [time_bins, velocity_bins].
    """
    num_time_bins = Sxx_db_T.shape[0]
    if num_time_bins <= max_cols:
        return Sxx_db_T

    stride = int(np.ceil(num_time_bins / max_cols))
    return Sxx_db_T[::stride, :]


def compute_band_spectrogram(records_band, t0_global, sample_rate, k, NFFT, NOVERLAP, stitch_in_memory=False):
    """
    For a given frequency band, reconstruct and save sparse waterfalls.

    Memory-light version:
    - reads the compressed txt row-by-row
    - writes the full sparse waterfall to .npy using an on-disk memmap
    - only creates a downsampled dB copy for plotting

    Returns:
        f_axis, t_full, Sxx_full

    By default, Sxx_full is None to avoid concatenating all captures into RAM.
    Set stitch_in_memory=True only for small datasets.
    """
    records_band = sorted(records_band, key=lambda r: r["capture_time"])

    print("Record bands size ", len(records_band))

    f_axis = None
    Sxx_segments = []
    t_segments = []

    # Fixed relative velocity axis (m/s)
    v_lim = 5000
    n_vel_bins = 500
    v = np.linspace(-v_lim, v_lim, n_vel_bins)

    for j, rec in enumerate(records_band):
        txt_path = rec["data"]
        cf = rec["cf"]
        t_capture = rec["capture_time"]

        num_rows = get_num_samples(txt_path)
        if num_rows == 0:
            print(f"[WARN] No samples in {txt_path}, skipping")
            continue

        step = NFFT - NOVERLAP
        t = (np.arange(num_rows) * step + NFFT / 2) / sample_rate
        t_global = t

        if f_axis is None:
            f_axis = v
        else:
            if len(v) != len(f_axis) or not np.allclose(v, f_axis, rtol=0, atol=1.0):
                print(
                    f"[WARN] Frequency axis mismatch in {os.path.basename(txt_path)}, "
                    "skipping this capture for stitched plot."
                )
                continue

        out_name = f"r{j}_waterfall_CF_{cf/1e9:.3f}GHz.png"
        out_tag = f"r{j}_waterfall_CF_{cf/1e9:.3f}GHz.npy"

        # Full output goes directly to disk. Shape matches the original saved output: [time, velocity].
        sparse_out = np.lib.format.open_memmap(
            out_tag,
            mode="w+",
            dtype=np.complex64,
            shape=(num_rows, len(v)),
        )
        sparse_out[:] = 0

        # Small/medium plotting buffer. If the capture is very long, use time downsampling.
        plot_stride = max(1, int(np.ceil(num_rows / MAX_PLOT_TIME_BINS)))
        plot_rows = int(np.ceil(num_rows / plot_stride))
        Sxx_plot_T = np.full((plot_rows, len(v)), MIN_DB, dtype=np.float32)

        for i, row in enumerate(_iter_compressed_rows(txt_path)):
            if row.size < 2 * k:
                print(f"[WARN] Row {i} in {txt_path} has only {row.size} values, expected {2 * k}; skipping")
                continue

            recon_freqs = row[0:k].real
            recon_vals = row[k:2*k]

            indices = np.searchsorted(v, recon_freqs)
            indices = np.clip(indices, 0, len(v) - 1)
            sparse_out[i, indices] = recon_vals.astype(np.complex64)

            if i % plot_stride == 0:
                p = i // plot_stride
                mag_db = 10.0 * np.log10(np.abs(sparse_out[i, :]) + 1e-15)
                Sxx_plot_T[p, :] = mag_db.astype(np.float32)

        sparse_out.flush()
        del sparse_out

        np.save("time_" + out_tag, t_global)
        np.save("rel_vel_" + out_tag, f_axis)

        # Convert relative time to absolute timestamps.
        stop_np = np.datetime64(t_capture)
        offset_ns = (np.asarray(t) * 1e9).astype(np.int64)
        start_time = stop_np - offset_ns[-1].astype("timedelta64[ns]")
        datetimestamps = start_time + offset_ns.astype("timedelta64[ns]")
        np.save("datetime_updated_" + out_tag, datetimestamps)

        # Save per-band waterfall plot. Plot may be downsampled in time for memory reasons.
        if plot_stride == 1:
            t_plot = t
        else:
            t_plot = t[::plot_stride]
            print(f"  Plot downsampled by {plot_stride}x in time to reduce memory")

        extent = [f_axis[0], f_axis[-1], t_plot[0], t_plot[-1]]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            Sxx_plot_T,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="turbo",
            interpolation="bilinear",
            vmin=MIN_DB,
            vmax=MAX_DB,
        )
        fig.colorbar(im, ax=ax, label="Power (dB)")
        ax.set_xlabel("Relative velocity (m/s)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Waterfall: CF ≈ {cf/1e9:.3f} GHz")
        fig.tight_layout()
        fig.savefig(out_name, dpi=200)
        plt.close(fig)
        print(f"  Saved r{j} plot: {out_name}")
        print(f"  Saved r{j} sparse waterfall: {out_tag}")

        t_segments.append(t_global)

        if stitch_in_memory:
            # Only do this for small datasets. It reloads the saved memmap and transposes to old shape.
            sparse_loaded = np.load(out_tag, mmap_mode="r")
            Sxx_db = 10.0 * np.log10(np.abs(sparse_loaded.T) + 1e-15)
            Sxx_segments.append(np.asarray(Sxx_db, dtype=np.float32))

    if f_axis is None or not t_segments:
        return None, None, None

    t_full = np.concatenate(t_segments)

    if stitch_in_memory and Sxx_segments:
        Sxx_full = np.concatenate(Sxx_segments, axis=1)
    else:
        Sxx_full = None

    return f_axis, t_full, Sxx_full


def find_txt_records():
    """
    Scan current working directory for .txt files and return a list of dicts:
    {
        'data': txt_path,
        'cf': center_freq (Hz),
        'capture_time': datetime
    }
    """
    txt_files = glob.glob("*.txt")
    records = []

    for txt_path in sorted(txt_files):
        filename = os.path.basename(txt_path)

        cf, capture_time = parse_filename_metadata(filename)
        if cf is None or capture_time is None:
            print(f"[WARN] Could not parse metadata from {filename}, skipping")
            continue

        records.append(
            {
                "data": txt_path,
                "cf": cf,
                "capture_time": capture_time,
            }
        )

    return records


def parse_filename_metadata(filename: str):
    """
    Extract center frequency and capture time from filename:
    r001_f11.200GHz_20251217T204137.txt
    """
    freq_match = re.search(r"_f([\d.]+)GHz_", filename)
    time_match = re.search(r"_(\d{8}T\d{6})(?:\D|$)", filename)

    if not freq_match or not time_match:
        return None, None

    cf = float(freq_match.group(1)) * 1e9
    capture_time = datetime.strptime(time_match.group(1), "%Y%m%dT%H%M%S")

    return cf, capture_time


def main():
    ### USER DEFINED INPUTS ###
    sample_rate = 500000.0

    # Specify the number of frequencies to keep.
    k = 10

    # Specify the STFT parameters.
    NFFT = 1024
    NOVERLAP = 512

    # Compress all .sigmf files in the current working directory.
    # Comment this out if the .txt compressed files are already generated.
    compress_all_sigmf_in_cwd(k, NFFT, NOVERLAP)

    records = find_txt_records()
    if not records:
        print("No valid compressed .txt captures found in working directory")
        return

    print(f"Found {len(records)} compressed .txt captures")

    t0_global = min(r["capture_time"] for r in records)
    print(f"Global start time: {t0_global}")

    bands = {}
    for r in records:
        bands.setdefault(r["cf"], []).append(r)

    cfs = sorted(bands.keys())
    print(f"Discovered {len(cfs)} distinct center frequencies")
    print(cfs)

    band_results = {}

    for cf in cfs:
        print(f"\n=== Processing band CF = {cf/1e9:.3f} GHz ===")
        f_axis, t_full, Sxx_full = compute_band_spectrogram(
            bands[cf],
            t0_global,
            sample_rate,
            k,
            NFFT,
            NOVERLAP,
            stitch_in_memory=False,
        )
        if f_axis is None:
            print(f"  [WARN] No data for CF {cf/1e9:.3f} GHz")
            continue

        band_results[cf] = (f_axis, t_full, Sxx_full)


if __name__ == "__main__":
    main()
