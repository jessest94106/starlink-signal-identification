#!/usr/bin/env python3
"""Minimal USRP raw-IQ capture for synchronization experiments."""

import json
import os
from datetime import datetime, timezone

import numpy as np
import uhd

# ======================== PARAMETERS ========================
USRP_ARGS = "type=b200"
# Starlink OFDM channel center, used as the frequency reference.
CHANNEL_CENTER_RF_HZ = 12.325e9

# Capture the upper edge-pilot band near the upper side of a 25 MHz
# usable passband, while sampling at 30 MS/s for filter margin and
# exact 8x resampling to the native 240 MHz Starlink grid.
PILOT_UPPER_EDGE_RF_HZ = CHANNEL_CENTER_RF_HZ + 116.015625e6
USABLE_BANDWIDTH_HZ = 25e6
EDGE_MARGIN_HZ = 1e6
RF_FREQUENCY_HZ = (
    PILOT_UPPER_EDGE_RF_HZ
    - USABLE_BANDWIDTH_HZ / 2
    + EDGE_MARGIN_HZ
)

LNB_LO_HZ = 9.75e9
SAMPLE_RATE_HZ = 30e6
GAIN_DB = 40.0
ANTENNA = "RX2"
CAPTURE_SECONDS = 1.0
CHUNK_SAMPLES = 1_000_000
OUTPUT_DIR = "/home/funlab/joanna_capture"
# ============================================================


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


os.makedirs(OUTPUT_DIR, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
base = os.path.join(OUTPUT_DIR, f"sync_{RF_FREQUENCY_HZ/1e9:.6f}GHz_{stamp}")
data_path = base + ".sigmf-data"
meta_path = base + ".sigmf-meta"

usrp = uhd.usrp.MultiUSRP(USRP_ARGS)
usrp.set_rx_rate(SAMPLE_RATE_HZ)
usrp.set_rx_gain(GAIN_DB)
usrp.set_rx_antenna(ANTENNA)

if_hz = RF_FREQUENCY_HZ - LNB_LO_HZ
if if_hz <= 0:
    raise ValueError("RF_FREQUENCY_HZ must be greater than LNB_LO_HZ")

usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(if_hz))
actual_rate = float(usrp.get_rx_rate())
actual_gain = float(usrp.get_rx_gain())
actual_if = float(usrp.get_rx_freq())

target_samples = int(round(actual_rate * CAPTURE_SECONDS))
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
streamer = usrp.get_rx_stream(stream_args)

buffer = np.empty(CHUNK_SAMPLES, dtype=np.complex64)
metadata = uhd.types.RXMetadata()
start_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
start_cmd.stream_now = True

start_utc = utc_now()
streamer.issue_stream_cmd(start_cmd)
received = 0
overflows = 0

try:
    with open(data_path, "wb", buffering=8 * 1024 * 1024) as output:
        while received < target_samples:
            count = min(CHUNK_SAMPLES, target_samples - received)
            n = streamer.recv(buffer[:count], metadata, timeout=3.0)

            if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                overflows += 1
                continue
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                raise RuntimeError(metadata.strerror())
            if n:
                buffer[:n].tofile(output)
                received += n
finally:
    stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stop_cmd)

end_utc = utc_now()

meta = {
    "global": {
        "core:datatype": "cf32_le",
        "core:sample_rate": actual_rate,
        "core:version": "1.0.0",
        "core:description": "Raw Starlink synchronization capture",
        "core:recorder": "USRP + LNB",
    },
    "captures": [{
        "core:sample_start": 0,
        "core:frequency": RF_FREQUENCY_HZ,
        "core:datetime": start_utc,
    }],
    "annotations": [],
    "custom": {
        "end_datetime": end_utc,
        "channel_center_rf_hz": CHANNEL_CENTER_RF_HZ,
        "pilot_upper_edge_rf_hz": PILOT_UPPER_EDGE_RF_HZ,
        "usable_bandwidth_hz": USABLE_BANDWIDTH_HZ,
        "edge_margin_hz": EDGE_MARGIN_HZ,
        "requested_rf_hz": RF_FREQUENCY_HZ,
        "lnb_lo_hz": LNB_LO_HZ,
        "requested_if_hz": if_hz,
        "actual_if_hz": actual_if,
        "gain_db": actual_gain,
        "sample_count": received,
        "overflow_count": overflows,
    },
}

with open(meta_path, "w") as output:
    json.dump(meta, output, indent=2)

print(data_path)
print(meta_path)
print(f"channel_center_rf={CHANNEL_CENTER_RF_HZ/1e9:.9f} GHz")
print(f"capture_center_rf={RF_FREQUENCY_HZ/1e9:.9f} GHz")
print(f"capture_center_if={actual_if/1e9:.9f} GHz")
print(f"sample_rate={actual_rate/1e6:.3f} MS/s")
print(f"samples={received}, duration={received/actual_rate:.6f}s, overflows={overflows}")
