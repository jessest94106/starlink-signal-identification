"""Reuse the supplied engine; never duplicate the detector or Kalman logic."""

import math
from dataclasses import asdict

from satellite_tracker_api import (
    DetectorConfig,
    KalmanConfig,
    TentativeTrackConfig,
    TrackingConfig,
    TrackingEngine,
    TrackManagerConfig,
)

GROUPS = {
    "detector": DetectorConfig,
    "tentative": TentativeTrackConfig,
    "kalman": KalmanConfig,
    "manager": TrackManagerConfig,
}


def default_processing():
    value = asdict(TrackingConfig())
    value["detector"].pop("sample_rate_hz")
    return value


def tracking_config(receiver, processing):
    groups = {
        name: cls(
            **(
                {**processing[name], "sample_rate_hz": receiver["sample_rate_hz"]}
                if name == "detector"
                else processing[name]
            )
        )
        for name, cls in GROUPS.items()
    }
    result = TrackingConfig(**groups)
    if result.detector.sample_count > 2_000_000:
        raise ValueError("Maximum supported epoch size is 2 million samples")
    if result.tentative.minimum_hits > result.tentative.confirmation_window_epochs:
        raise ValueError("minimum_hits exceeds confirmation_window_epochs")
    if (
        result.manager.maximum_confirmed_tracks > 100
        or (result.manager.maximum_tentative_tracks or 0) > 4096
    ):
        raise ValueError("Track capacity exceeds service limits")
    return result


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


class TrackerAdapter:
    def __init__(self, config):
        self.engine = TrackingEngine(config)
        self.active = set()

    def process(self, samples, elapsed, metadata):
        result = self.engine.process_iq_block(samples, elapsed)
        tracks = []
        # Project recent snapshots; the service prunes retained histories independently.
        for track in self.engine.track_snapshots(elapsed - 0.1):
            if not len(track.elapsed_times_s):
                continue
            tracks.append(
                {
                    "track_id": track.track_id,
                    "active": track.active,
                    "measured_frequency_offset_hz": finite(
                        track.measured_frequency_offsets_hz[-1]
                    ),
                    "estimated_frequency_offset_hz": finite(
                        track.estimated_frequency_offsets_hz[-1]
                    ),
                    "estimated_frequency_rate_hz_per_s": finite(
                        track.estimated_frequency_rates_hz_per_s[-1]
                    ),
                    "density": finite(track.density),
                    "maximum_gap_epochs": track.maximum_gap_epochs,
                }
            )
        current = {t["track_id"] for t in tracks if t["active"]}
        lifecycle = [
            {"type": "track_confirmed", "track_id": int(t)}
            for t in result.newly_confirmed_track_ids
        ] + [{"type": "track_terminated", "track_id": t} for t in self.active - current]
        self.active = current
        event = {
            **metadata,
            "type": "track_update",
            "epoch_index": result.epoch_index,
            "strongest_detection_snr_db": finite(result.strongest_detection_snr_db),
            "detections": [
                {
                    "frequency_offset_hz": float(f),
                    "track_id": int(t) if t >= 0 else None,
                }
                for f, t in zip(
                    result.detected_frequency_offsets_hz, result.assigned_track_ids
                )
            ],
            "tracks": tracks,
            "processing_latency_ms": result.timing.total_ns / 1e6,
        }
        return event, [{**metadata, **item} for item in lifecycle]
