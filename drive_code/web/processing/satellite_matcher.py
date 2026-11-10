from collections import defaultdict, deque

import numpy as np


class SatelliteMatcher:
    def __init__(self, store, tolerance_hz=1500):
        self.store = store
        self.tolerance = tolerance_hz
        self.points = defaultdict(lambda: deque(maxlen=300))
        self.last_attempt = {}
        self.matches = {}

    def update(self, event):
        output = []
        now = event["timestamp_s"]
        active = {t["track_id"] for t in event["tracks"] if t["active"]}
        for stale in set(self.points) - active:
            self.points.pop(stale, None)
            self.last_attempt.pop(stale, None)
            self.matches.pop(stale, None)
        for track in event["tracks"]:
            tid = track["track_id"]
            if not track["active"] or track["estimated_frequency_offset_hz"] is None:
                continue
            self.points[tid].append((now, track["estimated_frequency_offset_hz"]))
            points = self.points[tid]
            if len(points) < 30 or now - self.last_attempt.get(tid, 0) < 2:
                continue
            self.last_attempt[tid] = now
            times, frequencies = np.asarray(points).T
            if times[-1] - times[0] < 0.25:
                continue
            best = None
            for record in self.store.overlapping(times[0], times[-1]):
                predicted = np.interp(
                    times, record["timestamps"], record["predicted_doppler_hz"]
                )
                error = float(np.sqrt(np.mean((frequencies - predicted) ** 2)))
                if best is None or error < best[0]:
                    best = error, record
            if best is None or best[0] > self.tolerance:
                continue
            error, record = best
            if self.matches.get(tid) == record["satellite_id"]:
                continue
            self.matches[tid] = record["satellite_id"]
            output.append(
                {
                    "type": "satellite_match",
                    "processing_session_id": event["processing_session_id"],
                    "capture_session_id": event["capture_session_id"],
                    "config_version": event["config_version"],
                    "sequence": event["sequence"],
                    "timestamp_s": now,
                    "timestamp_utc": event["timestamp_utc"],
                    "track_id": tid,
                    "satellite": {
                        "id": str(record["satellite_id"]),
                        "name": record["satellite_name"],
                        "match_score": float(np.exp(-error / self.tolerance)),
                        "rmse_hz": error,
                    },
                    "predicted_trajectory": {
                        "timestamps": record["timestamps"],
                        "doppler_hz": record["predicted_doppler_hz"],
                        "azimuth_deg": record["azimuth_deg"],
                        "elevation_deg": record["elevation_deg"],
                    },
                }
            )
        return output
