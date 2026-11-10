"""Precomputed predictions only. No orbit propagation in the live path."""

import json
from pathlib import Path

import numpy as np


class PredictionStore:
    def __init__(self, directory):
        self.records = []
        self.errors = []
        for path in sorted(Path(directory).glob("*.json")):
            try:
                raw = json.loads(path.read_text())
                for record in raw if isinstance(raw, list) else [raw]:
                    arrays = [
                        np.asarray(record[k], dtype=float)
                        for k in (
                            "timestamps",
                            "predicted_doppler_hz",
                            "azimuth_deg",
                            "elevation_deg",
                        )
                    ]
                    if len(arrays[0]) < 2 or any(
                        a.ndim != 1
                        or len(a) != len(arrays[0])
                        or not np.all(np.isfinite(a))
                        for a in arrays
                    ):
                        raise ValueError(
                            "Prediction arrays must be finite, aligned, and have at least two samples"
                        )
                    if not np.all(np.diff(arrays[0]) > 0):
                        raise ValueError("Prediction timestamps must increase")
                    if not record.get("satellite_id") or not record.get(
                        "satellite_name"
                    ):
                        raise ValueError("Satellite identity is required")
                    self.records.append(record)
            except (ValueError, KeyError, TypeError) as exc:
                self.errors.append(f"{path.name}: {exc}")

    def overlapping(self, start, end):
        return [
            r
            for r in self.records
            if r["timestamps"][0] <= start and r["timestamps"][-1] >= end
        ]
