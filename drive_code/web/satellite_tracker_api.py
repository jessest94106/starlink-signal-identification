from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from time import perf_counter_ns
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import fft as scipy_fft


ComplexArray: TypeAlias = NDArray[np.complex64] | NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
IqEpoch: TypeAlias = tuple[float, ComplexArray]
ObservationEpoch: TypeAlias = tuple[float, FloatArray, float]


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _readonly(array: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_maximum_epochs(maximum_epochs: int | None) -> None:
    if maximum_epochs is not None and maximum_epochs < 0:
        raise ValueError("maximum_epochs must be nonnegative")


# region 1. Configuration

@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """FFT and CA-CFAR configuration."""

    sample_rate_hz: float = 500_000.0
    epoch_duration_s: float = 0.010
    cfar_training_bins_per_side: int = 220
    cfar_guard_bins_per_side: int = 4
    cfar_excluded_lower_hz: float | None = -12_500.0
    cfar_excluded_upper_hz: float | None = 22_500.0
    cfar_false_alarm_probability: float = 3e-5

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")
        if not np.isfinite(self.epoch_duration_s) or self.epoch_duration_s <= 0:
            raise ValueError("epoch_duration_s must be positive and finite")
        if self.cfar_training_bins_per_side <= 0:
            raise ValueError("cfar_training_bins_per_side must be positive")
        if self.cfar_guard_bins_per_side < 0:
            raise ValueError("cfar_guard_bins_per_side must be nonnegative")
        if self.sample_count <= 2 * (
            self.cfar_training_bins_per_side + self.cfar_guard_bins_per_side
        ):
            raise ValueError("epoch is too short for the requested CFAR geometry")
        if not 0 < self.cfar_false_alarm_probability < 1:
            raise ValueError("cfar_false_alarm_probability must lie in (0, 1)")
        has_lower = self.cfar_excluded_lower_hz is not None
        has_upper = self.cfar_excluded_upper_hz is not None
        if has_lower != has_upper:
            raise ValueError("both CFAR exclusion bounds must be set or absent")
        if has_lower:
            lower = _finite(self.cfar_excluded_lower_hz, "cfar_excluded_lower_hz")
            upper = _finite(self.cfar_excluded_upper_hz, "cfar_excluded_upper_hz")
            if lower >= upper:
                raise ValueError("lower CFAR exclusion bound must be below upper")

    @property
    def sample_count(self) -> int:
        return int(round(self.sample_rate_hz * self.epoch_duration_s))

    @property
    def actual_epoch_duration_s(self) -> float:
        return self.sample_count / self.sample_rate_hz

    @property
    def fft_bin_width_hz(self) -> float:
        return self.sample_rate_hz / self.sample_count


@dataclass(frozen=True, slots=True)
class TentativeTrackConfig:
    """Tentative-track association, confirmation, and removal rules."""

    confirmation_window_epochs: int = 100
    minimum_hits: int = 10
    association_radius_bins: float = 3.0
    minimum_decrease_rate_hz_per_s: float | None = 1_600.0

    def __post_init__(self) -> None:
        if self.confirmation_window_epochs <= 0:
            raise ValueError("confirmation_window_epochs must be positive")
        if self.minimum_hits < 2:
            raise ValueError("minimum_hits must be at least two")
        if not np.isfinite(self.association_radius_bins) or self.association_radius_bins < 0:
            raise ValueError("association_radius_bins must be finite and nonnegative")
        if self.minimum_decrease_rate_hz_per_s is not None:
            value = _finite(
                self.minimum_decrease_rate_hz_per_s,
                "minimum_decrease_rate_hz_per_s",
            )
            if value < 0:
                raise ValueError("minimum_decrease_rate_hz_per_s must be nonnegative")


@dataclass(frozen=True, slots=True)
class KalmanConfig:
    """Confirmed-track Kalman and innovation-gate configuration."""

    gate_sigma: float = 3.0
    measurement_std_bins: float = 1.0
    initial_rate_std_bins_per_epoch: float = 0.10
    acceleration_std_bins_per_epoch2: float = 0.003

    def __post_init__(self) -> None:
        values = {
            "gate_sigma": self.gate_sigma,
            "measurement_std_bins": self.measurement_std_bins,
            "initial_rate_std_bins_per_epoch": self.initial_rate_std_bins_per_epoch,
            "acceleration_std_bins_per_epoch2": self.acceleration_std_bins_per_epoch2,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class TrackManagerConfig:
    """Track capacity, termination, and shared association limits."""

    maximum_confirmed_tracks: int = 5
    maximum_tentative_tracks: int | None = 256
    confirmed_maximum_missed_epochs: int = 100
    maximum_upward_jitter_bins: int = 1

    def __post_init__(self) -> None:
        if self.maximum_confirmed_tracks <= 0:
            raise ValueError("maximum_confirmed_tracks must be positive")
        if self.maximum_tentative_tracks is not None and self.maximum_tentative_tracks <= 0:
            raise ValueError("maximum_tentative_tracks must be positive or None")
        if self.confirmed_maximum_missed_epochs < 0:
            raise ValueError("confirmed_maximum_missed_epochs must be nonnegative")
        if self.maximum_upward_jitter_bins < 0:
            raise ValueError("maximum_upward_jitter_bins must be nonnegative")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Complete detector and tracking configuration."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tentative: TentativeTrackConfig = field(default_factory=TentativeTrackConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    manager: TrackManagerConfig = field(default_factory=TrackManagerConfig)


@dataclass(frozen=True, slots=True)
class DisplayConfig:
    """GUI history, queue, and point-budget configuration."""

    detail_history_s: float = 3.0
    overview_history_s: float = 60.0
    telemetry_history_s: float = 50.0
    track_trail_s: float = 60.0
    render_rate_hz: float = 30.0
    display_queue_epochs: int = 512
    display_queue_drain_ms: float = 3.0
    maximum_overview_points: int = 4_000
    maximum_detail_points: int = 1_000
    maximum_track_points_per_panel: int = 1_000
    camera_smoothing_s: float = 0.180

    def __post_init__(self) -> None:
        positive_floats = {
            "detail_history_s": self.detail_history_s,
            "overview_history_s": self.overview_history_s,
            "telemetry_history_s": self.telemetry_history_s,
            "track_trail_s": self.track_trail_s,
            "render_rate_hz": self.render_rate_hz,
            "display_queue_drain_ms": self.display_queue_drain_ms,
            "camera_smoothing_s": self.camera_smoothing_s,
        }
        for name, value in positive_floats.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        positive_ints = {
            "display_queue_epochs": self.display_queue_epochs,
            "maximum_overview_points": self.maximum_overview_points,
            "maximum_detail_points": self.maximum_detail_points,
            "maximum_track_points_per_panel": self.maximum_track_points_per_panel,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

# endregion


# region 2. Track management and engine

class TrackManager:
    """Own association, centralized lifecycle transitions, IDs, and snapshots."""

    def __init__(self, config: TrackingConfig = TrackingConfig()) -> None:
        self._config = config
        self._fft_bin_width_hz = config.detector.fft_bin_width_hz
        self._epoch_duration_s = config.detector.actual_epoch_duration_s
        self.reset()

    @property
    def overload_removals(self) -> int:
        return self._overload_removals

    def reset(self) -> None:
        self._active_tracks: list[ConfirmedTrack] = []
        self._confirmed_by_id: dict[int, ConfirmedTrack] = {}
        self._tentative_by_id: dict[int, TentativeTrack] = {}
        self._next_track_id = 1
        self._overload_removals = 0

    def _start_tentative(
        self,
        epoch_index: int,
        elapsed_time_s: float,
        observed_bin: int,
    ) -> int:
        """Transition a new observation into a tentative track."""
        track_id = self._next_track_id
        track = TentativeTrack.start(
            track_id,
            epoch_index,
            elapsed_time_s,
            observed_bin,
        )
        self._tentative_by_id[track_id] = track
        self._next_track_id += 1
        return track_id

    def _confirm_tentative(
        self,
        track_id: int,
        epoch_index: int,
        elapsed_time_s: float,
    ) -> ConfirmedTrack:
        """Transition one tentative track into an active confirmed track."""
        tentative = self._tentative_by_id[track_id]
        confirmed = ConfirmedTrack.from_tentative(
            tentative,
            epoch_index,
            elapsed_time_s,
            self._config.kalman,
        )
        self._confirmed_by_id[track_id] = confirmed
        self._active_tracks.append(confirmed)
        del self._tentative_by_id[track_id]
        return confirmed

    def _discard_tentative(self, track_id: int) -> None:
        """Transition one tentative track into the unretained discarded state."""
        del self._tentative_by_id[track_id]

    def _terminate_confirmed(self, track: ConfirmedTrack) -> None:
        """Transition one active confirmed track into retained history."""
        track.terminate()

    def _remove_stale_tracks(self, epoch_index: int) -> None:
        survivors: list[ConfirmedTrack] = []
        for track in self._active_tracks:
            if track.should_terminate(epoch_index, self._config.manager):
                self._terminate_confirmed(track)
            else:
                survivors.append(track)
        self._active_tracks = survivors

        stale_ids = [
            track_id for track_id, track in self._tentative_by_id.items()
            if track.should_remove(epoch_index, self._config.tentative)
        ]
        for track_id in stale_ids:
            self._discard_tentative(track_id)

    @staticmethod
    def _nearest_unused(
        bins: IntArray,
        unused: NDArray[np.bool_],
        lower: float,
        upper: float,
        predicted_bin: float,
    ) -> int | None:
        if not bins.size or upper < bins[0] or lower > bins[-1]:
            return None
        first = int(np.searchsorted(bins, lower, side="left"))
        stop = int(np.searchsorted(bins, upper, side="right"))
        if first >= stop:
            return None
        candidates = np.flatnonzero(unused[first:stop]) + first
        if not candidates.size:
            return None
        return int(
            candidates[np.argmin(np.abs(bins[candidates] - predicted_bin))]
        )

    def _associate_confirmed(
        self,
        epoch_index: int,
        elapsed_time_s: float,
        bins: IntArray,
        unused: NDArray[np.bool_],
        candidate_track_ids: IntArray,
    ) -> None:
        if not bins.size or not np.any(unused):
            return
        self._active_tracks.sort(
            key=lambda track: track.observation_count,
            reverse=True,
        )
        for track in self._active_tracks:
            prediction = track.predict(epoch_index)
            lower, upper = track.validation_gate(
                prediction,
                self._config.kalman,
                self._config.manager,
            )
            chosen = self._nearest_unused(
                bins,
                unused,
                lower,
                upper,
                prediction.estimated_frequency_bin,
            )
            if chosen is None:
                continue
            unused[chosen] = False
            track.update(
                epoch_index,
                elapsed_time_s,
                int(bins[chosen]),
                prediction,
                self._config.kalman,
            )
            candidate_track_ids[chosen] = track._track_id
            if not np.any(unused):
                return

    def _associate_tentative(
        self,
        epoch_index: int,
        elapsed_time_s: float,
        bins: IntArray,
        unused: NDArray[np.bool_],
        candidate_track_ids: IntArray,
    ) -> None:
        if not bins.size or not np.any(unused):
            return
        ordered_tracks = sorted(
            self._tentative_by_id.values(),
            key=lambda track: len(track._epoch_indices),
            reverse=True,
        )
        for track in ordered_tracks:
            predicted_bin = track.predict_bin(epoch_index)
            lower, upper = track.validation_gate(
                predicted_bin,
                self._config.tentative,
                self._config.manager,
            )
            chosen = self._nearest_unused(
                bins,
                unused,
                lower,
                upper,
                predicted_bin,
            )
            if chosen is None:
                continue
            unused[chosen] = False
            track.append_observation(
                epoch_index,
                elapsed_time_s,
                int(bins[chosen]),
            )
            candidate_track_ids[chosen] = track._track_id
            if not np.any(unused):
                return

    def _start_tentative_tracks(
        self,
        epoch_index: int,
        elapsed_time_s: float,
        bins: IntArray,
        unused: NDArray[np.bool_],
        candidate_track_ids: IntArray,
    ) -> None:
        for index in np.flatnonzero(unused):
            track_id = self._start_tentative(
                epoch_index,
                elapsed_time_s,
                int(bins[index]),
            )
            candidate_track_ids[index] = track_id

    def _confirm_ready_tracks(
        self,
        epoch_index: int,
        elapsed_time_s: float,
    ) -> tuple[int, ...]:
        slots = (
            self._config.manager.maximum_confirmed_tracks
            - len(self._active_tracks)
        )
        if slots <= 0:
            return ()
        tentative_config = self._config.tentative
        ready = [
            track
            for track in self._tentative_by_id.values()
            if (
                len(track._epoch_indices) >= tentative_config.minimum_hits
                and track.is_ready(
                    tentative_config,
                    self._fft_bin_width_hz,
                    self._epoch_duration_s,
                )
            )
        ]
        ready.sort(
            key=lambda track: (
                track.density,
                -track.fitted_rate_bins_per_epoch,
                track._observed_bins[0] - track._observed_bins[-1],
                len(track._epoch_indices),
            ),
            reverse=True,
        )
        confirmed_ids: list[int] = []
        for tentative in ready[:slots]:
            track_id = tentative._track_id
            self._confirm_tentative(
                track_id,
                epoch_index,
                elapsed_time_s,
            )
            confirmed_ids.append(track_id)
        return tuple(confirmed_ids)

    def _enforce_tentative_limit(self) -> None:
        limit = self._config.manager.maximum_tentative_tracks
        if limit is None or len(self._tentative_by_id) <= limit:
            return
        ordered = sorted(
            self._tentative_by_id.values(),
            key=lambda track: (
                track.density,
                len(track._epoch_indices),
                track._last_epoch_index,
            ),
            reverse=True,
        )
        keep_ids = {track._track_id for track in ordered[:limit]}
        discard_ids = [
            track_id
            for track_id in self._tentative_by_id
            if track_id not in keep_ids
        ]
        self._overload_removals += len(discard_ids)
        for track_id in discard_ids:
            self._discard_tentative(track_id)

    def process_epoch(
        self,
        epoch_index: int,
        elapsed_time_s: float,
        frequency_offsets_hz: FloatArray,
    ) -> _TrackManagerOutput:
        if epoch_index < 0:
            raise ValueError("epoch_index must be nonnegative")
        time_s = _finite(elapsed_time_s, "elapsed_time_s")
        frequencies = np.asarray(frequency_offsets_hz, dtype=np.float64).reshape(-1)
        candidates = _form_candidates(
            frequencies,
            self._fft_bin_width_hz,
        )
        bins = candidates.representative_bins
        self._remove_stale_tracks(epoch_index)
        unused = np.ones(bins.size, dtype=bool)
        candidate_track_ids = np.full(bins.size, -1, dtype=np.int64)
        self._associate_confirmed(
            epoch_index,
            time_s,
            bins,
            unused,
            candidate_track_ids,
        )
        self._associate_tentative(
            epoch_index,
            time_s,
            bins,
            unused,
            candidate_track_ids,
        )
        self._start_tentative_tracks(
            epoch_index,
            time_s,
            bins,
            unused,
            candidate_track_ids,
        )
        newly_confirmed = self._confirm_ready_tracks(epoch_index, time_s)
        self._enforce_tentative_limit()
        if not candidates.raw_to_candidate.size:
            assigned = np.empty(0, dtype=np.int64)
        else:
            assigned = candidate_track_ids[candidates.raw_to_candidate]
        return _TrackManagerOutput(
            assigned_track_ids=_readonly(assigned, np.int64),
            newly_confirmed_track_ids=newly_confirmed,
        )

    def snapshots(
        self,
        minimum_elapsed_time_s: float | None = None,
    ) -> list[TrackSnapshot]:
        if minimum_elapsed_time_s is not None:
            minimum_elapsed_time_s = _finite(
                minimum_elapsed_time_s, "minimum_elapsed_time_s"
            )
        active_ids = {track._track_id for track in self._active_tracks}
        detector = self._config.detector
        snapshots: list[TrackSnapshot] = []
        for track in self._confirmed_by_id.values():
            start = (
                bisect_left(track._elapsed_times_s, minimum_elapsed_time_s)
                if minimum_elapsed_time_s is not None
                else 0
            )
            if start >= len(track._elapsed_times_s):
                continue
            elapsed = np.asarray(track._elapsed_times_s[start:], dtype=np.float64)
            measured_bins = np.asarray(track._observed_bins[start:], dtype=np.float64)
            estimated_bins = np.asarray(track._estimated_bins[start:], dtype=np.float64)
            estimated_rates = np.asarray(track._estimated_rates[start:], dtype=np.float64)
            full_span = (
                track._last_epoch_index
                - track._first_epoch_index
                + 1
            )
            snapshots.append(
                TrackSnapshot(
                    track_id=track._track_id,
                    active=track._track_id in active_ids,
                    first_elapsed_time_s=track._first_elapsed_time_s,
                    confirmation_elapsed_time_s=track._confirmation_elapsed_time_s,
                    elapsed_times_s=_readonly(elapsed, np.float64),
                    measured_frequency_offsets_hz=_readonly(
                        measured_bins * detector.fft_bin_width_hz,
                        np.float64,
                    ),
                    estimated_frequency_offsets_hz=_readonly(
                        estimated_bins * detector.fft_bin_width_hz,
                        np.float64,
                    ),
                    estimated_frequency_rates_hz_per_s=_readonly(
                        estimated_rates
                        * detector.fft_bin_width_hz
                        / detector.actual_epoch_duration_s,
                        np.float64,
                    ),
                    density=track.observation_count / full_span,
                    maximum_gap_epochs=track._maximum_gap_epochs,
                )
            )
        return snapshots

    def finish(self) -> None:
        for track in self._active_tracks:
            self._terminate_confirmed(track)
        self._active_tracks = []


class TrackingEngine:
    """Thread-safe synchronous detector/tracker boundary."""

    def __init__(self, config: TrackingConfig = TrackingConfig()) -> None:
        self._config = config
        self._detector = Detector(config.detector)
        self._manager = TrackManager(config)
        self._lock = threading.RLock()
        self._epoch_duration_s = config.detector.actual_epoch_duration_s
        self._next_epoch_index = 0

    @property
    def config(self) -> TrackingConfig:
        return self._config

    @property
    def next_epoch_index(self) -> int:
        with self._lock:
            return self._next_epoch_index

    def _elapsed_time(self, elapsed_time_s: float | None) -> float:
        if elapsed_time_s is None:
            return self._next_epoch_index * self._epoch_duration_s
        return _finite(elapsed_time_s, "elapsed_time_s")

    def process_iq_block(
        self,
        samples: ComplexArray,
        elapsed_time_s: float | None = None,
    ) -> ProcessedEpoch:
        with self._lock:
            started = perf_counter_ns()
            epoch_index = self._next_epoch_index
            time_s = self._elapsed_time(elapsed_time_s)
            detected = self._detector.process_block(samples)
            tracking_started = perf_counter_ns()
            manager_output = self._manager.process_epoch(
                epoch_index,
                time_s,
                detected.detected_frequency_offsets_hz,
            )
            finished = perf_counter_ns()
            self._next_epoch_index += 1
            return ProcessedEpoch(
                epoch_index=epoch_index,
                elapsed_time_s=time_s,
                detected_frequency_offsets_hz=detected.detected_frequency_offsets_hz,
                assigned_track_ids=manager_output.assigned_track_ids,
                strongest_detection_snr_db=detected.strongest_detection_snr_db,
                timing=ProcessingTiming(
                    fft_power_ns=detected.timing.fft_power_ns,
                    cfar_ns=detected.timing.cfar_ns,
                    tracking_ns=finished - tracking_started,
                    total_ns=finished - started,
                ),
                newly_confirmed_track_ids=(
                    manager_output.newly_confirmed_track_ids
                ),
            )

    def process_frequency_observations(
        self,
        frequency_offsets_hz: Iterable[float] | FloatArray,
        elapsed_time_s: float | None = None,
        strongest_detection_snr_db: float = float("nan"),
    ) -> ProcessedEpoch:
        with self._lock:
            started = perf_counter_ns()
            epoch_index = self._next_epoch_index
            time_s = self._elapsed_time(elapsed_time_s)
            frequencies = np.asarray(
                frequency_offsets_hz,
                dtype=np.float64,
            ).reshape(-1)
            snr_db = float(strongest_detection_snr_db)
            if not (np.isnan(snr_db) or np.isfinite(snr_db)):
                raise ValueError("strongest_detection_snr_db must be finite or NaN")
            tracking_started = perf_counter_ns()
            manager_output = self._manager.process_epoch(
                epoch_index,
                time_s,
                frequencies,
            )
            finished = perf_counter_ns()
            self._next_epoch_index += 1
            return ProcessedEpoch(
                epoch_index=epoch_index,
                elapsed_time_s=time_s,
                detected_frequency_offsets_hz=_readonly(frequencies, np.float64),
                assigned_track_ids=manager_output.assigned_track_ids,
                strongest_detection_snr_db=snr_db,
                timing=ProcessingTiming(
                    fft_power_ns=0,
                    cfar_ns=0,
                    tracking_ns=finished - tracking_started,
                    total_ns=finished - started,
                ),
                newly_confirmed_track_ids=(
                    manager_output.newly_confirmed_track_ids
                ),
            )

    def track_snapshots(
        self,
        minimum_elapsed_time_s: float | None = None,
    ) -> list[TrackSnapshot]:
        with self._lock:
            return self._manager.snapshots(minimum_elapsed_time_s)

    def try_confirmed_monitor_state(
        self,
        minimum_elapsed_time_s: float | None = None,
    ) -> MonitorState | None:
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return MonitorState(
                confirmed_tracks=self._manager.snapshots(minimum_elapsed_time_s),
                overload_removals=self._manager.overload_removals,
                next_epoch_index=self._next_epoch_index,
            )
        finally:
            self._lock.release()

    def skip_epochs(self, count: int) -> None:
        """Advance the epoch clock for dropped acquisition blocks.

        The next association predicts across the actual gap instead of treating
        missing I/Q as contiguous samples. Does not fabricate observations.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("skipped epoch count must be a nonnegative integer")
        with self._lock:
            self._next_epoch_index += count

    def prune_confirmed_history(self, minimum_elapsed_time_s: float) -> None:
        """Bound streaming history without changing Kalman/association state.

        Optional: existing offline callers retain complete histories by default.
        Preserve lifetime count/density and at least the latest observation.
        """
        cutoff = _finite(minimum_elapsed_time_s, "minimum_elapsed_time_s")
        with self._lock:
            manager = self._manager
            active_ids = {track._track_id for track in manager._active_tracks}
            for track_id, track in list(manager._confirmed_by_id.items()):
                if track_id not in active_ids and track._elapsed_times_s[-1] < cutoff:
                    del manager._confirmed_by_id[track_id]
                    continue
                count = min(bisect_left(track._elapsed_times_s, cutoff), len(track._elapsed_times_s) - 1)
                if count <= 0:
                    continue
                for name in ("_epoch_indices", "_elapsed_times_s", "_observed_bins", "_estimated_bins", "_estimated_rates"):
                    del getattr(track, name)[:count]
                track._pruned_observation_count += count

    def finish(self) -> None:
        with self._lock:
            self._manager.finish()

    def reset(self) -> None:
        with self._lock:
            self._detector = Detector(self._config.detector)
            self._manager.reset()
            self._next_epoch_index = 0

# endregion


# region 3. Data records

@dataclass(frozen=True, slots=True)
class DetectorTiming:
    """Detector compute time; excludes tracking, I/O, and drawing."""

    fft_power_ns: int
    cfar_ns: int


@dataclass(frozen=True, slots=True)
class ProcessingTiming:
    """Complete synchronous epoch timing; excludes I/O and drawing."""

    fft_power_ns: int
    cfar_ns: int
    tracking_ns: int
    total_ns: int

    def as_microseconds(self) -> dict[str, float]:
        return {
            "fft_power": self.fft_power_ns / 1_000.0,
            "cfar": self.cfar_ns / 1_000.0,
            "tracking": self.tracking_ns / 1_000.0,
            "total": self.total_ns / 1_000.0,
        }


@dataclass(frozen=True, slots=True)
class DetectorOutput:
    """Output from one detector epoch."""

    detected_frequency_offsets_hz: FloatArray
    strongest_detection_snr_db: float
    timing: DetectorTiming


@dataclass(frozen=True, slots=True)
class ProcessedEpoch:
    """Output from one TrackingEngine processing call."""

    epoch_index: int
    elapsed_time_s: float
    detected_frequency_offsets_hz: FloatArray
    assigned_track_ids: IntArray
    strongest_detection_snr_db: float
    timing: ProcessingTiming
    newly_confirmed_track_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    """Minimal cached identity and lifecycle indices."""

    track_id: int
    first_epoch_index: int
    last_epoch_index: int
    maximum_gap_epochs: int


@dataclass(frozen=True, slots=True)
class KalmanState:
    """Two-state frequency/rate estimate and covariance."""

    estimated_frequency_bin: float
    estimated_rate_bins_per_epoch: float
    covariance_ff: float
    covariance_fr: float
    covariance_rr: float


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    """Immutable confirmed-track projection for output and display."""

    track_id: int
    active: bool
    first_elapsed_time_s: float
    confirmation_elapsed_time_s: float
    elapsed_times_s: FloatArray
    measured_frequency_offsets_hz: FloatArray
    estimated_frequency_offsets_hz: FloatArray
    estimated_frequency_rates_hz_per_s: FloatArray
    density: float
    maximum_gap_epochs: int


@dataclass(frozen=True, slots=True)
class MonitorState:
    """Atomic nonblocking state returned to the GUI."""

    confirmed_tracks: list[TrackSnapshot]
    overload_removals: int
    next_epoch_index: int


@dataclass(frozen=True, slots=True)
class _CandidateBatch:
    """Private adjacent-bin grouping result."""

    representative_bins: IntArray
    raw_to_candidate: IntArray


@dataclass(frozen=True, slots=True)
class _TrackManagerOutput:
    """Private manager output for one epoch."""

    assigned_track_ids: IntArray
    newly_confirmed_track_ids: tuple[int, ...]

# endregion


# region 4. Detection and candidate formation

class Detector:
    """Hann-windowed FFT and CA-CFAR detector."""

    def __init__(self, config: DetectorConfig = DetectorConfig()) -> None:
        self._config = config
        self._window = np.hanning(config.sample_count)
        self._frequency_offsets_hz = np.fft.fftshift(
            np.fft.fftfreq(config.sample_count, d=1.0 / config.sample_rate_hz)
        )

        edge = config.cfar_training_bins_per_side + config.cfar_guard_bins_per_side
        cut_bins = np.arange(edge, config.sample_count - edge)
        reference_mask = np.ones(config.sample_count, dtype=bool)
        if config.cfar_excluded_lower_hz is not None:
            cut_frequencies = self._frequency_offsets_hz[cut_bins]
            cut_bins = cut_bins[
                (cut_frequencies < config.cfar_excluded_lower_hz)
                | (cut_frequencies > config.cfar_excluded_upper_hz)
            ]
            reference_mask[
                (self._frequency_offsets_hz >= config.cfar_excluded_lower_hz)
                & (self._frequency_offsets_hz <= config.cfar_excluded_upper_hz)
            ] = False
        if not cut_bins.size:
            raise ValueError("CFAR exclusion leaves no valid CUT")

        counts = np.concatenate(([0], np.cumsum(reference_mask, dtype=np.int64)))
        guard = config.cfar_guard_bins_per_side
        left_count = counts[cut_bins - guard] - counts[cut_bins - edge]
        right_count = counts[cut_bins + edge + 1] - counts[cut_bins + guard + 1]
        reference_count = left_count + right_count
        if np.any(reference_count <= 0):
            raise ValueError("CFAR exclusion leaves a CUT without reference cells")

        self._cut_bins = cut_bins
        self._reference_mask = reference_mask
        self._reference_count = reference_count
        self._cfar_alpha = reference_count * (
            config.cfar_false_alarm_probability ** (-1.0 / reference_count) - 1.0
        )
        self._masked_power = np.empty(config.sample_count, dtype=np.float64)
        self._cumulative = np.empty(config.sample_count + 1, dtype=np.float64)

    @property
    def config(self) -> DetectorConfig:
        return self._config

    def _apply_cfar(self, power: FloatArray) -> tuple[IntArray, float]:
        config = self._config
        bins = self._cut_bins
        edge = config.cfar_training_bins_per_side + config.cfar_guard_bins_per_side
        guard = config.cfar_guard_bins_per_side

        self._cumulative[0] = 0.0
        np.multiply(power, self._reference_mask, out=self._masked_power)
        np.cumsum(self._masked_power, dtype=np.float64, out=self._cumulative[1:])
        left_sum = self._cumulative[bins - guard] - self._cumulative[bins - edge]
        right_sum = (
            self._cumulative[bins + edge + 1]
            - self._cumulative[bins + guard + 1]
        )
        noise_floor = (left_sum + right_sum) / self._reference_count
        detected_positions = np.flatnonzero(
            power[bins] > self._cfar_alpha * noise_floor
        )
        detected_bins = bins[detected_positions]
        if not detected_bins.size:
            return detected_bins, float("nan")
        strongest_position = int(np.argmax(power[detected_bins]))
        strongest_bin = detected_bins[strongest_position]
        strongest_noise = noise_floor[detected_positions[strongest_position]]
        snr_db = float(10.0 * np.log10(power[strongest_bin] / strongest_noise))
        return detected_bins, snr_db

    def process_block(self, samples: ComplexArray) -> DetectorOutput:
        values = np.asarray(samples)
        if values.ndim != 1 or values.size != self._config.sample_count:
            raise ValueError(
                f"expected {self._config.sample_count} samples, got {values.shape}"
            )
        if not np.iscomplexobj(values):
            raise TypeError("samples must have a complex dtype")

        started = perf_counter_ns()
        spectrum = scipy_fft.fft(values * self._window, workers=1)
        spectrum = np.fft.fftshift(spectrum)
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        after_fft = perf_counter_ns()
        detected_bins, snr_db = self._apply_cfar(power)
        finished = perf_counter_ns()
        frequencies = _readonly(
            self._frequency_offsets_hz[detected_bins], np.float64
        )
        return DetectorOutput(
            detected_frequency_offsets_hz=frequencies,
            strongest_detection_snr_db=snr_db,
            timing=DetectorTiming(
                fft_power_ns=after_fft - started,
                cfar_ns=finished - after_fft,
            ),
        )


def _form_candidates(
    frequency_offsets_hz: FloatArray,
    fft_bin_width_hz: float,
) -> _CandidateBatch:
    bin_width = _finite(fft_bin_width_hz, "fft_bin_width_hz")
    if bin_width <= 0:
        raise ValueError("fft_bin_width_hz must be positive")
    frequencies = np.asarray(frequency_offsets_hz, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequency offsets must be finite")
    if not frequencies.size:
        return _CandidateBatch(
            representative_bins=np.empty(0, dtype=np.int64),
            raw_to_candidate=np.empty(0, dtype=np.int64),
        )

    raw_bins = np.rint(frequencies / bin_width).astype(np.int64)
    unique_bins, raw_to_unique = np.unique(raw_bins, return_inverse=True)
    groups = np.empty(unique_bins.size, dtype=np.int64)
    groups[0] = 0
    if unique_bins.size > 1:
        groups[1:] = np.cumsum(np.diff(unique_bins) > 1)
    starts = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
    stops = np.r_[starts[1:], unique_bins.size]
    representatives = np.asarray(
        [unique_bins[start + (stop - start) // 2] for start, stop in zip(starts, stops)],
        dtype=np.int64,
    )
    return _CandidateBatch(
        representative_bins=representatives,
        raw_to_candidate=groups[raw_to_unique],
    )

# endregion


# region 5. Track models

class TentativeTrack:
    """Unconfirmed observation history with regression statistics."""

    @classmethod
    def start(
        cls,
        track_id: int,
        epoch_index: int,
        elapsed_time_s: float,
        observed_bin: int,
    ) -> TentativeTrack:
        if track_id <= 0:
            raise ValueError("track_id must be positive")
        if epoch_index < 0:
            raise ValueError("epoch_index must be nonnegative")
        time_s = _finite(elapsed_time_s, "elapsed_time_s")
        self = cls.__new__(cls)
        self._track_id = int(track_id)
        self._first_epoch_index = int(epoch_index)
        self._last_epoch_index = int(epoch_index)
        self._maximum_gap_epochs = 0
        self._epoch_indices = [int(epoch_index)]
        self._elapsed_times_s = [time_s]
        self._observed_bins = [int(observed_bin)]
        self._sum_x = 0.0
        self._sum_y = float(observed_bin)
        self._sum_xx = 0.0
        self._sum_xy = 0.0
        self._fitted_rate_bins_per_epoch = 0.0
        return self

    @property
    def metadata(self) -> TrackMetadata:
        return TrackMetadata(
            track_id=self._track_id,
            first_epoch_index=self._first_epoch_index,
            last_epoch_index=self._last_epoch_index,
            maximum_gap_epochs=self._maximum_gap_epochs,
        )

    @property
    def observation_count(self) -> int:
        return len(self._epoch_indices)

    @property
    def density(self) -> float:
        return self.observation_count / (
            self._last_epoch_index - self._first_epoch_index + 1
        )

    @property
    def fitted_rate_bins_per_epoch(self) -> float:
        return self._fitted_rate_bins_per_epoch

    def predict_bin(self, epoch_index: int) -> float:
        if epoch_index < self._last_epoch_index:
            raise ValueError("epoch_index precedes the latest observation")
        gap = epoch_index - self._last_epoch_index
        rate = min(0.0, self._fitted_rate_bins_per_epoch)
        return self._observed_bins[-1] + gap * rate

    def validation_gate(
        self,
        predicted_bin: float,
        tentative_config: TentativeTrackConfig,
        manager_config: TrackManagerConfig,
    ) -> tuple[float, float]:
        radius = tentative_config.association_radius_bins
        return (
            predicted_bin - radius,
            min(
                predicted_bin + radius,
                self._observed_bins[-1]
                + manager_config.maximum_upward_jitter_bins,
            ),
        )

    def append_observation(
        self,
        epoch_index: int,
        elapsed_time_s: float,
        observed_bin: int,
    ) -> None:
        if epoch_index <= self._last_epoch_index:
            raise ValueError("observations must have increasing epoch indices")
        time_s = _finite(elapsed_time_s, "elapsed_time_s")
        if time_s < self._elapsed_times_s[-1]:
            raise ValueError("observation times must be nondecreasing")
        gap = epoch_index - self._last_epoch_index
        self._maximum_gap_epochs = max(self._maximum_gap_epochs, gap)
        self._last_epoch_index = int(epoch_index)
        self._epoch_indices.append(int(epoch_index))
        self._elapsed_times_s.append(time_s)
        self._observed_bins.append(int(observed_bin))
        x = float(epoch_index - self._first_epoch_index)
        self._sum_x += x
        self._sum_y += observed_bin
        self._sum_xx += x * x
        self._sum_xy += x * observed_bin
        count = len(self._epoch_indices)
        denominator = count * self._sum_xx - self._sum_x * self._sum_x
        self._fitted_rate_bins_per_epoch = (
            (count * self._sum_xy - self._sum_x * self._sum_y) / denominator
            if denominator > 0
            else 0.0
        )

    def is_ready(
        self,
        config: TentativeTrackConfig,
        fft_bin_width_hz: float,
        epoch_duration_s: float,
    ) -> bool:
        bin_width = _finite(fft_bin_width_hz, "fft_bin_width_hz")
        duration = _finite(epoch_duration_s, "epoch_duration_s")
        if bin_width <= 0 or duration <= 0:
            raise ValueError("bin width and epoch duration must be positive")
        if self.observation_count < config.minimum_hits:
            return False
        fitted_rate_hz_per_s = (
            self.fitted_rate_bins_per_epoch * bin_width / duration
        )
        if (
            config.minimum_decrease_rate_hz_per_s is not None
            and -fitted_rate_hz_per_s < config.minimum_decrease_rate_hz_per_s
        ):
            return False
        return True

    def should_remove(
        self,
        epoch_index: int,
        config: TentativeTrackConfig,
    ) -> bool:
        if epoch_index < self._last_epoch_index:
            raise ValueError("epoch_index precedes the latest observation")
        return (
            epoch_index - self._first_epoch_index
            >= config.confirmation_window_epochs
        )


class ConfirmedTrack:
    """Confirmed track with a two-state Kalman estimate."""

    @classmethod
    def from_tentative(
        cls,
        track: TentativeTrack,
        confirmation_epoch_index: int,
        confirmation_elapsed_time_s: float,
        config: KalmanConfig,
    ) -> ConfirmedTrack:
        confirmation_time = _finite(
            confirmation_elapsed_time_s, "confirmation_elapsed_time_s"
        )
        if confirmation_epoch_index < track._last_epoch_index:
            raise ValueError("confirmation epoch precedes the latest observation")
        self = cls.__new__(cls)
        first_bin = track._observed_bins[0]
        self._track_id = track._track_id
        self._first_epoch_index = track._first_epoch_index
        self._last_epoch_index = track._epoch_indices[0]
        self._maximum_gap_epochs = 0
        self._confirmation_elapsed_time_s = confirmation_time
        self._first_elapsed_time_s = track._elapsed_times_s[0]
        self._pruned_observation_count = 0
        self._active = True
        self._epoch_indices = [track._epoch_indices[0]]
        self._elapsed_times_s = [track._elapsed_times_s[0]]
        self._observed_bins = [first_bin]
        self._estimated_bins = [float(first_bin)]
        self._estimated_rates = [0.0]
        self._state = KalmanState(
            estimated_frequency_bin=float(first_bin),
            estimated_rate_bins_per_epoch=0.0,
            covariance_ff=config.measurement_std_bins**2,
            covariance_fr=0.0,
            covariance_rr=config.initial_rate_std_bins_per_epoch**2,
        )
        self._acceleration_variance = config.acceleration_std_bins_per_epoch2**2
        for epoch, elapsed, observed_bin in zip(
            track._epoch_indices[1:],
            track._elapsed_times_s[1:],
            track._observed_bins[1:],
        ):
            prediction = self.predict(epoch)
            self.update(epoch, elapsed, observed_bin, prediction, config)
        return self

    @property
    def metadata(self) -> TrackMetadata:
        return TrackMetadata(
            track_id=self._track_id,
            first_epoch_index=self._first_epoch_index,
            last_epoch_index=self._last_epoch_index,
            maximum_gap_epochs=self._maximum_gap_epochs,
        )

    @property
    def state(self) -> KalmanState:
        return self._state

    @property
    def observation_count(self) -> int:
        return len(self._epoch_indices) + self._pruned_observation_count

    def predict(self, epoch_index: int) -> KalmanState:
        gap = epoch_index - self._last_epoch_index
        if gap < 0:
            raise ValueError("epoch_index precedes the latest observation")
        state = self._state
        q = self._acceleration_variance
        return KalmanState(
            estimated_frequency_bin=(
                state.estimated_frequency_bin
                + gap * state.estimated_rate_bins_per_epoch
            ),
            estimated_rate_bins_per_epoch=state.estimated_rate_bins_per_epoch,
            covariance_ff=(
                state.covariance_ff
                + 2 * gap * state.covariance_fr
                + gap**2 * state.covariance_rr
                + q * gap**4 / 4
            ),
            covariance_fr=(
                state.covariance_fr
                + gap * state.covariance_rr
                + q * gap**3 / 2
            ),
            covariance_rr=state.covariance_rr + q * gap**2,
        )

    def validation_gate(
        self,
        prediction: KalmanState,
        kalman_config: KalmanConfig,
        manager_config: TrackManagerConfig,
    ) -> tuple[float, float]:
        innovation_variance = (
            prediction.covariance_ff + kalman_config.measurement_std_bins**2
        )
        gate = kalman_config.gate_sigma * np.sqrt(innovation_variance)
        return (
            prediction.estimated_frequency_bin - gate,
            min(
                prediction.estimated_frequency_bin + gate,
                self._observed_bins[-1]
                + manager_config.maximum_upward_jitter_bins,
            ),
        )

    def update(
        self,
        epoch_index: int,
        elapsed_time_s: float,
        observed_bin: int,
        prediction: KalmanState,
        config: KalmanConfig,
    ) -> None:
        if not self._active:
            raise RuntimeError("cannot update a terminated track")
        if epoch_index <= self._last_epoch_index:
            raise ValueError("observations must have increasing epoch indices")
        time_s = _finite(elapsed_time_s, "elapsed_time_s")
        if time_s < self._elapsed_times_s[-1]:
            raise ValueError("observation times must be nondecreasing")

        measurement_variance = config.measurement_std_bins**2
        innovation_variance = prediction.covariance_ff + measurement_variance
        innovation = observed_bin - prediction.estimated_frequency_bin
        frequency_gain = prediction.covariance_ff / innovation_variance
        rate_gain = prediction.covariance_fr / innovation_variance
        self._state = KalmanState(
            estimated_frequency_bin=(
                prediction.estimated_frequency_bin + frequency_gain * innovation
            ),
            estimated_rate_bins_per_epoch=(
                prediction.estimated_rate_bins_per_epoch + rate_gain * innovation
            ),
            covariance_ff=max(
                0.0, (1.0 - frequency_gain) * prediction.covariance_ff
            ),
            covariance_fr=(1.0 - frequency_gain) * prediction.covariance_fr,
            covariance_rr=max(
                0.0,
                prediction.covariance_rr
                - rate_gain * prediction.covariance_fr,
            ),
        )
        gap = epoch_index - self._last_epoch_index
        self._maximum_gap_epochs = max(self._maximum_gap_epochs, gap)
        self._last_epoch_index = int(epoch_index)
        self._epoch_indices.append(int(epoch_index))
        self._elapsed_times_s.append(time_s)
        self._observed_bins.append(int(observed_bin))
        self._estimated_bins.append(self._state.estimated_frequency_bin)
        self._estimated_rates.append(self._state.estimated_rate_bins_per_epoch)

    def should_terminate(
        self,
        epoch_index: int,
        config: TrackManagerConfig,
    ) -> bool:
        if epoch_index < self._last_epoch_index:
            raise ValueError("epoch_index precedes the latest observation")
        return (
            epoch_index - self._last_epoch_index
            > config.confirmed_maximum_missed_epochs
        )

    def terminate(self) -> None:
        self._active = False

# endregion


# region 6. Session readers

_SIGMF_DTYPES: dict[str, np.dtype] = {
    "cf32_le": np.dtype("<c8"),
    "cf32_be": np.dtype(">c8"),
    "cf64_le": np.dtype("<c16"),
    "cf64_be": np.dtype(">c16"),
}


def _parse_utc(value: str | float | int | datetime | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return _finite(float(value), "start_time_utc")
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def _read_sigmf_metadata(data_path: Path) -> dict[str, Any]:
    metadata_path = data_path.with_suffix(".sigmf-meta")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("SigMF metadata root must be an object")
    return metadata


class SigMFReader:
    """Memory-mapped SigMF I/Q reader."""

    def __init__(
        self,
        data_path: Path,
        fallback_config: DetectorConfig = DetectorConfig(),
    ) -> None:
        self._data_path = Path(data_path)
        if not self._data_path.exists():
            raise FileNotFoundError(self._data_path)
        metadata = _read_sigmf_metadata(self._data_path)
        global_metadata = metadata.get("global", {})
        if not isinstance(global_metadata, dict):
            raise ValueError("SigMF global metadata must be an object")
        sample_rate_hz = float(
            global_metadata.get("core:sample_rate", fallback_config.sample_rate_hz)
        )
        datatype_name = str(global_metadata.get("core:datatype", "cf32_le"))
        try:
            self._sample_dtype = _SIGMF_DTYPES[datatype_name]
        except KeyError as error:
            raise ValueError(
                f"unsupported SigMF datatype {datatype_name!r}; expected cf32/cf64"
            ) from error
        self._detector_config = replace(
            fallback_config,
            sample_rate_hz=sample_rate_hz,
        )
        self._start_time_utc_s = self._resolve_start_time(metadata)

    def _resolve_start_time(self, metadata: dict[str, Any]) -> float | None:
        captures = metadata.get("captures", [])
        if not isinstance(captures, list):
            raise ValueError("SigMF captures metadata must be a list")
        dated = [
            capture
            for capture in captures
            if isinstance(capture, dict)
            and capture.get("core:datetime") is not None
        ]
        if not dated:
            return None
        capture = min(
            dated,
            key=lambda item: int(item.get("core:sample_start", 0)),
        )
        timestamp = _parse_utc(capture["core:datetime"])
        if timestamp is None:
            return None
        sample_start = int(capture.get("core:sample_start", 0))
        return timestamp - sample_start / self._detector_config.sample_rate_hz

    @property
    def detector_config(self) -> DetectorConfig:
        return self._detector_config

    @property
    def start_time_utc_s(self) -> float | None:
        return self._start_time_utc_s

    def iter_epochs(
        self,
        maximum_epochs: int | None = None,
    ) -> Iterator[IqEpoch]:
        _validate_maximum_epochs(maximum_epochs)
        samples = np.memmap(self._data_path, dtype=self._sample_dtype, mode="r")
        count = samples.size // self._detector_config.sample_count
        if maximum_epochs is not None:
            count = min(count, maximum_epochs)
        for epoch_index in range(count):
            start = epoch_index * self._detector_config.sample_count
            stop = start + self._detector_config.sample_count
            yield (
                epoch_index * self._detector_config.actual_epoch_duration_s,
                samples[start:stop],
            )


def _group_point_rows(
    times_s: np.ndarray,
    frequency_offsets_hz: np.ndarray,
) -> Iterator[ObservationEpoch]:
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    frequencies = np.asarray(frequency_offsets_hz, dtype=np.float64).reshape(-1)
    if times.size != frequencies.size:
        raise ValueError("time and frequency columns have different lengths")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("time and frequency values must be finite")
    if not times.size:
        return
    order = np.argsort(times, kind="stable")
    times = times[order]
    frequencies = frequencies[order]
    starts = np.r_[0, np.flatnonzero(np.diff(times) != 0) + 1]
    stops = np.r_[starts[1:], times.size]
    for start, stop in zip(starts, stops):
        yield (
            float(times[start]),
            np.asarray(frequencies[start:stop], dtype=np.float64),
            float("nan"),
        )


class PointSessionReader:
    """Normalized NPZ, NPY, CSV, JSONL, or NDJSON observation reader."""

    _SUPPORTED_SUFFIXES = {".npz", ".npy", ".csv", ".jsonl", ".ndjson"}

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        if self._path.suffix.lower() not in self._SUPPORTED_SUFFIXES:
            raise ValueError(
                "unsupported point-session format; use NPZ, NPY, CSV, JSONL, or NDJSON"
            )

    @staticmethod
    def _validated_record(
        elapsed_time_s: float,
        frequencies: np.ndarray,
        strongest_snr_db: float,
    ) -> ObservationEpoch:
        time_s = _finite(elapsed_time_s, "elapsed_time_s")
        values = np.asarray(frequencies, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("frequency offsets must be finite")
        snr_db = float(strongest_snr_db)
        if not (np.isnan(snr_db) or np.isfinite(snr_db)):
            raise ValueError("strongest SNR must be finite or NaN")
        return time_s, values, snr_db

    def _iter_npz(self) -> Iterator[ObservationEpoch]:
        archive = np.load(self._path, allow_pickle=False)
        try:
            times = np.asarray(archive["time_s"], dtype=np.float64).reshape(-1)
            frequency_key = (
                "frequency_hz"
                if "frequency_hz" in archive
                else "frequencies_hz"
            )
            frequencies = np.asarray(archive[frequency_key], dtype=np.float64).reshape(-1)
            offsets = np.asarray(archive["offsets"], dtype=np.int64).reshape(-1)
            snr = (
                np.asarray(archive["snr_db"], dtype=np.float64).reshape(-1)
                if "snr_db" in archive
                else np.full(times.size, np.nan)
            )
            if offsets.size != times.size + 1 or not offsets.size or offsets[0] != 0:
                raise ValueError("NPZ offsets must have len(time_s) + 1 and start at zero")
            if np.any(np.diff(offsets) < 0):
                raise ValueError("NPZ offsets must be nondecreasing")
            if offsets[-1] != frequencies.size or snr.size != times.size:
                raise ValueError("NPZ arrays are inconsistent")
            for index, time_s in enumerate(times):
                yield self._validated_record(
                    float(time_s),
                    frequencies[offsets[index] : offsets[index + 1]],
                    float(snr[index]),
                )
        finally:
            archive.close()

    def _iter_json(self) -> Iterator[ObservationEpoch]:
        with self._path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    yield self._validated_record(
                        float(item["time_s"]),
                        np.asarray(item["frequencies_hz"], dtype=np.float64),
                        float(item.get("snr_db", np.nan)),
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid point record on line {line_number}"
                    ) from error

    def iter_epochs(
        self,
        maximum_epochs: int | None = None,
    ) -> Iterator[ObservationEpoch]:
        _validate_maximum_epochs(maximum_epochs)
        suffix = self._path.suffix.lower()
        if suffix == ".npz":
            records: Iterable[ObservationEpoch] = self._iter_npz()
        elif suffix == ".npy":
            rows = np.load(self._path, allow_pickle=False)
            if rows.ndim != 2 or rows.shape[1] != 2:
                raise ValueError("NPY sessions must have shape (N, 2)")
            records = _group_point_rows(rows[:, 0], rows[:, 1])
        elif suffix == ".csv":
            rows = np.genfromtxt(
                self._path,
                delimiter=",",
                names=True,
                dtype=None,
                encoding=None,
            )
            names = rows.dtype.names or ()
            if "time_s" not in names or "frequency_hz" not in names:
                raise ValueError("CSV requires time_s and frequency_hz columns")
            records = _group_point_rows(
                np.atleast_1d(rows["time_s"]),
                np.atleast_1d(rows["frequency_hz"]),
            )
        else:
            records = self._iter_json()

        for index, record in enumerate(records):
            if maximum_epochs is not None and index >= maximum_epochs:
                break
            yield record

# endregion


# region 7. Reporting and runtime

def latency_summary(
    timings: Iterable[ProcessingTiming],
) -> dict[str, dict[str, float]]:
    values = list(timings)
    if not values:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for field_name in ("fft_power_ns", "cfar_ns", "tracking_ns", "total_ns"):
        samples_us = np.fromiter(
            (getattr(item, field_name) / 1_000.0 for item in values),
            dtype=np.float64,
            count=len(values),
        )
        summary[field_name.removesuffix("_ns")] = {
            "count": float(samples_us.size),
            "mean_us": float(np.mean(samples_us)),
            "p50_us": float(np.percentile(samples_us, 50)),
            "p95_us": float(np.percentile(samples_us, 95)),
            "p99_us": float(np.percentile(samples_us, 99)),
            "max_us": float(np.max(samples_us)),
        }
    return summary


def _configured_engine_for_path(
    path: Path,
    config: TrackingConfig,
) -> tuple[TrackingEngine, SigMFReader | PointSessionReader]:
    if path.name.lower().endswith(".sigmf-data"):
        reader = SigMFReader(path, config.detector)
        resolved = replace(config, detector=reader.detector_config)
        return TrackingEngine(resolved), reader
    reader = PointSessionReader(path)
    return TrackingEngine(config), reader


def run_headless_session(
    path: Path,
    config: TrackingConfig = TrackingConfig(),
    maximum_epochs: int | None = None,
) -> tuple[list[TrackSnapshot], dict[str, dict[str, float]]]:
    _validate_maximum_epochs(maximum_epochs)
    engine, reader = _configured_engine_for_path(Path(path), config)
    timings: list[ProcessingTiming] = []
    if isinstance(reader, SigMFReader):
        for elapsed_time_s, samples in reader.iter_epochs(maximum_epochs):
            timings.append(
                engine.process_iq_block(samples, elapsed_time_s).timing
            )
    else:
        for elapsed_time_s, frequencies, snr_db in reader.iter_epochs(maximum_epochs):
            timings.append(
                engine.process_frequency_observations(
                    frequencies,
                    elapsed_time_s,
                    snr_db,
                ).timing
            )
    engine.finish()
    return engine.track_snapshots(), latency_summary(timings)


def run_benchmark(
    config: TrackingConfig,
    epoch_count: int,
    data_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    warmup_count = min(32, max(4, epoch_count // 10))
    required = warmup_count + epoch_count

    if data_path is not None:
        path = Path(data_path)
        if not path.name.lower().endswith(".sigmf-data"):
            raise ValueError("benchmark data_path must be a SigMF data file")
        reader = SigMFReader(path, config.detector)
        resolved = replace(config, detector=reader.detector_config)
        engine = TrackingEngine(resolved)
        blocks: Iterator[ComplexArray] = (
            samples for _, samples in reader.iter_epochs(required)
        )
    else:
        engine = TrackingEngine(config)
        rng = np.random.default_rng(61956)
        noise = (
            rng.standard_normal(
                (required, config.detector.sample_count),
                dtype=np.float32,
            )
            + 1j
            * rng.standard_normal(
                (required, config.detector.sample_count),
                dtype=np.float32,
            )
        ).astype(np.complex64, copy=False)
        blocks = iter(noise)

    try:
        for _ in range(warmup_count):
            engine.process_iq_block(next(blocks))
        timings = [
            engine.process_iq_block(next(blocks)).timing
            for _ in range(epoch_count)
        ]
    except StopIteration as error:
        raise ValueError(
            f"benchmark requires at least {required} complete epochs"
        ) from error
    return latency_summary(timings)

# endregion
