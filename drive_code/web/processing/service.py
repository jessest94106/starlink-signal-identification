import json
import logging
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import zmq

from processing.ipc import CAPTURE_COMMAND, EVENTS, IQ, PROCESS_COMMAND, request, socket
from processing.prediction_store import PredictionStore
from processing.satellite_matcher import SatelliteMatcher
from processing.tracker_adapter import TrackerAdapter, tracking_config


def run():
    commands = socket(zmq.REP, PROCESS_COMMAND, bind=True)
    incoming = socket(zmq.PULL, IQ, hwm=16)
    outgoing = socket(zmq.PUB, EVENTS, bind=True, hwm=256)
    config = adapter = None
    matcher = SatelliteMatcher(PredictionStore("prediction_data"))
    last_health = time.monotonic()
    sequence = dropped = 0
    fault = None
    latencies = deque(maxlen=10000)
    pipeline_latencies = deque(maxlen=10000)
    dropped_iq = 0
    last_iq_sequence = None

    def publish(event):
        nonlocal dropped
        try:
            outgoing.send_json(event, flags=zmq.NOBLOCK)
        except zmq.Again:
            dropped += 1

    while True:
        if commands.poll(1):
            message = commands.recv_json()
            try:
                action = message["action"]
                if action == "HEALTH":
                    result = {"configured": config is not None, "fault": fault}
                elif action == "APPLY":
                    candidate = message["config"]
                    tracking_config(candidate["receiver"], candidate["processing"])
                    request(CAPTURE_COMMAND, {"action": "STOP"})
                    applied = request(
                        CAPTURE_COMMAND, {"action": "APPLY", "config": candidate}
                    )["config"]
                    config, adapter = (
                        applied,
                        TrackerAdapter(
                            tracking_config(applied["receiver"], applied["processing"])
                        ),
                    )
                    matcher = SatelliteMatcher(PredictionStore("prediction_data"))
                    fault = None
                    last_iq_sequence = None
                    result = {"config": applied}
                elif action == "START":
                    if adapter is None or fault:
                        raise ValueError(
                            "Processing unavailable; apply configuration/reset first"
                        )
                    result = request(CAPTURE_COMMAND, {"action": "START"})
                elif action == "STOP":
                    result = request(CAPTURE_COMMAND, {"action": "STOP"})
                else:
                    raise ValueError("Unknown command")
                commands.send_json({"ok": True, **result})
            except Exception as exc:
                commands.send_json({"ok": False, "error": str(exc)})
        if incoming.poll(1):
            try:
                header_raw, raw = incoming.recv_multipart()
                header = json.loads(header_raw)
                if (
                    config is None
                    or header["processing_session_id"]
                    != config["processing_session_id"]
                ):
                    continue
                if fault:
                    continue
                samples = np.frombuffer(raw, dtype=np.complex64)
                if len(samples) != adapter.engine.config.detector.sample_count:
                    raise ValueError(
                        "I/Q block size does not match active detector configuration"
                    )
                sequence += 1
                if last_iq_sequence is not None:
                    gap = max(0, header["sequence_number"] - last_iq_sequence - 1)
                    dropped_iq += gap
                    adapter.engine.skip_epochs(gap)
                last_iq_sequence = header["sequence_number"]
                processing_start_ns = time.time_ns()
                timestamp = header["timestamp_ns"] / 1e9
                metadata = {
                    k: config[k]
                    for k in (
                        "capture_session_id",
                        "processing_session_id",
                        "config_version",
                    )
                }
                metadata.update(
                    receiver_metadata=header.get("receiver_metadata"),
                    sequence=sequence,
                    iq_sequence=header["sequence_number"],
                    timestamp_s=timestamp,
                    block_complete_ns=header["block_complete_ns"],
                    processing_start_ns=processing_start_ns,
                    timestamp_utc=datetime.fromtimestamp(
                        timestamp, timezone.utc
                    ).isoformat(),
                )
                elapsed = (
                    adapter.engine.next_epoch_index
                    * adapter.engine.config.detector.actual_epoch_duration_s
                )
                event, lifecycle = adapter.process(samples, elapsed, metadata)
                event["processing_complete_ns"] = time.time_ns()
                event["pipeline_latency_ms"] = max(
                    0,
                    (event["processing_complete_ns"] - header["block_complete_ns"])
                    / 1e6,
                )
                pipeline_latencies.append(event["pipeline_latency_ms"])
                latencies.append(event["processing_latency_ms"])
                publish(event)
                for extra in lifecycle + matcher.update(event):
                    publish(extra)
                if adapter.engine.next_epoch_index % 100 == 0:
                    adapter.engine.prune_confirmed_history(elapsed - 60)
            except Exception as exc:
                fault = str(exc)
                logging.exception("Processing fault")
                try:
                    request(CAPTURE_COMMAND, {"action": "STOP"})
                except Exception:
                    logging.exception("Failed to stop capture after processing fault")
        if time.monotonic() - last_health > 2:
            last_health = time.monotonic()
            try:
                capture = request(
                    CAPTURE_COMMAND, {"action": "HEALTH"}, timeout_ms=1000
                )
            except Exception as exc:
                capture = {"running": False, "fault": str(exc)}
            stats = (
                dict(
                    zip(
                        ("p50", "p95", "p99", "maximum"),
                        np.percentile(latencies, [50, 95, 99, 100]).tolist(),
                    )
                )
                if latencies
                else {}
            )
            publish(
                {
                    "type": "system_status",
                    "timestamp_s": time.time(),
                    "capture": capture,
                    "processing": {
                        "healthy": fault is None and adapter is not None,
                        "fault": fault,
                        "latency_ms": stats,
                        "iq_to_processing_latency_ms": dict(
                            zip(
                                ("p50", "p95", "p99", "maximum"),
                                np.percentile(
                                    pipeline_latencies, [50, 95, 99, 100]
                                ).tolist(),
                            )
                        )
                        if pipeline_latencies
                        else {},
                        "iq_pending": bool(incoming.poll(0)),
                        "iq_queue_capacity_blocks": 16,
                        "dropped_iq_sequence_gaps": dropped_iq,
                        "dropped_processed_events": dropped,
                        "prediction_records": len(matcher.store.records),
                        "prediction_errors": matcher.store.errors,
                    },
                }
            )


if __name__ == "__main__":
    from processing.logging_setup import configure

    configure("processing")
    run()
