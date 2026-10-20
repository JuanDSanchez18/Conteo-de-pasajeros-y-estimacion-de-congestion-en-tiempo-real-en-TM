# -*- coding: utf-8 -*-
# aws_iot.py — AWS IoT Core MQTT connection, publisher and retry logic
# Transmilenio Edge AI Engine — v1.2.0-prod
# Protocol : MQTT over mutual TLS (X.509) — Port 8883
# SDK      : awsiotsdk (AWS IoT Device SDK v2 for Python)
# Console output owned by main.py — this module only prints on failure or boot.

import json
import os
import time
from datetime import datetime, timezone

import config

# ---------------------------------------------------------------------------
# Module state — private
# ---------------------------------------------------------------------------
_iot_connection                = None   # Active MQTT connection handle
_last_published_capacity_state = None   # Last state sent (production mode)
_last_published_roi_count      = None   # Last count sent (testing mode)

MQTT_ERROR_LOG = os.path.join(config.RESULTS_DIR, "mqtt_errors.log")


# ---------------------------------------------------------------------------
# Capacity resolver — single source of truth for capacity labels
# ---------------------------------------------------------------------------
def resolve_capacity_state(roi_count: int) -> str:
    # Maps a people count to a label using config.CAPACITY_THRESHOLDS.
    # Returns "Unknown" if count falls outside all defined ranges.
    for label, (low, high) in config.CAPACITY_THRESHOLDS.items():
        if low <= roi_count <= high:
            return label
    return "Unknown"


# ---------------------------------------------------------------------------
# Init — MQTT connection bootstrap (call once at boot from main.py)
# ---------------------------------------------------------------------------
def init_aws_iot() -> bool:
    global _iot_connection

    if not config.AWS_IOT_ENABLED:
        print("[AWS IoT] Disabled in config — running offline.")
        return False

    try:
        from awsiot import mqtt_connection_builder

        _iot_connection = mqtt_connection_builder.mtls_from_path(
            endpoint         = config.AWS_IOT_ENDPOINT,
            port             = config.AWS_IOT_PORT,
            cert_filepath    = config.AWS_IOT_CERT_PATH,
            pri_key_filepath = config.AWS_IOT_KEY_PATH,
            ca_filepath      = config.AWS_IOT_CA_PATH,
            client_id        = config.AWS_IOT_CLIENT_ID,
            clean_session    = False,
            keep_alive_secs  = config.AWS_IOT_KEEPALIVE_SECONDS,
        )
        _iot_connection.connect().result(timeout=10)

        mode = (
            "ALL count changes (testing)"
            if config.AWS_IOT_PUBLISH_ALL_COUNT_CHANGES
            else "STATE changes only (production)"
        )
        print(f"[AWS IoT] Connected | {config.AWS_IOT_ENDPOINT} | {config.AWS_IOT_CLIENT_ID} | {mode}")
        return True

    except ImportError:
        print("[AWS IoT] WARNING — 'awsiotsdk' not installed. Run: pip install awsiotsdk")
        _iot_connection = None
        return False

    except Exception as e:
        print(f"[AWS IoT] WARNING — Connection failed, offline mode: {e}")
        _iot_connection = None
        return False


# ---------------------------------------------------------------------------
# Safe publish — decision + payload build + retry logic
# ---------------------------------------------------------------------------
def safe_aws_publish(
    roi_count:      int,
    outside_count:  int,
    capacity_state: str,
    soc_temp:       float,
) -> bool:
    global _last_published_capacity_state, _last_published_roi_count

    # Guard: offline or disabled
    if not config.AWS_IOT_ENABLED or _iot_connection is None:
        return False

    # Publish decision:
    #   TESTING  (PUBLISH_ALL_COUNT_CHANGES=True)  — skip if same count
    #   PRODUCTION (PUBLISH_ALL_COUNT_CHANGES=False) — skip if same state
    if config.AWS_IOT_PUBLISH_ALL_COUNT_CHANGES:
        if roi_count == _last_published_roi_count:
            return False
    else:
        if capacity_state == _last_published_capacity_state:
            return False

    payload = {
        "station_id"    : config.AWS_IOT_STATION_ID,
        "timestamp"     : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "roi_count"     : roi_count,
        "outside_count" : outside_count,
        "capacity_state": capacity_state,
        "capacity_pct"  : round(roi_count / config.MAX_CAPACITY * 100, 1),
        "soc_temp_c"    : round(soc_temp, 1),
        "publish_mode"  : (
            "all_count_changes"
            if config.AWS_IOT_PUBLISH_ALL_COUNT_CHANGES
            else "state_change_only"
        ),
    }

    for attempt in range(1, config.AWS_IOT_MAX_RETRIES + 1):
        try:
            from awsiot import mqtt

            pub_future, _ = _iot_connection.publish(
                topic   = config.AWS_IOT_TOPIC,
                payload = json.dumps(payload),
                qos     = mqtt.QoS.AT_LEAST_ONCE,
            )
            pub_future.result(timeout=5)  # Block until broker PUBACK

            # Delivery confirmed — update trackers
            _last_published_roi_count      = roi_count
            _last_published_capacity_state = capacity_state
            return True

        except Exception as e:
            _log_failed_mqtt_event(payload, e, attempt)
            if attempt < config.AWS_IOT_MAX_RETRIES:
                time.sleep(config.AWS_IOT_RETRY_BACKOFF_SECONDS)
            else:
                # Only print on total failure — main.py owns all normal output
                print(
                    f"\n[AWS IoT] FAILED — all {config.AWS_IOT_MAX_RETRIES} attempts exhausted "
                    f"| ROI: {roi_count} | State: {capacity_state} | Log: {MQTT_ERROR_LOG}"
                )
                return False

    return False


# ---------------------------------------------------------------------------
# Fallback log — persists failed publish events to Results/mqtt_errors.log
# ---------------------------------------------------------------------------
def _log_failed_mqtt_event(payload: dict, error: Exception, attempt: int) -> None:
    # Private — called only from safe_aws_publish() after each failed attempt.
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    msg = str(error) if str(error).strip() else repr(error)
    try:
        with open(MQTT_ERROR_LOG, "a") as f:
            f.write(
                f"{ts} | Attempt:{attempt}/{config.AWS_IOT_MAX_RETRIES} "
                f"| Error:{msg} | Payload:{json.dumps(payload)}\n"
            )
            f.flush()
    except OSError as e:
        # Log failure must never propagate to the inference loop
        print(f"[AWS IoT] WARNING — could not write error log: {e}")


# ---------------------------------------------------------------------------
# Disconnect — graceful MQTT teardown (call in finally block from main.py)
# ---------------------------------------------------------------------------
def disconnect_aws_iot() -> None:
    global _iot_connection
    if _iot_connection is None:
        return
    try:
        _iot_connection.disconnect().result(timeout=5)
        print("[AWS IoT] Disconnected cleanly from broker.")
    except Exception as e:
        print(f"[AWS IoT] WARNING — error during disconnect: {e}")
    finally:
        _iot_connection = None


# ---------------------------------------------------------------------------
# Status check — connection handle liveness
# ---------------------------------------------------------------------------
def is_connected() -> bool:
    return _iot_connection is not None