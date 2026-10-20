# -*- coding: utf-8 -*-
# config.py — Central configuration and boot validation
# Transmilenio Edge AI Engine — v1.0.0-prod

import numpy as np
import os

# ---------------------------------------------------------------------------
# Camera — Innomaker-U20CAM-720P on /dev/video0 via V4L2
# ---------------------------------------------------------------------------
CAMERA_SOURCE      = 0        # /dev/video0
CAMERA_WIDTH       = 1280
CAMERA_HEIGHT      = 720
CAMERA_FORCE_MJPEG = True     # Forces 30 FPS — YUYV is limited to 10 FPS
TARGET_LIVE_FPS    = 3.0      # One inference cycle every ~333ms

# Confirmed count requires this many consecutive stable cycles before firing.
# 4 cycles × (1 / 3.0 FPS) ≈ 1.33s confirmation window
STABILIZATION_THRESHOLD = 4

# ---------------------------------------------------------------------------
# Watchdog — camera stream reconnection
# ---------------------------------------------------------------------------
WATCHDOG_MAX_FAILED_FRAMES      = 30
WATCHDOG_RECONNECT_COOLDOWN_SEC = 2.0

# ---------------------------------------------------------------------------
# OpenVINO — inference on ARM Cortex-A76 (Raspberry Pi 5)
# Runtime: openvino 2024.1.0 — ARM64 CPU plugin
# ---------------------------------------------------------------------------
MODEL_XML      = "/home/siscotm1/DetectorIA/Modelo/Roboflow/weights_openvino_model_f32/weights.xml"
CONF_THRESHOLD = 0.45   # Detection confidence threshold
NMS_THRESHOLD  = 0.45   # Non-Maximum Suppression overlap threshold
OV_NUM_THREADS = 3      # Leaves 1 core free for USB camera capture
OV_PRECISION   = "f32"  # Full precision — confirmed: half=False

# ---------------------------------------------------------------------------
# Capacity thresholds — single source of truth for main.py and aws_iot.py
# Format: "label": (min_count, max_count) inclusive
# ---------------------------------------------------------------------------
MAX_CAPACITY = 20

CAPACITY_THRESHOLDS = {
    "Empty"     : (0,                           0),
    "Low"       : (1,                           int(MAX_CAPACITY * 0.25)),
    "Medium"    : (int(MAX_CAPACITY * 0.25) + 1, int(MAX_CAPACITY * 0.55)),
    "High"      : (int(MAX_CAPACITY * 0.55) + 1, int(MAX_CAPACITY * 0.90)),
    "Super High": (int(MAX_CAPACITY * 0.90) + 1, 9999),
}

# ---------------------------------------------------------------------------
# Region of Interest (ROI)
# ⚠️ Recalibrate with perspective calibration script before deployment.
# Current polygon covers the full 1280×720 frame (testing default).
# ---------------------------------------------------------------------------
ROI_VERTICES = np.array(
    #[[0, 0], [1279, 0], [1279, 719], [0, 719]], np.int32
    [[50, 160], [1230, 160], [1280, 720], [0, 720]], dtype=np.int32
)

# ---------------------------------------------------------------------------
# Thermal safety — BCM2712 SoC thresholds
# ---------------------------------------------------------------------------
THERMAL_WARNING_THRESHOLD_C  = 70.0   # Yellow alert — elevated temperature
THERMAL_THROTTLE_THRESHOLD_C = 80.0   # Red alert — CPU throttle risk

# ---------------------------------------------------------------------------
# Logging and storage
# ---------------------------------------------------------------------------
RESULTS_DIR                  = "csv_results"           # All output artifacts go here
CHUNK_INTERVAL_SECONDS       = 300                     # Summary row every 5 minutes
HARDWARE_SAMPLE_INTERVAL_SEC = 30.0                    # Telemetry sampled every 30s
MAIN_LOOP_IDLE_SLEEP_SEC     = 0.002                   # Prevents busy-wait CPU burn
IMAGE_SAVE_COOLDOWN_SECONDS  = 5.0                     # Min seconds between saved frames
SAVE_AUDIT_IMAGES            = True                    # Save annotated HUD frames
SAVE_CLEAN_DATASET           = True                   # Save raw frames for dataset
BASE_DEBUG_DIR               = "Debug_Frames_Live_Run"
DATASET_CLEAN_DIR            = "Dataset_Clean"

# ---------------------------------------------------------------------------
# AWS IoT Core — MQTT over mutual TLS (X.509), port 8883
# ---------------------------------------------------------------------------
AWS_IOT_ENABLED               = True
AWS_IOT_ENDPOINT              = "a18i7o4n7yywv1-ats.iot.us-east-2.amazonaws.com"
AWS_IOT_PORT                  = 8883
AWS_IOT_CLIENT_ID             = "TM_Camara1"
AWS_IOT_TOPIC                 = "transmilenio/stations/estacion1/occupancy"
AWS_IOT_CERT_PATH             = "/home/siscotm1/DetectorIA/certs/TM_Camara1.cert.pem"
AWS_IOT_KEY_PATH              = "/home/siscotm1/DetectorIA/certs/TM_Camara1.private.key"
AWS_IOT_CA_PATH               = "/home/siscotm1/DetectorIA/certs/root-CA.crt"
AWS_IOT_STATION_ID            = "TM_ESTACION1_CAM01"
AWS_IOT_KEEPALIVE_SECONDS     = 30

# True  → TESTING    — publishes on every confirmed count change
# False → PRODUCTION — publishes only when capacity state label changes
AWS_IOT_PUBLISH_ALL_COUNT_CHANGES = True

AWS_IOT_MAX_RETRIES           = 3
AWS_IOT_RETRY_BACKOFF_SECONDS = 1.5


# ---------------------------------------------------------------------------
# Boot validation — called from main.py before any subsystem initializes
# ---------------------------------------------------------------------------
def validate_config() -> None:
    errors = []

    if not os.path.exists(MODEL_XML):
        errors.append(f"[CONFIG] Model XML not found: {MODEL_XML}")
    if not os.path.exists(MODEL_XML.replace(".xml", ".bin")):
        errors.append(f"[CONFIG] Model BIN not found: {MODEL_XML.replace('.xml', '.bin')}")
    if not (0 < CONF_THRESHOLD < 1):
        errors.append(f"[CONFIG] CONF_THRESHOLD out of range (0–1): {CONF_THRESHOLD}")
    if TARGET_LIVE_FPS <= 0:
        errors.append(f"[CONFIG] TARGET_LIVE_FPS must be positive: {TARGET_LIVE_FPS}")
    if len(ROI_VERTICES) < 3:
        errors.append("[CONFIG] ROI_VERTICES requires at least 3 points")
    if not isinstance(CAPACITY_THRESHOLDS, dict) or len(CAPACITY_THRESHOLDS) < 2:
        errors.append("[CONFIG] CAPACITY_THRESHOLDS must have at least 2 entries")
    if AWS_IOT_ENABLED:
        for path in [AWS_IOT_CERT_PATH, AWS_IOT_KEY_PATH, AWS_IOT_CA_PATH]:
            if not os.path.exists(path):
                errors.append(f"[CONFIG] Certificate not found: {path}")
        if "REPLACE" in AWS_IOT_ENDPOINT or "REEMPLAZAR" in AWS_IOT_ENDPOINT:
            errors.append("[CONFIG] AWS_IOT_ENDPOINT has not been configured")
    if not isinstance(AWS_IOT_PUBLISH_ALL_COUNT_CHANGES, bool):
        errors.append("[CONFIG] AWS_IOT_PUBLISH_ALL_COUNT_CHANGES must be True or False")

    if errors:
        for e in errors:
            print(e)
        raise SystemExit("[BOOT ABORTED] Fix configuration errors before starting.")

    mode = (
        "ALL COUNT CHANGES (testing)"
        if AWS_IOT_PUBLISH_ALL_COUNT_CHANGES
        else "STATE CHANGES ONLY (production)"
    )
    print("[CONFIG] ✅ Validation passed.")
    print(f"[CONFIG] 📡 Publish mode: {mode}")
    print(
        f"[CONFIG] ⏱  Confirmation window: "
        f"{STABILIZATION_THRESHOLD / TARGET_LIVE_FPS:.1f}s "
        f"({STABILIZATION_THRESHOLD} cycles × {1 / TARGET_LIVE_FPS:.2f}s)"
    )
