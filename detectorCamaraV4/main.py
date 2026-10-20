# -*- coding: utf-8 -*-
# main.py — Transmilenio Edge AI Engine v1.4.0-prod

import os
import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from openvino import Core

import config
import aws_iot
import hardware_monitor
import csv_handler

# ── UI state map: capacity label → display color (BGR) ───────────────────────
_STATE_UI = {
    #              B    G    R
    "Empty"     : {"label": "Empty",      "bgr": (139,  69,   0)},  # Blue
    "Low"       : {"label": "Low",        "bgr": (  0, 160,   0)},  # Green
    "Medium"    : {"label": "Medium",     "bgr": (  0, 140, 255)},  # Orange
    "High"      : {"label": "High",       "bgr": (  0,   0, 200)},  # Red
    "Super High": {"label": "Super High", "bgr": (160,   0, 160)},  # Violet
    "Unknown"   : {"label": "Unknown",    "bgr": ( 80,  80,  80)},  # Gray
}

# Fallback UI state shown before the first confirmed reading
_STATE_UI_DEFAULT = {"label": "Init", "bgr": (50, 50, 50)}

def _ui(state: str) -> dict:
    # Returns the UI dict for the given capacity state string
    return _STATE_UI.get(state, _STATE_UI_DEFAULT)


# ── Boot ──────────────────────────────────────────────────────────────────────
config.validate_config()                            # Abort if any config value is invalid
aws_iot.init_aws_iot()                              # Connect to AWS IoT Core via MQTT
os.makedirs(config.RESULTS_DIR, exist_ok=True)      # Ensure output root folder exists

# Build session-scoped CSV paths and write their headers
_SESSION_TS = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
CSV_RAW_PATH, CSV_SUMMARY_PATH = csv_handler.build_session_filenames(_SESSION_TS)
csv_handler.initialize_csv_files(CSV_RAW_PATH, CSV_SUMMARY_PATH)

# Create optional output directories based on config flags
if config.SAVE_AUDIT_IMAGES:
    os.makedirs(config.BASE_DEBUG_DIR, exist_ok=True)
if config.SAVE_CLEAN_DATASET:
    os.makedirs(config.DATASET_CLEAN_DIR, exist_ok=True)

# Pre-compute reshaped ROI vertices once — avoids per-frame reshape and type cast
_roi_vertices_f32 = config.ROI_VERTICES.reshape(-1, 1, 2).astype(np.float32)


# ── Video Capture ─────────────────────────────────────────────────────────────
class LiveVideoStreamThread:
    def __init__(self, source: int):
        self.source                = source
        self.capture               = self._open_capture(source)
        self.frame_queue           = queue.Queue(maxsize=1)
        self.is_stopped            = False
        self.frame_idx             = 0
        self.failed_frames_counter = 0
        self._log_camera_info()

    def _open_capture(self, source: int) -> cv2.VideoCapture:
        # Open V4L2 device; optionally force MJPEG for 30 FPS
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if config.CAMERA_FORCE_MJPEG:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _log_camera_info(self):
        # Print resolved codec, resolution, and driver-reported FPS
        fcc_int = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        fcc_str = "".join([chr((fcc_int >> 8 * i) & 0xFF) for i in range(4)])
        w   = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        print(f"[CAMERA] {fcc_str} | {w}x{h} | {fps:.0f} FPS (driver-reported)")

    def start(self):
        # Launch the capture loop in a background daemon thread
        threading.Thread(
            target=self._stream_worker, daemon=True, name="CaptureThread"
        ).start()
        return self

    def _stream_worker(self):
        while not self.is_stopped:
            success, frame = self.capture.read()
            if not success:
                self.failed_frames_counter += 1
                if self.failed_frames_counter % 10 == 0:
                    print(
                        f"\n[WATCHDOG] Frame drop streak: "
                        f"{self.failed_frames_counter}/"
                        f"{config.WATCHDOG_MAX_FAILED_FRAMES}"
                    )
                if self.failed_frames_counter >= config.WATCHDOG_MAX_FAILED_FRAMES:
                    print(f"\n[WATCHDOG] Reconnecting /dev/video{self.source}...")
                    self.capture.release()
                    time.sleep(config.WATCHDOG_RECONNECT_COOLDOWN_SEC)
                    self.capture = self._open_capture(self.source)
                    if self.capture.isOpened():
                        print("[WATCHDOG] Camera reconnected.")
                        self.failed_frames_counter = 0
                    else:
                        print("[WATCHDOG] Reconnection failed — retrying...")
                time.sleep(0.1)
                continue

            self.failed_frames_counter = 0
            self.frame_idx += 1

            # Drop stale frame to keep latency minimal, then enqueue the new one
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((self.frame_idx, int(time.time() * 1000), frame))

    def read_frame(self) -> tuple:
        # Return latest frame tuple, or (None, None, None) if queue is empty
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None, None, None

    def release_resources(self):
        # Signal worker thread to stop and release the camera device
        self.is_stopped = True
        self.capture.release()


# ── OpenVINO ──────────────────────────────────────────────────────────────────
print("[OPENVINO] Loading model...")
_ov_core = Core()
_ov_core.set_property("CPU", {
    "INFERENCE_PRECISION_HINT": config.OV_PRECISION,
    "PERFORMANCE_HINT"        : "LATENCY",
    "NUM_STREAMS"             : "1",
    "INFERENCE_NUM_THREADS"   : str(config.OV_NUM_THREADS),
})

_compiled_model    = _ov_core.compile_model(_ov_core.read_model(config.MODEL_XML), "CPU")
_infer_request     = _compiled_model.create_infer_request()
_input_tensor_node = _compiled_model.input(0)
_letterbox_canvas  = np.zeros((640, 640, 3), dtype=np.uint8)
print("[OPENVINO] Model ready.")


# ── NMS ───────────────────────────────────────────────────────────────────────
def apply_nms(boxes: np.ndarray, scores: np.ndarray) -> list:
    # Suppress overlapping detections; keep highest-scoring non-overlapping boxes
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1 + 1) * (y2 - y1 + 1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        inter_x1 = np.maximum(x1[i], x1[order[1:]])
        inter_y1 = np.maximum(y1[i], y1[order[1:]])
        inter_x2 = np.minimum(x2[i], x2[order[1:]])
        inter_y2 = np.minimum(y2[i], y2[order[1:]])
        inter_w  = np.maximum(0, inter_x2 - inter_x1 + 1)
        inter_h  = np.maximum(0, inter_y2 - inter_y1 + 1)
        iou      = (inter_w * inter_h) / (
            areas[i] + areas[order[1:]] - inter_w * inter_h
        )
        order    = order[np.where(iou <= config.NMS_THRESHOLD)[0] + 1]
    return keep


# ── Audit HUD ─────────────────────────────────────────────────────────────────
def _draw_audit_hud(
    frame,
    confirmed_count: int,   # Last confirmed count → CSV/AWS
    live_roi_count:  int,   # Current frame IN count
    live_out_count:  int,   # Current frame OUT count
    cap_state:       str,   # Confirmed state → panel color
    live_fps:        float,
    infer_ms:        float,
    soc_temp:        float,
    detections:      list,
    frame_w:         int,
    frame_h:         int,
    date_str:        str,
):
    out = frame.copy()

    # Panel color reflects the CONFIRMED stable state
    ui = _ui(cap_state)

    # ── ROI polygon ───────────────────────────────────────────────────────────
    cv2.polylines(
        out, [config.ROI_VERTICES], isClosed=True,
        color=(0, 255, 255), thickness=2
    )

    # ── Detection boxes ───────────────────────────────────────────────────────
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = (0, 190, 0) if det["inside"] else (0, 140, 255)
        tag   = f"{'IN' if det['inside'] else 'OUT'} {det['score']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        chip_y = max(y1 - 4, th + 6)
        cv2.rectangle(out, (x1, chip_y - th - 4), (x1 + tw + 6, chip_y), color, -1)
        cv2.putText(
            out, tag, (x1 + 3, chip_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.circle(out, (det["cx"], det["cy"]), 5, (0, 230, 230), -1)

    # ── Top panel ─────────────────────────────────────────────────────────────
    PANEL_H = 62
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, PANEL_H), ui["bgr"], -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.line(out, (0, PANEL_H), (frame_w, PANEL_H), (255, 255, 255), 1)

    div1_x = frame_w // 3
    div2_x = (frame_w * 2) // 3
    for dx in [div1_x, div2_x]:
        cv2.line(out, (dx, 8), (dx, PANEL_H - 8), (255, 255, 255), 1)

    # ── LEFT: Last / Live / Out ───────────────────────────────────────────────
    if confirmed_count < 0:
        cv2.putText(
            out, "Detecting...",
            (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
            (255, 255, 255), 2, cv2.LINE_AA
        )
    else:
        # Row 1 — last confirmed count (stable, published to CSV/AWS)
        cv2.putText(
            out, f"Last: {max(confirmed_count, 0):02d}",
            (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        # Row 2 — current frame live count (green boxes)
        cv2.putText(
            out, f"Live: {live_roi_count:02d}",
            (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            (180, 255, 180), 1, cv2.LINE_AA
        )
        # Row 3 — detections outside ROI
        cv2.putText(
            out, f"Out:  {live_out_count:02d}",
            (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (180, 180, 180), 1, cv2.LINE_AA
        )

    # ── CENTER: always show current live state — no "Confirming..." ──────────
    if confirmed_count < 0:
        center_label = "Detecting..."
    else:
        # Resolve state from live count — always one of the 5 defined states
        live_cap_state = aws_iot.resolve_capacity_state(live_roi_count)
        center_label   = _ui(live_cap_state)["label"]

    (cw, _), _ = cv2.getTextSize(center_label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    cx_label   = div1_x + (div2_x - div1_x - cw) // 2
    cv2.putText(
        out, center_label,
        (cx_label, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    # ── RIGHT: time ───────────────────────────────────────────────────────────
    time_str = datetime.now().strftime("%H:%M:%S")
    (tw, _), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(
        out, time_str,
        (frame_w - tw - 12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 1, cv2.LINE_AA
    )

    # ── Bottom strip ──────────────────────────────────────────────────────────
    STRIP_Y = frame_h - 26
    cv2.rectangle(out, (0, STRIP_Y), (frame_w, frame_h), (15, 15, 15), -1)
    cv2.line(out, (0, STRIP_Y), (frame_w, STRIP_Y), (120, 120, 120), 1)
    bottom_text = (
        f"FPS: {live_fps:.1f}  |  "
        f"Inference: {infer_ms:.0f}ms  |  "
        f"SoC: {soc_temp:.1f}C  |  "
        f"{date_str}  |  "
        f"Transmilenio Edge AI v1.4.0"
    )
    cv2.putText(
        out, bottom_text,
        (10, frame_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
        (185, 185, 185), 1, cv2.LINE_AA
    )

    return out


# ── Loop State Variables ──────────────────────────────────────────────────────
_buf_roi_counts = []
_buf_cap_states = []
_buf_cpu_temp   = []
_buf_cpu_usage  = []
_buf_ram_usage  = []

_raw_count_window = deque(maxlen=5)

_hysteresis_candidate_count = -1
_hysteresis_stable_cycles   =  0
_last_confirmed_roi_count   = -1
_last_confirmed_cap_state   = "Unknown"   # Initialized to valid state, not ""

_next_chunk_deadline  = time.time() + config.CHUNK_INTERVAL_SECONDS
_next_frame_time      = time.time()
_time_per_frame       = 1.0 / config.TARGET_LIVE_FPS
_last_hw_sample_time  = 0.0
_last_audit_save_time = 0.0
_last_clean_save_time = 0.0

_fps_window_start  = time.time()
_fps_frame_count   = 0
_live_fps          = 0.0
_last_inference_ms = 0.0


# ── Startup Banner ────────────────────────────────────────────────────────────
print("=" * 68)
print("  Transmilenio Edge AI Engine — v1.4.0-prod — ACTIVE")
print("=" * 68)
print(f"  Target FPS   : {config.TARGET_LIVE_FPS} ({_time_per_frame*1000:.0f}ms/cycle)")
print(f"  Hysteresis   : {config.STABILIZATION_THRESHOLD} cycles = "
      f"{config.STABILIZATION_THRESHOLD / config.TARGET_LIVE_FPS:.1f}s")
print(f"  Moving avg   : 5-frame window")
print(f"  AWS IoT      : {'ONLINE' if aws_iot.is_connected() else 'OFFLINE'} | "
      f"Mode: {'ALL changes' if config.AWS_IOT_PUBLISH_ALL_COUNT_CHANGES else 'STATE only'}")
print(f"  RAW CSV      : {CSV_RAW_PATH}")
print(f"  SUMMARY CSV  : {CSV_SUMMARY_PATH}")
print("=" * 68)
print("  Console: one line per confirmed occupancy change.")
print("=" * 68)

stream = LiveVideoStreamThread(config.CAMERA_SOURCE).start()
time.sleep(1.0)


# ── Main Loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        now = time.time()
        frame_idx, _, raw_frame = stream.read_frame()

        # No frame available yet — idle and retry
        if raw_frame is None:
            time.sleep(config.MAIN_LOOP_IDLE_SLEEP_SEC)
            continue

        # Rate-limit processing to TARGET_LIVE_FPS
        if now < _next_frame_time:
            time.sleep(config.MAIN_LOOP_IDLE_SLEEP_SEC)
            continue
        _next_frame_time = now + _time_per_frame

        # Compute rolling FPS over 1-second windows
        _fps_frame_count += 1
        elapsed_fps = now - _fps_window_start
        if elapsed_fps >= 1.0:
            _live_fps         = _fps_frame_count / elapsed_fps
            _fps_frame_count  = 0
            _fps_window_start = now

        frame_h, frame_w = raw_frame.shape[:2]

        # Sample hardware telemetry at the configured interval
        if now - _last_hw_sample_time >= config.HARDWARE_SAMPLE_INTERVAL_SEC:
            hw = hardware_monitor.get_hardware_snapshot()
            _buf_cpu_temp.append(hw["soc_temp_c"])
            _buf_cpu_usage.append(hw["cpu_pct"])
            _buf_ram_usage.append(hw["ram_gb"])
            _last_hw_sample_time = now
            if hw["thermal_status"] == "CRITICAL":
                print(f"\n[THERMAL] CRITICAL — {hw['soc_temp_c']}C")
            elif hw["thermal_status"] == "WARNING":
                print(f"\n[THERMAL] WARNING — {hw['soc_temp_c']}C")

        _latest_temp = _buf_cpu_temp[-1] if _buf_cpu_temp else 0.0

        # ── Preprocessing: letterbox 1280×720 → 640×640 ───────────────────────
        _letterbox_canvas.fill(0)
        scale    = min(640 / frame_w, 640 / frame_h)
        new_w    = int(frame_w * scale)
        new_h    = int(frame_h * scale)
        pad_x    = (640 - new_w) // 2
        pad_y    = (640 - new_h) // 2
        resized  = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        _letterbox_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = (
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        )
        input_tensor = (
            _letterbox_canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        )[np.newaxis, ...]

        # ── Inference ─────────────────────────────────────────────────────────
        _infer_t0 = time.time()
        _infer_request.infer({_input_tensor_node: input_tensor})
        _last_inference_ms = (time.time() - _infer_t0) * 1000.0

        predictions = np.squeeze(_infer_request.get_output_tensor(0).data, axis=0)

        # ── Post-processing ────────────────────────────────────────────────────
        confidence_scores = predictions[4, :]
        mask     = confidence_scores >= config.CONF_THRESHOLD
        filtered = predictions[:, mask]

        if filtered.shape[1] > 0:
            cxs  = filtered[0, :]
            cys  = filtered[1, :]
            bws  = filtered[2, :]
            bhs  = filtered[3, :]
            scrs = filtered[4, :]

            # Convert center-format → corner-format, unproject from letterbox
            x1s = np.clip((cxs - bws / 2 - pad_x) / scale, 0.0, frame_w - 1)
            y1s = np.clip((cys - bhs / 2 - pad_y) / scale, 0.0, frame_h - 1)
            x2s = np.clip((cxs + bws / 2 - pad_x) / scale, 0.0, frame_w - 1)
            y2s = np.clip((cys + bhs / 2 - pad_y) / scale, 0.0, frame_h - 1)

            boxes_arr  = np.stack([x1s, y1s, x2s, y2s], axis=1).astype(np.float32)
            scores_arr = scrs.astype(np.float32)

            keep         = apply_nms(boxes_arr, scores_arr)
            final_boxes  = boxes_arr[keep]
            final_scores = scores_arr[keep]
        else:
            final_boxes  = np.empty((0, 4), dtype=np.float32)
            final_scores = np.empty((0,),   dtype=np.float32)

        # ── ROI Classification ────────────────────────────────────────────────
        frame_roi_count     = 0
        frame_outside_count = 0
        detections_meta     = []

        for box, score in zip(final_boxes, final_scores):
            x1, y1, x2, y2 = box
            cx = max(0, min(int((x1 + x2) / 2), frame_w - 1))
            cy = max(0, min(int(y1 + (y2 - y1) * 0.75), frame_h - 1))

            inside = cv2.pointPolygonTest(
                _roi_vertices_f32,
                (float(cx), float(cy)),
                measureDist=False,
            ) >= 0

            if inside:
                frame_roi_count += 1
            else:
                frame_outside_count += 1

            detections_meta.append({
                "box"   : (int(x1), int(y1), int(x2), int(y2)),
                "cx"    : cx,
                "cy"    : cy,
                "score" : score,
                "inside": inside,
            })

        # ── Moving average ────────────────────────────────────────────────────
        _raw_count_window.append(frame_roi_count)
        smoothed_roi_count = round(
            sum(_raw_count_window) / len(_raw_count_window)
        )

        # ── Hysteresis gate ───────────────────────────────────────────────────
        if smoothed_roi_count == _hysteresis_candidate_count:
            _hysteresis_stable_cycles += 1
        else:
            _hysteresis_candidate_count = smoothed_roi_count
            _hysteresis_stable_cycles   = 1

        count_confirmed = _hysteresis_stable_cycles >= config.STABILIZATION_THRESHOLD
        confirmed_roi_count = (
            _hysteresis_candidate_count if count_confirmed
            else _last_confirmed_roi_count
        )

        # ── Map confirmed count to capacity state ─────────────────────────────
        current_cap_state = aws_iot.resolve_capacity_state(
            confirmed_roi_count if confirmed_roi_count >= 0 else 0
        )

        # ── Fire event only when confirmed count genuinely changes ─────────────
        if count_confirmed and confirmed_roi_count != _last_confirmed_roi_count:
            _last_confirmed_roi_count = confirmed_roi_count
            _last_confirmed_cap_state = current_cap_state
            _buf_roi_counts.append(confirmed_roi_count)
            _buf_cap_states.append(current_cap_state)

            ts_str = datetime.now().strftime("%H:%M:%S")

            csv_handler.save_raw_detection_row(
                raw_path       = CSV_RAW_PATH,
                timestamp_str  = ts_str,
                frame_index    = frame_idx,
                roi_count      = confirmed_roi_count,
                outside_count  = frame_outside_count,
                capacity_state = current_cap_state,
            )

            aws_iot.safe_aws_publish(
                roi_count      = confirmed_roi_count,
                outside_count  = frame_outside_count,
                capacity_state = current_cap_state,
                soc_temp       = _latest_temp,
            )

            aws_status = "Cloud OK" if aws_iot.is_connected() else "Cloud OFF"
            print(
                f"\r[{ts_str}]  "
                f"ROI: {confirmed_roi_count:02d} |  "
                f"{_ui(current_cap_state)['label']:<12s}|  "
                f"Out: {frame_outside_count:02d}  |  "
                f"{aws_status}  |  "
                f"{_live_fps:.1f}fps  {_latest_temp:.1f}C   ",
                end="", flush=True,
            )

        # ── 5-minute summary ──────────────────────────────────────────────────
        if now >= _next_chunk_deadline:
            window_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hw_avgs   = hardware_monitor.compute_buffer_averages(
                _buf_cpu_temp, _buf_cpu_usage, _buf_ram_usage
            )
            analytics = csv_handler.save_summary_row(
                summary_path      = CSV_SUMMARY_PATH,
                window_timestamp  = window_ts,
                roi_counts_buffer = _buf_roi_counts,
                states_buffer     = _buf_cap_states,
                avg_temp_c        = hw_avgs["avg_temp_c"],
                avg_cpu_pct       = hw_avgs["avg_cpu_pct"],
                avg_ram_gb        = hw_avgs["avg_ram_gb"],
            )

            if analytics:
                print(
                    f"\n[5-MIN] {window_ts} "
                    f"| avg:{analytics['avg_roi']} peak:{analytics['max_peak']} "
                    f"sd:{analytics['std_dev']} pax"
                    f" | {analytics['dominant_state']}"
                    f" | {hw_avgs['avg_temp_c']}C "
                    f"CPU:{hw_avgs['avg_cpu_pct']}% RAM:{hw_avgs['avg_ram_gb']}GB"
                )

            _buf_roi_counts.clear()
            _buf_cap_states.clear()
            _buf_cpu_temp.clear()
            _buf_cpu_usage.clear()
            _buf_ram_usage.clear()
            _next_chunk_deadline = now + config.CHUNK_INTERVAL_SECONDS

        # ── Save annotated audit frame ─────────────────────────────────────────
        if config.SAVE_AUDIT_IMAGES and (
            now - _last_audit_save_time >= config.IMAGE_SAVE_COOLDOWN_SECONDS
        ):
            _now_dt  = datetime.now()
            _hour_dir = os.path.join(
                config.BASE_DEBUG_DIR,
                _now_dt.strftime("%Y-%m-%d_%Hh"),
            )
            os.makedirs(_hour_dir, exist_ok=True)

            audit_frame = _draw_audit_hud(
                raw_frame,
                confirmed_count = _last_confirmed_roi_count,
                live_roi_count  = frame_roi_count,
                live_out_count  = frame_outside_count,
                cap_state       = _last_confirmed_cap_state,
                live_fps        = _live_fps,
                infer_ms        = _last_inference_ms,
                soc_temp        = _latest_temp,
                detections      = detections_meta,
                frame_w         = frame_w,
                frame_h         = frame_h,
                date_str        = _now_dt.strftime("%Y-%m-%d"),
            )

            # Filename uses live count — reflects what is visible in the image
            fname = os.path.join(
                _hour_dir,
                f"audit_{_now_dt.strftime('%H%M%S')}"
                f"_pax{max(frame_roi_count, 0):02d}.jpg",
            )
            cv2.imwrite(fname, audit_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            _last_audit_save_time = now

        # ── Save raw frame for dataset collection ─────────────────────────────
        if config.SAVE_CLEAN_DATASET and (
            now - _last_clean_save_time >= config.IMAGE_SAVE_COOLDOWN_SECONDS
        ):
            fname = os.path.join(
                config.DATASET_CLEAN_DIR,
                f"clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_f{frame_idx:06d}.jpg",
            )
            cv2.imwrite(fname, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _last_clean_save_time = now

except KeyboardInterrupt:
    print("\n\n[SHUTDOWN] Stopping engine...")

except Exception:
    print("\n\n[FATAL] Unhandled exception:")
    traceback.print_exc()

finally:
    print("[SHUTDOWN] Releasing camera...")
    stream.release_resources()
    print("[SHUTDOWN] Disconnecting AWS IoT...")
    aws_iot.disconnect_aws_iot()
    print("[SHUTDOWN] Writing final summary...")
    if _buf_roi_counts:
        hw_avgs_final = hardware_monitor.compute_buffer_averages(
            _buf_cpu_temp, _buf_cpu_usage, _buf_ram_usage
        )
        csv_handler.save_summary_row(
            summary_path      = CSV_SUMMARY_PATH,
            window_timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            roi_counts_buffer = _buf_roi_counts,
            states_buffer     = _buf_cap_states,
            avg_temp_c        = hw_avgs_final["avg_temp_c"],
            avg_cpu_pct       = hw_avgs_final["avg_cpu_pct"],
            avg_ram_gb        = hw_avgs_final["avg_ram_gb"],
        )
        print("[SHUTDOWN] Final summary written.")
    print("[SHUTDOWN] Engine stopped.")