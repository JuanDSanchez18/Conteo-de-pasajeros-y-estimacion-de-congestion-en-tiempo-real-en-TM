# -*- coding: utf-8 -*-
# ============================================================
# Fixed-Station Sequence Engine V8.2
# Run mode: ONE execution = ONE sequence folder
# Output  : Single CSV for all videos in the sequence,
#           differentiated only by Video_Source column
#
# To process multiple sequences:
#   Step 1 → Set SEQUENCE_DIR to Sequence1 → run
#   Step 2 → Set SEQUENCE_DIR to Sequence2 → run
#   Step 3 → ...
# ============================================================
import cv2
import numpy as np
from openvino import Core
import csv
import os
import sys
import io
import queue
import threading
import time
import glob
from collections import Counter

# ── UTF-8 Console Fix ────────────────────────────────────────
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace'
)

# ─────────────────────────────────────────────────────────────
# SECTION 1 — ONLY CHANGE THIS TO SWITCH SEQUENCES
# ─────────────────────────────────────────────────────────────
# Point directly to the sequence folder you want to process.
# The script auto-detects the sequence name and parent folder.
# Examples:
#   SEQUENCE_DIR = ".../SingleDoorMiddle/Sequence1"
#   SEQUENCE_DIR = ".../SingleDoorMiddle/Sequence2"
#   SEQUENCE_DIR = ".../SingleDoorMiddle/Sequence3"
SEQUENCE_DIR = "/home/siscotm1/Documents/SingleDoorMiddle/Sequence3"

MODEL_XML = "/home/siscotm1/DetectorIA/Modelo/best_openvino_model_DetectorIA_v1/best.xml"

# ─────────────────────────────────────────────────────────────
# SECTION 2 — AUTO PATH DERIVATION (DO NOT EDIT)
# ─────────────────────────────────────────────────────────────
# Extracts names automatically from SEQUENCE_DIR:
#   SEQUENCE_DIR  = .../SingleDoorMiddle/Sequence1
#   sequence_name = Sequence1
#   parent_name   = SingleDoorMiddle
#
# Output folder sits next to the sequence folder:
#   .../SingleDoorMiddle/Output_Sequence1/
SEQUENCE_DIR    = SEQUENCE_DIR.rstrip("/")
sequence_name   = os.path.basename(SEQUENCE_DIR)
parent_dir      = os.path.dirname(SEQUENCE_DIR)
parent_name     = os.path.basename(parent_dir)

# All outputs go here — created automatically
output_dir      = os.path.join(parent_dir, f"Output_{sequence_name}")
debug_dir       = os.path.join(output_dir, "Debug_Frames")

# Single CSV for the entire sequence
# Video_Source column differentiates each video inside
csv_raw_path     = os.path.join(
    output_dir,
    f"Validation_{parent_name}_{sequence_name}_RAW.csv"
)
csv_summary_path = os.path.join(
    output_dir,
    f"Validation_{parent_name}_{sequence_name}_Resumen_5Min.csv"
)

# ─────────────────────────────────────────────────────────────
# SECTION 3 — INFERENCE PARAMETERS
# ─────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.45
NMS_THRESHOLD  = 0.45
SKIP_FRAMES    = 30        # 1 inference per video second at 30 FPS
VIDEO_FPS      = 30.0
MAX_CAPACITY   = 20

# ─────────────────────────────────────────────────────────────
# SECTION 4 — OPENVINO CONFIG
# ─────────────────────────────────────────────────────────────
OV_NUM_THREADS = 4
OV_NUM_STREAMS = 2

# ─────────────────────────────────────────────────────────────
# SECTION 5 — TEMPERATURE SAMPLING
# ─────────────────────────────────────────────────────────────
TEMP_SAMPLE_EVERY_N_FRAMES = 150   # Every ~2.5 min of video content

# ─────────────────────────────────────────────────────────────
# SECTION 6 — ROI PRESETS (UNCHANGED)
# ─────────────────────────────────────────────────────────────
ROI_PRESETS = {
    "OPTION_A": np.array([[50, 105], [590, 105], [635, 640], [5, 640]], np.int32),
    "OPTION_B": np.array([[15, 50],  [625, 50],  [638, 640], [2, 640]], np.int32)
}
ACTIVE_ROI_KEY = "OPTION_A"
ROI_VERTICES   = ROI_PRESETS[ACTIVE_ROI_KEY]

# ─────────────────────────────────────────────────────────────
# SECTION 7 — DEBUG FLAGS
# ─────────────────────────────────────────────────────────────
DEBUG_MODE        = True
DEBUG_SKIP_FRAMES = 150    # 1 JPEG every 5s of virtual video

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def initialize_raw_csv(filepath):
    """
    Creates a single clean CSV for the entire sequence.
    All videos append rows here — differentiated by Video_Source.
    Called ONCE before the video loop, not per video.
    """
    with open(filepath, mode='w', newline='') as f:
        csv.writer(f).writerow([
            'Video_Source', 'Frame', 'Relative_msec',
            'ROI_Count', 'Outside_Count', 'Capacity_State', 'Temp_C'
        ])


def save_atomic_csv_row(filepath, row_data):
    """
    Thread-safe atomic append with fsync.
    Event-driven — called only when ROI_Count changes.
    """
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)
        f.flush()
        os.fsync(f.fileno())


def compute_capacity_state(current_count, max_cap):
    if current_count == 0:                        return "Empty"
    elif current_count <= int(max_cap * 0.25):    return "Low"
    elif current_count <= int(max_cap * 0.55):    return "Medium"
    elif current_count <= int(max_cap * 0.90):    return "High"
    else:                                         return "Super High"


def read_soc_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except:
        return 0.0


# ─────────────────────────────────────────────────────────────
# ASYNC VIDEO STREAM ENGINE
# ─────────────────────────────────────────────────────────────
class AsyncVideoStreamThread:
    def __init__(self, file_source):
        self.capture            = cv2.VideoCapture(file_source)
        self.frame_queue        = queue.Queue(maxsize=128)
        self.is_stopped         = False
        self.internal_frame_idx = 0

    def start(self):
        threading.Thread(target=self._stream_worker, daemon=True).start()
        return self

    def _stream_worker(self):
        while not self.is_stopped:
            if not self.frame_queue.full():
                success, frame = self.capture.read()
                if not success:
                    self.is_stopped = True
                    break
                self.internal_frame_idx += 1
                rel_msec = int(self.capture.get(cv2.CAP_PROP_POS_MSEC))
                self.frame_queue.put((self.internal_frame_idx, rel_msec, frame))
            else:
                time.sleep(0.001)

    def read_frame(self):
        return (self.frame_queue.get()
                if not self.frame_queue.empty()
                else (None, None, None))

    def release_resources(self):
        self.is_stopped = True
        self.capture.release()


# ─────────────────────────────────────────────────────────────
# OPENVINO INIT — Model loaded ONCE, reused for all videos
# ─────────────────────────────────────────────────────────────
openvino_core    = Core()
compiled_network = openvino_core.read_model(MODEL_XML)
openvino_core.set_property("CPU", {
    "INFERENCE_PRECISION_HINT" : "f32",
    "PERFORMANCE_HINT"         : "THROUGHPUT",
    "NUM_STREAMS"              : str(OV_NUM_STREAMS),
    "INFERENCE_NUM_THREADS"    : str(OV_NUM_THREADS)
})
hardware_pipeline = openvino_core.compile_model(compiled_network, "CPU")
inference_request = hardware_pipeline.create_infer_request()
input_tensor_node = hardware_pipeline.input(0)

# ─────────────────────────────────────────────────────────────
# SCAN VIDEOS IN THE SEQUENCE FOLDER
# ─────────────────────────────────────────────────────────────
video_list = sorted(glob.glob(os.path.join(SEQUENCE_DIR, "*.mkv")))

if not video_list:
    print(f"[ERROR] No .mkv files found in: {SEQUENCE_DIR}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# CREATE OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────
os.makedirs(output_dir, exist_ok=True)
if DEBUG_MODE:
    os.makedirs(debug_dir, exist_ok=True)

# Single CSV initialized ONCE here — all videos append to it
initialize_raw_csv(csv_raw_path)

# ─────────────────────────────────────────────────────────────
# SYSTEM BANNER
# ─────────────────────────────────────────────────────────────
seconds_per_record = SKIP_FRAMES / VIDEO_FPS

print("=" * 80)
print(f"[SYSTEM] Fixed-Station Sequence Engine V8.2")
print(f"[SYSTEM] OpenVINO    : THROUGHPUT | Threads: {OV_NUM_THREADS} | Streams: {OV_NUM_STREAMS}")
print(f"[SYSTEM] Sequence    : {sequence_name}")
print(f"[SYSTEM] Videos found: {len(video_list)}")
print(f"[SYSTEM] Output dir  : {output_dir}")
print(f"[SYSTEM] Single CSV  : {os.path.basename(csv_raw_path)}")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# TELEMETRY ACCUMULATORS — Shared across ALL videos
# Flushed only at end of sequence for summary generation
# ─────────────────────────────────────────────────────────────
telemetry_temperatures    = []
telemetry_roi_counts      = []
telemetry_capacity_states = []
individual_video_history  = []

sequence_start_time    = time.time()
total_sequence_frames  = 0

# Initial temperature read — fixes the 0.0 CSV issue
last_known_temp = read_soc_temperature()
if last_known_temp > 0:
    telemetry_temperatures.append(last_known_temp)

# ─────────────────────────────────────────────────────────────
# VIDEO LOOP — Single level, no outer sequence loop
# ─────────────────────────────────────────────────────────────
try:
    for v_idx, video_path in enumerate(video_list, start=1):

        video_name  = os.path.basename(video_path)
        video_base  = video_name.rsplit('.', 1)[0]

        print(f"\n  [{v_idx:02d}/{len(video_list):02d}] Starting: {video_name}", flush=True)

        stream = AsyncVideoStreamThread(video_path).start()
        time.sleep(0.5)

        video_run_start_time   = None
        video_frames_processed = 0

        # ── Reset per video — CRITICAL BUG FIX ──────────────────
        # Ensures the first frame of every video writes to CSV
        # regardless of the ROI count the previous video ended on.
        last_recorded_roi_count = -1

        fps_timer_base = time.time()
        fps_counter    = 0
        live_fps       = 0.0

        # ── FRAME LOOP ───────────────────────────────────────────
        while not stream.is_stopped or not stream.frame_queue.empty():

            frame_idx, rel_msec, raw_frame = stream.read_frame()
            if raw_frame is None:
                time.sleep(0.002)
                continue

            if frame_idx % SKIP_FRAMES != 0:
                continue

            if video_run_start_time is None:
                video_run_start_time = time.time()
                fps_timer_base       = time.time()

            video_frames_processed += 1
            total_sequence_frames  += 1
            fps_counter            += 1

            frame_h, frame_w = raw_frame.shape[:2]

            # ── Temperature Sampling ─────────────────────────────
            if video_frames_processed % TEMP_SAMPLE_EVERY_N_FRAMES == 0:
                last_known_temp = read_soc_temperature()
                if last_known_temp > 0:
                    telemetry_temperatures.append(last_known_temp)

            current_hardware_temp = last_known_temp

            # ── Console Monitor ──────────────────────────────────
            elapsed = time.time() - fps_timer_base
            if elapsed >= 1.0:
                live_fps       = fps_counter / elapsed
                fps_counter    = 0
                fps_timer_base = time.time()

            print(
                f"    ↳ [{video_base}] Frm:{frame_idx:05d} | "
                f"ROI:{last_recorded_roi_count:02d} | "
                f"FPS:{live_fps:.2f} | "
                f"Temp:{current_hardware_temp:.1f}C",
                end="\r", flush=True
            )

            # ── Letterbox Preprocessing ──────────────────────────
            input_tensor_canvas = np.zeros((640, 640, 3), dtype=np.uint8)
            rescale_factor      = min(640 / frame_w, 640 / frame_h)
            scaled_w            = int(frame_w * rescale_factor)
            scaled_h            = int(frame_h * rescale_factor)
            resized_frame       = cv2.resize(raw_frame, (scaled_w, scaled_h),
                                             interpolation=cv2.INTER_LINEAR)
            resized_rgb         = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            x_pad               = (640 - scaled_w) // 2
            y_pad               = (640 - scaled_h) // 2
            input_tensor_canvas[y_pad:y_pad+scaled_h, x_pad:x_pad+scaled_w] = resized_rgb

            # ── Inference ────────────────────────────────────────
            blob = (input_tensor_canvas
                    .transpose(2, 0, 1)
                    .reshape(1, 3, 640, 640)
                    .astype(np.float32) / 255.0)
            inference_request.infer({input_tensor_node: blob})
            raw_outputs = inference_request.get_output_tensor(0).data

            # ── Post-Processing ──────────────────────────────────
            detections_matrix   = np.squeeze(raw_outputs)
            confidence_mask     = detections_matrix[4, :] > CONF_THRESHOLD
            filtered_detections = detections_matrix[:, confidence_mask].T

            roi_count, outside_count = 0, 0
            bounding_boxes, confidences = [], []
            nms_indices = []

            if len(filtered_detections) > 0:
                for det in filtered_detections:
                    cx, cy, w_d, h_d = det[0:4]
                    orig_cx  = int((cx  - x_pad) / rescale_factor)
                    orig_cy  = int((cy  - y_pad) / rescale_factor)
                    orig_w   = int(w_d  / rescale_factor)
                    orig_h   = int(h_d  / rescale_factor)
                    bounding_boxes.append([
                        int(orig_cx - orig_w / 2),
                        int(orig_cy - orig_h / 2),
                        orig_w, orig_h
                    ])
                    confidences.append(float(det[4]))

                nms_indices = cv2.dnn.NMSBoxes(
                    bounding_boxes, confidences,
                    CONF_THRESHOLD, NMS_THRESHOLD
                )

                if len(nms_indices) > 0:
                    for idx_nms in nms_indices.flatten():
                        bx, by, bw, bh = bounding_boxes[idx_nms]
                        pts = [
                            (int(bx),          int(by)),
                            (int(bx + bw),     int(by)),
                            (int(bx + bw),     int(by + bh)),
                            (int(bx),          int(by + bh)),
                            (int(bx + bw / 2), int(by + bh / 2))
                        ]
                        if any(cv2.pointPolygonTest(ROI_VERTICES, p, False) >= 0
                               for p in pts):
                            roi_count += 1
                        else:
                            outside_count += 1

            current_state = compute_capacity_state(roi_count, MAX_CAPACITY)
            telemetry_roi_counts.append(roi_count)
            telemetry_capacity_states.append(current_state)

            # ── Event-Driven CSV Write ───────────────────────────
            # Writes to the SINGLE sequence CSV.
            # Video_Source column identifies which video the row belongs to.
            if roi_count != last_recorded_roi_count:
                save_atomic_csv_row(csv_raw_path, [
                    video_name, frame_idx, rel_msec,
                    roi_count, outside_count,
                    current_state, current_hardware_temp
                ])
                last_recorded_roi_count = roi_count

            # ── Debug Frame ──────────────────────────────────────
            if DEBUG_MODE and (frame_idx % DEBUG_SKIP_FRAMES == 0):
                debug_canvas = raw_frame.copy()
                cv2.polylines(debug_canvas, [ROI_VERTICES],
                              isClosed=True, color=(180, 180, 180), thickness=2)

                if len(filtered_detections) > 0 and len(nms_indices) > 0:
                    for idx_nms in nms_indices.flatten():
                        bx, by, bw, bh = bounding_boxes[idx_nms]
                        pts = [
                            (int(bx), int(by)), (int(bx+bw), int(by)),
                            (int(bx+bw), int(by+bh)), (int(bx), int(by+bh)),
                            (int(bx+bw/2), int(by+bh/2))
                        ]
                        color = (0, 215, 0) if any(
                            cv2.pointPolygonTest(ROI_VERTICES, p, False) >= 0
                            for p in pts
                        ) else (0, 140, 255)
                        cv2.rectangle(debug_canvas,
                                      (bx, by), (bx+bw, by+bh),
                                      color, 2 if color == (0, 215, 0) else 1)

                overlay = debug_canvas.copy()
                cv2.rectangle(overlay, (0, 0), (frame_w, 45), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.6, debug_canvas, 0.4, 0, debug_canvas)
                cv2.putText(debug_canvas,
                            f"{sequence_name} | {video_name} | "
                            f"Frm:{frame_idx:05d} | "
                            f"ROI:{roi_count} | {current_state} | "
                            f"{current_hardware_temp:.1f}C",
                            (20, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (240, 240, 240), 2, cv2.LINE_AA)

                # Prefix with video_base to avoid filename collisions
                cv2.imwrite(
                    os.path.join(debug_dir, f"{video_base}_frame_{frame_idx:06d}.jpg"),
                    debug_canvas
                )

        # ── End of video ─────────────────────────────────────────
        stream.release_resources()

        video_net_time = time.time() - (
            video_run_start_time if video_run_start_time else time.time()
        )
        video_fps = (video_frames_processed / video_net_time
                     if video_net_time > 0 else 0)

        print(f"\n    ✔ {video_name} | "
              f"Frames: {video_frames_processed} | "
              f"FPS: {video_fps:.2f} | "
              f"Time: {video_net_time:.1f}s")

        individual_video_history.append({
            "name": video_name, "frames": video_frames_processed,
            "duration": video_net_time, "fps": video_fps
        })

    # ─────────────────────────────────────────────────────────
    # POST-SEQUENCE — 5-MIN BLOCK SUMMARY
    # ─────────────────────────────────────────────────────────
    total_records = len(telemetry_roi_counts)
    if total_records > 0:
        print(f"\n[ANALYTICS] Building 5-min summary → "
              f"{os.path.basename(csv_summary_path)}...")
        try:
            records_per_5min = int(300 / seconds_per_record)
            records_per_5min = max(records_per_5min, 1)

            with open(csv_summary_path, mode='w', newline='') as f_s:
                sw = csv.writer(f_s)
                sw.writerow([
                    'Interval_Minutes', 'ROI_Count_Average',
                    'ROI_Count_Max_Peak', 'ROI_Standard_Deviation',
                    'Predominant_Capacity_State', 'CPU_Temperature_Average'
                ])

                for i in range(0, total_records, records_per_5min):
                    s_roi    = telemetry_roi_counts[i : i + records_per_5min]
                    s_states = telemetry_capacity_states[i : i + records_per_5min]
                    s_temps  = telemetry_temperatures[
                        i // TEMP_SAMPLE_EVERY_N_FRAMES :
                        (i + records_per_5min) // TEMP_SAMPLE_EVERY_N_FRAMES
                    ]
                    if not s_roi:
                        continue

                    avg_roi = sum(s_roi) / len(s_roi)
                    std_dev = (sum((x - avg_roi)**2 for x in s_roi) / len(s_roi)) ** 0.5
                    sw.writerow([
                        int((i * seconds_per_record) / 60),
                        round(avg_roi, 2),
                        max(s_roi),
                        round(std_dev, 2),
                        Counter(s_states).most_common(1)[0][0],
                        round(sum(s_temps)/len(s_temps), 1) if s_temps else 0.0
                    ])

            print("   ✔ 5-min summary written successfully.")
        except Exception as se:
            print(f"   ⚠ Summary failed: {se}")

    # ─────────────────────────────────────────────────────────
    # FINAL CONSOLIDATED REPORT
    # ─────────────────────────────────────────────────────────
    seq_total_time = time.time() - sequence_start_time
    virt_scope     = total_sequence_frames * seconds_per_record
    accel          = virt_scope / seq_total_time if seq_total_time > 0 else 0
    batch_fps      = total_sequence_frames / seq_total_time if seq_total_time > 0 else 0

    print("\n" + "=" * 80)
    print(f" [SEQUENCE REPORT — {sequence_name} — V8.2]")
    print("=" * 80)
    print(f" Sequence folder   : {SEQUENCE_DIR}")
    print(f" Output folder     : {output_dir}")
    print(f" Single CSV        : {os.path.basename(csv_raw_path)}")
    print(f" Physical time     : {seq_total_time:.2f}s ({seq_total_time/60:.2f} min)")
    print(f" Virtual scope     : {virt_scope:.2f}s ({virt_scope/3600:.2f} hrs)")
    print(f" Acceleration      : {accel:.2f}x vs real time")
    print(f"\n (INPUT SCHEMA)")
    print(f"  -> Model         : YOLO11n OpenVINO FP32")
    print(f"  -> OV Threads    : {OV_NUM_THREADS} | Streams: {OV_NUM_STREAMS} | THROUGHPUT")
    print(f"  -> SKIP_FRAMES   : {SKIP_FRAMES} → 1 inference/video second")
    print(f"  -> Inferences    : {total_sequence_frames} cycles")
    print(f"\n (TRAFFIC METRICS)")
    print(f"  -> Peak Occupancy   : {max(telemetry_roi_counts) if telemetry_roi_counts else 0} pax")
    print(f"  -> Avg Occupancy    : {sum(telemetry_roi_counts)/len(telemetry_roi_counts):.2f} pax/s"
          if telemetry_roi_counts else "  -> Avg Occupancy    : N/A")
    print(f"  -> Dominant State   : "
          f"{Counter(telemetry_capacity_states).most_common(1)[0][0] if telemetry_capacity_states else 'N/A'}")
    print(f"\n (HARDWARE — RPi5)")
    print(f"  -> Net Throughput   : {batch_fps:.2f} FPS")
    print(f"  -> SoC Temp Avg     : {sum(telemetry_temperatures)/len(telemetry_temperatures):.1f}C"
          if telemetry_temperatures else "  -> SoC Temp Avg     : N/A")
    print(f"  -> SoC Temp Peak    : {max(telemetry_temperatures):.1f}C"
          if telemetry_temperatures else "  -> SoC Temp Peak    : N/A")
    print(f"\n FILE BREAKDOWN:")
    for vs in individual_video_history:
        print(f"  -> {vs['name']}: {vs['frames']}f | "
              f"{vs['duration']:.1f}s | {vs['fps']:.2f} FPS")
    print("=" * 80)

except Exception as e:
    print(f"\n[FATAL ERROR]: {e}")
    raise