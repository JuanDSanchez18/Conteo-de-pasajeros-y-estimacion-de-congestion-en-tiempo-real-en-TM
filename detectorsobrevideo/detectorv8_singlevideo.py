# -*- coding: utf-8 -*-
# ============================================================
# Fixed-Station Single Engine V7.7
# Optimizations:
#   [1] OpenVINO 4-core THROUGHPUT pipeline
#   [2] SoC temperature sampled every 150 processed frames
#   [3] ROI polygon test unchanged (any-point strategy)
#   [4] SKIP_FRAMES = 30 (1 inference per video second)
# ============================================================
import cv2
import numpy as np
from openvino import Core
import csv
import os
import queue
import threading
import time
from collections import Counter

import sys
import io

# Forzar stdout a UTF-8
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace'
)

# ─────────────────────────────────────────────────────────────
# SECTION 1 — FILE PATHS
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = "/home/siscotm1/Documents/SingleDoorMiddle/Sequence3/videoTM_01.mkv"
MODEL_XML  = "/home/siscotm1/DetectorIA/Modelo/best_openvino_model_DetectorIA_v1/best.xml"

# ─────────────────────────────────────────────────────────────
# SECTION 2 — INFERENCE PARAMETERS
# ─────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.45
NMS_THRESHOLD  = 0.45
SKIP_FRAMES    = 30        # 1 inference per video second at 30 FPS
VIDEO_FPS      = 30.0
MAX_CAPACITY   = 20

# ─────────────────────────────────────────────────────────────
# SECTION 3 — OPENVINO CORE CONFIGURATION  [CHANGE #1]
# ─────────────────────────────────────────────────────────────
# Using 4 cores with THROUGHPUT hint to maximize offline
# video processing speed on Raspberry Pi 5.
# LATENCY hint was designed for live streaming (minimize
# single-frame delay). For offline video analysis where
# we want to process the full file as fast as possible,
# THROUGHPUT is the correct strategy.
OV_NUM_THREADS = 4         # RPi5 has 4 performance cores
OV_NUM_STREAMS = 2         # 2 parallel inference streams
                           # (streams × threads/stream = 4 cores)

# ─────────────────────────────────────────────────────────────
# SECTION 4 — TEMPERATURE SAMPLING INTERVAL  [CHANGE #2]
# ─────────────────────────────────────────────────────────────
# Reading /sys/class/thermal every frame adds unnecessary
# OS I/O overhead in a tight inference loop.
# At SKIP_FRAMES=30 and VIDEO_FPS=30, each processed frame
# represents exactly 1 second of video content.
# Sampling every 150 processed frames = every ~2.5 minutes
# of video content. Temperature changes slowly on hardware,
# so this interval captures thermal trends accurately.
TEMP_SAMPLE_EVERY_N_FRAMES = 150   # ~2.5 min of video content
                                   # Adjust to 30 for every 30s
                                   # Adjust to 60 for every 1min

# ─────────────────────────────────────────────────────────────
# SECTION 5 — ROI PRESETS (UNCHANGED)
# ─────────────────────────────────────────────────────────────
ROI_PRESETS = {
    "OPTION_A": np.array([[50, 105], [590, 105], [635, 640], [5, 640]], np.int32),
    "OPTION_B": np.array([[15, 50],  [625, 50],  [638, 640], [2, 640]], np.int32)
}
ACTIVE_ROI_KEY = "OPTION_A"
ROI_VERTICES   = ROI_PRESETS[ACTIVE_ROI_KEY]

# ─────────────────────────────────────────────────────────────
# SECTION 6 — DEBUG FLAGS
# ─────────────────────────────────────────────────────────────
DEBUG_MODE        = True
DEBUG_SKIP_FRAMES = 150    # 1 debug JPEG every 5s of video content

# ─────────────────────────────────────────────────────────────
# SECTION 7 — AUTO OUTPUT NAMING
# ─────────────────────────────────────────────────────────────
video_filename    = os.path.basename(VIDEO_PATH)
video_base_name   = video_filename.rsplit('.', 1)[0]
csv_raw_filename  = f"Validation_SingleVideo_{video_base_name}_RAW.csv"
csv_summary_filename = f"Validation_SingleVideo_{video_base_name}_Resumen_5Min.csv"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def initialize_raw_csv(filepath):
    """Generates a clean CSV file structure with standard analytics headers."""
    with open(filepath, mode='w', newline='') as f:
        csv.writer(f).writerow([
            'Video_Source', 'Frame', 'Relative_msec',
            'ROI_Count', 'Outside_Count', 'Capacity_State', 'Temp_C'
        ])

initialize_raw_csv(csv_raw_filename)


def save_atomic_csv_row(filepath, row_data):
    """
    Thread-safe atomic write. Called ONLY on ROI_Count change events,
    not on every processed frame. This keeps CSV writes event-driven
    rather than continuous, reducing disk I/O significantly.
    """
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)
        f.flush()
        os.fsync(f.fileno())


def compute_capacity_state(current_count, max_cap):
    """Classifies station occupancy based on calibrated threshold layers."""
    if current_count == 0:
        return "Empty"
    elif current_count <= int(max_cap * 0.25):   # ≤ 5
        return "Low"
    elif current_count <= int(max_cap * 0.55):   # ≤ 11
        return "Medium"
    elif current_count <= int(max_cap * 0.90):   # ≤ 18
        return "High"
    else:                                         # > 18
        return "Super High"


def read_soc_temperature():
    """
    Reads SoC temperature from Linux thermal zone.
    Called every TEMP_SAMPLE_EVERY_N_FRAMES processed frames,
    NOT on every frame. Reduces OS filesystem I/O in hot loop.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except:
        return 0.0


# ─────────────────────────────────────────────────────────────
# ASYNC VIDEO STREAM (UNCHANGED ARCHITECTURE)
# ─────────────────────────────────────────────────────────────
class AsyncVideoStreamThread:
    """
    Asynchronous video frame preloading engine for ARM architectures.
    Decouples USB/disk read from inference loop, preventing I/O stalls
    from blocking the OpenVINO pipeline.
    """
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
        return self.frame_queue.get() \
               if not self.frame_queue.empty() \
               else (None, None, None)

    def release_resources(self):
        self.is_stopped = True
        self.capture.release()


# ─────────────────────────────────────────────────────────────
# OPENVINO ENGINE INITIALIZATION  [CHANGE #1 — APPLIED HERE]
# ─────────────────────────────────────────────────────────────
print("-" * 80)
print(f"[SYSTEM] Fixed-Station Single Engine V8 - Dedicated Mode")
print(f"[SYSTEM] OpenVINO: THROUGHPUT | Threads: {OV_NUM_THREADS} | Streams: {OV_NUM_STREAMS}")
print(f"[SYSTEM] Temp sampling: every {TEMP_SAMPLE_EVERY_N_FRAMES} processed frames")
print(f"[SYSTEM] Target File  : {video_filename}")
print(f"[SYSTEM] Output RAW   : {csv_raw_filename}")
print("-" * 80)

openvino_core      = Core()
compiled_network   = openvino_core.read_model(MODEL_XML)

# KEY CHANGE: THROUGHPUT + explicit thread/stream allocation
# ┌─────────────────────────────────────────────────────────┐
# │  PERFORMANCE_HINT  : THROUGHPUT                         │
# │    → OpenVINO batches inference requests internally      │
# │    → Maximizes frames/second over the full video file    │
# │    → Trades individual frame latency for total speed     │
# │                                                          │
# │  NUM_STREAMS       : 2                                   │
# │    → 2 parallel execution queues inside OpenVINO         │
# │    → Each stream uses 2 threads → 2×2 = 4 cores total    │
# │                                                          │
# │  INFERENCE_NUM_THREADS : 4                               │
# │    → Hard cap at 4 threads matching RPi5 core count      │
# │    → Prevents OS scheduler contention                    │
# └─────────────────────────────────────────────────────────┘
openvino_core.set_property("CPU", {
    "INFERENCE_PRECISION_HINT" : "f32",
    "PERFORMANCE_HINT"         : "THROUGHPUT",
    "NUM_STREAMS"              : str(OV_NUM_STREAMS),
    "INFERENCE_NUM_THREADS"    : str(OV_NUM_THREADS)
})

hardware_pipeline   = openvino_core.compile_model(compiled_network, "CPU")
inference_request   = hardware_pipeline.create_infer_request()
input_tensor_node   = hardware_pipeline.input(0)


# ─────────────────────────────────────────────────────────────
# TELEMETRY ACCUMULATORS
# ─────────────────────────────────────────────────────────────
telemetry_temperatures    = []
telemetry_roi_counts      = []
telemetry_capacity_states = []

# Temperature state — persists last known value between samples
last_known_temp = read_soc_temperature()     # Updated every TEMP_SAMPLE_EVERY_N_FRAMES
telemetry_temperatures.append(last_known_temp) 

lote_start_time           = time.time()
video_run_start_time      = None


# ─────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────
try:
    output_debug_dir = f"Debug_Frames_SingleRun_{video_base_name}"
    if DEBUG_MODE:
        os.makedirs(output_debug_dir, exist_ok=True)

    print(f"[STARTING INFERENCE] -> Processing: {video_filename}", flush=True)

    stream = AsyncVideoStreamThread(VIDEO_PATH).start()
    time.sleep(0.5)   # Buffer cushion — preload initial frames into RAM

    video_run_start_time   = time.time()
    video_frames_processed = 0
    last_recorded_roi_count = -1

    # Live FPS console monitor variables
    fps_timer_base = time.time()
    fps_counter    = 0
    live_fps       = 0.0

    while not stream.is_stopped or not stream.frame_queue.empty():

        frame_idx, rel_msec, raw_frame = stream.read_frame()
        if raw_frame is None:
            time.sleep(0.002)
            continue

        # ── Frame skip filter (lightweight modulo in RAM) ────────
        if frame_idx % SKIP_FRAMES != 0:
            continue

        video_frames_processed += 1
        fps_counter            += 1

        frame_h, frame_w = raw_frame.shape[:2]

        # ── TEMPERATURE SAMPLING  [CHANGE #2] ───────────────────
        # Read hardware temperature only every N processed frames.
        # Between samples, reuse last_known_temp to avoid /sys I/O.
        # processed frame 1   → READ  (video second 1)
        # processed frame 2   → SKIP  (reuse last value)
        # ...
        # processed frame 150 → READ  (video second 150 ≈ 2.5 min)
        if video_frames_processed % TEMP_SAMPLE_EVERY_N_FRAMES == 0:
            last_known_temp = read_soc_temperature()
            if last_known_temp > 0:
                telemetry_temperatures.append(last_known_temp)

        current_hardware_temp = last_known_temp   # Use cached value always

        # ── Console FPS Monitor ──────────────────────────────────
        fps_elapsed_window = time.time() - fps_timer_base
        if fps_elapsed_window >= 1.0:
            live_fps    = fps_counter / fps_elapsed_window
            fps_counter = 0
            fps_timer_base = time.time()

        print(
            f"   ↳ Frm: {frame_idx:05d} | "
            f"ROI Pax: {last_recorded_roi_count:02d} | "
            f"Throughput: {live_fps:.2f} FPS | "
            f"SoC Temp: {current_hardware_temp:.1f}C",
            end="\r", flush=True
        )

        # ── Letterbox Preprocessing ──────────────────────────────
        input_tensor_canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        rescale_factor      = min(640 / frame_w, 640 / frame_h)
        scaled_w            = int(frame_w * rescale_factor)
        scaled_h            = int(frame_h * rescale_factor)

        resized_frame = cv2.resize(raw_frame, (scaled_w, scaled_h),
                                   interpolation=cv2.INTER_LINEAR)
        resized_rgb   = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        x_pad_offset  = (640 - scaled_w) // 2
        y_pad_offset  = (640 - scaled_h) // 2
        input_tensor_canvas[
            y_pad_offset : y_pad_offset + scaled_h,
            x_pad_offset : x_pad_offset + scaled_w
        ] = resized_rgb

        # ── OpenVINO Inference ───────────────────────────────────
        blob = (
            input_tensor_canvas
            .transpose(2, 0, 1)
            .reshape(1, 3, 640, 640)
            .astype(np.float32) / 255.0
        )
        inference_request.infer({input_tensor_node: blob})
        raw_outputs = inference_request.get_output_tensor(0).data

        # ── Vectorized Post-Processing ───────────────────────────
        detections_matrix = np.squeeze(raw_outputs)
        confidence_mask   = detections_matrix[4, :] > CONF_THRESHOLD
        filtered_detections = detections_matrix[:, confidence_mask].T

        roi_count, outside_count = 0, 0
        bounding_boxes, confidences = [], []
        nms_indices = []

        if len(filtered_detections) > 0:
            for object_row in filtered_detections:
                cx_pixel, cy_pixel, w_pixel, h_pixel = object_row[0:4]
                orig_cx    = int((cx_pixel - x_pad_offset) / rescale_factor)
                orig_cy    = int((cy_pixel - y_pad_offset) / rescale_factor)
                orig_w_box = int(w_pixel   / rescale_factor)
                orig_h_box = int(h_pixel   / rescale_factor)

                bounding_boxes.append([
                    int(orig_cx - orig_w_box / 2),
                    int(orig_cy - orig_h_box / 2),
                    orig_w_box, orig_h_box
                ])
                confidences.append(float(object_row[4]))

            nms_indices = cv2.dnn.NMSBoxes(
                bounding_boxes, confidences,
                CONF_THRESHOLD, NMS_THRESHOLD
            )

            if len(nms_indices) > 0:
                for idx_nms in nms_indices.flatten():
                    bx, by, bw, bh = bounding_boxes[idx_nms]

                    # ── ROI Point Test [CHANGE #3 — UNCHANGED] ──
                    # Evaluates 5 strategic points per bounding box.
                    # ANY point inside polygon → passenger counts as ROI.
                    # This is intentionally more inclusive than centroid-only.
                    box_evaluation_points = [
                        (int(bx),          int(by)),
                        (int(bx + bw),     int(by)),
                        (int(bx + bw),     int(by + bh)),
                        (int(bx),          int(by + bh)),
                        (int(bx + bw / 2), int(by + bh / 2))   # centroid
                    ]
                    if any(
                        cv2.pointPolygonTest(ROI_VERTICES, pt, False) >= 0
                        for pt in box_evaluation_points
                    ):
                        roi_count += 1
                    else:
                        outside_count += 1

        current_capacity_state = compute_capacity_state(roi_count, MAX_CAPACITY)

        telemetry_roi_counts.append(roi_count)
        telemetry_capacity_states.append(current_capacity_state)

        # ── EVENT-DRIVEN CSV WRITE (unchanged — only on ROI change) ─
        if roi_count != last_recorded_roi_count:
            save_atomic_csv_row(csv_raw_filename, [
                video_filename, frame_idx, rel_msec,
                roi_count, outside_count,
                current_capacity_state, current_hardware_temp
            ])
            last_recorded_roi_count = roi_count

        # ── Debug Frame Output ───────────────────────────────────
        if DEBUG_MODE and (frame_idx % DEBUG_SKIP_FRAMES == 0):
            debug_canvas = raw_frame.copy()
            cv2.polylines(debug_canvas, [ROI_VERTICES],
                          isClosed=True, color=(180, 180, 180), thickness=2)

            if len(filtered_detections) > 0 and len(nms_indices) > 0:
                for idx_nms in nms_indices.flatten():
                    bx, by, bw, bh = bounding_boxes[idx_nms]
                    box_pts = [
                        (int(bx), int(by)), (int(bx + bw), int(by)),
                        (int(bx + bw), int(by + bh)), (int(bx), int(by + bh)),
                        (int(bx + bw / 2), int(by + bh / 2))
                    ]
                    color = (0, 215, 0) if any(
                        cv2.pointPolygonTest(ROI_VERTICES, pt, False) >= 0
                        for pt in box_pts
                    ) else (0, 140, 255)
                    thickness = 2 if color == (0, 215, 0) else 1
                    cv2.rectangle(debug_canvas,
                                  (bx, by), (bx + bw, by + bh),
                                  color, thickness)

            hud_overlay = debug_canvas.copy()
            cv2.rectangle(hud_overlay, (0, 0), (frame_w, 45), (20, 20, 20), -1)
            cv2.addWeighted(hud_overlay, 0.6, debug_canvas, 0.4, 0, debug_canvas)

            hud_string = (
                f"VID SINGLE | FRAME: {frame_idx:05d} | "
                f"ROI: {roi_count} | SEMAPHORE: {current_capacity_state} | "
                f"TEMP: {current_hardware_temp:.1f}C"
            )
            cv2.putText(debug_canvas, hud_string,
                        (20, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.imwrite(
                os.path.join(output_debug_dir, f"frame_{frame_idx:06d}.jpg"),
                debug_canvas
            )

    # Release stream OUTSIDE the while loop
    stream.release_resources()

    video_net_duration = time.time() - video_run_start_time
    video_net_fps      = (video_frames_processed / video_net_duration
                          if video_net_duration > 0 else 0)
    print(f"\n   -> Finished: {video_filename} | "
          f"Inference Throughput: {video_net_fps:.2f} FPS")

except Exception as run_error:
    print(f"\n\n[FATAL ERROR]: {run_error}")

finally:
    video_run_end_time   = time.time()
    total_physical_time  = video_run_end_time - (
        video_run_start_time if video_run_start_time else video_run_end_time
    )
    fps_promedio = (video_frames_processed / total_physical_time
                    if total_physical_time > 0 else 0)

    seconds_per_record         = SKIP_FRAMES / VIDEO_FPS
    tiempo_video_equivalente_seg = video_frames_processed * seconds_per_record
    factor_aceleracion           = (
        tiempo_video_equivalente_seg / total_physical_time
        if total_physical_time > 0 else 0
    )

    pico_max_personas = max(telemetry_roi_counts)  if telemetry_roi_counts  else 0
    promedio_personas = (sum(telemetry_roi_counts) / len(telemetry_roi_counts)
                         if telemetry_roi_counts else 0)

    estado_predominante = "N/A"
    if telemetry_capacity_states:
        conteo_estados      = Counter(telemetry_capacity_states)
        estado_predominante = conteo_estados.most_common(1)[0][0]

    temp_max  = max(telemetry_temperatures)  if telemetry_temperatures else 0.0
    temp_prom = (sum(telemetry_temperatures) / len(telemetry_temperatures)
                 if telemetry_temperatures else 0.0)

    # ─────────────────────────────────────────────────────────
    # POST-RUN 5-MINUTE BLOCK AGGREGATION
    # ─────────────────────────────────────────────────────────
    total_sequence_records = len(telemetry_roi_counts)
    if total_sequence_records > 0:
        print(f"\n[ANALYTICS] Building 5-min blocks → {csv_summary_filename}...")
        try:
            records_per_5min_block = int(300 / seconds_per_record)
            if records_per_5min_block < 1:
                records_per_5min_block = 1

            with open(csv_summary_filename, mode='w', newline='') as f_summary:
                summary_writer = csv.writer(f_summary)
                summary_writer.writerow([
                    'Interval_Minutes', 'ROI_Count_Average', 'ROI_Count_Max_Peak',
                    'ROI_Standard_Deviation', 'Predominant_Capacity_State',
                    'CPU_Temperature_Average'
                ])

                for i in range(0, total_sequence_records, records_per_5min_block):
                    slice_roi    = telemetry_roi_counts[i : i + records_per_5min_block]
                    slice_states = telemetry_capacity_states[i : i + records_per_5min_block]
                    slice_temps  = telemetry_temperatures[
                        i // TEMP_SAMPLE_EVERY_N_FRAMES :
                        (i + records_per_5min_block) // TEMP_SAMPLE_EVERY_N_FRAMES
                    ]

                    if not slice_roi:
                        continue

                    virtual_minute_stamp = int((i * seconds_per_record) / 60)
                    avg_roi              = sum(slice_roi) / len(slice_roi)
                    max_peak_roi         = max(slice_roi)
                    variance             = sum((x - avg_roi) ** 2 for x in slice_roi) / len(slice_roi)
                    std_deviation        = variance ** 0.5

                    state_freq           = Counter(slice_states)
                    dominant_state       = state_freq.most_common(1)[0][0]
                    avg_temperature      = (sum(slice_temps) / len(slice_temps)
                                            if slice_temps else 0.0)

                    summary_writer.writerow([
                        virtual_minute_stamp,
                        round(avg_roi, 2),
                        max_peak_roi,
                        round(std_deviation, 2),
                        dominant_state,
                        round(avg_temperature, 1)
                    ])

            print("   ✔ 5-minute statistical blocks written successfully.")
        except Exception as summary_error:
            print(f"   ⚠ Summary generation failed: {summary_error}")
    else:
        print("\n[ANALYTICS] Insufficient data to generate summary profile.")

    # ─────────────────────────────────────────────────────────
    # CONSOLIDATED FINAL REPORT
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" [CONSOLIDATED SUMMARY REPORT — V8]")
    print("=" * 80)
    print(f" Target Video File              : {video_filename}")
    print(f" Total Physical Processing Time : {total_physical_time:.2f}s "
          f"({total_physical_time/60:.2f} min)")
    print(f" Virtual Video Scope Monitored  : {tiempo_video_equivalente_seg:.2f}s "
          f"({tiempo_video_equivalente_seg/60:.2f} min)")
    print(f" Acceleration Factor            : {factor_aceleracion:.2f}x vs real time")
    print("\n (INPUT SCHEMA)")
    print(f"  -> Model        : YOLO11n Intel OpenVINO FP32")
    print(f"  -> OV Threads   : {OV_NUM_THREADS} cores | Streams: {OV_NUM_STREAMS} | THROUGHPUT mode")
    print(f"  -> SKIP_FRAMES  : {SKIP_FRAMES} → 1 inference/video second")
    print(f"  -> Temp Sampling: every {TEMP_SAMPLE_EVERY_N_FRAMES} frames "
          f"({int(TEMP_SAMPLE_EVERY_N_FRAMES * seconds_per_record)}s of video)")
    print(f"  -> Inferences   : {video_frames_processed} cycles executed")
    print("\n (TRAFFIC METRICS)")
    print(f"  -> Peak Occupancy    : {pico_max_personas} passengers in ROI")
    print(f"  -> Average Occupancy : {promedio_personas:.2f} pax/second")
    print(f"  -> Dominant State    : {estado_predominante}")
    print("\n (HARDWARE PROFILE — RPi5)")
    print(f"  -> Net Throughput    : {fps_promedio:.2f} FPS")
    print(f"  -> SoC Temp Average  : {temp_prom:.1f} C")
    print(f"  -> SoC Temp Peak     : {temp_max:.1f} C")
    print(f"  -> RAW Log           : {csv_raw_filename}")
    print(f"  -> Summary Profile   : {csv_summary_filename}")
    print("=" * 80)