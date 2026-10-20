# -*- coding: utf-8 -*-
# csv_handler.py — CSV lifecycle management
# Transmilenio Edge AI Engine — v1.0.0-prod

import csv
import os
from collections import Counter

import config

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
_RAW_CSV_HEADERS = [
    "Time",
    "Frame_Index",
    "ROI_Count",
    "Outside_Count",
    "Capacity_State",
]

_SUMMARY_CSV_HEADERS = [
    "Timestamp_Window",
    "ROI_Count_Average",
    "ROI_Count_Max_Peak",
    "ROI_Standard_Deviation",
    "Predominant_Capacity_State",
    "CPU_Temperature_Average_C",
    "CPU_Usage_Average_Pct",
    "RAM_Usage_Average_GB",
]


# ---------------------------------------------------------------------------
# Filename generation
# ---------------------------------------------------------------------------
def build_session_filenames(startup_timestamp: str) -> tuple:
    """Returns (raw_csv_path, summary_csv_path) for this session."""
    raw_path = os.path.join(
        config.RESULTS_DIR,
        f"Live_Station_Analytics_{startup_timestamp}_RAW.csv",
    )
    summary_path = os.path.join(
        config.RESULTS_DIR,
        f"Live_Station_Resumen_{startup_timestamp}_5Min.csv",
    )
    return raw_path, summary_path


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def initialize_csv_files(raw_path: str, summary_path: str) -> None:
    """Creates both CSV files with headers if they do not already exist."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if not os.path.exists(raw_path):
        _write_header(raw_path, _RAW_CSV_HEADERS)
        print(f"[CSV] RAW log created     : {raw_path}")

    if not os.path.exists(summary_path):
        _write_header(summary_path, _SUMMARY_CSV_HEADERS)
        print(f"[CSV] SUMMARY log created : {summary_path}")


def _write_header(filepath: str, headers: list) -> None:
    with open(filepath, mode="w", newline="") as f:
        csv.writer(f).writerow(headers)


# ---------------------------------------------------------------------------
# Atomic row append — crash-resilient (fsync on every write)
# ---------------------------------------------------------------------------
def _append_row(filepath: str, row_data: list) -> None:
    with open(filepath, mode="a", newline="") as f:
        csv.writer(f).writerow(row_data)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# RAW row — one per confirmed count change
# ---------------------------------------------------------------------------
def save_raw_detection_row(
    raw_path:       str,
    timestamp_str:  str,
    frame_index:    int,
    roi_count:      int,
    outside_count:  int,
    capacity_state: str,
) -> None:
    _append_row(raw_path, [
        timestamp_str,
        frame_index,
        roi_count,
        outside_count,
        capacity_state,
    ])


# ---------------------------------------------------------------------------
# SUMMARY row — one per 5-minute window
# ---------------------------------------------------------------------------
def save_summary_row(
    summary_path:      str,
    window_timestamp:  str,
    roi_counts_buffer: list,
    states_buffer:     list,
    avg_temp_c:        float,
    avg_cpu_pct:       float,
    avg_ram_gb:        float,
) -> dict:
    """
    Computes window analytics and appends one row to the summary CSV.
    Returns a dict with computed values for main.py console output.
    Returns {} if the buffer is empty (nothing to write).
    """
    if not roi_counts_buffer:
        return {}

    avg_roi  = sum(roi_counts_buffer) / len(roi_counts_buffer)
    max_peak = max(roi_counts_buffer)
    variance = sum((x - avg_roi) ** 2 for x in roi_counts_buffer) / len(roi_counts_buffer)
    std_dev  = variance ** 0.5

    state_counter  = Counter(states_buffer)
    dominant_state = state_counter.most_common(1)[0][0] if state_counter else "N/A"

    _append_row(summary_path, [
        window_timestamp,
        round(avg_roi,  2),
        max_peak,
        round(std_dev,  2),
        dominant_state,
        round(avg_temp_c,  1),
        round(avg_cpu_pct, 1),
        round(avg_ram_gb,  3),
    ])

    return {
        "avg_roi"        : round(avg_roi,  2),
        "max_peak"       : max_peak,
        "std_dev"        : round(std_dev,  2),
        "dominant_state" : dominant_state,
    }