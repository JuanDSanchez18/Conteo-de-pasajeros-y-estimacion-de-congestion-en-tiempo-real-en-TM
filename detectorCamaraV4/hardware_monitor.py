# -*- coding: utf-8 -*-
# hardware_monitor.py — Hardware telemetry for Raspberry Pi 5 BCM2712
# Transmilenio Edge AI Engine — v1.0.0-prod
# No state stored here — all buffers live in main.py
# All thresholds sourced from config.py — no hardcoded values

import psutil
import config

# Linux sysfs thermal interface for BCM2712
_THERMAL_ZONE_PATH = "/sys/class/thermal/thermal_zone0/temp"


# ---------------------------------------------------------------------------
# SoC temperature
# ---------------------------------------------------------------------------
def read_soc_temperature() -> float:
    # Raw value is in millidegrees (e.g. 52000 → 52.0°C). Returns 0.0 on failure.
    try:
        with open(_THERMAL_ZONE_PATH, "r") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, IOError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# CPU & RAM
# ---------------------------------------------------------------------------
def read_cpu_usage() -> float:
    # Non-blocking — returns usage since last call. First call may return 0.0.
    return psutil.cpu_percent(interval=None)


def read_ram_usage_gb() -> float:
    # Returns actual application memory pressure (excludes OS buffers/cache).
    return psutil.virtual_memory().used / (1024 ** 3)


def read_system_resource_metrics() -> tuple:
    # Convenience wrapper — returns (cpu_percent, ram_used_gb) in one call.
    return read_cpu_usage(), read_ram_usage_gb()


# ---------------------------------------------------------------------------
# Thermal alert evaluator
# ---------------------------------------------------------------------------
def check_thermal_alerts(temp_celsius: float) -> str:
    # Returns severity string only — printing is handled by main.py.
    if temp_celsius >= config.THERMAL_THROTTLE_THRESHOLD_C:
        return "CRITICAL"
    elif temp_celsius >= config.THERMAL_WARNING_THRESHOLD_C:
        return "WARNING"
    return "OK"


# ---------------------------------------------------------------------------
# Hardware snapshot — composite dict for payloads and CSV summaries
# ---------------------------------------------------------------------------
def get_hardware_snapshot() -> dict:
    # Single call for a consistent point-in-time hardware state.
    temp_c = read_soc_temperature()
    cpu_pct, ram_gb = read_system_resource_metrics()

    return {
        "soc_temp_c"    : round(temp_c,  1),
        "cpu_pct"       : round(cpu_pct, 1),
        "ram_gb"        : round(ram_gb,  3),
        "thermal_status": check_thermal_alerts(temp_c),
    }


# ---------------------------------------------------------------------------
# Buffer aggregation — averages from main.py ring buffers (5-min window)
# ---------------------------------------------------------------------------
def compute_buffer_averages(
    temp_buffer: list,
    cpu_buffer:  list,
    ram_buffer:  list,
) -> dict:
    # Falls back to a live read if any buffer is empty (e.g. first boot window).
    avg_temp = (sum(temp_buffer) / len(temp_buffer)) if temp_buffer else read_soc_temperature()
    avg_cpu  = (sum(cpu_buffer)  / len(cpu_buffer))  if cpu_buffer  else read_cpu_usage()
    avg_ram  = (sum(ram_buffer)  / len(ram_buffer))  if ram_buffer  else read_ram_usage_gb()

    return {
        "avg_temp_c" : round(avg_temp, 1),
        "avg_cpu_pct": round(avg_cpu,  1),
        "avg_ram_gb" : round(avg_ram,  3),
    }