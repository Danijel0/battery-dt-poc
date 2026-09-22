#!/usr/bin/env python3
"""
PoC Fleet Simulator — MQTT Publisher
======================================
Simulates N vehicles publishing BMS telemetry to Eclipse Mosquitto.
Topic: dt/telemetry/{vehicle_id}
QoS: 1

Changes from v1:
  - SoC resets to random(0.7, 1.0) when it falls below soc_min (simulates
    opportunity charging at depot — realistic for urban bus operation)
  - Temperature follows a smooth random walk instead of uniform sampling
    each second. Abrupt per-second temperature jumps are not physical and
    degrade CNN-LSTM SoC prediction quality.
  - Discharge rate varies with C-rate drawn from training distribution
    [0.5, 0.75, 1.0, 1.5, 2.0] to match PyBaMM training scenarios.

Usage:
    pip install paho-mqtt
    python simulate_fleet.py --vehicles 10 --hz 1 --duration 300
"""

import argparse
import json
import random
import statistics
import time
import uuid
from datetime import datetime, timezone
from threading import Thread, Lock

import paho.mqtt.client as mqtt

MQTT_HOST = "localhost"
MQTT_PORT = 1883
MQTT_QOS  = 1

_stats_lock = Lock()
_stats = {"ok": 0, "errors": 0, "latencies": []}

CHEMISTRY_PROFILES = {
    0: {  # NMC — Chen2020
        "name":       "NMC",
        "v_full":     4.2,
        "v_empty":    2.5,
        "i_nominal":  5.0,
        "temp_range": (-10.0, 45.0),
    },
    1: {  # LFP — Prada2013
        "name":       "LFP",
        "v_full":     3.6,
        "v_empty":    2.0,
        "i_nominal":  2.3,
        "temp_range": (-10.0, 45.0),
    },
}

# C-rates matching PyBaMM training scenarios
TRAINING_C_RATES = [0.5, 0.75, 1.0, 1.5, 2.0]

# Minimum SoC before simulating opportunity charge (depot recharge)
SOC_MIN = 0.10


# Piecewise linear OCV approximations derived from PyBaMM Chen2020 (NMC)
# and Prada2013 (LFP) discharge curves.
# These breakpoints ensure simulated voltage-SoC relationship matches
# training data, so CNN-LSTM predictions are physically consistent.
OCV_BREAKPOINTS = {
    0: [  # NMC — Chen2020: strongly nonlinear
        (0.00, 2.50), (0.05, 3.00), (0.10, 3.50), (0.20, 3.70),
        (0.40, 3.85), (0.60, 3.95), (0.80, 4.05), (0.90, 4.12),
        (1.00, 4.20),
    ],
    1: [  # LFP — Prada2013: flat plateau between 20-90% SoC
        (0.00, 2.00), (0.05, 2.80), (0.10, 3.10), (0.20, 3.20),
        (0.80, 3.30), (0.90, 3.40), (0.95, 3.50), (1.00, 3.60),
    ],
}


def ocv_from_soc(soc: float, chem_id: int) -> float:
    """Piecewise linear OCV interpolation."""
    pts = OCV_BREAKPOINTS[chem_id]
    soc = max(0.0, min(1.0, soc))
    for i in range(len(pts) - 1):
        s0, v0 = pts[i]
        s1, v1 = pts[i + 1]
        if s0 <= soc <= s1:
            return v0 + (v1 - v0) * (soc - s0) / (s1 - s0)
    return pts[-1][1]


def generate_telemetry(vehicle_id: str, battery_id: str,
                       soc: float, chem_id: int,
                       c_rate: float, temp: float) -> dict:
    profile = CHEMISTRY_PROFILES[chem_id]
    # Nonlinear OCV + small measurement noise
    voltage  = ocv_from_soc(soc, chem_id) + random.gauss(0, 0.005)
    current  = profile["i_nominal"] * c_rate + random.gauss(0, 0.05)

    return {
        "vehicle_id":  vehicle_id,
        "battery_id":  battery_id,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "voltage":     round(voltage, 4),
        "current":     round(current, 4),
        "temperature": round(temp, 2),
        "chem_id":     chem_id,
    }


def vehicle_thread(vehicle_id: str, battery_id: str,
                   hz: float, duration: float, chem_id: int):
    client = mqtt.Client(client_id=f"sim-{vehicle_id}", clean_session=True)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    topic    = f"dt/telemetry/{vehicle_id}"
    interval = 1.0 / hz
    profile  = CHEMISTRY_PROFILES[chem_id]

    # Initial state
    soc     = random.uniform(0.7, 1.0)
    c_rate  = random.choice(TRAINING_C_RATES)
    t_min, t_max = profile["temp_range"]
    temp    = random.uniform(15.0, 35.0)   # start at operational temp

    # SoC discharge rate: c_rate * i_nominal * (1/3600) per second
    # For 1Hz: soc_drop_per_step = c_rate / (capacity_Ah * 3600)
    # Simplified: use fraction of capacity per second
    nominal_cap = profile["i_nominal"]  # Ah
    t_end = time.monotonic() + duration

    while time.monotonic() < t_end:
        payload = generate_telemetry(vehicle_id, battery_id,
                                     soc, chem_id, c_rate, temp)

        # SoC discharge — proportional to C-rate
        soc_drop = (c_rate * profile["i_nominal"]) / (nominal_cap * 3600) / hz
        soc = max(0.0, soc - soc_drop)

        # Opportunity charging: reset SoC when below minimum
        # Simulates bus returning to depot for charging
        if soc < SOC_MIN:
            soc    = random.uniform(0.7, 1.0)
            c_rate = random.choice(TRAINING_C_RATES)

        # Temperature random walk: ±0.5°C per second, bounded
        temp = float(max(t_min, min(t_max, temp + random.gauss(0, 0.3))))

        t0     = time.monotonic()
        result = client.publish(topic, json.dumps(payload), qos=MQTT_QOS)
        result.wait_for_publish(timeout=5.0)
        latency_ms = (time.monotonic() - t0) * 1000

        with _stats_lock:
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                _stats["ok"] += 1
                _stats["latencies"].append(latency_ms)
            else:
                _stats["errors"] += 1

        time.sleep(interval)

    client.loop_stop()
    client.disconnect()


def run(n_vehicles: int, hz: float, duration: float):
    vehicles = [
        (
            f"HV-{str(uuid.uuid4())[:6].upper()}",
            f"BAT-{str(uuid.uuid4())[:6].upper()}",
            i % 2,
        )
        for i in range(n_vehicles)
    ]

    print(f"\nBattery DT PoC — MQTT Fleet Simulator")
    print(f"  Vehicles   : {n_vehicles} ({n_vehicles//2} NMC + {n_vehicles - n_vehicles//2} LFP)")
    print(f"  Rate       : {hz} Hz per vehicle")
    print(f"  Duration   : {duration}s")
    print(f"  Broker     : {MQTT_HOST}:{MQTT_PORT}\n")

    threads = [
        Thread(target=vehicle_thread,
               args=(vid, bid, hz, duration, chem_id),
               daemon=True)
        for vid, bid, chem_id in vehicles
    ]

    for i, (vid, bid, chem_id) in enumerate(vehicles):
        chem = CHEMISTRY_PROFILES[chem_id]["name"]
        print(f"  Vehicle {i+1:02d}: {vid} ({chem})")

    t_start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.monotonic() - t_start
    lats    = sorted(_stats["latencies"])
    total   = _stats["ok"] + _stats["errors"]

    def pct(p):
        if not lats: return 0
        return lats[min(int(len(lats)*p/100), len(lats)-1)]

    print(f"── Results ──────────────────────────────────────────")
    print(f"  Total      : {total}")
    print(f"  Success    : {_stats['ok']}  ({100*_stats['ok']/max(total,1):.1f}%)")
    print(f"  Errors     : {_stats['errors']}")
    if lats:
        print(f"  Lat p50    : {pct(50):.1f} ms")
        print(f"  Lat p95    : {pct(95):.1f} ms")
        print(f"  Lat p99    : {pct(99):.1f} ms")
        print(f"  Lat mean   : {statistics.mean(lats):.1f} ms")
    print(f"  Throughput : {_stats['ok']/elapsed:.1f} msg/s")
    print(f"────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicles", type=int,   default=10)
    parser.add_argument("--hz",       type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--host",     type=str,   default="localhost")
    parser.add_argument("--port",     type=int,   default=1883)
    args = parser.parse_args()

    MQTT_HOST = args.host
    MQTT_PORT = args.port
    run(args.vehicles, args.hz, args.duration)
