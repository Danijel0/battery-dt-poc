"""
Historical Telemetry Batch Injector
=====================================
Simulates 30 days of realistic battery telemetry for a fleet of vehicles
and injects directly into InfluxDB with historical timestamps.

Sampling: 1 point per minute (vs 1Hz realtime) = 43,200 points/vehicle/30days
Realistic daily cycle:
  - 06:00-22:00: operational (discharge cycles, partial recharge at midday)
  - 22:00-06:00: overnight full recharge
  - Weekend: reduced operation

Per Architecture_draft.pdf Section 1.3:
  Target: 10-20 vehicles, 30-day history for meaningful SoH estimation
"""

import os
import math
import random
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ── Config ────────────────────────────────────────────────────────────────────
INFLUX_URL   = os.environ.get("INFLUX_URL",   "http://localhost:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "dt-super-secret-token")
INFLUX_ORG   = os.environ.get("INFLUX_ORG",   "battery-dt")
BUCKET       = "telemetry"
SOC_BUCKET   = "soc-estimates"

SAMPLE_INTERVAL_MIN = 1      # 1 point per minute
DAYS                = 30
BATCH_SIZE          = 5000   # points per InfluxDB write batch

# Chemistry profiles (aligned with training data)
CHEMISTRY_PROFILES = {
    0: {  # NMC — Chen2020
        "name":      "NMC",
        "v_full":    4.2,
        "v_empty":   2.5,
        "i_nominal": 5.0,    # 1C = 5A
        "capacity":  5.0,    # Ah
    },
    1: {  # LFP — Prada2013
        "name":      "LFP",
        "v_full":    3.6,
        "v_empty":   2.0,
        "i_nominal": 2.3,    # 1C = 2.3A
        "capacity":  2.3,    # Ah
    },
}


def get_operating_mode(hour: int, day_of_week: int) -> str:
    """Determine vehicle operating mode based on time of day and day of week."""
    is_weekend = day_of_week >= 5
    if is_weekend:
        if 8 <= hour < 18:
            return "reduced"    # weekend reduced operation
        else:
            return "charging"
    else:
        if 6 <= hour < 10:
            return "morning_peak"   # high discharge
        elif 10 <= hour < 12:
            return "cruising"       # moderate discharge
        elif 12 <= hour < 13:
            return "midday_charge"  # partial recharge at depot
        elif 13 <= hour < 17:
            return "afternoon"      # moderate discharge
        elif 17 <= hour < 20:
            return "evening_peak"   # high discharge
        elif 20 <= hour < 22:
            return "return"         # low discharge, returning to depot
        else:
            return "charging"       # overnight charging


def simulate_step(soc: float, mode: str, chem_id: int,
                  temp_base: float, dt_min: float) -> tuple[float, dict]:
    """
    Simulate one timestep of battery operation.
    Returns (new_soc, telemetry_dict).
    """
    profile  = CHEMISTRY_PROFILES[chem_id]
    capacity = profile["capacity"]   # Ah
    dt_h     = dt_min / 60.0         # hours

    # Current based on operating mode
    if mode == "charging":
        # CCCV charging: 0.5C until full
        current  = -profile["i_nominal"] * 0.5   # negative = charging
        dsoc     = abs(current) * dt_h / capacity
        soc      = min(1.0, soc + dsoc)
    elif mode == "midday_charge":
        current  = -profile["i_nominal"] * 0.3
        dsoc     = abs(current) * dt_h / capacity
        soc      = min(0.85, soc + dsoc)   # partial charge to 85%
    elif mode == "morning_peak" or mode == "evening_peak":
        c_rate   = random.uniform(1.5, 2.0)
        current  = profile["i_nominal"] * c_rate
        dsoc     = current * dt_h / capacity
        soc      = max(0.1, soc - dsoc)
    elif mode == "cruising" or mode == "afternoon":
        c_rate   = random.uniform(0.8, 1.2)
        current  = profile["i_nominal"] * c_rate
        dsoc     = current * dt_h / capacity
        soc      = max(0.1, soc - dsoc)
    elif mode == "return":
        c_rate   = random.uniform(0.3, 0.6)
        current  = profile["i_nominal"] * c_rate
        dsoc     = current * dt_h / capacity
        soc      = max(0.15, soc - dsoc)
    else:  # reduced
        c_rate   = random.uniform(0.5, 1.0)
        current  = profile["i_nominal"] * c_rate
        dsoc     = current * dt_h / capacity
        soc      = max(0.1, soc - dsoc)

    # Voltage from SoC (simplified OCV curve)
    v_range  = profile["v_full"] - profile["v_empty"]
    voltage  = profile["v_empty"] + soc * v_range + random.gauss(0, 0.005)

    # Temperature: rises during high current, ambient overnight
    if mode == "charging":
        temp = temp_base + random.gauss(2, 0.5)
    elif mode in ("morning_peak", "evening_peak"):
        temp = temp_base + random.gauss(8, 1.0)
    else:
        temp = temp_base + random.gauss(4, 0.8)

    return soc, {
        "voltage":     round(float(voltage),  4),
        "current":     round(float(current),  4),
        "temperature": round(float(temp),     2),
        "chem_id":     float(chem_id),
    }


def inject_vehicle(client: InfluxDBClient, vehicle_id: str,
                   battery_id: str, chem_id: int,
                   days: int, temp_base: float):
    """Inject historical telemetry for one vehicle."""
    write_api = client.write_api(write_options=SYNCHRONOUS)
    profile   = CHEMISTRY_PROFILES[chem_id]

    now       = datetime.now(timezone.utc)
    start     = now - timedelta(days=days)
    soc       = random.uniform(0.85, 1.0)   # start near full

    tel_points = []
    soc_points = []
    total      = 0

    t = start
    while t <= now:
        hour        = t.hour
        day_of_week = t.weekday()
        mode        = get_operating_mode(hour, day_of_week)

        soc, fields = simulate_step(soc, mode, chem_id, temp_base, SAMPLE_INTERVAL_MIN)

        # Telemetry point
        tel_points.append(
            Point("battery_telemetry")
            .tag("vehicle_id",  vehicle_id)
            .tag("battery_id",  battery_id)
            .field("voltage",     fields["voltage"])
            .field("current",     fields["current"])
            .field("temperature", fields["temperature"])
            .field("chem_id",     fields["chem_id"])
            .time(t, WritePrecision.S)
        )

        # SoC estimate point (simulating what soc-service would write)
        soc_points.append(
            Point("soc_estimates")
            .tag("vehicle_id", vehicle_id)
            .field("soc",     soc)
            .field("chem_id", float(chem_id))
            .time(t, WritePrecision.S)
        )

        total += 1

        # Batch write
        if len(tel_points) >= BATCH_SIZE:
            write_api.write(bucket=BUCKET,     org=INFLUX_ORG, record=tel_points)
            write_api.write(bucket=SOC_BUCKET, org=INFLUX_ORG, record=soc_points)
            tel_points = []
            soc_points = []
            print(f"    {vehicle_id}: {total} points written ...", end="\r")

        t += timedelta(minutes=SAMPLE_INTERVAL_MIN)

    # Write remaining
    if tel_points:
        write_api.write(bucket=BUCKET,     org=INFLUX_ORG, record=tel_points)
        write_api.write(bucket=SOC_BUCKET, org=INFLUX_ORG, record=soc_points)

    print(f"    {vehicle_id} ({profile['name']}): {total} points written ✓")
    return total


def main(n_vehicles: int, days: int):
    print(f"Battery DT — Historical Telemetry Injector")
    print(f"  Vehicles : {n_vehicles}")
    print(f"  Days     : {days}")
    print(f"  Interval : {SAMPLE_INTERVAL_MIN} min")
    print(f"  Points   : ~{n_vehicles * days * 24 * 60 // SAMPLE_INTERVAL_MIN:,} total\n")

    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    # Define fleet — alternate NMC/LFP
    fleet = []
    for i in range(n_vehicles):
        chem_id    = i % 2
        profile    = CHEMISTRY_PROFILES[chem_id]
        vehicle_id = f"HIST-{profile['name']}-{i+1:02d}"
        battery_id = f"BAT-{profile['name']}-{i+1:02d}"
        temp_base  = random.uniform(15.0, 35.0)   # ambient temp per vehicle
        fleet.append((vehicle_id, battery_id, chem_id, temp_base))

    print("Fleet:")
    for vid, bid, chem_id, temp in fleet:
        print(f"  {vid}  T_base={temp:.1f}°C")
    print()

    grand_total = 0
    for vehicle_id, battery_id, chem_id, temp_base in fleet:
        print(f"  Injecting {vehicle_id} ...")
        n = inject_vehicle(client, vehicle_id, battery_id,
                           chem_id, days, temp_base)
        grand_total += n

    client.close()
    print(f"\nTotal points written: {grand_total:,}")
    print(f"\nVehicle IDs for testing:")
    for vid, _, chem_id, _ in fleet:
        print(f"  {vid}  ({CHEMISTRY_PROFILES[chem_id]['name']})")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicles", type=int,   default=4)
    parser.add_argument("--days",     type=int,   default=30)
    args = parser.parse_args()
    main(args.vehicles, args.days)