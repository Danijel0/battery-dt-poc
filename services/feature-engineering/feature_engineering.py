"""
Feature Engineering Service
=============================
Data access abstraction layer per Architecture_draft.pdf Section 3.2.3:
  "Services do not access InfluxDB directly; all queries
   are routed through the Feature Engineering API."

Provides:
  GET /features/soc/{vehicle_id}
      → 60-second window [voltage, current, temperature] at 1Hz
      → shape: (60, 4)

  GET /features/soh/{vehicle_id}
      → 30-day history: cycle count, capacity fade, temp stats
      → dict of aggregated features

  GET /features/energy/{vehicle_id}
      → Current SoC, recent temperature for energy forecasting

Technology: FastAPI, InfluxDB client, NumPy/Pandas
Per Section 3.2.3: time-series windowing,
derived features, feature selection
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from prometheus_client import Histogram, Counter, make_asgi_app

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("feature-engineering")

# ── Config ────────────────────────────────────────────────────────────────────
INFLUX_URL     = os.environ["INFLUX_URL"]
INFLUX_TOKEN   = os.environ["INFLUX_TOKEN"]
INFLUX_ORG     = os.environ["INFLUX_ORG"]
INFLUX_BUCKET  = os.environ.get("INFLUX_BUCKET_TELEMETRY", "telemetry")

# ── Prometheus ────────────────────────────────────────────────────────────────
query_latency = Histogram("feateng_query_latency_seconds", "InfluxDB query latency", ["endpoint"])
query_errors  = Counter("feateng_query_errors_total", "Query errors", ["endpoint"])


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Battery DT — Feature Engineering Service",
    version="0.1.0",
    description="Data access abstraction and ML feature preparation",
)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

_influx: Optional[InfluxDBClientAsync] = None


@app.on_event("startup")
async def startup():
    global _influx
    _influx = InfluxDBClientAsync(
        url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG,
        timeout=120_000
    )
    log.info("InfluxDB client initialised: %s", INFLUX_URL)


@app.on_event("shutdown")
async def shutdown():
    if _influx:
        await _influx.close()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "feature-engineering"}


# ── SoC features: 60s window ──────────────────────────────────────────────────
@app.get("/features/soc/{vehicle_id}")
async def get_soc_features(
    vehicle_id: str,
    window_seconds: int = Query(default=300, ge=10, le=600),
):
    """
    Returns raw (window_seconds × 4) array for CNN-LSTM input.
    Columns: [voltage, current, temperature, chem_id]
    """
    with query_latency.labels("soc").time():
        try:
            query_api = _influx.query_api()
            flux = f"""
                from(bucket: "{INFLUX_BUCKET}")
                  |> range(start: -{window_seconds}s)
                  |> filter(fn: (r) => r._measurement == "battery_telemetry")
                  |> filter(fn: (r) => r.vehicle_id == "{vehicle_id}")
                  |> filter(fn: (r) => r._field == "voltage" or
                                       r._field == "current" or
                                       r._field == "temperature" or
                                       r._field == "chem_id")
                  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
                  |> sort(columns: ["_time"])
            """
            tables = await query_api.query(flux, org=INFLUX_ORG)
        except Exception as e:
            query_errors.labels("soc").inc()
            log.error("InfluxDB SoC query failed: %s", e)
            raise HTTPException(status_code=503, detail="Database query failed")

    records = []
    for table in tables:
        for record in table.records:
            records.append({
                "time":        record.get_time(),
                "voltage":     record.values.get("voltage",     np.nan),
                "current":     record.values.get("current",     np.nan),
                "temperature": record.values.get("temperature", np.nan),
                "chem_id":     record.values.get("chem_id",     0.0),
            })

    if len(records) < 10:
        raise HTTPException(
            status_code=404,
            detail=f"Insufficient telemetry for vehicle {vehicle_id} (got {len(records)} points, need ≥10)",
        )

    df = pd.DataFrame(records).set_index("time").sort_index()
    df = df.resample("1s").mean().interpolate(method="time").tail(window_seconds)
    chem_id = float(df["chem_id"].dropna().iloc[0]) if "chem_id" in df.columns and not df["chem_id"].dropna().empty else 0.0

    # Return raw values — normalisation done in soc-service using training stats
    chem_col = np.full(len(df), chem_id)
    window = np.column_stack([
        df["voltage"].values,
        df["current"].values,
        df["temperature"].values,
        chem_col,
    ])

    return {
        "vehicle_id":    vehicle_id,
        "window_seconds": window_seconds,
        "shape":         list(window.shape),
        "features":      window.tolist(),
        "feature_names": ["voltage", "current", "temperature", "chem_id"],
        "chem_id":       chem_id,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }



# Vehicle SoC target policy (operational minimum SoC).
# In production this would be retrieved from a vehicle metadata store.
# For PoC: configurable per vehicle via SOC_TARGETS env var (JSON dict)
# e.g. SOC_TARGETS={"HV-001": 0.1, "HV-002": 0.2}
import json as _json
_SOC_TARGETS: dict = _json.loads(os.environ.get("SOC_TARGETS", "{}"))
_SOC_TARGET_DEFAULT: float = float(os.environ.get("SOC_TARGET_DEFAULT", "0.1"))

def _get_soc_target(vehicle_id: str) -> float:
    return _SOC_TARGETS.get(vehicle_id, _SOC_TARGET_DEFAULT)


# ── SoH features: 30-day aggregated history ───────────────────────────────────
@app.get("/features/soh/{vehicle_id}")
async def get_soh_features(vehicle_id: str):
    """
    Returns aggregated features over the last 30 days for GPR SoH estimation.
    Reads telemetry from InfluxDB.
    """
    with query_latency.labels("soh").time():
        try:
            query_api = _influx.query_api()

            # Query telemetry
            flux_telemetry = f"""
                from(bucket: "telemetry")
                  |> range(start: -30d)
                  |> filter(fn: (r) => r._measurement == "battery_telemetry")
                  |> filter(fn: (r) => r.vehicle_id == "{vehicle_id}")
                  |> filter(fn: (r) => r._field == "voltage" or
                                       r._field == "current" or
                                       r._field == "temperature" or
                                       r._field == "chem_id")
                  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
            """
            tables_t = await query_api.query(flux_telemetry, org=INFLUX_ORG)

        except Exception as e:
            query_errors.labels("soh").inc()
            raise HTTPException(status_code=503, detail="Database query failed")

    # Parse telemetry
    tel_records = [
        {
            "voltage":     r.values.get("voltage"),
            "current":     r.values.get("current"),
            "temperature": r.values.get("temperature"),
            "chem_id":     r.values.get("chem_id"),
        }
        for table in tables_t for r in table.records
    ]

    if not tel_records:
        raise HTTPException(status_code=404,
                            detail=f"No telemetry for vehicle {vehicle_id}")

    df_t = pd.DataFrame(tel_records).dropna(subset=["voltage", "current", "temperature"])

    # chem_id — take first non-null value (constant per vehicle)
    chem_id = float(df_t["chem_id"].dropna().iloc[0]) if "chem_id" in df_t.columns and not df_t["chem_id"].dropna().empty else 0.0

    # Chemistry-specific parameters
    nominal_cap = 5.0 if chem_id == 0.0 else 2.3   # Chen2020=5Ah, Prada2013=2.3Ah
    i_nominal   = 5.0 if chem_id == 0.0 else 2.3   # 1C current per chemistry

    # Active discharge only — exclude near-zero current (rest periods)
    # so c_rate and current_mean match training scenario conditions
    active_mask = abs(df_t["current"]) > 0.1
    df_active   = df_t[active_mask] if active_mask.sum() > 10 else df_t

    features = {
        "vehicle_id":      vehicle_id,
        # GPR SoH features — must match SOH_FEATURES in train_soh_model.py v3:
        # [c_rate, temp_mean_c, temp_min_c, temp_max_c,
        #  current_mean_a, soc_target, nominal_cap_ah]
        "c_rate":          float(abs(df_active["current"]).mean() / i_nominal),
        "temp_mean_c":     float(df_active["temperature"].mean()),
        "temp_min_c":      float(df_t["temperature"].min()),
        "temp_max_c":      float(df_t["temperature"].max()),
        "current_mean_a":  float(abs(df_active["current"]).mean()),
        "soc_target":      _get_soc_target(vehicle_id),  # soc_mean removed
        "nominal_cap_ah":  nominal_cap,
        # chem_id retained for chemistry routing in soh-service (not passed to GPR)
        "chem_id":         chem_id,
        "n_observations":  len(df_t),
        "timestamp":       datetime.now(timezone.utc).isoformat()
    }
    return features


# ── Energy features: current vehicle state ───────────────────────────────────
@app.get("/features/energy/{vehicle_id}")
async def get_energy_features(vehicle_id: str):
    """
    Returns current vehicle state for energy consumption forecasting.
    Per Section 4.3: current SoC (from InfluxDB soc-estimates bucket) + temperature.
    """
    with query_latency.labels("energy").time():
        try:
            query_api = _influx.query_api()
            flux = f"""
                from(bucket: "{INFLUX_BUCKET}")
                  |> range(start: -5m)
                  |> filter(fn: (r) => r._measurement == "battery_telemetry")
                  |> filter(fn: (r) => r.vehicle_id == "{vehicle_id}")
                  |> filter(fn: (r) => r._field == "temperature" or r._field == "voltage")
                  |> last()
                  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
            """
            tables = await query_api.query(flux, org=INFLUX_ORG)
        except Exception as e:
            query_errors.labels("energy").inc()
            raise HTTPException(status_code=503, detail="Database query failed")

    latest = {}
    for table in tables:
        for record in table.records:
            latest.update(record.values)

    if not latest:
        raise HTTPException(status_code=404, detail=f"No recent data for vehicle {vehicle_id}")

    return {
        "vehicle_id":      vehicle_id,
        "temperature_c":   latest.get("temperature"),
        "voltage_v":       latest.get("voltage"),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }