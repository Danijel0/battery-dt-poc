"""
Data Ingestion / Validator Service
===================================
Subscribes to Eclipse Mosquitto on topic dt/telemetry/#
Validates each message (JSON Schema + range + outlier detection),
writes valid telemetry to InfluxDB.

Per Architecture_draft.pdf Section 3.2.1:
  - Schema validation
  - Range validation
  - Outlier detection
  - Missing data handling
  - Real-time, negligible latency (<< 1s total pipeline)

MQTT topic convention: dt/telemetry/{vehicle_id}
Payload (JSON):
  {
    "vehicle_id":    "HV-001",
    "battery_id":    "BAT-001",
    "timestamp":     "2026-03-15T14:30:00Z",
    "voltage":       320.5,   # V
    "current":       150.0,   # A (positive = discharge)
    "temperature":   32.1     # °C
  }
"""

import os
import json
import logging
import time
from datetime import datetime

import paho.mqtt.client as mqtt
import jsonschema
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from prometheus_client import Counter, Histogram, start_http_server

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("data-ingestion")

# ── Config ────────────────────────────────────────────────────────────────────
MQTT_HOST      = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT      = int(os.environ.get("MQTT_PORT", 1883))
MQTT_TOPIC     = os.environ.get("MQTT_TOPIC", "dt/telemetry/#")
MQTT_QOS       = int(os.environ.get("MQTT_QOS", 1))
INFLUX_URL     = os.environ["INFLUX_URL"]
INFLUX_TOKEN   = os.environ["INFLUX_TOKEN"]
INFLUX_ORG     = os.environ["INFLUX_ORG"]
INFLUX_BUCKET  = os.environ["INFLUX_BUCKET"]

# ── JSON Schema (validation) ──────────────────────────────────────────────────
TELEMETRY_SCHEMA = {
    "type": "object",
    "required": ["vehicle_id", "battery_id", "timestamp", "voltage", "current", "temperature", "chem_id"],
    "properties": {
        "vehicle_id":  {"type": "string", "minLength": 1, "maxLength": 64},
        "battery_id":  {"type": "string", "minLength": 1, "maxLength": 64},
        "timestamp":   {"type": "string", "format": "date-time"},
        "voltage":     {"type": "number", "minimum": 0,     "maximum": 1000},
        "current":     {"type": "number", "minimum": -2000, "maximum": 2000},
        "temperature": {"type": "number", "minimum": -40,   "maximum": 100},
        "chem_id": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}

# ── Prometheus metrics ────────────────────────────────────────────────────────
msgs_received     = Counter("ingestion_mqtt_received_total",   "MQTT messages received")
msgs_valid        = Counter("ingestion_valid_total",           "Messages passed validation")
msgs_invalid      = Counter("ingestion_invalid_total",         "Messages failed validation")
msgs_written      = Counter("ingestion_influx_written_total",  "Points written to InfluxDB")
write_errors      = Counter("ingestion_write_errors_total",    "InfluxDB write failures")
pipeline_latency  = Histogram("ingestion_pipeline_latency_seconds", "Validate + write latency")

# ── InfluxDB client (module-level, reused across callbacks) ──────────────────
_influx_client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
_write_api        = _influx_client.write_api(write_options=SYNCHRONOUS)


def validate_message(payload: dict) -> tuple[bool, str]:
    """
    Two-stage validation:
      1. JSON Schema (structure + type + range)
      2. Outlier detection (z-score placeholder — expandable)
    Returns (is_valid, reason).
    """
    try:
        jsonschema.validate(instance=payload, schema=TELEMETRY_SCHEMA)
    except jsonschema.ValidationError as e:
        return False, f"Schema: {e.message}"

    # Basic outlier guard: voltage should not be near zero if current is high
    if payload["voltage"] < 10 and abs(payload["current"]) > 10:
        return False, "Outlier: near-zero voltage with non-zero current"

    return True, "ok"


def write_to_influx(payload: dict):
    """Write validated telemetry point to InfluxDB."""
    ts = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
    point = (
        Point("battery_telemetry")
        .tag("vehicle_id",  payload["vehicle_id"])
        .tag("battery_id",  payload["battery_id"])
        .field("voltage",     payload["voltage"])
        .field("current",     payload["current"])
        .field("temperature", payload["temperature"])
        .field("chem_id", float(payload["chem_id"]))
        .time(ts, WritePrecision.S)
    )
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


# ── MQTT callbacks ────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("Connected to Mosquitto. Subscribing to %s (QoS %d)", MQTT_TOPIC, MQTT_QOS)
        client.subscribe(MQTT_TOPIC, qos=MQTT_QOS)
    else:
        log.error("MQTT connection failed with code %d", rc)


def on_message(client, userdata, msg):
    msgs_received.inc()
    t0 = time.monotonic()

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        msgs_invalid.inc()
        log.warning("Malformed JSON on topic %s: %s", msg.topic, e)
        return

    valid, reason = validate_message(payload)
    if not valid:
        msgs_invalid.inc()
        log.warning("Validation failed [%s]: %s", msg.topic, reason)
        return

    msgs_valid.inc()

    try:
        write_to_influx(payload)
        msgs_written.inc()
        latency = time.monotonic() - t0
        pipeline_latency.observe(latency)
        log.debug(
            "Stored: vehicle=%s battery=%s ts=%s (%.1fms)",
            payload["vehicle_id"],
            payload["battery_id"],
            payload["timestamp"],
            latency * 1000,
        )
    except Exception as e:
        write_errors.inc()
        log.error("InfluxDB write failed: %s", e)


def on_disconnect(client, userdata, rc):
    if rc != 0:
        log.warning("Unexpected MQTT disconnect (rc=%d), will reconnect", rc)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Start Prometheus metrics server
    start_http_server(8001)
    log.info("Prometheus metrics exposed on :8001/metrics")

    client = mqtt.Client(client_id="dt-data-ingestion", clean_session=True)
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    log.info("Connecting to Mosquitto at %s:%d …", MQTT_HOST, MQTT_PORT)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()