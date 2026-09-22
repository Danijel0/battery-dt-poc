"""
create_grafana_dashboard.py
===========================
Creates a PoC Validation dashboard in Grafana via API.
Run once before benchmark — dashboard persists in Grafana.

Usage:
    python scripts/create_grafana_dashboard.py
    # Then open http://localhost:3000 and find "Battery DT PoC Validation"
"""

import json
import requests

GRAFANA_URL   = "http://localhost:3000"
GRAFANA_USER  = "admin"
GRAFANA_PASS  = "dtgrafana"  

session = requests.Session()
session.auth = (GRAFANA_USER, GRAFANA_PASS)


def panel(title, expr, unit, panel_id, x, y, w=12, h=8):
    return {
        "id":    panel_id,
        "title": title,
        "type":  "timeseries",
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {"lineWidth": 2, "fillOpacity": 10},
            }
        },
        "options": {"tooltip": {"mode": "multi"}},
        "targets": [
            {
                "datasource": {"type": "prometheus"},
                "expr":       expr,
                "legendFormat": title,
                "refId": "A",
            }
        ],
    }


dashboard = {
    "dashboard": {
        "title":       "Battery DT PoC Validation",
        "tags":        ["battery-dt", "poc", "benchmark"],
        "timezone":    "browser",
        "refresh":     "5s",
        "schemaVersion": 38,
        "panels": [
            panel(
                "SoC Inference p95 (ML only, ms)",
                "histogram_quantile(0.95, rate(soc_inference_seconds_bucket[1m])) * 1000",
                "ms", 1, 0, 0,
            ),
            panel(
                "SoC Request Latency p95 (end-to-end, ms)",
                "histogram_quantile(0.95, rate(soc_request_latency_seconds_bucket[1m])) * 1000",
                "ms", 2, 12, 0,
            ),
            panel(
                "SoH Inference p95 (GPR, ms)",
                "histogram_quantile(0.95, rate(soh_inference_seconds_bucket[1m])) * 1000",
                "ms", 3, 0, 8,
            ),
            panel(
                "SoH Request Latency p95 (end-to-end, ms)",
                "histogram_quantile(0.95, rate(soh_request_latency_seconds_bucket[1m])) * 1000",
                "ms", 4, 12, 8,
            ),
            panel(
                "SoC Cache Hit Rate (%)",
                "rate(soc_cache_hits_total[1m]) / (rate(soc_cache_hits_total[1m]) + rate(soc_cache_misses_total[1m])) * 100",
                "percent", 5, 0, 16,
            ),
            panel(
                "SoH Cache Hit Rate (%)",
                "rate(soh_cache_hits_total[1m]) / (rate(soh_cache_hits_total[1m]) + rate(soh_cache_misses_total[1m])) * 100",
                "percent", 6, 12, 16,
            ),
            panel(
                "SoC Requests/sec",
                "rate(soc_predict_requests_total[1m])",
                "reqps", 7, 0, 24,
            ),
            panel(
                "SoH Requests/sec",
                "rate(soh_predict_requests_total[1m])",
                "reqps", 8, 12, 24,
            ),
            panel(
                "Ingestion Pipeline p95 (MQTT→InfluxDB, ms)",
                "histogram_quantile(0.95, rate(ingestion_pipeline_latency_seconds_bucket[1m])) * 1000",
                "ms", 9, 0, 32,
            ),
            panel(
                "Feature Engineering Query p95 (InfluxDB, ms)",
                "histogram_quantile(0.95, rate(feateng_query_latency_seconds_bucket[1m])) * 1000",
                "ms", 10, 12, 32,
            ),
            panel(
                "MQTT Messages Received/sec",
                "rate(ingestion_mqtt_received_total[1m])",
                "reqps", 11, 0, 40,
            ),
            panel(
                "Invalid Messages/sec",
                "rate(ingestion_invalid_total[1m])",
                "reqps", 12, 12, 40,
            ),
        ],
    },
    "overwrite": True,
    "folderId":  0,
}

r = session.post(
    f"{GRAFANA_URL}/api/dashboards/db",
    json=dashboard,
    timeout=10,
)

if r.status_code == 200:
    data = r.json()
    uid  = data.get("uid", "")
    url  = data.get("url", "")
    print(f"Dashboard created: {GRAFANA_URL}{url}")
    print(f"UID: {uid}")
else:
    print(f"Error {r.status_code}: {r.text}")
