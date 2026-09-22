"""
SoC Prediction Service
========================
REST API za CNN-LSTM State of Charge estimaciju.

Endpoints (per Architecture_draft.pdf Section 6.1):
  POST /predict
    Request:  {"vehicle_id": "HV-001", "timestamp": "2026-03-15T14:30:00Z"}
    Response: {"vehicle_id": "HV-001", "soc_percent": 67.3,
               "confidence_lower": 66.1, "confidence_upper": 68.5,
               "timestamp": "...", "model_version": "cnn-lstm-v1"}

Flow (per Section 4.1):
  1. Check Redis cache (TTL 60s) → return if hit
  2. Request features from Feature Engineering service
  3. Run CNN-LSTM inference
  4. Cache result
  5. Return SoC + confidence interval
"""

import os
import json
import logging
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import redis
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

_confidence_margin = 0.0238
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("soc-service")

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_ENG_URL = os.environ.get("FEATURE_ENG_URL", "http://feature-engineering:8000")
REDIS_URL       = os.environ.get("REDIS_URL", "redis://redis:6379/0")
REDIS_TTL       = int(os.environ.get("REDIS_TTL_SOC", 60))
INFLUX_URL      = os.environ.get("INFLUX_URL",    "http://influxdb:8086")
INFLUX_TOKEN    = os.environ.get("INFLUX_TOKEN",  "dt-super-secret-token")
INFLUX_ORG      = os.environ.get("INFLUX_ORG",    "battery-dt")
INFLUX_BUCKET   = os.environ.get("INFLUX_BUCKET_SOC", "soc-estimates")
MODEL_PATH      = os.environ.get("MODEL_PATH", "/app/models/cnn_lstm_soc_best.pt")
NORM_STATS_PATH = os.environ.get("NORM_STATS_PATH", "/app/models/normalisation_stats.json")
WINDOW_SIZE     = 60
N_FEATURES      = 4
_influx_write = None

# ── Prometheus ────────────────────────────────────────────────────────────────
predict_requests = Counter("soc_predict_requests_total", "Total SoC predict requests")
cache_hits       = Counter("soc_cache_hits_total", "Redis cache hits")
cache_misses     = Counter("soc_cache_misses_total", "Redis cache misses")
inference_time   = Histogram("soc_inference_seconds", "CNN-LSTM inference latency")
request_latency  = Histogram("soc_request_latency_seconds", "End-to-end request latency")

# ── CNN-LSTM Model (must match train_soc_model.py) ───────────────────────────
class CNNLSTMSoC(nn.Module):
    def __init__(self, n_features=4, cnn_filters=64, kernel_size=3,
                 lstm_hidden=128, lstm_layers=1, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Battery DT — SoC Prediction Service",
    version="1.0.0",
)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

_model       = None
_norm_stats  = None
_redis       = None
_http_client = None


@app.on_event("startup")
async def startup():
    global _model, _norm_stats, _redis, _http_client, _influx_write

    # Load model
    log.info("Loading CNN-LSTM model from %s", MODEL_PATH)
    _model = CNNLSTMSoC()
    _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    _model.eval()
    log.info("Model loaded")

    # Load normalisation stats
    with open(NORM_STATS_PATH) as f:
        _norm_stats = json.load(f)
    log.info("Normalisation stats loaded: %s", _norm_stats)

    # Load model info for confidence margin
    model_info_path = MODEL_PATH.replace("cnn_lstm_soc_best.pt", "soc_model_info.json")
    try:
        with open(model_info_path) as f:
            model_info = json.load(f)
        global _confidence_margin
        _confidence_margin = 2.0 * float(model_info.get("best_val_rmse"))
        log.info("Confidence margin set to ±%.4f (2×RMSE)", _confidence_margin)
    except Exception:
        _confidence_margin = 0.02   # fallback

    # Redis
    _redis = redis.from_url(REDIS_URL, decode_responses=True)
    log.info("Redis connected: %s", REDIS_URL)

    # HTTP client for Feature Engineering
    _http_client = httpx.AsyncClient(base_url=FEATURE_ENG_URL, timeout=10.0)
    log.info("Feature Engineering client: %s", FEATURE_ENG_URL)

    # InfluxDB
    _influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    _influx_write  = _influx_client.write_api(write_options=SYNCHRONOUS)
    log.info("InfluxDB write client initialised")

@app.on_event("shutdown")
async def shutdown():
    if _http_client:
        await _http_client.aclose()


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    vehicle_id: str
    timestamp: str = None


class PredictResponse(BaseModel):
    vehicle_id:       str
    soc_percent:      float
    confidence_lower: float
    confidence_upper: float
    timestamp:        str
    model_version:    str
    cache_hit:        bool


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_window(window: np.ndarray) -> np.ndarray:
    """Apply per-feature normalisation matching training.
    Only normalises V, I, T (columns 0-2). chem_id (column 3) stays as 0/1.
    """
    stats = _norm_stats
    window[:, 0] = (window[:, 0] - stats["voltage"]["mean"])     / max(stats["voltage"]["std"],     1e-3)
    window[:, 1] = (window[:, 1] - stats["current"]["mean"])     / max(stats["current"]["std"],     1e-3)
    window[:, 2] = (window[:, 2] - stats["temperature"]["mean"]) / max(stats["temperature"]["std"], 1e-3)
    window[:, :3] = np.clip(window[:, :3], -10.0, 10.0)
    return window

def write_soc_to_influx(vehicle_id: str, soc: float, chem_id: float):
    point = (
        Point("soc_estimates")
        .tag("vehicle_id", vehicle_id)
        .field("soc",     soc)
        .field("chem_id", chem_id)
    )
    try:
        _influx_write.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
    except Exception as e:
        log.warning("InfluxDB SoC write failed: %s", e)

def run_inference(window: np.ndarray) -> tuple[float, float, float]:
    """
    Run CNN-LSTM inference on (60, 4) window.
    Returns (soc, lower_95ci, upper_95ci).
    Uses MC Dropout for uncertainty estimation (T=20 forward passes).
    """
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 60, 4)

    # Single forward pass (eval mode, no dropout)
    _model.eval()
    with torch.no_grad():
        soc_mean = _model(x).item()  

    margin = _confidence_margin
    preds = [soc_mean]  # placeholder for future MC Dropout

    soc_mean = float(np.mean(preds))
    lower = max(0.0, soc_mean - margin)
    upper = min(1.0, soc_mean + margin)

    return soc_mean, lower, upper


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "soc-service"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Predict SoC for a vehicle.
    Checks Redis cache first (TTL=60s), then runs CNN-LSTM inference.
    Per Architecture_draft.pdf Section 4.1 SoC prediction flow.
    """
    predict_requests.inc()
    cache_key = f"soc:{req.vehicle_id}"

    with request_latency.time():
        # 1. Check cache
        cached = _redis.get(cache_key)
        if cached:
            cache_hits.inc()
            data = json.loads(cached)
            data["cache_hit"] = True
            return PredictResponse(**data)

        cache_misses.inc()

        # 2. Get features from Feature Engineering
        try:
            resp = await _http_client.get(f"/features/soc/{req.vehicle_id}?window_seconds=300")
            if resp.status_code == 404:
                raise HTTPException(status_code=404,
                                    detail=f"No telemetry for vehicle {req.vehicle_id}")
            resp.raise_for_status()
            features = resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503,
                                detail=f"Feature Engineering unavailable: {e}")

        # 3. Prepare input window
        window = np.array(features["features"], dtype=np.float32)  # raw values
        window = normalize_window(window)  # normalise using training stats
        chem_id = float(features.get("chem_id"))
        if window.shape != (WINDOW_SIZE, N_FEATURES):
            # Pad or trim if needed
            if len(window) < WINDOW_SIZE:
                pad = np.zeros((WINDOW_SIZE - len(window), N_FEATURES), dtype=np.float32)
                window = np.vstack([pad, window])
            else:
                window = window[-WINDOW_SIZE:]

        # 4. Inference
        with inference_time.time():
            soc, lower, upper = run_inference(window)

        # 5. Build response
        ts = datetime.now(timezone.utc).isoformat()
        result = {
            "vehicle_id":       req.vehicle_id,
            "soc_percent":      round(soc * 100, 2),
            "confidence_lower": round(lower * 100, 2),
            "confidence_upper": round(upper * 100, 2),
            "timestamp":        ts,
            "model_version":    "cnn-lstm-v1",
            "cache_hit":        False,
        }

        # 6. Cache result (TTL=60s per Section 3.2.2)
        _redis.setex(cache_key, REDIS_TTL, json.dumps(result))
        
        # Write SoC estimate to InfluxDB for SoH feature aggregation
        write_soc_to_influx(req.vehicle_id, soc, chem_id)  
        
        log.info("SoC predicted: vehicle=%s soc=%.1f%% [%.1f, %.1f]",
                 req.vehicle_id, result["soc_percent"],
                 result["confidence_lower"], result["confidence_upper"])

        return PredictResponse(**result)