"""
SoH Prediction Service
========================
REST API za GPR State of Health estimaciju.

Endpoints (per Architecture_draft.pdf Section 6.1):
  GET /status/{vehicle_id}
    Response: {"vehicle_id": "HV-001", "soh_percent": 92.3,
               "resistance_ohm": 0.025, "confidence_lower": 91.1,
               "confidence_upper": 93.5, "abnormal_degradation": false,
               "last_updated": "...", "cache_hit": false}

Flow (per Section 4.2 Batch SoH Update Flow):
  1. Check Redis cache (TTL 24h) → return if hit
  2. Request SoH features from Feature Engineering
  3. Run GPR inference → resistance estimate + uncertainty
  4. Compare vs expected resistance (anomaly detection)
  5. Cache result
  6. Return SoH + confidence interval
"""

import os
import json
import pickle
import logging
from datetime import datetime, timezone

import numpy as np
import redis
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("soh-service")

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_ENG_URL = os.environ.get("FEATURE_ENG_URL", "http://feature-engineering:8000")
REDIS_URL       = os.environ.get("REDIS_URL", "redis://redis:6379/1")
REDIS_TTL       = int(os.environ.get("REDIS_TTL_SOH", 86400))   # 24h
MODELS_DIR      = os.environ.get("MODELS_DIR", "/app/models")
INFO_PATH       = os.environ.get("INFO_PATH",  "/app/models/gpr_soh_info.json")

# EOL criterion: SoH = 80% when R = 2 × R_nominal
# (100% DCIR increase = 20% capacity loss, standard battery EOL criterion).
SOH_PARAMS = {
    # R_nominal = median resistance across all PyBaMM scenarios at soc_target=0.1,
    # representing beginning-of-life performance under representative operational
    # conditions. SoH is defined relative to this reference, consistent with
    # Lipu et al. (2018) who define SoH relative to nominal conditions.
    "NMC": {"r_nominal": 0.04366, "r_eol_factor": 2.0, "soh_eol": 0.80,
            "alert_ratio": 1.50, "critical_ratio": 2.0},
    "LFP": {"r_nominal": 0.04536, "r_eol_factor": 2.0, "soh_eol": 0.80,
            "alert_ratio": 1.50, "critical_ratio": 2.0},
}

# ── Prometheus ────────────────────────────────────────────────────────────────
predict_requests = Counter("soh_predict_requests_total", "Total SoH requests")
cache_hits       = Counter("soh_cache_hits_total",       "Redis cache hits")
cache_misses     = Counter("soh_cache_misses_total",     "Redis cache misses")
inference_time   = Histogram("soh_inference_seconds",   "GPR inference latency")
request_latency  = Histogram("soh_request_latency_seconds", "End-to-end latency")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Battery DT — SoH Prediction Service",
    version="1.0.0",
)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Per-chemistry model registry: {"NMC": (gpr, scaler), "LFP": (gpr, scaler)}
_models      = {}
_model_info  = None
_redis       = None
_http_client = None


@app.on_event("startup")
async def startup():
    global _models, _model_info, _redis, _http_client

    # Load per-chemistry GPR models (v2)
    for chem in ("NMC", "LFP"):
        model_path  = os.path.join(MODELS_DIR, f"gpr_soh_{chem}.pkl")
        scaler_path = os.path.join(MODELS_DIR, f"gpr_soh_{chem}_scaler.pkl")
        with open(model_path,  "rb") as f: gpr    = pickle.load(f)
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
        _models[chem] = (gpr, scaler)
        log.info("Loaded GPR-%s from %s", chem, model_path)

    with open(INFO_PATH) as f:
        _model_info = json.load(f)
    log.info("GPR models loaded. Version=%s", _model_info.get("model_version", "unknown"))

    _redis = redis.from_url(REDIS_URL, decode_responses=True)
    log.info("Redis connected: %s", REDIS_URL)

    _http_client = httpx.AsyncClient(base_url=FEATURE_ENG_URL, timeout=120.0)
    log.info("Feature Engineering client: %s", FEATURE_ENG_URL)


@app.on_event("shutdown")
async def shutdown():
    if _http_client:
        await _http_client.aclose()


# ── Schemas ───────────────────────────────────────────────────────────────────
class SoHResponse(BaseModel):
    vehicle_id:           str
    resistance_ohm:       float
    resistance_lower:     float
    resistance_upper:     float
    soh_percent:          float
    soh_lower:            float
    soh_upper:            float
    abnormal_degradation: bool
    degradation_level:    str   # "normal" | "alert" | "critical"
    last_updated:         str
    model_version:        str
    cache_hit:            bool


# ── Helpers ───────────────────────────────────────────────────────────────────
def resistance_to_soh(r: float, chem: str) -> float:
    """
    Linear R → SoH% mapping.
    Anchor: SoH=100% at R=R_nominal, SoH=80% at R=2×R_nominal.
    The EOL criterion (100% DCIR increase = 20% capacity loss) is a
    widely accepted standard for battery retirement decisions.
    """
    p     = SOH_PARAMS[chem]
    r_nom = p["r_nominal"]
    r_eol = r_nom * p["r_eol_factor"]
    soh   = 1.0 - (1.0 - p["soh_eol"]) * (r - r_nom) / (r_eol - r_nom)
    return float(np.clip(soh * 100, 0.0, 100.0))


def classify_degradation(r: float, chem: str) -> str:
    p     = SOH_PARAMS[chem]
    ratio = r / p["r_nominal"]
    if ratio >= p["critical_ratio"]:
        return "critical"
    elif ratio >= p["alert_ratio"]:
        return "alert"
    return "normal"


def run_inference(features: list, chem: str) -> tuple[float, float, float]:
    """
    Per-chemistry GPR inference.
    Returns (r_mean, r_lower_95ci, r_upper_95ci).
    """
    gpr, scaler = _models[chem]
    X        = np.array(features, dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X)

    with inference_time.time():
        r_mean, r_std = gpr.predict(X_scaled, return_std=True)

    r_mean  = float(r_mean[0])
    r_std   = float(r_std[0])
    r_lower = max(0.0, r_mean - 1.96 * r_std)
    r_upper = r_mean + 1.96 * r_std
    return r_mean, r_lower, r_upper


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "soh-service"}


@app.get("/status/{vehicle_id}", response_model=SoHResponse)
async def get_soh_status(vehicle_id: str):
    """
    Get SoH estimate for a vehicle.
    Checks Redis cache first (TTL=24h), then runs GPR inference.
    Per Architecture_draft.pdf Section 4.2 Batch SoH Update Flow.
    """
    predict_requests.inc()
    cache_key = f"soh:{vehicle_id}"

    with request_latency.time():
        # 1. Check cache
        cached = _redis.get(cache_key)
        if cached:
            cache_hits.inc()
            data = json.loads(cached)
            data["cache_hit"] = True
            return SoHResponse(**data)

        cache_misses.inc()

        # 2. Get SoH features from Feature Engineering
        try:
            resp = await _http_client.get(f"/features/soh/{vehicle_id}")
            if resp.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"No telemetry for vehicle {vehicle_id}"
                )
            resp.raise_for_status()
            features_data = resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503,
                                detail=f"Feature Engineering unavailable: {e}")

        # 3. Determine chemistry (for model routing, not passed to GPR)
        chem_id = float(features_data.get("chem_id", 0.0))
        chem    = "LFP" if chem_id == 1.0 else "NMC"

        # 4. Build feature vector — must match SOH_FEATURES in train_soh_model.py:
        # NMC: [c_rate, temp_mean_c, temp_min_c, temp_max_c,
        #        current_mean_a, soc_target, nominal_cap_ah]  (7 features)
        # LFP: [c_rate, temp_mean_c, current_mean_a, soc_target,
        #        nominal_cap_ah]  (5 features — temp_min/max dropped:
        #        SPM isothermal makes them identical to temp_mean)
        if chem == "NMC":
            features = [
                float(features_data.get("c_rate")),
                float(features_data.get("temp_mean_c")),
                float(features_data.get("temp_min_c")),
                float(features_data.get("temp_max_c")),
                float(features_data.get("current_mean_a")),
                float(features_data.get("soc_target", 0.1)),
                float(features_data.get("nominal_cap_ah")),
            ]
        else:  # LFP
            features = [
                float(features_data.get("c_rate")),
                float(features_data.get("temp_mean_c")),
                float(features_data.get("current_mean_a")),
                float(features_data.get("soc_target", 0.1)),
                float(features_data.get("nominal_cap_ah")),
            ]

        # 5. GPR inference — per-chemistry model
        r_mean, r_lower, r_upper = run_inference(features, chem)

        # 6. Convert to SoH and classify
        soh       = resistance_to_soh(r_mean, chem)
        soh_lower = resistance_to_soh(r_upper, chem)   # higher R = lower SoH
        soh_upper = resistance_to_soh(r_lower, chem)
        level     = classify_degradation(r_mean, chem)
        abnormal  = level in ("alert", "critical")

        ts = datetime.now(timezone.utc).isoformat()
        result = {
            "vehicle_id":           vehicle_id,
            "resistance_ohm":       round(r_mean,  5),
            "resistance_lower":     round(r_lower, 5),
            "resistance_upper":     round(r_upper, 5),
            "soh_percent":          round(soh, 2),
            "soh_lower":            round(soh_lower, 2),
            "soh_upper":            round(soh_upper, 2),
            "abnormal_degradation": abnormal,
            "degradation_level":    level,
            "last_updated":         ts,
            "model_version":        "gpr-soh-v2",
            "cache_hit":            False,
        }

        # 7. Cache 24h
        _redis.setex(cache_key, REDIS_TTL, json.dumps(result))

        log.info(
            "SoH: vehicle=%s R=%.5fΩ SoH=%.1f%% level=%s",
            vehicle_id, r_mean, soh, level
        )

        return SoHResponse(**result)