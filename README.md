# Battery Digital Twin — PoC

A microservices-based digital twin architecture for real-time battery health monitoring in heavy electric vehicle fleets. Implements State of Charge (SoC) estimation and State of Health (SoH) monitoring as independent REST services, aligned with the ISO 23247 digital twin framework.

---

## Architecture

Five-layer microservices stack:

```
Vehicles (BMS @ 1Hz)
    │ MQTT
    ▼
Ingestion Layer     Eclipse Mosquitto + Data Validator
    │
    ▼
Data Layer          InfluxDB (time-series) + Redis (prediction cache)
    │
    ▼
ML Infrastructure   Feature Engineering API + PyBaMM simulator
    │
    ▼
Service Layer       SoC Service (CNN-LSTM) + SoH Service (GPR)
    │
    ▼
Application Layer   REST API Gateway
```

---

## Services

| Service | Port | Description |
|---|---|---|
| API Gateway | 8080 | Single entry point for fleet management clients |
| SoC Service | 8001 | CNN-LSTM real-time State of Charge prediction |
| SoH Service | 8002 | GPR-based daily State of Health monitoring |
| Feature Engineering | 8004 | InfluxDB abstraction and feature preparation |
| InfluxDB | 8086 | Time-series telemetry storage |
| Redis | 6379 | Prediction cache (SoC TTL=60s, SoH TTL=24h) |
| Mosquitto | 1883 | MQTT broker for vehicle telemetry |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Monitoring dashboard |

---

## Results

Benchmark conducted with 10 simulated vehicles at 1 Hz, 4 concurrent workers, 60s throughput test.

| Metric | Value | Target |
|---|---|---|
| SoC warm-cache p95 latency | 21 ms | < 1000 ms ✓ |
| SoH warm-cache p95 latency | 26 ms | < 5000 ms ✓ |
| SoC cold latency (mean) | 124 ms | — |
| SoH cold latency (mean) | 8,909 ms | — |
| Throughput | 177 req/s | — |
| Error rate | 0% | < 1% ✓ |
| Cache hit rate | 100% | > 80% ✓ |
| SoC cache speedup | 12× | — |
| SoH cache speedup | 581× | — |

---

## Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Docker Compose v2
- Python 3.10+ (for training scripts and benchmark)

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/Danijel0/battery-dt-poc.git
cd battery-dt-poc

# 2. Start all services
docker compose up -d

# 3. Wait for services to be healthy (~30s)
docker compose ps

# 4. Run fleet simulator (Terminal 1)
pip install paho-mqtt
python scripts/simulate_fleet.py --vehicles 10 --duration 300

# 5. Run benchmark (Terminal 2, after 35s)
pip install requests influxdb-client
python scripts/benchmark.py --duration 60 --workers 4
```

---

## Training Models

### Generate training data (requires PyBaMM)

```bash
pip install pybamm
python scripts/generate_training_data.py
```

### Train SoC model (CNN-LSTM)

```bash
pip install torch numpy h5py scikit-learn
python scripts/train_soc_model.py
```

### Train SoH models (GPR, per chemistry)

```bash
python scripts/train_soh_model.py
```

Trained models are saved to `data/models/`.

---

## API Usage

### SoC Prediction

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"vehicle_id": "HV-001", "timestamp": "2026-06-01T10:00:00Z"}'
```

```json
{
  "vehicle_id": "HV-001",
  "soc_percent": 67.3,
  "confidence_lower": 62.5,
  "confidence_upper": 72.1,
  "model_version": "cnn-lstm-v1",
  "cache_hit": false
}
```

### SoH Status

```bash
curl http://localhost:8002/status/HIST-NMC-01
```

```json
{
  "vehicle_id": "HIST-NMC-01",
  "resistance_ohm": 0.04651,
  "soh_percent": 98.7,
  "soh_lower": 95.8,
  "soh_upper": 100.0,
  "degradation_level": "normal",
  "abnormal_degradation": false,
  "model_version": "gpr-soh-v2",
  "cache_hit": false
}
```

---

## Validation

Partial validation against the Vilsen & Stroe (2023) forklift LFP degradation dataset:

```bash
# Place dataset at forklift_validation/Cell{1,2,3}/Round{00..58}/RPT.csv
python scripts/validate_soh_forklift.py --data forklift_validation
```

Output plot: `data/validation/soh_validation_forklift.png`

---

## Project Structure

```
battery-dt-poc/
├── docker-compose.yml
├── services/
│   ├── soc-service/          # CNN-LSTM SoC inference
│   ├── soh-service/          # GPR SoH inference
│   ├── feature-engineering/  # InfluxDB abstraction
│   └── data-ingestion/       # MQTT → InfluxDB pipeline
├── scripts/
│   ├── generate_training_data.py
│   ├── train_soc_model.py
│   ├── train_soh_model.py
│   ├── simulate_fleet.py
│   ├── benchmark.py
│   ├── validate_soh_forklift.py
│   ├── create_grafana_dashboard.py
|   ├── upload_to_minio.py
|   └── inject_historical_telemetry.py
├── data/
│   ├── models/               # Trained model weights
│   └── validation/           # Validation outputs
└── config/
    ├── influxdb/
    ├── grafana/
    ├── prometheus/
    ├── nginx/
    └── mosquitto/
```

---

## Tech Stack

**ML:** PyTorch (CNN-LSTM), scikit-learn (GPR), PyBaMM (battery simulation)  
**Data:** InfluxDB 2.7, Redis 7  
**Messaging:** Eclipse Mosquitto 2.0 (MQTT)  
**API:** FastAPI, Uvicorn  
**Monitoring:** Prometheus, Grafana  
**Deployment:** Docker, Docker Compose  
**Languages:** Python 3.10