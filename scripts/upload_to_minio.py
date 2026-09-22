"""
MinIO Upload Script
====================
Uploads PyBaMM training data and trained models to MinIO object storage.

Per Architecture_draft.pdf Section 3.2.2:
  Training Data Storage (MinIO):
    - PyBaMM outputs (HDF5)
    - Model weights (.pt)
    - Training logs
  Retention: indefinite (models versioned)

Buckets:
  training-data/  — HDF5 datasets + normalisation stats
  models/         — trained model weights + metadata
"""

import os
import json
from datetime import datetime, timezone

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print("Installing minio...")
    import subprocess
    subprocess.run(["pip", "install", "minio"], check=True)
    from minio import Minio
    from minio.error import S3Error

# ── Config ────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT  = os.environ.get("MINIO_ENDPOINT",  "localhost:9000")
MINIO_ACCESS    = os.environ.get("MINIO_ACCESS",    "dtminio")
MINIO_SECRET    = os.environ.get("MINIO_SECRET",    "dtminiopass")
MINIO_SECURE    = False

BUCKET_TRAINING = "training-data"
BUCKET_MODELS   = "models"

DATA_DIR   = "data"
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Files to upload
TRAINING_FILES = [
    "soc_training_data.h5",
    "soh_training_data.h5",
    "normalisation_stats.json",
]
MODEL_FILES = [
    "cnn_lstm_soc_best.pt",
    "soc_model_info.json",
    "gpr_soh.pkl",
    "gpr_soh_scaler.pkl",
    "gpr_soh_info.json",
    "normalisation_stats.json",
]


def ensure_bucket(client: Minio, bucket: str):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"  Created bucket: {bucket}")
    else:
        print(f"  Bucket exists: {bucket}")


def upload_file(client: Minio, bucket: str, local_path: str, object_name: str):
    size = os.path.getsize(local_path)
    client.fput_object(
        bucket, object_name, local_path,
        metadata={
            "uploaded-at": datetime.now(timezone.utc).isoformat(),
            "source":      "battery-dt-poc",
        }
    )
    print(f"  ✓ {object_name}  ({size/1024:.1f} KB)")


def main():
    print("Battery DT — MinIO Upload")
    print(f"  Endpoint: {MINIO_ENDPOINT}\n")

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=MINIO_SECURE,
    )

    # ── Training data ─────────────────────────────────────────────────────────
    print("── Training Data ────────────────────────────────────")
    ensure_bucket(client, BUCKET_TRAINING)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for fname in TRAINING_FILES:
        local = os.path.join(DATA_DIR, fname)
        if not os.path.exists(local):
            print(f"  ✗ {fname} not found, skipping")
            continue
        object_name = f"{ts}/{fname}"
        upload_file(client, BUCKET_TRAINING, local, object_name)
        # Also upload as "latest/"
        upload_file(client, BUCKET_TRAINING, local, f"latest/{fname}")

    # ── Models ────────────────────────────────────────────────────────────────
    print("\n── Models ───────────────────────────────────────────")
    ensure_bucket(client, BUCKET_MODELS)

    for fname in MODEL_FILES:
        local = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(local):
            print(f"  ✗ {fname} not found, skipping")
            continue
        # Route to appropriate subfolder based on model type
        if "gpr" in fname:
            prefix = "gpr-soh"
        else:
            prefix = "cnn-lstm-soc"
        upload_file(client, BUCKET_MODELS, local, f"{prefix}/{ts}/{fname}")
        upload_file(client, BUCKET_MODELS, local, f"{prefix}/latest/{fname}")

    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Training data: s3://{BUCKET_TRAINING}/latest/")
    print(f"  Models CNN-LSTM: s3://{BUCKET_MODELS}/cnn-lstm-soc/latest/")
    print(f"  Models GPR:      s3://{BUCKET_MODELS}/gpr-soh/latest/")
    print(f"  Versioned:       s3://{BUCKET_TRAINING}/{ts}/")
    print(f"\nView at: http://localhost:9001")
    print("Done.")

if __name__ == "__main__":
    main()
