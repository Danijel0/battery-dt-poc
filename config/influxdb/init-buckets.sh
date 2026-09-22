#!/bin/bash
# Initialize additional InfluxDB buckets after setup
# Default bucket "telemetry" is created via env vars

INFLUX_URL="http://localhost:8086"
TOKEN="dt-super-secret-token"
ORG="battery-dt"

buckets=("soc-estimates" "soh-estimates" "simulation" "metrics")

for bucket in "${buckets[@]}"; do
    influx bucket create \
        --name "$bucket" \
        --org "$ORG" \
        --token "$TOKEN" \
        --retention 0 \
        --host "$INFLUX_URL" \
        2>/dev/null || echo "Bucket $bucket already exists"
done

echo "InfluxDB buckets initialized."
