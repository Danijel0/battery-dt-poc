"""
PoC Validation Benchmark
========================
Measures the four metrics defined in Architecture_draft.pdf Section 7.3:
  1. Inference latency     (target: < 1s for SoC)
  2. Throughput            (requests/second per instance)
  3. Cache effectiveness   (Redis hit rate)
  4. Resource utilization  (CPU, memory)

Usage:
  python scripts/benchmark.py [--host localhost] [--vehicles 20] [--duration 60]

Output:
  - Console summary table
  - benchmark_results.json  (machine-readable, for thesis appendix)
"""

import argparse
import json
import time
import statistics
import concurrent.futures
from datetime import datetime, timezone

import requests

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--host",      default="localhost")
parser.add_argument("--soc-port",  default=8001, type=int)
parser.add_argument("--soh-port",  default=8002, type=int)
parser.add_argument("--gw-port",   default=8080, type=int)
parser.add_argument("--vehicles",  default=20,   type=int,
                    help="Number of simulated vehicles")
parser.add_argument("--duration",  default=60,   type=int,
                    help="Throughput test duration in seconds")
parser.add_argument("--workers",   default=4,    type=int,
                    help="Concurrent workers for throughput test")
parser.add_argument("--output",    default="benchmark_results.json")
args = parser.parse_args()

INFLUX_URL   = f"http://{args.host}:8086"
INFLUX_TOKEN = "dt-super-secret-token"
INFLUX_ORG   = "battery-dt"


def discover_active_vehicles(window_seconds: int = 300, max_vehicles: int = 20) -> list[str]:
    """
    Query InfluxDB for vehicle_ids with telemetry in the last window_seconds.
    Uses the HTTP API directly (no influx CLI dependency).
    """
    import urllib.request

    flux = (
        f'from(bucket: "telemetry")'
        f' |> range(start: -{window_seconds}s)'
        f' |> keep(columns: ["vehicle_id"])'
        f' |> distinct(column: "vehicle_id")'
        f' |> limit(n: {max_vehicles})'
    )
    url  = f"{INFLUX_URL}/api/v2/query?org={INFLUX_ORG}"
    data = flux.encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Token {INFLUX_TOKEN}",
            "Content-Type":  "application/vnd.flux",
            "Accept":        "application/csv",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            csv_text = resp.read().decode("utf-8")
        vehicles = []
        for line in csv_text.splitlines():
            parts = line.strip().split(",")
            if parts and parts[-1].startswith("HV-"):
                vehicles.append(parts[-1])
        return vehicles
    except Exception as e:
        print(f"  WARNING: Could not discover vehicles: {e}")
        return []

BASE_SOC = f"http://{args.host}:{args.soc_port}"
BASE_SOH = f"http://{args.host}:{args.soh_port}"

# Vehicle IDs discovered dynamically at runtime from InfluxDB
# Populated in main() before tests run
VEHICLE_IDS_SOC = []   # vehicles with telemetry in last 300s (for SoC)
VEHICLE_IDS_SOH = [    # vehicles with 30-day history (for SoH)
    "HIST-NMC-01", "HIST-NMC-03", "HIST-LFP-02", "HIST-LFP-04",
]
VEHICLE_IDS = []

# ─────────────────────────────────────────────────────────────────────────────

def divider(title=""):
    w = 60
    if title:
        print(f"\n{'─'*3} {title} {'─'*(w-len(title)-5)}")
    else:
        print("─" * w)


def check_health():
    """Verify both services are reachable before benchmarking."""
    divider("Health check")
    ok = True
    for name, url in [("soc-service", f"{BASE_SOC}/health"),
                      ("soh-service", f"{BASE_SOH}/health")]:
        try:
            r = requests.get(url, timeout=5)
            status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            status = f"UNREACHABLE ({e})"
            ok = False
        print(f"  {name:20s}: {status}")
    return ok


# ── 1. Latency ────────────────────────────────────────────────────────────────

def measure_latency(n_warmup=5, n_samples=50, n_cold=10):
    """
    Cold-cache latency: flush Redis n_cold times, measure each independently.
    Warm-cache latency: n_samples repeated calls on same vehicle (Redis hit).
    Reports mean and p95 for both cold and warm paths.
    """
    divider("1. Inference latency")
    import subprocess
    results = {}

    for service, base, endpoint, vehicle in [
        ("SoC", BASE_SOC, "predict", VEHICLE_IDS_SOC[0] if VEHICLE_IDS_SOC else "HV-UNKNOWN"),
        ("SoH", BASE_SOH, "status",  "HIST-NMC-01"),
    ]:
        db = 0 if service == "SoC" else 1

        # ── Cold latency loop ────────────────────────────────────────────────
        # Flush cache before each cold call for representative measurement.
        # n_cold=10 gives stable mean/p95 rather than a single lucky/unlucky sample.
        cold_times = []
        for i in range(n_cold):
            try:
                subprocess.run(
                    ["docker", "compose", "exec", "-T", "redis",  
                    "redis-cli", "-n", str(db), "FLUSHDB"],
                    capture_output=True, timeout=5
                )
            except Exception:
                pass

            t0 = time.perf_counter()
            if service == "SoC":
                requests.post(f"{base}/{endpoint}",
                              json={"vehicle_id": vehicle,
                                    "timestamp": datetime.now(timezone.utc).isoformat()},
                              timeout=60)
            else:
                requests.get(f"{base}/{endpoint}/{vehicle}", timeout=60)
            cold_times.append((time.perf_counter() - t0) * 1000)

        cold_mean = statistics.mean(cold_times)
        cold_p95  = sorted(cold_times)[int(0.95 * len(cold_times))]

        # ── Warm-up (not measured) ───────────────────────────────────────────
        for _ in range(n_warmup):
            if service == "SoC":
                requests.post(f"{base}/{endpoint}",
                              json={"vehicle_id": vehicle,
                                    "timestamp": datetime.now(timezone.utc).isoformat()},
                              timeout=10)
            else:
                requests.get(f"{base}/{endpoint}/{vehicle}", timeout=10)

        # ── Warm latency loop ────────────────────────────────────────────────
        warm_times = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            if service == "SoC":
                requests.post(f"{base}/{endpoint}",
                              json={"vehicle_id": vehicle,
                                    "timestamp": datetime.now(timezone.utc).isoformat()},
                              timeout=10)
            else:
                requests.get(f"{base}/{endpoint}/{vehicle}", timeout=10)
            warm_times.append((time.perf_counter() - t0) * 1000)

        p50  = statistics.median(warm_times)
        p95  = sorted(warm_times)[int(0.95 * len(warm_times))]
        p99  = sorted(warm_times)[int(0.99 * len(warm_times))]
        mean = statistics.mean(warm_times)

        target_ms = 1000 if service == "SoC" else 5000
        status = "PASS" if p95 < target_ms else "FAIL"

        print(f"\n  {service}:")
        print(f"    cold mean={cold_mean:.1f}ms  cold p95={cold_p95:.1f}ms  (n={n_cold})")
        print(f"    warm mean={mean:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  (n={n_samples})")
        print(f"    target < {target_ms}ms  [{status}]")
        if cold_mean > 0:
            print(f"    cache speedup: {cold_mean/mean:.1f}x")

        results[service] = {
            "cold_mean_ms": round(cold_mean, 2),
            "cold_p95_ms":  round(cold_p95,  2),
            "cold_n":       n_cold,
            "mean_ms":      round(mean, 2),
            "p50_ms":       round(p50,  2),
            "p95_ms":       round(p95,  2),
            "p99_ms":       round(p99,  2),
            "cache_speedup_x": round(cold_mean / mean, 2) if mean > 0 else None,
            "target_ms":    target_ms,
            "pass":         status == "PASS",
            "n_samples":    n_samples,
        }

    return results


# ── 2. Throughput ─────────────────────────────────────────────────────────────

def measure_throughput():
    """
    Sustained request rate over --duration seconds with --workers threads.
    Uses SoC service (real-time, most latency-sensitive).
    """
    divider("2. Throughput")
    import itertools

    # Warm up cache for ALL vehicles before throughput test
    # Latency warm-up only covers one vehicle — remaining vehicles would
    # trigger cold inference at the start of the throughput test, skewing results.
    print("  Warming up cache for all vehicles...")
    for vid in VEHICLE_IDS_SOC:
        try:
            requests.post(f"{BASE_SOC}/predict",
                          json={"vehicle_id": vid,
                                "timestamp": datetime.now(timezone.utc).isoformat()},
                          timeout=15)
        except Exception:
            pass
    time.sleep(2)
    print(f"  Cache populated for {len(VEHICLE_IDS_SOC)} vehicles.")

    vehicle_cycle = itertools.cycle(VEHICLE_IDS_SOC)
    completed  = []
    errors     = 0
    stop_flag  = [False]

    def worker():
        nonlocal errors
        while not stop_flag[0]:
            vid = next(vehicle_cycle)
            t0  = time.perf_counter()
            try:
                r = requests.post(
                    f"{BASE_SOC}/predict",
                    json={"vehicle_id": vid,
                          "timestamp": datetime.now(timezone.utc).isoformat()},
                    timeout=5,
                )
                if r.status_code == 200:
                    completed.append((time.perf_counter() - t0) * 1000)
                else:
                    errors += 1
            except Exception:
                errors += 1

    print(f"  Running {args.workers} workers for {args.duration}s ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker) for _ in range(args.workers)]
        # Wait halfway, then capture CPU under load
        time.sleep(args.duration / 2)
        print("  Capturing resource utilization under load...")
        under_load = read_docker_stats("under_load")
        measure_resources._under_load = under_load
        time.sleep(args.duration / 2)
        stop_flag[0] = True

    rps      = len(completed) / args.duration
    err_rate = errors / max(1, len(completed) + errors) * 100

    print(f"  Total requests : {len(completed) + errors}")
    print(f"  Successful     : {len(completed)}")
    print(f"  Errors         : {errors}  ({err_rate:.1f}%)")
    print(f"  Throughput     : {rps:.1f} req/s  ({args.workers} workers)")
    if completed:
        p95 = sorted(completed)[int(0.95 * len(completed))]
        print(f"  p95 latency    : {p95:.1f} ms  (under load)")

    return {
        "requests_total":    len(completed) + errors,
        "requests_ok":       len(completed),
        "errors":            errors,
        "error_rate_pct":    round(err_rate, 2),
        "rps":               round(rps, 2),
        "workers":           args.workers,
        "duration_s":        args.duration,
        "p95_ms_under_load": round(sorted(completed)[int(0.95*len(completed))], 2) if completed else None,
    }


# ── 3. Cache effectiveness ────────────────────────────────────────────────────

def measure_cache():
    """
    Hit rate: call same vehicle twice (second should hit cache).
    Cold rate: call each vehicle once (all misses).
    """
    divider("3. Cache effectiveness")

    # Repeated calls — should all hit cache after first
    vehicle  = VEHICLE_IDS_SOH[0]  # HIST-NMC-01 has 30d data for SoH cache test
    n        = 20
    hits     = 0
    latencies_hit  = []
    latencies_miss = []

    for i in range(n):
        t0 = time.perf_counter()
        r  = requests.get(f"{BASE_SOH}/status/{vehicle}", timeout=10)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            data = r.json()
            if data.get("cache_hit"):
                hits += 1
                latencies_hit.append(ms)
            else:
                latencies_miss.append(ms)

    hit_rate = hits / n * 100
    print(f"  Repeated calls on {vehicle} (n={n}):")
    print(f"    Cache hit rate : {hit_rate:.0f}%")
    if latencies_miss:
        print(f"    Miss latency   : {statistics.mean(latencies_miss):.1f} ms (avg)")
    if latencies_hit:
        print(f"    Hit latency    : {statistics.mean(latencies_hit):.1f} ms (avg)")
        speedup = statistics.mean(latencies_miss) / statistics.mean(latencies_hit) if latencies_miss else None
        if speedup:
            print(f"    Cache speedup  : {speedup:.1f}×")

    # Per-vehicle first call (cold misses across fleet)
    cold_latencies = []
    for vid in VEHICLE_IDS_SOH[:8]:
        t0 = time.perf_counter()
        r  = requests.get(f"{BASE_SOH}/status/{vid}", timeout=60)
        cold_latencies.append((time.perf_counter() - t0) * 1000)

    print(f"\n  Cold calls across {len(VEHICLE_IDS)} vehicles:")
    print(f"    Mean latency   : {statistics.mean(cold_latencies):.1f} ms")
    print(f"    p95 latency    : {sorted(cold_latencies)[int(0.95*len(cold_latencies))]:.1f} ms")

    return {
        "hit_rate_pct":          round(hit_rate, 1),
        "mean_hit_latency_ms":   round(statistics.mean(latencies_hit), 2)  if latencies_hit  else None,
        "mean_miss_latency_ms":  round(statistics.mean(latencies_miss), 2) if latencies_miss else None,
        "cache_speedup_x":       round(statistics.mean(latencies_miss)/statistics.mean(latencies_hit), 2)
                                  if latencies_hit and latencies_miss else None,
        "cold_mean_ms":          round(statistics.mean(cold_latencies), 2),
        "n_vehicles":            len(VEHICLE_IDS),
    }


# ── 4. Resource utilization ───────────────────────────────────────────────────

def read_docker_stats(label: str) -> dict:
    """Read current CPU and memory from Docker stats."""
    import subprocess
    containers = ["dt-soc", "dt-soh", "dt-feature-eng",
                  "dt-influxdb", "dt-redis", "dt-mosquitto", "dt-ingestion"]
    results = {}
    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format",
             "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"] + containers,
            text=True, timeout=15
        )
        print(f"  {label}:")
        for line in out.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                name = parts[0].strip()
                cpu  = parts[1].strip()
                mem  = parts[2].strip()
                print(f"    {name:25s}: CPU={cpu:6s}  MEM={mem}")
                results[name] = {f"{label}_cpu": cpu, f"{label}_mem": mem}
    except Exception as e:
        print(f"    Could not read Docker stats: {e}")
    return results


def measure_resources():
    """
    Reports resource utilization in two conditions:
    - Under load: captured mid-way through throughput test (stored in _under_load)
    - Idle baseline: measured now, after cooldown
    """
    divider("4. Resource utilization")

    # Under-load stats captured during throughput test
    under_load = getattr(measure_resources, "_under_load", {})
    if under_load:
        print("  Under load (mid-throughput):")
        for name, vals in under_load.items():
            cpu = vals.get("under_load_cpu", "?")
            mem = vals.get("under_load_mem", "?")
            print(f"    {name:25s}: CPU={cpu:6s}  MEM={mem}")
    else:
        print("  Under-load stats not available (throughput test may have skipped)")

    # Idle baseline after cooldown
    idle = read_docker_stats("idle_baseline")

    results = {}
    for name in set(list(idle.keys()) + list(under_load.keys())):
        results[name] = {**idle.get(name, {}), **under_load.get(name, {})}
    return results



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Battery Digital Twin — PoC Validation Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Vehicles: {args.vehicles}  Duration: {args.duration}s  Workers: {args.workers}")
    print("=" * 60)

    # Discover active vehicles from InfluxDB
    global VEHICLE_IDS_SOC, VEHICLE_IDS
    print("\nDiscovering active vehicles (last 300s)...")
    VEHICLE_IDS_SOC = discover_active_vehicles(window_seconds=300)
    VEHICLE_IDS     = VEHICLE_IDS_SOC
    if not VEHICLE_IDS_SOC:
        print("  WARNING: No active vehicles found.")
        print("  Run: python scripts/simulate_fleet.py --duration 300")
        print("  Then wait 30s and retry benchmark.")
    else:
        print(f"  Found {len(VEHICLE_IDS_SOC)} active vehicles: {VEHICLE_IDS_SOC}")

    if not check_health():
        print("\nERROR: Services not reachable. Is docker compose up?")
        return

    results = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_vehicles": args.vehicles,
            "duration_s": args.duration,
            "workers":    args.workers,
        }
    }

    results["latency"]    = measure_latency()
    results["throughput"] = measure_throughput()

    # Allow services to recover after throughput stress test
    print("\n  Cooling down 15s after throughput test...")
    time.sleep(15)

    results["cache"]      = measure_cache()
    results["resources"]  = measure_resources()

    # ── Summary table ──────────────────────────────────────────────────────────
    divider("SUMMARY")
    print()

    soc_lat  = results["latency"].get("SoC", {})
    soh_lat  = results["latency"].get("SoH", {})
    tput     = results["throughput"]
    cache    = results["cache"]

    rows = [
        ("SoC p95 latency",    f"{soc_lat.get('p95_ms','?')} ms",
         "< 1000 ms", "PASS" if soc_lat.get("pass") else "FAIL"),
        ("SoH p95 latency",    f"{soh_lat.get('p95_ms','?')} ms",
         "< 5000 ms", "PASS" if soh_lat.get("pass") else "FAIL"),
        ("Throughput",         f"{tput.get('rps','?')} req/s",
         f"{args.workers} workers", "—"),
        ("Error rate",         f"{tput.get('error_rate_pct','?')}%",
         "< 1%", "PASS" if tput.get("error_rate_pct", 99) < 1 else "FAIL"),
        ("Cache hit rate",     f"{cache.get('hit_rate_pct','?')}%",
         "> 80%", "PASS" if cache.get("hit_rate_pct", 0) > 80 else "CHECK"),
        ("Cache speedup",      f"{cache.get('cache_speedup_x','?')}×",
         "", "—"),
    ]

    print(f"  {'Metric':<25} {'Value':>12}  {'Target':>12}  {'Status':>6}")
    print(f"  {'─'*25} {'─'*12}  {'─'*12}  {'─'*6}")
    for metric, value, target, status in rows:
        print(f"  {metric:<25} {value:>12}  {target:>12}  {status:>6}")

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
