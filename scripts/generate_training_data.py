"""
PyBaMM Training Data Generator v2 — Heavy-Duty NMC + LFP
==========================================================
Compatible with PyBaMM 26.4.0

Changes from v1:
  - Expanded SCENARIOS: systematic (c_rate × temp) grid coverage
    NMC: 5 C-rates × 9 temps = 45 scenarios (was 14 unique combos)
    LFP: 5 C-rates × 9 temps = 45 scenarios (was 14 unique combos)
  - SoH aggregation: one datapoint per scenario (median R, mean features)
    instead of overlapping 300s sliding windows. GPR needs independent
    observations — sliding windows from the same discharge are correlated
    and cause inflated confidence (low uncertainty on training set,
    enormous uncertainty elsewhere).
  - temp_init_c removed from SoH features: replaced by temp_min_c.
    In production, temp_init_c = df.iloc[0] is an arbitrary value from
    a 30-day window. temp_min_c is stable and physically meaningful
    (cold-start conditions drive worst-case resistance).

Models:
  NMC: SPMe + lumped thermal (Chen2020, 5Ah cell)
  LFP: SPM  isothermal     (Prada2013, 2.3Ah cell)

SoH features per scenario (7, one row per scenario):
  [c_rate, temp_mean_c, temp_min_c, temp_max_c,
   current_mean_a, soc_target, nominal_cap_ah]
  chem_id excluded from SoH features — separate model per chemistry.

SoC features per window (60s × 4):
  [voltage_V, current_A, temperature_C, chem_id]
"""

import os
import json
import h5py
import numpy as np
import pybamm

OUTPUT_DIR  = "data"
WINDOW_SIZE = 60
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHEMISTRIES = {
    "NMC": {
        "param_set": "Chen2020",
        "v_min":     "2.5 V",
        "model":     "SPMe",
        "chem_id":   0,
        "label":     "NMC_Chen2020_SPMe",
    },
    "LFP": {
        "param_set": "Prada2013",
        "v_min":     "2.0 V",
        "model":     "SPM",
        "chem_id":   1,
        "label":     "LFP_Prada2013_SPM",
    },
}

# Systematic grid: 5 C-rates × 9 temperatures = 45 scenarios per chemistry
# C-rates cover typical heavy-duty operation: slow depot charge to fast
# opportunity charge. Intermediate rates (0.75, 1.5) fill gaps where
# the v1 model had high uncertainty.
C_RATES = [0.5, 0.75, 1.0, 1.5, 2.0]
TEMPS   = [-10.0, -5.0, 0.0, 10.0, 20.0, 25.0, 35.0, 40.0, 45.0]

# Grid dimensions:
# soc_init=1.0, soc_target=0.0 → full discharge to V_min → soc_min≈0 (deep DoD)
# soc_init=1.0, soc_target=0.1 → stop at 10% SoC → soc_min≈0.10 (urban bus policy)
# soc_init=1.0, soc_target=0.2 → stop at 20% SoC → soc_min≈0.20 (conservative policy)
# This gives GPR coverage of the full operational soc_min range [0, 0.5]
SOC_TARGETS_GRID = [0.0, 0.1, 0.2]

EXTRA_SCENARIOS = [
    # (label, c_rate, temp_c, soc_init, soc_target)
    # 3C fast-charge scenarios (LFP only — NMC SPMe solver fails >2C)
    ("fast_3C_25C",  3.0, 25.0, 1.0, 0.0),
    ("fast_3C_35C",  3.0, 35.0, 1.0, 0.0),
    ("fast_3C_45C",  3.0, 45.0, 1.0, 0.0),
]

def build_scenarios():
    """Build full scenario list: (c_rate × temp × soc_target) grid + extras."""
    scenarios = []
    for c in C_RATES:
        for t in TEMPS:
            for st in SOC_TARGETS_GRID:
                st_tag = str(int(st*100))
                label = (f"grid_{str(c).replace('.','p')}C_{int(t):+d}C_min{st_tag}"
                         .replace("+", "p").replace("-", "m"))
                scenarios.append((label, c, t, 1.0, st))  # soc_init always 1.0
    scenarios.extend(EXTRA_SCENARIOS)
    return scenarios


def run_simulation(chemistry: dict, c_rate: float, temp_c: float,
                   soc_init: float, soc_target: float = 0.0) -> dict | None:
    options = {"thermal": "lumped"} if chemistry["model"] == "SPMe" else {}
    if chemistry["model"] == "SPMe":
        model = pybamm.lithium_ion.SPMe(options=options)
    else:
        model = pybamm.lithium_ion.SPM(options=options)

    param = pybamm.ParameterValues(chemistry["param_set"])
    param["Initial temperature [K]"] = 273.15 + temp_c
    param["Ambient temperature [K]"] = 273.15 + temp_c

    # Duration to discharge from soc_init to soc_target.
    # soc_target controls the minimum SoC in the simulation — critical for
    # covering the operational soc_min range that vehicles actually experience.
    # Default soc_target=0.0 → drain to V_min (deep DoD scenarios).
    # soc_target=0.1 → stop at 10% SoC (typical urban bus min SoC policy).
    soc_to_discharge = soc_init - soc_target
    duration_s = int((3600 / c_rate) * soc_to_discharge)  # exact: 1C discharges at ~1Ah/h

    experiment = pybamm.Experiment([
        pybamm.step.c_rate(
            c_rate,
            duration=duration_s,
            termination=chemistry["v_min"],
            period="1 second",
        )
    ])

    try:
        sim = pybamm.Simulation(
            model, parameter_values=param,
            experiment=experiment,
            solver=pybamm.CasadiSolver(mode="safe"),
        )
        sim.solve(initial_soc=soc_init)
        sol = sim.solution
    except Exception as e:
        print(f"✗ Solver: {e}")
        return None

    t   = sol["Time [s]"].entries
    V   = sol["Terminal voltage [V]"].entries
    I   = sol["Current [A]"].entries
    cap = sol["Discharge capacity [A.h]"].entries

    nominal_cap = float(param["Nominal cell capacity [A.h]"])
    soc = np.clip(soc_init - cap / nominal_cap, 0.0, 1.0)

    T_raw = sol["Cell temperature [K]"].entries
    T = (T_raw.mean(axis=0) if T_raw.ndim == 2 else T_raw) - 273.15

    try:
        ocv = sol["Battery open-circuit voltage [V]"].entries
        R   = np.where(np.abs(I) > 1e-6,
                       np.abs(ocv - V) / np.abs(I), np.nan)
        med = np.nanmedian(R)
        R   = np.where(np.isnan(R), med, R)
    except Exception:
        R = np.zeros_like(t)

    n = min(len(t), len(V), len(I), len(T), len(soc), len(R))
    if n < WINDOW_SIZE + 10:
        return None

    return {
        "time":        t[:n],
        "voltage":     V[:n],
        "current":     I[:n],
        "temp":        T[:n],
        "soc":         soc[:n],
        "resistance":  R[:n],
        "chem_id":     chemistry["chem_id"],
        "nominal_cap": nominal_cap,
        "c_rate":      c_rate,
        "temp_init":   temp_c,
    }


def extract_soc_windows(data: dict):
    """X: (N, 60, 4)  y: (N,) SoC"""
    V, I, T, soc = data["voltage"], data["current"], data["temp"], data["soc"]
    cid      = float(data["chem_id"])
    chem_col = np.full(WINDOW_SIZE, cid, dtype=np.float32)
    n = len(V)
    X, y = [], []
    for i in range(WINDOW_SIZE, n):
        window = np.stack([V[i-WINDOW_SIZE:i], I[i-WINDOW_SIZE:i],
                           T[i-WINDOW_SIZE:i]], axis=1)
        X.append(np.column_stack((window, chem_col)))
        y.append(float(soc[i]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def extract_soh_scenario(data: dict, soc_target: float = 0.0) -> dict | None:
    """
    One aggregated datapoint per scenario.
    Features match what feature-engineering returns in production:
      [c_rate, temp_mean_c, temp_min_c, temp_max_c,
       current_mean_a, soc_target, nominal_cap_ah]
    Target: median internal resistance [Ohm]
    Using median (not mean) — robust to transient spikes at end-of-discharge.
    """
    I   = data["current"]
    T   = data["temp"]
    soc = data["soc"]
    R   = data["resistance"]

    # Skip windows with near-zero current (rest periods)
    active = np.abs(I) > 0.1
    if active.sum() < 10:
        return None

    return {
        "c_rate":         data["c_rate"],
        "temp_mean_c":    float(T.mean()),
        "temp_min_c":     float(T.min()),
        "temp_max_c":     float(T.max()),
        "current_mean_a": float(np.abs(I[active]).mean()),
        "soc_target":     float(soc_target),  # soc_mean removed: artefact of simulation depth
        "nominal_cap_ah": data["nominal_cap"],
        "resistance_ohm": float(np.nanmedian(R[active])),
    }


def main():
    SCENARIOS = build_scenarios()
    total     = len(CHEMISTRIES) * len(SCENARIOS)
    print(f"Battery DT — Training Data Generator v2")
    print(f"  Chemistries : NMC (SPMe+thermal) + LFP (SPM)")
    print(f"  Scenarios   : {len(SCENARIOS)} × 2 = {total} total")
    print(f"  Grid        : {len(C_RATES)} C-rates × {len(TEMPS)} temps")
    print(f"  Extra       : {len(EXTRA_SCENARIOS)} operational scenarios\n")

    all_soc_X, all_soc_y = [], []
    soh_by_chem = {"NMC": {"X": [], "y": []}, "LFP": {"X": [], "y": []}}

    idx = 0
    for chem_name, chemistry in CHEMISTRIES.items():
        print(f"── {chem_name} ({chemistry['label']}) ──────────────────────────────")
        for (label, c_rate, temp_c, soc_init, soc_target) in SCENARIOS:
            idx += 1
            tag = f"[{idx:03d}/{total}] {chem_name}_{label}"
            print(f"{tag} ...", end=" ", flush=True)

            data = run_simulation(chemistry, c_rate, temp_c, soc_init, soc_target)
            if data is None:
                print("✗ failed")
                continue

            # SoC windows
            X, y = extract_soc_windows(data)
            if len(X) > 0:
                all_soc_X.append(X)
                all_soc_y.append(y)

            # SoH: one aggregated point per scenario
            soh_point = extract_soh_scenario(data, soc_target)
            if soh_point:
                feat = [
                    soh_point["c_rate"],
                    soh_point["temp_mean_c"],
                    soh_point["temp_min_c"],
                    soh_point["temp_max_c"],
                    soh_point["current_mean_a"],
                    soh_point["soc_target"],
                    soh_point["nominal_cap_ah"],
                ]
                soh_by_chem[chem_name]["X"].append(feat)
                soh_by_chem[chem_name]["y"].append(soh_point["resistance_ohm"])

            print(f"✓  n={len(data['voltage'])}  R={np.nanmedian(data['resistance']):.5f}Ω"
                  f"  T=[{data['temp'].min():.1f},{data['temp'].max():.1f}]°C")
        print()

    # ── SoC dataset ───────────────────────────────────────────────────────────
    if all_soc_X:
        X_all = np.concatenate(all_soc_X, axis=0)
        y_all = np.concatenate(all_soc_y, axis=0)

        X_cont = X_all[:, :, :3].reshape(-1, 3)
        X_mean = X_cont.mean(axis=0)
        X_std  = np.maximum(X_cont.std(axis=0), 1e-3)
        X_norm = X_all.copy()
        X_norm[:, :, :3] = (X_all[:, :, :3] - X_mean) / X_std

        soc_path = os.path.join(OUTPUT_DIR, "soc_training_data.h5")
        with h5py.File(soc_path, "w") as f:
            f.create_dataset("X",      data=X_norm)
            f.create_dataset("y",      data=y_all)
            f.create_dataset("X_mean", data=X_mean)
            f.create_dataset("X_std",  data=X_std)
            f.attrs["window_size"] = WINDOW_SIZE
            f.attrs["features"]    = ["voltage_V", "current_A", "temperature_C", "chem_id"]
            f.attrs["n_scenarios"] = len(all_soc_X)
            f.attrs["version"]     = "v2"
        print(f"SoC dataset: {soc_path}  shape={X_norm.shape}")

        stats = {
            "voltage":     {"mean": float(X_mean[0]), "std": float(X_std[0])},
            "current":     {"mean": float(X_mean[1]), "std": float(X_std[1])},
            "temperature": {"mean": float(X_mean[2]), "std": float(X_std[2])},
            "features":    ["voltage_V", "current_A", "temperature_C", "chem_id"],
            "window_size": WINDOW_SIZE,
            "version":     "v2",
        }
        for path in [
            os.path.join(OUTPUT_DIR, "normalisation_stats.json"),
            os.path.join(OUTPUT_DIR, "models", "normalisation_stats.json"),
        ]:
            if os.path.exists(os.path.dirname(path)):
                with open(path, "w") as f:
                    json.dump(stats, f, indent=2)

    # ── SoH dataset — separate per chemistry ─────────────────────────────────
    SOH_FEATURES = [
        "c_rate", "temp_mean_c", "temp_min_c", "temp_max_c",
        "current_mean_a", "soc_target", "nominal_cap_ah",
    ]
    for chem_name in ["NMC", "LFP"]:
        data = soh_by_chem[chem_name]
        if not data["X"]:
            continue
        Xs = np.array(data["X"], dtype=np.float32)
        ys = np.array(data["y"], dtype=np.float32)
        path = os.path.join(OUTPUT_DIR, f"soh_training_data_{chem_name}.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=Xs)
            f.create_dataset("y", data=ys)
            f.attrs["features"] = SOH_FEATURES
            f.attrs["target"]   = "median_resistance_ohm"
            f.attrs["version"]  = "v2"
            f.attrs["n_scenarios"] = len(Xs)
        print(f"SoH {chem_name}: {path}  shape={Xs.shape}"
              f"  R=[{ys.min():.5f},{ys.max():.5f}]Ω")

    # Keep combined file for backward compat
    all_X = np.concatenate([np.array(soh_by_chem[c]["X"]) for c in ["NMC","LFP"]
                             if soh_by_chem[c]["X"]], axis=0).astype(np.float32)
    all_y = np.concatenate([np.array(soh_by_chem[c]["y"]) for c in ["NMC","LFP"]
                             if soh_by_chem[c]["y"]], axis=0).astype(np.float32)
    chem_col_nmc = np.zeros(len(soh_by_chem["NMC"]["y"]), dtype=np.float32)
    chem_col_lfp = np.ones(len(soh_by_chem["LFP"]["y"]),  dtype=np.float32)
    chem_col = np.concatenate([chem_col_nmc, chem_col_lfp])
    all_X_with_chem = np.column_stack([all_X, chem_col])

    combined_path = os.path.join(OUTPUT_DIR, "soh_training_data.h5")
    with h5py.File(combined_path, "w") as f:
        f.create_dataset("X", data=all_X_with_chem)
        f.create_dataset("y", data=all_y)
        f.attrs["features"] = SOH_FEATURES + ["chem_id"]
        f.attrs["target"]   = "median_resistance_ohm"
        f.attrs["version"]  = "v2"
    print(f"SoH combined: {combined_path}  shape={all_X_with_chem.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
