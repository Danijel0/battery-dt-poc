"""
GPR SoH Model Training v3 — per-chemistry, scenario-aggregated
===============================================================
Reads soh_training_data_NMC.h5 and soh_training_data_LFP.h5
produced by generate_training_data.py v2.

Key changes from v2:
  - One datapoint per PyBaMM scenario (not sliding windows).
    Sliding windows are correlated → GPR learns false structure
    → enormous uncertainty outside training points.
  - Updated feature set (8 features, no chem_id — separate models):
      [c_rate, temp_mean_c, temp_min_c, temp_max_c,
       current_mean_a, soc_target, nominal_cap_ah]
  - temp_init_c removed: in production this equals df.iloc[0] from
    a 30-day InfluxDB query — an arbitrary value, not the physical
    start temperature. temp_min_c is stable and meaningful.
  - LOO cross-validation (Leave-One-Out) instead of train/test split:
    correct for small n (45-58 scenarios per chemistry).

SoH% conversion (unchanged from v2):
  SoH = 80% at R = 2 × R_nominal (100% DCIR increase = EOL).
"""

import os
import json
import h5py
import pickle
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODEL_DIR = "data/models"
DATA_DIR  = "data"
SEED      = 42
os.makedirs(MODEL_DIR, exist_ok=True)

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

SOH_FEATURES = [
    "c_rate", "temp_mean_c", "temp_min_c", "temp_max_c",
    "current_mean_a", "soc_target", "nominal_cap_ah",
]


def resistance_to_soh(r: float, chem: str) -> float:
    p     = SOH_PARAMS[chem]
    r_nom = p["r_nominal"]
    r_eol = r_nom * p["r_eol_factor"]
    soh   = 1.0 - (1.0 - p["soh_eol"]) * (r - r_nom) / (r_eol - r_nom)
    return float(np.clip(soh * 100, 0.0, 100.0))


def load_chemistry(chem: str):
    path = os.path.join(DATA_DIR, f"soh_training_data_{chem}.h5")
    if not os.path.exists(path):
        # Fallback: slice from combined file (v1 data)
        print(f"  WARNING: {path} not found, slicing from combined file")
        with h5py.File(os.path.join(DATA_DIR, "soh_training_data.h5"), "r") as f:
            X_all = f["X"][:]
            y_all = f["y"][:]
        cid  = 0 if chem == "NMC" else 1
        mask = X_all[:, -1] == cid
        return X_all[mask][:, :-1], y_all[mask]   # drop chem_id column

    with h5py.File(path, "r") as f:
        X        = f["X"][:]
        y        = f["y"][:]
        features = list(f.attrs.get("features", SOH_FEATURES))
        n_scen   = int(f.attrs.get("n_scenarios", len(X)))
    print(f"  Loaded {chem}: {X.shape}  ({n_scen} scenarios)  "
          f"R=[{y.min():.5f},{y.max():.5f}] Ω")
    return X, y


def train_chemistry(chem: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  {chem}")
    print(f"{'='*55}")

    X, y = load_chemistry(chem)
    n    = len(X)

    # LFP uses SPM (isothermal): temp_min == temp_mean == temp_max (zero variance).
    # These redundant columns cause GPR kernel optimisation to collapse to a
    # degenerate flat prior (length_scale → ∞). Drop them for LFP only.
    # NMC uses SPMe+thermal so all three temp columns carry information.
    if chem == "LFP":
        # SOH_FEATURES idx: 0=c_rate, 1=temp_mean, 2=temp_min, 3=temp_max,
        #                    4=current_mean, 5=soc_target, 6=nominal_cap
        X = np.delete(X, [2, 3], axis=1)  # drop temp_min_c, temp_max_c
        features_used = [f for f in SOH_FEATURES if f not in ("temp_min_c", "temp_max_c")]
    else:
        features_used = SOH_FEATURES

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    ls_bounds = (0.1, 50.0)
    kernel = (
        1.0 * Matern(length_scale=1.0, nu=2.5,
                     length_scale_bounds=ls_bounds)
        + WhiteKernel(noise_level=1e-3,
                      noise_level_bounds=(1e-6, 0.1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=20,
        normalize_y=True,
        random_state=SEED,
    )
    gpr.fit(X_s, y)
    print(f"  Kernel: {gpr.kernel_}")

    # LOO cross-validation — correct for small n
    loo         = LeaveOneOut()
    loo_preds   = np.zeros(n)
    loo_actuals = np.zeros(n)
    for i, (tr, te) in enumerate(loo.split(X_s)):
        g = GaussianProcessRegressor(
            kernel=gpr.kernel_, normalize_y=True, random_state=SEED
        )
        g.fit(X_s[tr], y[tr])
        loo_preds[i]   = g.predict(X_s[te])[0]
        loo_actuals[i] = y[te[0]]

    rmse     = float(np.sqrt(mean_squared_error(loo_actuals, loo_preds)))
    mae      = float(mean_absolute_error(loo_actuals, loo_preds))
    y_range  = float(y.max() - y.min())
    rmse_pct = rmse / y_range * 100

    # In-sample uncertainty (sanity check — should be small)
    _, y_std_train = gpr.predict(X_s, return_std=True)

    print(f"\n  LOO-CV RMSE : {rmse:.6f} Ω  ({rmse_pct:.2f}%)")
    print(f"  LOO-CV MAE  : {mae:.6f} Ω")
    print(f"  In-sample σ : {y_std_train.mean():.6f} Ω (mean), "
          f"{y_std_train.max():.6f} Ω (max)")
    print(f"  n scenarios : {n}")
    print(f"  y range     : {y_range:.5f} Ω")

    # Sample predictions with SoH%
    y_pred_all, y_std_all = gpr.predict(X_s, return_std=True)
    print(f"\n  Sample in-sample predictions (not LOO):")
    print(f"  {'R_actual':>10}  {'R_pred':>10}  {'SoH_act%':>9}  {'SoH_pred%':>10}  {'2σ':>8}")
    for i in range(min(8, n)):
        a = y[i]; p = y_pred_all[i]; s = y_std_all[i]
        print(f"  {a:10.5f}  {p:10.5f}  {resistance_to_soh(a,chem):9.1f}  "
              f"{resistance_to_soh(p,chem):10.1f}  {2*s:8.5f}")

    # Save
    model_path  = os.path.join(MODEL_DIR, f"gpr_soh_{chem}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"gpr_soh_{chem}_scaler.pkl")
    with open(model_path,  "wb") as f: pickle.dump(gpr,    f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

    info = {
        "chemistry":       chem,
        "model_class":     "GaussianProcessRegressor",
        "kernel":          str(gpr.kernel_),
        "features":        features_used,
        "target":          "median_resistance_ohm",
        "aggregation":     "per_scenario",
        "n_scenarios":     n,
        "loo_rmse_ohm":    rmse,
        "loo_rmse_pct":    rmse_pct,
        "loo_mae_ohm":     mae,
        "y_range_ohm":     y_range,
        "r_nominal_ohm":   SOH_PARAMS[chem]["r_nominal"],
        "r_eol_ohm":       SOH_PARAMS[chem]["r_nominal"] * SOH_PARAMS[chem]["r_eol_factor"],
        "soh_eol_pct":     SOH_PARAMS[chem]["soh_eol"] * 100,
        "model_path":      model_path,
        "scaler_path":     scaler_path,
    }
    with open(os.path.join(MODEL_DIR, f"gpr_soh_{chem}_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n  Saved: {model_path}")
    return info


def train():
    results = {}
    for chem in ["NMC", "LFP"]:
        results[chem] = train_chemistry(chem)

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    for chem, info in results.items():
        status = "PASS" if info["loo_rmse_pct"] < 10.0 else "CHECK"
        print(f"  {chem}: LOO-RMSE={info['loo_rmse_ohm']:.6f} Ω"
              f"  ({info['loo_rmse_pct']:.2f}%)"
              f"  n={info['n_scenarios']}  [{status}]")
    print()
    print("  Note: LOO-CV RMSE < 10% is acceptable for small-n GPR.")
    print("  In-sample RMSE is not meaningful — GPR interpolates exactly.")

    combined = {
        "model_version": "gpr-soh-v3",
        "approach":      "per-chemistry, scenario-aggregated",
        "features":      SOH_FEATURES,
        "chemistries":   results,
        "soh_params":    SOH_PARAMS,
    }
    with open(os.path.join(MODEL_DIR, "gpr_soh_info.json"), "w") as f:
        json.dump(combined, f, indent=2)


if __name__ == "__main__":
    train()
