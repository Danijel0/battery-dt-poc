"""
SoH Validation — Forklift LFP Degradation Dataset
===================================================
Extracts DCIR from HPPC pulses in RPT files across 58 aging rounds,
computes relative resistance increase R/R_nominal, and compares against
GPR SoH model predictions.

Dataset: Vilsen and Stroe (2023), Mendeley Data
  - 3 LFP prismatic cells, 180Ah, forklift duty cycle
  - 58 aging rounds, RPT every 2 weeks
  - HPPC pulses: ±90A (0.5C) and ±45A (0.25C)

Usage:
    python scripts/validate_soh_forklift.py --data
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data",      default="forklift_dataset")
parser.add_argument("--models",    default="data/models")
parser.add_argument("--output",    default="data/validation")
parser.add_argument("--pulse_a",   default=90.0, type=float,
                    help="HPPC discharge pulse magnitude [A]")
parser.add_argument("--dt_dcir",   default=10.0, type=float,
                    help="Time window [s] for DCIR calculation")
args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

# ── DCIR extraction ───────────────────────────────────────────────────────────

def extract_dcir_from_rpt(csv_path: Path, pulse_threshold: float = 80.0,
                           dt: float = 10.0) -> list[dict]:
    """
    Extract DCIR from HPPC discharge pulses in an RPT file.

    DCIR = ΔV / ΔI where:
      ΔI = current at pulse start - current before pulse (rest)
      ΔV = voltage drop dt seconds into pulse vs. pre-pulse OCV

    Returns list of dicts: {round, soc_approx, dcir_ohm, temp_c}
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # Identify discharge pulse starts: current crosses threshold downward
    I = df["current"].values
    V = df["voltage"].values
    T = df["temperature"].values
    t = df["time"].values

    results = []
    in_pulse = False

    for i in range(1, len(I)):
        # Pulse start: current crosses -pulse_threshold (discharge = negative)
        if not in_pulse and I[i] < -pulse_threshold and I[i-1] > -pulse_threshold:
            in_pulse = True
            t_start = t[i]
            temp    = T[i]
            i_pre   = I[i-1]   # ~0A at rest

            # Find stable OCV: last rest sample where |I| < 1A within 30s before pulse
            # This avoids SoC-reset voltage jumps at t=0 of each HPPC sub-test
            rest_voltages = []
            for k in range(max(0, i-30), i):
                if abs(I[k]) < 1.0:
                    rest_voltages.append(V[k])
            if len(rest_voltages) < 3:
                in_pulse = False
                continue
            v_pre = np.median(rest_voltages)  # robust OCV estimate

            # Find sample dt seconds into pulse
            j = i
            while j < len(t) and t[j] - t_start < dt:
                j += 1
            if j >= len(I):
                continue

            v_pulse = V[j]
            i_pulse = I[j]

            delta_v = v_pre - v_pulse           # IR drop (positive for discharge)
            delta_i = abs(i_pulse) - abs(i_pre) # current step (positive)

            if delta_i < 10.0:  # ignore if current step too small
                continue
            if delta_v <= 0:    # ignore if voltage increases (charging artifact)
                continue
            if v_pre < 3.0:     # ignore pulses at very low SoC (below LFP plateau)
                continue

            dcir = delta_v / delta_i            # Ohm

            # Approximate SoC from energy balance (rough)
            energy_wh = df["energy"].values[i] if "energy" in df.columns else np.nan
            cap_ah    = 180.0
            soc_approx = max(0.0, min(1.0, 1.0 - energy_wh / (cap_ah * 3.2)))

            if 0.001 < dcir < 1.0:   # sanity filter
                results.append({
                    "dcir_ohm":   dcir,
                    "soc_approx": soc_approx,
                    "temp_c":     temp,
                    "t_pulse":    t_start,
                })

        elif in_pulse and abs(I[i]) < 5.0:
            in_pulse = False

    return results


# ── Load all RPT files ────────────────────────────────────────────────────────

data_root = Path(args.data)
records   = []

for cell_dir in sorted(data_root.glob("Cell*")):
    cell_id = int(cell_dir.name.replace("Cell", ""))
    for round_dir in sorted(cell_dir.glob("Round*")):
        round_id = int(round_dir.name.replace("Round", ""))
        rpt_file = round_dir / "RPT.csv"
        if not rpt_file.exists():
            continue

        pulses = extract_dcir_from_rpt(rpt_file,
                                        pulse_threshold=args.pulse_a * 0.8,
                                        dt=args.dt_dcir)
        if not pulses:
            continue

        # Use median DCIR per round (robust to outlier pulses)
        dcir_vals = [p["dcir_ohm"] for p in pulses]
        temp_vals = [p["temp_c"]   for p in pulses]

        records.append({
            "cell":      cell_id,
            "round":     round_id,
            "dcir_ohm":  np.median(dcir_vals),
            "dcir_std":  np.std(dcir_vals),
            "n_pulses":  len(dcir_vals),
            "temp_mean": np.mean(temp_vals),
        })
        print(f"  Cell{cell_id} Round{round_id:02d}: "
              f"DCIR={np.median(dcir_vals)*1000:.3f}mΩ "
              f"(n={len(dcir_vals)}, T={np.mean(temp_vals):.1f}°C)")

df_all = pd.DataFrame(records)
if df_all.empty:
    print("No DCIR data extracted — check pulse threshold or file paths.")
    raise SystemExit(1)

print(f"\nExtracted {len(df_all)} round measurements across "
      f"{df_all['cell'].nunique()} cells")

# ── Normalise to R/R_nominal ──────────────────────────────────────────────────
# R_nominal = median DCIR at Round 1 (first RPT with HPPC pulses).
# Round 0 contains only capacity test without HPPC pulses.
r_nom = df_all[df_all["round"] == 1]["dcir_ohm"].median()
print(f"R_nominal (BOL median): {r_nom*1000:.3f} mΩ")

df_all["r_rel"]   = df_all["dcir_ohm"] / r_nom    # R / R_nominal
df_all["soh_pct"] = (1.0 - (1.0 - 0.80) *
                     (df_all["dcir_ohm"] - r_nom) /
                     (2 * r_nom - r_nom)).clip(0, 1) * 100

# ── GPR prediction for comparison ────────────────────────────────────────────
# Build feature vectors matching training: [c_rate, temp_mean_c, temp_min_c,
# temp_max_c, current_mean_a, soc_target, nominal_cap_ah]
# Forklift dataset: LFP, 180Ah, 0.5C HPPC pulses
# GPR was trained on Prada2013 2.3Ah — use same feature structure
# but note capacity mismatch: comparison is relative, not absolute.

try:
    with open(os.path.join(args.models, "gpr_soh_LFP.pkl"),  "rb") as f:
        gpr = pickle.load(f)
    with open(os.path.join(args.models, "gpr_soh_LFP_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    gpr_preds = []
    for _, row in df_all.iterrows():
        # LFP GPR features: [c_rate, temp_mean_c, current_mean_a,
        #                     soc_target, nominal_cap_ah]
        # Use 0.5C (90A / 180Ah), temperature from dataset,
        # soc_target=0.5 (mid-discharge HPPC), nominal_cap=2.3 (training cell)
        feat = np.array([[
            0.5,               # c_rate (HPPC pulse = 0.5C)
            row["temp_mean"],  # temp_mean_c
            90.0 * 2.3 / 180.0,  # current_mean_a scaled to training cell
            0.5,               # soc_target (mid-discharge)
            2.3,               # nominal_cap_ah (training cell)
        ]])
        feat_s      = scaler.transform(feat)
        r_pred, r_std = gpr.predict(feat_s, return_std=True)
        gpr_preds.append({
            "round":   row["round"],
            "cell":    row["cell"],
            "r_pred":  float(r_pred[0]),
            "r_std":   float(r_std[0]),
        })

    df_gpr = pd.DataFrame(gpr_preds)
    # Normalise GPR predictions the same way
    r_gpr_nom = df_gpr[df_gpr["round"] == 0]["r_pred"].mean()
    df_gpr["r_rel_pred"] = df_gpr["r_pred"] / r_gpr_nom
    have_gpr = True
    print(f"GPR R_nominal (LFP model): {r_gpr_nom*1000:.3f} mΩ")

except Exception as e:
    print(f"GPR model not loaded: {e}")
    have_gpr = False

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SoH Validation — Forklift LFP Dataset vs GPR Model\n"
             "Vilsen and Stroe (2023), 180Ah prismatic cells, forklift duty cycle",
             fontsize=11)

colors = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800"}

# Panel A: Relative resistance increase over rounds
ax = axes[0]
for cell_id, grp in df_all.groupby("cell"):
    grp = grp.sort_values("round")
    ax.errorbar(grp["round"], grp["r_rel"],
                yerr=grp["dcir_std"] / r_nom,
                label=f"Cell {cell_id} (measured)",
                color=colors[cell_id], marker="o", markersize=4,
                capsize=3, linewidth=1.5)

if have_gpr:
    # GPR predicts constant R for same operating conditions — show as band
    r_rel_gpr_mean = df_gpr["r_rel_pred"].mean()
    r_rel_gpr_std  = df_gpr["r_rel_pred"].std()
    ax.axhline(r_rel_gpr_mean, color="red", linestyle="--",
               label=f"GPR prediction (mean={r_rel_gpr_mean:.3f})")
    ax.axhspan(r_rel_gpr_mean - r_rel_gpr_std,
               r_rel_gpr_mean + r_rel_gpr_std,
               alpha=0.15, color="red", label="GPR ±1σ")

ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="R_nominal (BOL)")
ax.axhline(2.0, color="darkred", linestyle=":", linewidth=1, label="EOL (R = 2×R_nom)")
ax.set_xlabel("Aging Round")
ax.set_ylabel("R / R_nominal")
ax.set_title("A. Relative Resistance Increase")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# Panel B: SoH% over rounds
ax2 = axes[1]
for cell_id, grp in df_all.groupby("cell"):
    grp = grp.sort_values("round")
    ax2.plot(grp["round"], grp["soh_pct"],
             label=f"Cell {cell_id} (measured)",
             color=colors[cell_id], marker="o", markersize=4, linewidth=1.5)

ax2.axhline(80.0, color="darkred", linestyle="--", linewidth=1.5,
            label="EOL threshold (80%)")
ax2.set_xlabel("Aging Round")
ax2.set_ylabel("SoH [%]")
ax2.set_title("B. State of Health over Aging Rounds")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(60, 105)

plt.tight_layout()
plot_path = os.path.join(args.output, "soh_validation_forklift.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {plot_path}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────")
print(f"R_nominal (BOL):      {r_nom*1000:.3f} mΩ")
print(f"R at last round:      {df_all.groupby('round')['dcir_ohm'].median().iloc[-1]*1000:.3f} mΩ")
print(f"R/R_nom at last round:{df_all.groupby('round')['r_rel'].median().iloc[-1]:.3f}")
print(f"SoH at last round:    {df_all.groupby('round')['soh_pct'].median().iloc[-1]:.1f}%")
print(f"Rounds to EOL (est):  not reached" if df_all["r_rel"].max() < 2.0
      else f"EOL reached at round: {df_all[df_all['r_rel']>=2.0]['round'].min()}")

# Save numeric results
results = {
    "r_nominal_mohm":     round(r_nom * 1000, 4),
    "r_eol_mohm":         round(2 * r_nom * 1000, 4),
    "n_cells":            int(df_all["cell"].nunique()),
    "n_rounds":           int(df_all["round"].nunique()),
    "r_rel_final":        round(df_all.groupby("round")["r_rel"].median().iloc[-1], 4),
    "soh_final_pct":      round(df_all.groupby("round")["soh_pct"].median().iloc[-1], 2),
    "eol_reached":        bool(df_all["r_rel"].max() >= 2.0),
    "gpr_available":      have_gpr,
}
if have_gpr:
    results["gpr_r_rel_mean"] = round(r_rel_gpr_mean, 4)
    results["gpr_r_rel_std"]  = round(r_rel_gpr_std, 4)

out_path = os.path.join(args.output, "soh_validation_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {out_path}")
