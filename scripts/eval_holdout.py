"""
scripts/eval_holdout.py
------------------------
Test the Hybrid model against a genuinely held-out attack family:
  - DoS slowloris   (slow HTTP attack — volumetric)
  - DoS Slowhttptest (slow headers/body)
  - Bot              (C2 beaconing — very different traffic pattern)

These three attack types are excluded from ALL training data in create_windows.py
(HELD_OUT_ATTACKS list), so this evaluation represents true zero-day generalisation.

Outputs:
  outputs/metrics/holdout_family_evaluation.json
  outputs/metrics/holdout_family_evaluation.csv
"""

import sys, json, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader

from src.data.sequence_dataset import NetworkSequenceDataset
from src.models.gru_world_model import GRUWorldModel
from src.models.fusion_model import RiskFusionEngine
from src.utils.metrics import evaluate_classification, per_class_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_DIR    = ROOT / "data"   / "processed" / "windows"
SCALER_PATH = ROOT / "models" / "scalers"   / "baseline_scaler.pkl"
GRU_PATH    = ROOT / "models" / "gru"       / "gru_world_model.pth"
FUSION_PFX  = str(ROOT / "models" / "gru"  / "fusion")
METRICS_DIR = ROOT / "outputs" / "metrics"
THRESHOLD_F = METRICS_DIR / "optimal_threshold.json"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

HELD_OUT_ATTACKS = ["DoS slowloris", "DoS Slowhttptest", "Bot"]

scaler    = joblib.load(SCALER_PATH)
threshold = json.load(open(THRESHOLD_F)).get("threshold", 0.10) if THRESHOLD_F.exists() else 0.10

train  = pd.read_parquet(DATA_DIR / "train_windows_30s.parquet")
test_u = pd.read_parquet(DATA_DIR / "test_unknown_windows_30s.parquet")

exclude  = {"Timestamp", "Label", "attack_stage", "future_attack_1", "future_attack_3", "future_attack_5"}
features = [c for c in train.columns if c not in exclude]

model = GRUWorldModel(input_dim=len(features), hidden_dim=128, num_layers=2)
model.load_state_dict(torch.load(str(GRU_PATH), map_location="cpu"))
model.eval()

engine = RiskFusionEngine()
engine.load(FUSION_PFX)

# ── Get risk predictions on the unknown test set ───────────────────────────────
X     = scaler.transform(test_u[features].fillna(0).values)
Y_fut = test_u[["future_attack_1", "future_attack_3", "future_attack_5"]].values
ds    = NetworkSequenceDataset(X, Y_fut, np.zeros(len(X)), 30)
loader = DataLoader(ds, batch_size=128, shuffle=False)

all_risk, all_p = [], []
with torch.no_grad():
    for x_seq, y_next, y_fut, _ in loader:
        next_pred, fut_logits, _, z_t = model(x_seq)
        p   = torch.sigmoid(fut_logits[:, 0]).cpu().numpy()
        err = torch.mean((next_pred - y_next) ** 2, dim=1).cpu().numpy()
        risk, _, _ = engine.predict_risk(p, z_t.numpy(), err)
        all_risk.extend(risk.tolist())
        all_p.extend(p.tolist())

risk_arr = np.array(all_risk)
pred_arr = (risk_arr > threshold).astype(int)

# Align with dataframe
seq_len     = 30
test_u_trim = test_u.iloc[seq_len : seq_len + len(pred_arr)].copy()
y_true_bin  = (test_u_trim["Label"] != "BENIGN").astype(int).values

# ── Overall on Unknown set ─────────────────────────────────────────────────────
overall = evaluate_classification(y_true_bin, pred_arr, risk_arr)

SEP = "=" * 70
print(f"\n{SEP}")
print("HOLDOUT FAMILY EVALUATION (Genuinely unseen attack families)")
print(SEP)
print(f"  Unknown test set size  : {len(test_u_trim)} windows (after seq-len offset)")
print(f"  Risk threshold used    : {threshold:.2f}")
print(f"  Overall F1             : {overall['F1-score']:.4f}")
print(f"  Overall Precision      : {overall['Precision']:.4f}")
print(f"  Overall Recall         : {overall['Recall']:.4f}")
print(f"  Overall FPR            : {overall['FPR']:.4f}")
print(f"  PR-AUC                 : {overall['PR-AUC'] or 0:.4f}")

# ── Per held-out attack family ─────────────────────────────────────────────────
present_attacks = [a for a in HELD_OUT_ATTACKS if a in test_u_trim["Label"].values]
per_attack_df   = per_class_evaluation(test_u_trim["Label"].tolist(), pred_arr, present_attacks)

print(f"\n  Per held-out attack family:")
print(per_attack_df.to_string(index=False))
print(SEP)

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "held_out_attacks": HELD_OUT_ATTACKS,
    "threshold":        float(threshold),
    "overall":          overall,
    "per_family":       per_attack_df.to_dict(orient="records"),
}
json_path = METRICS_DIR / "holdout_family_evaluation.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

csv_path = METRICS_DIR / "holdout_family_evaluation.csv"
per_attack_df.to_csv(csv_path, index=False)

logging.info("Saved %s and %s", json_path, csv_path)
