"""
scripts/eval_lead_time.py
--------------------------
Measure detection lead time: how many windows (and seconds) before
the first true attack onset the Hybrid model raises a True Positive alert.

Outputs:
  outputs/metrics/lead_time_results.json
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
from src.utils.metrics import compute_lead_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_DIR    = ROOT / "data"   / "processed" / "windows"
SCALER_PATH = ROOT / "models" / "scalers"   / "baseline_scaler.pkl"
GRU_PATH    = ROOT / "models" / "gru"       / "gru_world_model.pth"
FUSION_PFX  = str(ROOT / "models" / "gru"  / "fusion")
METRICS_DIR = ROOT / "outputs" / "metrics"
THRESHOLD_F = METRICS_DIR / "optimal_threshold.json"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

scaler    = joblib.load(SCALER_PATH)
threshold = json.load(open(THRESHOLD_F)).get("threshold", 0.10) if THRESHOLD_F.exists() else 0.10

train   = pd.read_parquet(DATA_DIR / "train_windows_30s.parquet")
test_k  = pd.read_parquet(DATA_DIR / "test_known_windows_30s.parquet")
test_u  = pd.read_parquet(DATA_DIR / "test_unknown_windows_30s.parquet")

exclude  = {"Timestamp", "Label", "attack_stage", "future_attack_1", "future_attack_3", "future_attack_5"}
features = [c for c in train.columns if c not in exclude]

model = GRUWorldModel(input_dim=len(features), hidden_dim=128, num_layers=2)
model.load_state_dict(torch.load(str(GRU_PATH), map_location="cpu"))
model.eval()

engine = RiskFusionEngine()
engine.load(FUSION_PFX)


def get_preds_full(df):
    """Return risk scores and binary predictions, aligned with df rows (with seq-len offset)."""
    X     = scaler.transform(df[features].fillna(0).values)
    Y_fut = df[["future_attack_1", "future_attack_3", "future_attack_5"]].values
    ds    = NetworkSequenceDataset(X, Y_fut, np.zeros(len(X)), 30)
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    all_risk = []
    with torch.no_grad():
        for x_seq, y_next, y_fut, _ in loader:
            next_pred, fut_logits, _, z_t = model(x_seq)
            p   = torch.sigmoid(fut_logits[:, 0]).cpu().numpy()
            err = torch.mean((next_pred - y_next) ** 2, dim=1).cpu().numpy()
            risk, _, _ = engine.predict_risk(p, z_t.numpy(), err)
            all_risk.extend(risk.tolist())

    risk_arr = np.array(all_risk)
    pred_arr = (risk_arr > threshold).astype(int)

    # Align with original dataframe (offset by seq_len)
    seq_len  = 30
    true_arr = df["current_attack_state"].values[seq_len : seq_len + len(pred_arr)]
    return true_arr, pred_arr


results = {}
SEP = "=" * 60
print()
print(SEP)
print("DETECTION LEAD TIME ANALYSIS")
print(SEP)

for split_name, df in [("Known", test_k), ("Unknown", test_u)]:
    if "current_attack_state" not in df.columns:
        df["current_attack_state"] = (df["Label"] != "BENIGN").astype(int)

    logging.info("Computing predictions for %s test set...", split_name)
    y_true, y_pred = get_preds_full(df)

    lt = compute_lead_time(y_true, y_pred, window_duration_s=30)
    results[split_name] = lt

    print(f"\n  {split_name} Test Set:")
    print(f"    Attack episodes detected : {lt['n_episodes']}")
    print(f"    Missed episodes          : {lt['missed_episodes']}")
    if lt["mean_lead_sec"] is not None:
        print(f"    Mean lead time           : {lt['mean_lead_sec']:.1f}s  ({lt['mean_lead_sec']/60:.1f} min)")
        print(f"    Median lead time         : {lt['median_lead_sec']:.1f}s")
        print(f"    Min lead time            : {lt['min_lead_sec']:.1f}s  (earliest warning)")
        print(f"    Max lead time            : {lt['max_lead_sec']:.1f}s")
    else:
        print("    No pre-attack alarms found (all episodes missed)")

print()
print(SEP)

out_path = METRICS_DIR / "lead_time_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
logging.info("Lead time results saved to %s", out_path)
