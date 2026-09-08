"""
scripts/eval_per_class.py
--------------------------
Evaluate the Hybrid model per attack type and per MITRE stage
on both Known and Unknown test sets.

Outputs:
  outputs/metrics/per_attack_evaluation.csv
  outputs/metrics/per_stage_evaluation.csv
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
from src.utils.metrics import per_class_evaluation, per_stage_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data" / "processed" / "windows"
SCALER_PATH = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
GRU_PATH    = ROOT / "models" / "gru" / "gru_world_model.pth"
FUSION_PFX  = str(ROOT / "models" / "gru" / "fusion")
METRICS_DIR = ROOT / "outputs" / "metrics"
THRESHOLD_F = ROOT / "outputs" / "metrics" / "optimal_threshold.json"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
scaler = joblib.load(SCALER_PATH)
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


def get_risk_scores(df):
    """Extract hybrid risk scores for every window in a dataframe."""
    X     = scaler.transform(df[features].fillna(0).values)
    Y_fut = df[["future_attack_1", "future_attack_3", "future_attack_5"]].values
    ds    = NetworkSequenceDataset(X, Y_fut, np.zeros(len(X)), 30)
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    all_risk, all_p, all_z, all_err = [], [], [], []
    with torch.no_grad():
        for x_seq, y_next, y_fut, _ in loader:
            next_pred, fut_logits, _, z_t = model(x_seq)
            p = torch.sigmoid(fut_logits[:, 0]).cpu().numpy()
            err = torch.mean((next_pred - y_next) ** 2, dim=1).cpu().numpy()
            risk, _, _ = engine.predict_risk(p, z_t.numpy(), err)
            all_risk.extend(risk.tolist())
            all_p.extend(p.tolist())
    return np.array(all_risk), np.array(all_p)


logging.info("Computing risk scores for Known test set...")
risk_k, p_k = get_risk_scores(test_k)
pred_k = (risk_k > threshold).astype(int)

logging.info("Computing risk scores for Unknown test set...")
risk_u, p_u = get_risk_scores(test_u)
pred_u = (risk_u > threshold).astype(int)


# ── Trim to sequence-dataset length (first 30 windows used as context) ────────
def trim_df(df, pred):
    offset = 30  # NetworkSequenceDataset skips first seq_len rows
    trimmed_df = df.iloc[offset : offset + len(pred)].copy()
    return trimmed_df, pred


test_k_trim, pred_k_trim = trim_df(test_k, pred_k)
test_u_trim, pred_u_trim = trim_df(test_u, pred_u)

# ── Per-attack evaluation ─────────────────────────────────────────────────────
all_attacks = sorted(set(test_k_trim["Label"].tolist() + test_u_trim["Label"].tolist()) - {"BENIGN"})

logging.info("Running per-attack evaluation...")

rows_attack = []
for split_name, df_trim, pred in [
    ("Known",   test_k_trim, pred_k_trim),
    ("Unknown", test_u_trim, pred_u_trim),
]:
    result = per_class_evaluation(df_trim["Label"].tolist(), pred, all_attacks)
    result.insert(0, "Split", split_name)
    rows_attack.append(result)

attack_df = pd.concat(rows_attack, ignore_index=True)
out_path  = METRICS_DIR / "per_attack_evaluation.csv"
attack_df.to_csv(out_path, index=False)
logging.info("Saved %s", out_path)

# ── Per-stage evaluation ──────────────────────────────────────────────────────
logging.info("Running per-stage evaluation...")

rows_stage = []
for split_name, df_trim, pred in [
    ("Known",   test_k_trim, pred_k_trim),
    ("Unknown", test_u_trim, pred_u_trim),
]:
    result = per_stage_evaluation(df_trim["attack_stage"].tolist(), pred)
    result.insert(0, "Split", split_name)
    rows_stage.append(result)

stage_df  = pd.concat(rows_stage, ignore_index=True)
out_path2 = METRICS_DIR / "per_stage_evaluation.csv"
stage_df.to_csv(out_path2, index=False)
logging.info("Saved %s", out_path2)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PER-ATTACK EVALUATION")
print("=" * 70)
print(attack_df.to_string(index=False))

print("\n" + "=" * 70)
print("PER-STAGE EVALUATION")
print("=" * 70)
print(stage_df.to_string(index=False))
