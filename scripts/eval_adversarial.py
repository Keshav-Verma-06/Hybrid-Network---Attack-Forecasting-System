"""
scripts/eval_adversarial.py
---------------------------
Evaluates the robustness of the Hybrid model against manipulated
or adversarial traffic.

Perturbations tested:
1. Gaussian Noise Injection (simulating missing/encrypted features)
2. Targeted Feature Manipulation (e.g. lowering SYN/ACK ratios to evade detection)
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sequence_dataset import NetworkSequenceDataset
from src.models.gru_world_model import GRUWorldModel
from src.models.fusion_model import RiskFusionEngine
from src.utils.metrics import evaluate_classification

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_preds(model, engine, X, Y_fut, seq_len=30, threshold=0.10):
    ds = NetworkSequenceDataset(X, Y_fut, np.zeros(len(X)), seq_len)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    all_risk = []
    with torch.no_grad():
        for x_seq, y_next, y_fut, _ in loader:
            next_pred, fut_logits, _, z_t = model(x_seq)
            p = torch.sigmoid(fut_logits[:, 0]).cpu().numpy()
            err = torch.mean((next_pred - y_next) ** 2, dim=1).cpu().numpy()
            risk, _, _ = engine.predict_risk(p, z_t.numpy(), err)
            all_risk.extend(risk.tolist())
            
    risk_arr = np.array(all_risk)
    pred_arr = (risk_arr > threshold).astype(int)
    true_arr = Y_fut[seq_len:seq_len+len(pred_arr), 0]
    return true_arr, pred_arr, risk_arr

def main():
    DATA_DIR    = ROOT / "data" / "processed" / "windows"
    SCALER_PATH = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
    GRU_PATH    = ROOT / "models" / "gru" / "gru_world_model.pth"
    FUSION_PFX  = str(ROOT / "models" / "gru" / "fusion")
    METRICS_DIR = ROOT / "outputs" / "metrics"
    THRESHOLD_F = METRICS_DIR / "optimal_threshold.json"

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    threshold = json.load(open(THRESHOLD_F)).get("threshold", 0.10) if THRESHOLD_F.exists() else 0.10

    scaler = joblib.load(SCALER_PATH)
    test_u = pd.read_parquet(DATA_DIR / "test_unknown_windows_30s.parquet")

    exclude = {"Timestamp", "Label", "attack_stage", "future_attack_1", "future_attack_3", "future_attack_5"}
    features = [c for c in test_u.columns if c not in exclude]
    
    X_base = scaler.transform(test_u[features].fillna(0).values)
    Y_fut = test_u[["future_attack_1", "future_attack_3", "future_attack_5"]].values

    model = GRUWorldModel(input_dim=X_base.shape[1], hidden_dim=128, num_layers=2)
    model.load_state_dict(torch.load(GRU_PATH, map_location="cpu"))
    engine = RiskFusionEngine()
    engine.load(FUSION_PFX)

    # 1. Baseline Performance
    logging.info("Evaluating Baseline (No Perturbation)...")
    y_true, y_pred, y_prob = get_preds(model, engine, X_base, Y_fut, threshold=threshold)
    base_metrics = evaluate_classification(y_true, y_pred, y_prob)

    # 2. Gaussian Noise (Simulating missing/encrypted features or sensor degradation)
    logging.info("Evaluating Gaussian Noise Perturbation (std=0.5)...")
    noise = np.random.normal(0, 0.5, X_base.shape)
    X_noise = X_base + noise
    y_true_n, y_pred_n, y_prob_n = get_preds(model, engine, X_noise, Y_fut, threshold=threshold)
    noise_metrics = evaluate_classification(y_true_n, y_pred_n, y_prob_n)

    # 3. Targeted Manipulation: Suppress SYN/ACK ratios and Volume
    logging.info("Evaluating Targeted Evasion (Suppressing TCP flags and volume features)...")
    X_evade = X_base.copy()
    
    if len(features) > 0:
        evade_feats = ["syn_to_ack_ratio", "flow_count", "outbound_bytes_ratio"]
        for feat in evade_feats:
            if feat in features:
                idx = list(features).index(feat)
                X_evade[:, idx] = X_evade[:, idx] * 0.1  # Reduce feature value by 90%
    else:
        # Fallback if feature names aren't tracked: just perturb first 3 features
        X_evade[:, :3] = X_evade[:, :3] * 0.1
        
    y_true_e, y_pred_e, y_prob_e = get_preds(model, engine, X_evade, Y_fut, threshold=threshold)
    evade_metrics = evaluate_classification(y_true_e, y_pred_e, y_prob_e)

    results = {
        "baseline_f1": float(base_metrics["F1-score"]),
        "baseline_recall": float(base_metrics["Recall"]),
        "noise_f1": float(noise_metrics["F1-score"]),
        "noise_recall": float(noise_metrics["Recall"]),
        "noise_f1_drop_pct": float((base_metrics["F1-score"] - noise_metrics["F1-score"]) / max(base_metrics["F1-score"], 1e-9) * 100),
        "evasion_f1": float(evade_metrics["F1-score"]),
        "evasion_recall": float(evade_metrics["Recall"]),
        "evasion_f1_drop_pct": float((base_metrics["F1-score"] - evade_metrics["F1-score"]) / max(base_metrics["F1-score"], 1e-9) * 100),
    }

    print("\n" + "="*60)
    print("ADVERSARIAL ROBUSTNESS RESULTS")
    print("="*60)
    print(f"Baseline F1: {results['baseline_f1']:.4f}")
    print(f"Noise F1:    {results['noise_f1']:.4f} (Drop: {results['noise_f1_drop_pct']:.1f}%)")
    print(f"Evasion F1:  {results['evasion_f1']:.4f} (Drop: {results['evasion_f1_drop_pct']:.1f}%)")
    print("="*60)

    out_path = METRICS_DIR / "adversarial_robustness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
