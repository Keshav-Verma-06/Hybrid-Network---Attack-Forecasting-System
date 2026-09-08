"""
scripts/eval_independent.py
---------------------------
Simulates independent validation using an enterprise or CII-relevant dataset
(e.g. CIC-IDS-2018 or UNSW-NB15). Since we don't have the raw pcaps for these
alternative datasets, this script generates a synthetic proxy dataset with 
different statistical distributions (different topologies) to test generalisation.
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

def generate_synthetic_enterprise_data(X_base, Y_fut):
    """
    Generates a synthetic enterprise dataset by altering the underlying feature
    distributions, mimicking a different network topology (e.g. higher background noise,
    different ratio of TCP/UDP traffic, larger payload sizes).
    """
    X_syn = X_base.copy()
    
    # 1. Increase baseline noise across all features (mimicking higher enterprise traffic volume)
    noise = np.random.normal(0, 0.2, X_syn.shape)
    X_syn += noise
    
    # 2. Shift means of certain features to simulate different topologies
    # (assuming standard scaled features, adding a constant shifts the mean)
    X_syn[:, 0::4] += 0.5  # E.g. larger average packet sizes
    X_syn[:, 1::4] -= 0.3  # E.g. different port distributions
    
    # Clip to keep somewhat realistic bounds (though scaler handles normalisation)
    X_syn = np.clip(X_syn, -5.0, 5.0)
    
    return X_syn, Y_fut

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
    
    # Use Unknown dataset as base to generate synthetic enterprise dataset
    test_u = pd.read_parquet(DATA_DIR / "test_unknown_windows_30s.parquet")

    exclude = {"Timestamp", "Label", "attack_stage", "future_attack_1", "future_attack_3", "future_attack_5"}
    features = [c for c in test_u.columns if c not in exclude]

    X_base = scaler.transform(test_u[features].fillna(0).values)
    Y_fut = test_u[["future_attack_1", "future_attack_3", "future_attack_5"]].values

    logging.info("Generating synthetic enterprise topology dataset (proxy for independent validation)...")
    X_ind, Y_ind = generate_synthetic_enterprise_data(X_base, Y_fut)

    model = GRUWorldModel(input_dim=X_base.shape[1], hidden_dim=128, num_layers=2)
    model.load_state_dict(torch.load(GRU_PATH, map_location="cpu"))
    engine = RiskFusionEngine()
    engine.load(FUSION_PFX)

    logging.info("Evaluating on Independent Synthetic Topology...")
    y_true, y_pred, y_prob = get_preds(model, engine, X_ind, Y_ind, threshold=threshold)
    ind_metrics = evaluate_classification(y_true, y_pred, y_prob)

    results = {
        "dataset": "Synthetic_Enterprise_Topology",
        "threshold": float(threshold),
        "independent_f1": float(ind_metrics["F1-score"]),
        "independent_precision": float(ind_metrics["Precision"]),
        "independent_recall": float(ind_metrics["Recall"]),
        "independent_fpr": float(ind_metrics["FPR"]),
        "independent_prauc": float(ind_metrics.get("PR-AUC", 0.0))
    }

    print("\n" + "="*60)
    print("INDEPENDENT TOPOLOGY VALIDATION")
    print("="*60)
    print(f"Dataset      : {results['dataset']}")
    print(f"Overall F1   : {results['independent_f1']:.4f}")
    print(f"Precision    : {results['independent_precision']:.4f}")
    print(f"Recall       : {results['independent_recall']:.4f}")
    print(f"FPR          : {results['independent_fpr']:.4f}")
    print("="*60)

    out_path = METRICS_DIR / "independent_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
