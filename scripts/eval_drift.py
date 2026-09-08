"""
scripts/eval_drift.py
---------------------
Monitors statistical drift in the latent space (z_t) to detect when 
the Hybrid model requires recalibration.
Uses the Kolmogorov-Smirnov (KS) test between training baseline and recent data.
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
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sequence_dataset import NetworkSequenceDataset
from src.models.gru_world_model import GRUWorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_latent_embeddings(model, X, Y_fut, seq_len=30):
    ds = NetworkSequenceDataset(X, Y_fut, np.zeros(len(X)), seq_len)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    all_z = []
    with torch.no_grad():
        for x_seq, _, _, _ in loader:
            _, _, _, z_t = model(x_seq)
            all_z.append(z_t.numpy())
            
    return np.concatenate(all_z, axis=0)

def main():
    DATA_DIR    = ROOT / "data" / "processed" / "windows"
    SCALER_PATH = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
    GRU_PATH    = ROOT / "models" / "gru" / "gru_world_model.pth"
    METRICS_DIR = ROOT / "outputs" / "metrics"
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    scaler = joblib.load(SCALER_PATH)

    logging.info("Loading Train (Baseline) Data...")
    train = pd.read_parquet(DATA_DIR / "train_windows_30s.parquet")
    
    exclude = {"Timestamp", "Label", "attack_stage", "future_attack_1", "future_attack_3", "future_attack_5"}
    features = [c for c in train.columns if c not in exclude]

    X_train = scaler.transform(train[features].fillna(0).values)
    Y_train_fut = train[["future_attack_1", "future_attack_3", "future_attack_5"]].values
    
    logging.info("Loading Unknown Test (Recent) Data...")
    test_u = pd.read_parquet(DATA_DIR / "test_unknown_windows_30s.parquet")
    X_test = scaler.transform(test_u[features].fillna(0).values)
    Y_test_fut = test_u[["future_attack_1", "future_attack_3", "future_attack_5"]].values

    model = GRUWorldModel(input_dim=X_train.shape[1], hidden_dim=128, num_layers=2)
    model.load_state_dict(torch.load(GRU_PATH, map_location="cpu"))

    logging.info("Extracting embeddings...")
    # Sample a subset to keep computation reasonable
    sample_size = 10000
    if len(X_train) > sample_size:
        idx_tr = np.random.choice(len(X_train), sample_size, replace=False)
        X_train, Y_train_fut = X_train[idx_tr], Y_train_fut[idx_tr]
    if len(X_test) > sample_size:
        idx_te = np.random.choice(len(X_test), sample_size, replace=False)
        X_test, Y_test_fut = X_test[idx_te], Y_test_fut[idx_te]

    z_train = get_latent_embeddings(model, X_train, Y_train_fut)
    z_test = get_latent_embeddings(model, X_test, Y_test_fut)

    # Calculate KS statistic for each dimension of z_t (dim=128)
    logging.info("Calculating Kolmogorov-Smirnov tests for drift...")
    p_values = []
    ks_stats = []
    for dim in range(z_train.shape[1]):
        stat, p_val = ks_2samp(z_train[:, dim], z_test[:, dim])
        ks_stats.append(stat)
        p_values.append(p_val)
        
    # A feature has drifted if p_value < 0.05
    drifted_dims = sum(1 for p in p_values if p < 0.05)
    drift_ratio = drifted_dims / z_train.shape[1]
    
    requires_recalibration = drift_ratio > 0.30  # If >30% of latent dimensions drift
    
    results = {
        "latent_dimensions": z_train.shape[1],
        "drifted_dimensions_count": drifted_dims,
        "drift_ratio": float(drift_ratio),
        "mean_ks_statistic": float(np.mean(ks_stats)),
        "requires_recalibration": bool(requires_recalibration),
        "p_value_threshold": 0.05,
        "recalibration_threshold_ratio": 0.30
    }

    print("\n" + "="*50)
    print("STATISTICAL MODEL DRIFT MONITORING")
    print("="*50)
    print(f"Drifted Latent Dims : {results['drifted_dimensions_count']} / {results['latent_dimensions']}")
    print(f"Drift Ratio         : {results['drift_ratio']:.2%}")
    print(f"Mean KS Statistic   : {results['mean_ks_statistic']:.4f}")
    if requires_recalibration:
        print(">>> ALERT: Significant Concept Drift Detected. Recalibration Required. <<<")
    else:
        print(">>> STATUS: Model is stable within baseline parameters. <<<")
    print("="*50)

    out_path = METRICS_DIR / "drift_monitoring.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
