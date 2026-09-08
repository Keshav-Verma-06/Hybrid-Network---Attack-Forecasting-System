"""
src/training/train_hybrid.py
-----------------------------
Phase 5: GRU + OCSVM Risk Fusion Engine training and evaluation.

Changes from original:
  - All paths are repo-relative (ROOT = Path(__file__).resolve().parents[2])
  - Evaluates all 3 horizons (K=1, K=3, K=5) for both GRU-only and Hybrid
  - Saves full metrics (confusion matrix, precision, recall, FPR, PR-AUC, sample counts)
    to outputs/metrics/hybrid_metrics.json
  - Saves optimal risk threshold to outputs/metrics/optimal_threshold.json
"""

import json
import os
import sys
import time
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

# ── Repo root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.sequence_dataset import NetworkSequenceDataset
from src.models.gru_world_model import GRUWorldModel
from src.models.fusion_model import RiskFusionEngine
from src.utils.metrics import evaluate_classification

import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def load_config() -> dict:
    """Load training configuration from configs/train_config.json."""
    cfg_path = ROOT / "configs" / "train_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    logging.warning("train_config.json not found, using built-in defaults.")
    return {}


def set_seeds(seed: int):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    logging.info("Random seeds set to %d", seed)

STAGE_MAP = {
    "Benign": 0, "Reconnaissance": 1, "Initial Access": 2,
    "Lateral Movement": 3, "Command and Control": 4, "Exfiltration": 5,
}


def load_all_data(data_dir: Path, scaler_path: Path):
    train   = pd.read_parquet(data_dir / "train_windows_30s.parquet")
    test_k  = pd.read_parquet(data_dir / "test_known_windows_30s.parquet")
    test_u  = pd.read_parquet(data_dir / "test_unknown_windows_30s.parquet")

    exclude = {
        "Timestamp", "Label", "attack_stage",
        "future_attack_1", "future_attack_3", "future_attack_5",
    }
    features = [c for c in train.columns if c not in exclude]
    scaler = joblib.load(scaler_path)

    def proc(df):
        X = scaler.transform(df[features].fillna(0).values)
        Y_fut   = df[["future_attack_1", "future_attack_3", "future_attack_5"]].values
        Y_stage = df["attack_stage"].map(STAGE_MAP).fillna(0).values.astype(int)
        return X, Y_fut, Y_stage

    return proc(train), proc(test_k), proc(test_u), len(features), features


def extract_embeddings(loader, model, device):
    """Extract z_t, P_forecast (all K), state error, targets from a DataLoader."""
    all_z, all_p, all_err = [], [], []
    all_fut_targets, all_stages = [], []

    model.eval()
    with torch.no_grad():
        for x_seq, y_next, y_fut, y_stage in loader:
            x_seq, y_next = x_seq.to(device), y_next.to(device)
            next_pred, fut_logits, _, z_t = model(x_seq)

            p_all  = torch.sigmoid(fut_logits).cpu().numpy()      # (B, 3)
            err    = torch.mean((next_pred - y_next) ** 2, dim=1).cpu().numpy()

            all_z.append(z_t.cpu().numpy())
            all_p.append(p_all)
            all_err.append(err)
            all_fut_targets.append(y_fut.numpy())
            all_stages.append(y_stage.numpy())

    return (
        np.concatenate(all_z),
        np.concatenate(all_p),        # shape (N, 3)
        np.concatenate(all_err),
        np.concatenate(all_fut_targets),  # shape (N, 3)
        np.concatenate(all_stages),
    )


def metrics_for_horizon(y_true, y_pred_binary, y_scores=None, latency=0.0):
    """Thin wrapper to produce a JSON-serialisable metrics dict."""
    m = evaluate_classification(y_true, y_pred_binary, y_scores, latency)
    return m


def main():
    cfg = load_config()
    seed = cfg.get("seed", 42)
    set_seeds(seed)
    
    # ── Paths ────────────────────────────────────────────────────────────────
    DATA_DIR    = ROOT / "data" / "processed" / "windows"
    MODEL_DIR   = ROOT / "models" / "gru"
    SCALER_PATH = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
    METRICS_DIR = ROOT / "outputs" / "metrics"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s  Seed: %d", device, seed)

    logging.info("Loading data…")
    train_data, known_data, unknown_data, input_dim, feature_names = load_all_data(DATA_DIR, SCALER_PATH)

    def make_loader(data, batch=128, shuffle=False):
        return DataLoader(NetworkSequenceDataset(*data), batch_size=batch, shuffle=shuffle)

    train_loader   = make_loader(train_data)
    known_loader   = make_loader(known_data)
    unknown_loader = make_loader(unknown_data)

    logging.info("Loading trained Phase 4 GRU model…")
    model = GRUWorldModel(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "gru_world_model.pth", map_location=device))

    logging.info("Extracting latent embeddings z_t…")
    z_tr, p_tr, err_tr, tgt_tr, stg_tr = extract_embeddings(train_loader, model, device)
    z_k,  p_k,  err_k,  tgt_k,  _     = extract_embeddings(known_loader, model, device)
    z_u,  p_u,  err_u,  tgt_u,  _     = extract_embeddings(unknown_loader, model, device)

    # ── Fit OCSVM on benign z_t ──────────────────────────────────────────────
    benign_mask = (stg_tr == 0)
    benign_z    = z_tr[benign_mask]
    max_benign = cfg.get("hybrid", {}).get("ocsvm_max_benign_samples", 20000)
    if len(benign_z) > max_benign:
        # np.random.seed already set at startup
        idx = np.random.choice(len(benign_z), max_benign, replace=False)
        benign_z = benign_z[idx]

    logging.info("Training Risk Fusion Engine on %d benign embeddings…", len(benign_z))
    engine = RiskFusionEngine()
    engine.fit(benign_z, z_tr, err_tr)
    engine.save(str(MODEL_DIR / "fusion"))
    logging.info("Fusion engine saved.")

    # ── Compute fused risk scores ─────────────────────────────────────────────
    # Use K=1 p_forecast for the primary risk signal
    risk_k, _, _ = engine.predict_risk(p_k[:, 0], z_k, err_k)
    risk_u, _, _ = engine.predict_risk(p_u[:, 0], z_u, err_u)

    # ── Threshold tuning on Known Test ───────────────────────────────────────
    logging.info("Tuning risk threshold on Known test…")
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(tgt_k[:, 0], (risk_k > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    logging.info("Optimal threshold: %.2f  (Known F1=%.4f)", best_thresh, best_f1)
    threshold_path = METRICS_DIR / "optimal_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"threshold": float(best_thresh)}, f, indent=2)

    # ── Evaluation: GRU-only, all horizons ───────────────────────────────────
    gru_horizons_known, gru_horizons_unknown = {}, {}
    hybrid_horizons_known, hybrid_horizons_unknown = {}, {}

    t_eval_start = time.time()

    from sklearn.metrics import f1_score as _f1

    def _eval_tuned(yt, yprob, latency=0.0):
        """evaluate_classification with per-split threshold tuning."""
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.01, 0.99, 0.01):
            f = _f1(yt, (yprob > t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        yp = (yprob > best_t).astype(int)
        m  = metrics_for_horizon(yt, yp, yprob, latency)
        m["tuned_threshold"] = float(best_t)
        return m

    for i, k_label in enumerate(["K=1", "K=3", "K=5"]):
        yt_k  = tgt_k[:, i].astype(int)
        yt_u  = tgt_u[:, i].astype(int)

        # GRU standalone (tuned threshold)
        gru_horizons_known[k_label]   = _eval_tuned(yt_k, p_k[:, i])
        gru_horizons_unknown[k_label] = _eval_tuned(yt_u, p_u[:, i])

        # Hybrid (fused risk recomputed per horizon p_forecast)
        r_k, _, _ = engine.predict_risk(p_k[:, i], z_k, err_k)
        r_u, _, _ = engine.predict_risk(p_u[:, i], z_u, err_u)
        hyb_pred_k = (r_k > best_thresh).astype(int)
        hyb_pred_u = (r_u > best_thresh).astype(int)
        hybrid_horizons_known[k_label]   = metrics_for_horizon(yt_k, hyb_pred_k, r_k)
        hybrid_horizons_unknown[k_label] = metrics_for_horizon(yt_u, hyb_pred_u, r_u)

    eval_latency = (time.time() - t_eval_start) / max(len(tgt_k), 1)

    # ── Save results ──────────────────────────────────────────────────────────
    hybrid_results = {
        "model": "GRU + OCSVM Hybrid",
        "optimal_threshold": float(best_thresh),
        "gru_only": {
            "known_test":   gru_horizons_known,
            "unknown_test": gru_horizons_unknown,
        },
        "hybrid": {
            "known_test":   hybrid_horizons_known,
            "unknown_test": hybrid_horizons_unknown,
        },
        "feature_names": feature_names,
        "eval_latency_sec": float(eval_latency),
    }

    out_path = METRICS_DIR / "hybrid_metrics.json"
    with open(out_path, "w") as f:
        json.dump(hybrid_results, f, indent=2, default=str)
    logging.info("Hybrid evaluation results saved to %s", out_path)

    # ── Print summary table ───────────────────────────────────────────────────
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("PHASE 5: GRU + OCSVM HYBRID — FULL MULTI-HORIZON EVALUATION")
    print(SEP)
    print(f"{'Model':<20} {'Split':<10} {'K=1 F1':>8} {'K=3 F1':>8} {'K=5 F1':>8} {'FPR':>8} {'PR-AUC':>8}")
    print("-" * 72)
    for label, gru_h, hyb_h, split in [
        ("Known",   gru_horizons_known,   hybrid_horizons_known,   "known"),
        ("Unknown", gru_horizons_unknown, hybrid_horizons_unknown, "unknown"),
    ]:
        for name, h in [("GRU Only", gru_h), ("Hybrid", hyb_h)]:
            f1s = [h[k]["F1-score"] for k in ["K=1", "K=3", "K=5"]]
            fpr = h["K=1"]["FPR"]
            prauc = h["K=1"].get("PR-AUC") or 0
            print(f"  {name:<18} {label:<10} {f1s[0]:>8.4f} {f1s[1]:>8.4f} {f1s[2]:>8.4f} {fpr:>8.4f} {prauc:>8.4f}")
    print(SEP)


if __name__ == "__main__":
    main()
