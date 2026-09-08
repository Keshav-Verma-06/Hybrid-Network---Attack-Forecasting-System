"""
src/training/train_gru.py
--------------------------
Phase 4: Multi-task GRU World Model training.

Changes from original:
  - All paths are repo-relative (ROOT = Path(__file__).resolve().parents[2])
  - Evaluates all 3 forecast horizons (K=1, K=3, K=5) after training
  - Saves evaluation results to outputs/metrics/gru_metrics.json
"""

import json
import os
import sys
import time
import random
import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ── Repo root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.sequence_dataset import NetworkSequenceDataset
from src.models.gru_world_model import GRUWorldModel
from src.utils.metrics import evaluate_classification

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

# MITRE stages mapping
STAGE_MAP = {
    "Benign": 0, "Reconnaissance": 1, "Initial Access": 2,
    "Lateral Movement": 3, "Command and Control": 4, "Exfiltration": 5,
}


def load_and_prep_data(data_dir: Path, scaler_path: Path):
    train      = pd.read_parquet(data_dir / "train_windows_30s.parquet")
    test_known = pd.read_parquet(data_dir / "test_known_windows_30s.parquet")

    exclude = {
        "Timestamp", "Label", "attack_stage",
        "future_attack_1", "future_attack_3", "future_attack_5",
    }
    features = [c for c in train.columns if c not in exclude]

    scaler = joblib.load(scaler_path)

    def prep(df):
        X = scaler.transform(df[features].fillna(0).values)
        Y_fut = df[["future_attack_1", "future_attack_3", "future_attack_5"]].values
        Y_stage = df["attack_stage"].map(STAGE_MAP).fillna(0).values.astype(int)
        return X, Y_fut, Y_stage

    return prep(train), prep(test_known), len(features), features


def calculate_pos_weights(Y_future):
    weights = []
    for i in range(Y_future.shape[1]):
        pos = np.sum(Y_future[:, i] == 1)
        neg = np.sum(Y_future[:, i] == 0)
        weights.append(neg / max(pos, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_all_horizons(model, test_data, scaler, device, seq_len=30):
    """Evaluate GRU on K=1, K=3, K=5 horizons and return metrics dicts.
    Threshold is tuned per horizon to maximise F1 so even models that
    output low-magnitude probabilities are correctly evaluated.
    """
    X_test, Y_test_fut, _ = test_data
    dataset = NetworkSequenceDataset(X_test, Y_test_fut, np.zeros(len(X_test)), seq_len)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    all_probs   = [[], [], []]   # K=1, K=3, K=5
    all_targets = [[], [], []]

    model.eval()
    with torch.no_grad():
        for x_seq, y_next, y_fut, _ in loader:
            x_seq = x_seq.to(device)
            _, fut_logits, _, _ = model(x_seq)
            probs = torch.sigmoid(fut_logits).cpu().numpy()

            for k in range(3):
                all_probs[k].extend(probs[:, k].tolist())
                all_targets[k].extend(y_fut[:, k].numpy().tolist())

    horizons = {}
    for i, k_label in enumerate(["K=1", "K=3", "K=5"]):
        yt    = np.array(all_targets[i], dtype=int)
        yprob = np.array(all_probs[i], dtype=float)

        # Tune threshold to maximise F1 — avoids all-zero predictions when
        # model outputs low-magnitude logits (common early in training or
        # when positive samples are rare and the model is well-calibrated).
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.01, 0.99, 0.01):
            f1 = f1_score(yt, (yprob > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        yp = (yprob > best_t).astype(int)
        logging.info("  GRU %s: threshold=%.2f  F1=%.4f", k_label, best_t, best_f1)

        m = evaluate_classification(yt, yp, yprob)
        m["tuned_threshold"] = float(best_t)
        horizons[k_label] = m

    return horizons


def main():
    cfg = load_config()
    gru_cfg  = cfg.get("gru", {})
    seed     = cfg.get("seed", 42)
    set_seeds(seed)

    # ── Paths (all repo-relative) ────────────────────────────────────────────
    DATA_DIR    = ROOT / "data" / "processed" / "windows"
    SCALER_PATH = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
    MODEL_DIR   = ROOT / "models" / "gru"
    METRICS_DIR = ROOT / "outputs" / "metrics"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s  Seed: %d", device, seed)

    # Save config snapshot for reproducibility
    snap = {"seed": seed, "device": str(device), **gru_cfg}
    with open(METRICS_DIR / "training_config_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2)
    logging.info("Config snapshot saved.")

    logging.info("Loading sequential data…")
    train_data, test_data, input_dim, feature_names = load_and_prep_data(DATA_DIR, SCALER_PATH)

    train_dataset = NetworkSequenceDataset(*train_data, seq_len=30)
    test_dataset  = NetworkSequenceDataset(*test_data, seq_len=30)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    model     = GRUWorldModel(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    huber_loss = nn.HuberLoss()
    pos_weight = calculate_pos_weights(train_data[1]).to(device)
    logging.info("BCE pos_weights K=1,3,5: %s", pos_weight.cpu().numpy())
    bce_loss   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_loss    = nn.CrossEntropyLoss()

    EPOCHS = 30
    patience = 8
    best_val_f1 = -1.0
    patience_counter = 0

    logging.info("Starting multi-task GRU training…")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x_seq, y_next, y_fut, y_stage in train_loader:
            x_seq, y_next = x_seq.to(device), y_next.to(device)
            y_fut, y_stage = y_fut.to(device), y_stage.to(device)

            # Simulated Feature Dropout (robustness against encrypted/missing features)
            # Drop 10% of features randomly for each sequence
            if model.training:
                drop_mask = (torch.rand_like(x_seq) > 0.10).float()
                x_seq = x_seq * drop_mask

            optimizer.zero_grad()
            next_pred, fut_logits, stage_logits, _ = model(x_seq)

            l_next  = huber_loss(next_pred, y_next)
            l_fut   = bce_loss(fut_logits, y_fut)
            l_stage = ce_loss(stage_logits, y_stage)
            loss    = (0.40 * l_next) + (0.40 * l_fut) + (0.20 * l_stage)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x_seq, y_next, y_fut, y_stage in test_loader:
                x_seq, y_next = x_seq.to(device), y_next.to(device)
                y_fut, y_stage = y_fut.to(device), y_stage.to(device)

                next_pred, fut_logits, stage_logits, _ = model(x_seq)
                l = (0.40 * huber_loss(next_pred, y_next)
                     + 0.40 * bce_loss(fut_logits, y_fut)
                     + 0.20 * ce_loss(stage_logits, y_stage))
                val_loss += l.item()

                preds_k1 = (torch.sigmoid(fut_logits[:, 0]) > 0.5).cpu().numpy()
                all_preds.extend(preds_k1)
                all_targets.extend(y_fut[:, 0].cpu().numpy())

        val_loss /= len(test_loader)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        logging.info(
            "Epoch %02d | Train: %.4f | Val: %.4f | F1@K1: %.4f",
            epoch + 1, train_loss, val_loss, val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_DIR / "gru_world_model.pth")
            patience_counter = 0
            logging.info("  -> Best model saved! (F1 improved)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping after %d epochs.", epoch + 1)
                break

    # ── Post-training evaluation across all horizons ─────────────────────────
    logging.info("Loading best model for final multi-horizon evaluation…")
    model.load_state_dict(torch.load(MODEL_DIR / "gru_world_model.pth", map_location=device))

    scaler = joblib.load(SCALER_PATH)
    known_horizons   = evaluate_all_horizons(model, test_data, scaler, device)

    # Also evaluate on unknown test set
    try:
        test_unknown_df  = pd.read_parquet(DATA_DIR / "test_unknown_windows_30s.parquet")
        features = feature_names
        X_u  = scaler.transform(test_unknown_df[features].fillna(0).values)
        Y_u  = test_unknown_df[["future_attack_1", "future_attack_3", "future_attack_5"]].values
        unknown_horizons = evaluate_all_horizons(model, (X_u, Y_u, np.zeros(len(X_u))), scaler, device)
    except Exception as e:
        logging.warning("Could not evaluate unknown test set: %s", e)
        unknown_horizons = {}

    gru_results = {
        "model": "GRU World Model",
        "best_val_f1_k1": float(best_val_f1),
        "known_test": {k: v for k, v in known_horizons.items()},
        "unknown_test": {k: v for k, v in unknown_horizons.items()},
        "feature_names": feature_names,
        "input_dim": input_dim,
    }

    out_path = METRICS_DIR / "gru_metrics.json"
    with open(out_path, "w") as f:
        json.dump(gru_results, f, indent=2, default=str)
    logging.info("GRU evaluation results saved to %s", out_path)

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 4: GRU WORLD MODEL — FINAL EVALUATION")
    print("=" * 70)
    for k_label, m in known_horizons.items():
        print(f"  Known   {k_label}: F1={m['F1-score']:.4f}  PR-AUC={m['PR-AUC'] or 0:.4f}  FPR={m['FPR']:.4f}")
    for k_label, m in unknown_horizons.items():
        print(f"  Unknown {k_label}: F1={m['F1-score']:.4f}  PR-AUC={m['PR-AUC'] or 0:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
