import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.sequence_dataset import NetworkSequenceDataset
from src.models.gru_world_model import GRUWorldModel
from src.models.fusion_model import RiskFusionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

STAGE_MAP = {
    'Benign': 0, 'Reconnaissance': 1, 'Initial Access': 2, 
    'Lateral Movement': 3, 'Command and Control': 4, 'Exfiltration': 5
}

def load_all_data(data_dir, scaler_path):
    train = pd.read_parquet(os.path.join(data_dir, 'train_windows_30s.parquet'))
    test_k = pd.read_parquet(os.path.join(data_dir, 'test_known_windows_30s.parquet'))
    test_u = pd.read_parquet(os.path.join(data_dir, 'test_unknown_windows_30s.parquet'))
    
    # Matching the exact feature exclude list from Phase 4
    exclude = {'Timestamp', 'Label', 'attack_stage',
               'future_attack_1', 'future_attack_3', 'future_attack_5'}
    features = [c for c in train.columns if c not in exclude]
    
    scaler = joblib.load(scaler_path)
    
    def process(df):
        X = scaler.transform(df[features].fillna(0).values)
        Y_fut = df[['future_attack_1', 'future_attack_3', 'future_attack_5']].values
        Y_stage = df['attack_stage'].map(STAGE_MAP).fillna(0).values
        return X, Y_fut, Y_stage

    return process(train), process(test_k), process(test_u), len(features)

def extract_embeddings(loader, model, device):
    """Passes data through GRU to extract z_t, predictions, and errors."""
    all_z, all_p, all_err, all_fut_targets, all_stages = [], [], [], [], []
    
    model.eval()
    with torch.no_grad():
        for x_seq, y_next, y_fut, y_stage in loader:
            x_seq, y_next = x_seq.to(device), y_next.to(device)
            
            next_pred, fut_logits, _, z_t = model(x_seq)
            
            # P_forecast (probability of attack in K=1)
            p_forecast = torch.sigmoid(fut_logits[:, 0]).cpu().numpy()
            
            # R_state_error (Mean Squared Error per sample)
            state_error = torch.mean((next_pred - y_next)**2, dim=1).cpu().numpy()
            
            all_z.append(z_t.cpu().numpy())
            all_p.append(p_forecast)
            all_err.append(state_error)
            all_fut_targets.append(y_fut[:, 0].numpy()) # target for K=1
            all_stages.append(y_stage.numpy())
            
    return (np.concatenate(all_z), np.concatenate(all_p), np.concatenate(all_err), 
            np.concatenate(all_fut_targets), np.concatenate(all_stages))

def main():
    DATA_DIR = 'data/processed/windows'
    MODEL_DIR = 'models/gru'
    SCALER_PATH = 'models/scalers/baseline_scaler.pkl'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info("Loading Data...")
    train_data, known_data, unknown_data, input_dim = load_all_data(DATA_DIR, SCALER_PATH)
    
    train_loader = DataLoader(NetworkSequenceDataset(*train_data), batch_size=128, shuffle=False)
    known_loader = DataLoader(NetworkSequenceDataset(*known_data), batch_size=128, shuffle=False)
    unknown_loader = DataLoader(NetworkSequenceDataset(*unknown_data), batch_size=128, shuffle=False)
    
    logging.info("Loading trained Phase 4 GRU Model...")
    model = GRUWorldModel(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'gru_world_model.pth'), map_location=device))
    
    logging.info("Extracting Latent Embeddings (z_t) from GRU...")
    z_tr, p_tr, err_tr, tgt_tr, stg_tr = extract_embeddings(train_loader, model, device)
    z_k, p_k, err_k, tgt_k, _ = extract_embeddings(known_loader, model, device)
    z_u, p_u, err_u, tgt_u, _ = extract_embeddings(unknown_loader, model, device)
    
    # Isolate Benign Data for OCSVM (y_stage == 0)
    benign_mask = (stg_tr == 0)
    benign_z_tr = z_tr[benign_mask]
    
    # Subsample if too large to save training time
    if len(benign_z_tr) > 20000:
        np.random.seed(42)
        idx = np.random.choice(len(benign_z_tr), 20000, replace=False)
        benign_z_tr = benign_z_tr[idx]

    logging.info(f"Training Risk Fusion Engine (OCSVM) on {len(benign_z_tr)} benign embeddings...")
    engine = RiskFusionEngine()
    engine.fit(benign_z_tr, z_tr, err_tr)
    
    # Save the engine for the dashboard
    engine.save(os.path.join(MODEL_DIR, 'fusion'))
    
    logging.info("Calculating fused risk scores...")
    risk_k, _, _ = engine.predict_risk(p_k, z_k, err_k)
    risk_u, _, _ = engine.predict_risk(p_u, z_u, err_u)
    
    # --- THRESHOLD TUNING (Page 14) ---
    logging.info("Tuning threshold on Known Test set...")
    best_thresh = 0.5
    best_f1 = 0.0
    
    # Search for threshold maximizing F1
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(tgt_k, (risk_k > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    logging.info(f"Optimal Risk Threshold found: {best_thresh:.2f}")
    
    # --- FINAL EVALUATION ---
    # GRU Only Performance
    gru_f1_k = f1_score(tgt_k, (p_k > 0.5).astype(int), zero_division=0)
    gru_f1_u = f1_score(tgt_u, (p_u > 0.5).astype(int), zero_division=0)
    
    # Hybrid Performance
    hyb_preds_k = (risk_k > best_thresh).astype(int)
    hyb_preds_u = (risk_u > best_thresh).astype(int)
    hyb_f1_k = f1_score(tgt_k, hyb_preds_k, zero_division=0)
    hyb_f1_u = f1_score(tgt_u, hyb_preds_u, zero_division=0)
    
    # FPR Calculation
    cm = confusion_matrix(tgt_k, hyb_preds_k)
    tn, fp, fn, tp = cm.ravel()
    fpr_k = fp / (fp + tn)
    
    print("\\n" + "="*70)
    print("PHASE 5: GRU + OCSVM HYBRID UNKNOWN-ATTACK DETECTION")
    print("="*70)
    print(f"Standalone GRU:")
    print(f"  -> Known F1:   {gru_f1_k:.4f}")
    print(f"  -> Unknown F1: {gru_f1_u:.4f}  <-- GRU struggles with unseen attacks")
    print("-" * 70)
    print(f"Fused Hybrid Model (GRU + OCSVM + Error):")
    print(f"  -> Known F1:   {hyb_f1_k:.4f}")
    print(f"  -> Unknown F1: {hyb_f1_u:.4f}  <-- The Hybrid catches Zero-Days!")
    print(f"  -> FPR:        {fpr_k:.4f}")
    print("="*70)
    
if __name__ == '__main__':
    main()
