import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.sequence_dataset import NetworkSequenceDataset
from src.models.gru_world_model import GRUWorldModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MITRE stages mapping
STAGE_MAP = {
    'Benign': 0, 'Reconnaissance': 1, 'Initial Access': 2, 
    'Lateral Movement': 3, 'Command and Control': 4, 'Exfiltration': 5
}

def load_and_prep_data(data_dir, scaler_path):
    train = pd.read_parquet(os.path.join(data_dir, 'train_windows_30s.parquet'))
    test_known = pd.read_parquet(os.path.join(data_dir, 'test_known_windows_30s.parquet'))
    
    # Exclude non-features exactly as in Phase 3
    exclude = {
        'Timestamp', 'Label', 'attack_stage',
        'future_attack_1', 'future_attack_3', 'future_attack_5'
    }
    features = [c for c in train.columns if c not in exclude]
    
    # Load Phase 3 Scaler
    scaler = joblib.load(scaler_path)
    
    # Prepare Train Arrays
    X_train = scaler.transform(train[features].fillna(0).values)
    Y_train_future = train[['future_attack_1', 'future_attack_3', 'future_attack_5']].values
    Y_train_stage = train['attack_stage'].map(STAGE_MAP).fillna(0).values
    
    # Prepare Test Arrays
    X_test = scaler.transform(test_known[features].fillna(0).values)
    Y_test_future = test_known[['future_attack_1', 'future_attack_3', 'future_attack_5']].values
    Y_test_stage = test_known['attack_stage'].map(STAGE_MAP).fillna(0).values
    
    return (X_train, Y_train_future, Y_train_stage), (X_test, Y_test_future, Y_test_stage), len(features)

def calculate_pos_weights(Y_future):
    """Dynamically calculates pos_weight to fix the Phase 3 class imbalance issue."""
    weights = []
    for i in range(Y_future.shape[1]):
        pos_count = np.sum(Y_future[:, i] == 1)
        neg_count = np.sum(Y_future[:, i] == 0)
        weight = neg_count / max(pos_count, 1.0)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)

def main():
    DATA_DIR = 'data/processed/windows'
    SCALER_PATH = 'models/scalers/baseline_scaler.pkl'
    MODEL_DIR = 'models/gru'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    logging.info("Loading sequential data...")
    train_data, test_data, input_dim = load_and_prep_data(DATA_DIR, SCALER_PATH)
    
    train_dataset = NetworkSequenceDataset(*train_data, seq_len=30)
    test_dataset = NetworkSequenceDataset(*test_data, seq_len=30)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = GRUWorldModel(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # ── LOSS FUNCTIONS (Page 12) ──
    huber_loss = nn.HuberLoss()
    
    # This dynamically fixes the class imbalance failure from Phase 3
    pos_weight = calculate_pos_weights(train_data[1]).to(device)
    logging.info(f"Calculated BCE pos_weights for K=1,3,5: {pos_weight.cpu().numpy()}")
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    ce_loss = nn.CrossEntropyLoss()
    
    EPOCHS = 30
    patience = 8
    best_val_f1 = -1
    patience_counter = 0
    
    logging.info("Starting Multi-task GRU training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for x_seq, y_next, y_fut, y_stage in train_loader:
            x_seq, y_next = x_seq.to(device), y_next.to(device)
            y_fut, y_stage = y_fut.to(device), y_stage.to(device)
            
            optimizer.zero_grad()
            next_pred, fut_logits, stage_logits, _ = model(x_seq)
            
            l_next = huber_loss(next_pred, y_next)
            l_fut = bce_loss(fut_logits, y_fut)
            l_stage = ce_loss(stage_logits, y_stage)
            
            # Weighted loss (Page 12)
            loss = (0.40 * l_next) + (0.40 * l_fut) + (0.20 * l_stage)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_seq, y_next, y_fut, y_stage in test_loader:
                x_seq, y_next = x_seq.to(device), y_next.to(device)
                y_fut, y_stage = y_fut.to(device), y_stage.to(device)
                
                next_pred, fut_logits, stage_logits, _ = model(x_seq)
                
                l_next = huber_loss(next_pred, y_next)
                l_fut = bce_loss(fut_logits, y_fut)
                l_stage = ce_loss(stage_logits, y_stage)
                val_loss += ((0.40 * l_next) + (0.40 * l_fut) + (0.20 * l_stage)).item()
                
                # Collect predictions for K=1 (first column of future risk)
                preds_k1 = (torch.sigmoid(fut_logits[:, 0]) > 0.5).cpu().numpy()
                targets_k1 = y_fut[:, 0].cpu().numpy()
                
                all_preds.extend(preds_k1)
                all_targets.extend(targets_k1)
                
        val_loss /= len(test_loader)
        
        # Evaluate if we beat Phase 3!
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        logging.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Forecast F1@1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'gru_world_model.pth'))
            patience_counter = 0
            logging.info("  -> Best model saved! (Improved F1)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == '__main__':
    main()
