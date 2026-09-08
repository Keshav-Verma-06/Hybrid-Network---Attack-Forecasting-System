"""
Standalone GRU evaluation script with threshold tuning.
Run: python scripts/eval_gru.py
"""
import sys, json, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sequence_dataset import NetworkSequenceDataset
from src.models.gru_world_model import GRUWorldModel
from src.utils.metrics import evaluate_classification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR    = ROOT / 'data' / 'processed' / 'windows'
SCALER_PATH = ROOT / 'models' / 'scalers' / 'baseline_scaler.pkl'
MODEL_PATH  = ROOT / 'models' / 'gru' / 'gru_world_model.pth'
METRICS_DIR = ROOT / 'outputs' / 'metrics'

scaler = joblib.load(SCALER_PATH)

train  = pd.read_parquet(DATA_DIR / 'train_windows_30s.parquet')
test_k = pd.read_parquet(DATA_DIR / 'test_known_windows_30s.parquet')
test_u = pd.read_parquet(DATA_DIR / 'test_unknown_windows_30s.parquet')

exclude = {'Timestamp','Label','attack_stage','future_attack_1','future_attack_3','future_attack_5'}
features  = [c for c in train.columns if c not in exclude]
input_dim = len(features)
logging.info("Features: %d  Model: %s", input_dim, MODEL_PATH)

model = GRUWorldModel(input_dim=input_dim, hidden_dim=128, num_layers=2)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location='cpu'))
model.eval()


def get_probs(df):
    X     = scaler.transform(df[features].fillna(0).values)
    Y_fut = df[['future_attack_1','future_attack_3','future_attack_5']].values
    ds    = NetworkSequenceDataset(X, Y_fut, np.zeros(len(X)), 30)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    all_probs, all_targets = [[], [], []], [[], [], []]
    with torch.no_grad():
        for x_seq, y_next, y_fut, _ in loader:
            _, fut_logits, _, _ = model(x_seq)
            probs = torch.sigmoid(fut_logits).cpu().numpy()
            for k in range(3):
                all_probs[k].extend(probs[:, k].tolist())
                all_targets[k].extend(y_fut[:, k].numpy().tolist())
    return all_probs, all_targets


def eval_with_tuning(probs, targets):
    yt    = np.array(targets, dtype=int)
    yprob = np.array(probs, dtype=float)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.01, 0.99, 0.01):
        f1 = f1_score(yt, (yprob > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    yp = (yprob > best_t).astype(int)
    m  = evaluate_classification(yt, yp, yprob)
    m['tuned_threshold'] = float(best_t)
    return m


logging.info("Evaluating on known test set...")
probs_k, tgt_k = get_probs(test_k)
logging.info("Evaluating on unknown test set...")
probs_u, tgt_u = get_probs(test_u)

known_h, unknown_h = {}, {}
SEP = '=' * 70
print()
print(SEP)
print("GRU STANDALONE EVALUATION (tuned thresholds)")
print(SEP)
for i, k in enumerate(['K=1', 'K=3', 'K=5']):
    mk = eval_with_tuning(probs_k[i], tgt_k[i])
    mu = eval_with_tuning(probs_u[i], tgt_u[i])
    known_h[k]   = mk
    unknown_h[k] = mu
    thr_k = mk['tuned_threshold']
    thr_u = mu['tuned_threshold']
    f1_k  = mk['F1-score']
    prauc_k = mk['PR-AUC'] or 0
    fpr_k  = mk['FPR']
    f1_u  = mu['F1-score']
    prauc_u = mu['PR-AUC'] or 0
    print(f"  Known   {k}: F1={f1_k:.4f}  thr={thr_k:.2f}  PR-AUC={prauc_k:.4f}  FPR={fpr_k:.4f}")
    print(f"  Unknown {k}: F1={f1_u:.4f}  thr={thr_u:.2f}  PR-AUC={prauc_u:.4f}")
print(SEP)

results = {
    'model':         'GRU World Model',
    'feature_names': features,
    'input_dim':     input_dim,
    'best_val_f1_k1': known_h['K=1']['F1-score'],
    'known_test':    known_h,
    'unknown_test':  unknown_h,
}

out = METRICS_DIR / 'gru_metrics.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=str)
logging.info("Saved %s", out)
