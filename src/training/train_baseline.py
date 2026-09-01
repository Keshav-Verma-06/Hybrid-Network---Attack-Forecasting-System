import os
import sys
import time
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.metrics import evaluate_classification
from src.models.logistic_baseline import get_supervised_models
from src.models.anomaly_baselines import get_anomaly_models, map_anomaly_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_datasets(data_dir):
    train        = pd.read_parquet(os.path.join(data_dir, 'train_windows_30s.parquet'))
    test_known   = pd.read_parquet(os.path.join(data_dir, 'test_known_windows_30s.parquet'))
    test_unknown = pd.read_parquet(os.path.join(data_dir, 'test_unknown_windows_30s.parquet'))
    return train, test_known, test_unknown


def main():
    DATA_DIR   = 'data/processed/windows'
    MODEL_DIR  = 'models/baselines'
    SCALER_DIR = 'models/scalers'
    REPORT_DIR = 'outputs/metrics'

    for d in [MODEL_DIR, SCALER_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)

    logging.info('Loading windowed datasets...')
    train, test_known, test_unknown = load_datasets(DATA_DIR)

    # TARGET: future_attack_1 = "will an attack occur in the NEXT 30s window?"
    # This is the actual forecasting task the GRU will solve (K=1 horizon).
    # current_attack_state is kept as a feature (strong temporal signal).
    TARGET = 'future_attack_1'
    exclude = {
        'Timestamp', 'Label', 'attack_stage',
        'future_attack_1', 'future_attack_3', 'future_attack_5'
    }
    features = [c for c in train.columns if c not in exclude]
    logging.info('Feature count: %d  Target: %s', len(features), TARGET)

    X_train  = train[features].fillna(0).values
    X_test_k = test_known[features].fillna(0).values
    X_test_u = test_unknown[features].fillna(0).values
    y_train  = train[TARGET].values
    y_test_k = test_known[TARGET].values
    y_test_u = test_unknown[TARGET].values

    logging.info('Splits -> Train:%d  KnownTest:%d  UnknownTest:%d',
                 len(X_train), len(X_test_k), len(X_test_u))
    logging.info('Train attack rate: %.2f%%  KnownTest: %.2f%%  UnknownTest: %.2f%%',
                 y_train.mean() * 100, y_test_k.mean() * 100, y_test_u.mean() * 100)

    logging.info('Fitting StandardScaler on training data only...')
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_k_sc = scaler.transform(X_test_k)
    X_test_u_sc = scaler.transform(X_test_u)
    joblib.dump(scaler, os.path.join(SCALER_DIR, 'baseline_scaler.pkl'))

    X_benign = X_train_sc[y_train == 0]
    logging.info('Benign windows for anomaly training: %d', len(X_benign))

    results = []

    # ── Supervised Models ──────────────────────────────────────────────────
    for name, model in get_supervised_models().items():
        logging.info('Training supervised: %s ...', name)
        model.fit(X_train_sc, y_train)
        joblib.dump(model, os.path.join(MODEL_DIR, name.replace(' ', '_').lower() + '.pkl'))

        t0       = time.time()
        preds_k  = model.predict(X_test_k_sc)
        lat_k    = (time.time() - t0) / max(len(X_test_k_sc), 1)
        preds_u  = model.predict(X_test_u_sc)
        probs_k  = model.predict_proba(X_test_k_sc)[:, 1] if hasattr(model, 'predict_proba') else None
        probs_u  = model.predict_proba(X_test_u_sc)[:, 1] if hasattr(model, 'predict_proba') else None

        mk = evaluate_classification(y_test_k, preds_k, probs_k, lat_k)
        mu = evaluate_classification(y_test_u, preds_u, probs_u)

        logging.info('%s  KnownF1=%.4f  UnknownF1=%.4f  FPR=%.4f',
                     name, mk['F1-score'], mu['F1-score'], mk['FPR'])
        results.append({
            'Model': name,
            'Known F1':              round(mk['F1-score'], 4),
            'Unknown F1':            round(mu['F1-score'], 4),
            'FPR':                   round(mk['FPR'], 4),
            'Inference latency (ms)':round(lat_k * 1000, 4),
        })

    # ── Anomaly Models ─────────────────────────────────────────────────────
    for name, model in get_anomaly_models().items():
        if len(X_benign) > 20000:
            np.random.seed(42)
            X_sub = X_benign[np.random.choice(len(X_benign), 20000, replace=False)]
            logging.info('Subsampled to 20000 benign windows for %s', name)
        else:
            X_sub = X_benign

        logging.info('Training anomaly: %s on %d windows...', name, len(X_sub))
        model.fit(X_sub)
        joblib.dump(model, os.path.join(MODEL_DIR, name.lower() + '.pkl'))

        t0      = time.time()
        preds_k = map_anomaly_predictions(model.predict(X_test_k_sc))
        lat_k   = (time.time() - t0) / max(len(X_test_k_sc), 1)
        preds_u = map_anomaly_predictions(model.predict(X_test_u_sc))
        scores_k = -model.decision_function(X_test_k_sc) if hasattr(model, 'decision_function') else None
        scores_u = -model.decision_function(X_test_u_sc) if hasattr(model, 'decision_function') else None

        mk = evaluate_classification(y_test_k, preds_k, scores_k, lat_k)
        mu = evaluate_classification(y_test_u, preds_u, scores_u)

        logging.info('%s  KnownF1=%.4f  UnknownF1=%.4f  FPR=%.4f',
                     name, mk['F1-score'], mu['F1-score'], mk['FPR'])
        results.append({
            'Model': name,
            'Known F1':              round(mk['F1-score'], 4),
            'Unknown F1':            round(mu['F1-score'], 4),
            'FPR':                   round(mk['FPR'], 4),
            'Inference latency (ms)':round(lat_k * 1000, 4),
        })

    # ── Output Benchmark Table ─────────────────────────────────────────────
    df = pd.DataFrame(results)
    df['Forecast F1@1'] = 'N/A (baseline)'
    df['Forecast F1@5'] = 'N/A (baseline)'
    df = df[['Model','Known F1','Unknown F1','Forecast F1@1','Forecast F1@5',
             'FPR','Inference latency (ms)']]

    sep = '=' * 90
    print()
    print(sep)
    print('  PHASE 3 MANDATORY BASELINE RESULTS')
    print(sep)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))
    print(sep)

    report = os.path.join(REPORT_DIR, 'baseline_metrics_table.csv')
    df.to_csv(report, index=False)
    logging.info('Results saved to %s', report)


if __name__ == '__main__':
    main()
