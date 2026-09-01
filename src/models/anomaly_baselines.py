from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


def get_anomaly_models():
    """Returns anomaly baseline models defined in Phase 3.
    Trained on benign traffic only; deviations flagged as attacks.
    """
    return {
        'OCSVM': OneClassSVM(kernel='rbf', gamma='scale', nu=0.05),
        'LOF': LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05),
    }


def map_anomaly_predictions(y_pred_sklearn):
    """Maps sklearn anomaly output to standard binary labels.
    sklearn: 1=inlier (benign), -1=outlier (attack)
    ours:    0=benign,           1=attack
    """
    return [0 if x == 1 else 1 for x in y_pred_sklearn]
