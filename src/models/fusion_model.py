import joblib
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

class RiskFusionEngine:
    def __init__(self):
        # nu=0.05 targets ~5% anomaly rate in training to build a tight boundary
        self.ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        self.anomaly_scaler = MinMaxScaler()
        self.error_scaler = MinMaxScaler()
        
    def fit(self, benign_z, all_z, all_errors):
        """Fits OCSVM on benign data, and fits scalers on all training data."""
        # 1. Train OCSVM on benign latent states only
        self.ocsvm.fit(benign_z)
        
        # 2. Fit scaler for anomaly scores
        # sklearn decision_function: >0 is normal, <0 is anomaly. 
        # We negate it so higher values = higher anomaly risk.
        raw_anomaly = -self.ocsvm.decision_function(all_z).reshape(-1, 1)
        self.anomaly_scaler.fit(raw_anomaly)
        
        # 3. Fit scaler for state reconstruction errors
        self.error_scaler.fit(all_errors.reshape(-1, 1))
        
    def predict_risk(self, p_forecast, z_t, state_error):
        """Calculates final R_t risk score using the Phase 5 equation."""
        p_forecast = np.array(p_forecast).flatten()
        
        # Normalize anomaly scores
        raw_anomaly = -self.ocsvm.decision_function(z_t).reshape(-1, 1)
        a_ocsvm = self.anomaly_scaler.transform(raw_anomaly).flatten()
        a_ocsvm = np.clip(a_ocsvm, 0, 1) # Ensure bounded 0-1
        
        # Normalize state errors
        r_error = self.error_scaler.transform(np.array(state_error).reshape(-1, 1)).flatten()
        r_error = np.clip(r_error, 0, 1)
        
        # PHASE 5 RISK FUSION EQUATION
        R_t = (0.50 * p_forecast) + (0.30 * a_ocsvm) + (0.20 * r_error)
        
        return R_t, a_ocsvm, r_error
        
    def save(self, path_prefix):
        joblib.dump(self.ocsvm, f"{path_prefix}_ocsvm.pkl")
        joblib.dump(self.anomaly_scaler, f"{path_prefix}_anomaly_scaler.pkl")
        joblib.dump(self.error_scaler, f"{path_prefix}_error_scaler.pkl")
        
    def load(self, path_prefix):
        self.ocsvm = joblib.load(f"{path_prefix}_ocsvm.pkl")
        self.anomaly_scaler = joblib.load(f"{path_prefix}_anomaly_scaler.pkl")
        self.error_scaler = joblib.load(f"{path_prefix}_error_scaler.pkl")
