import torch
import numpy as np
from torch.utils.data import Dataset

class NetworkSequenceDataset(Dataset):
    def __init__(self, features, future_risks, attack_stages, seq_len=30):
        """
        features: np.array of shape (N, feature_dim)
        future_risks: np.array of shape (N, 3) for K=1, 3, 5
        attack_stages: np.array of shape (N,) containing integer class labels
        """
        self.features = features
        self.future_risks = future_risks
        self.attack_stages = attack_stages
        self.seq_len = seq_len

    def __len__(self):
        # We need seq_len windows for input, and 1 subsequent window for next-state target
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # Sequence from t-29 to t
        x_seq = self.features[idx : idx + self.seq_len]
        
        # Next state S_{t+1} (continuous features)
        y_next_state = self.features[idx + self.seq_len]
        
        # Future risk and stage aligned with the target window
        y_future = self.future_risks[idx + self.seq_len]
        y_stage = self.attack_stages[idx + self.seq_len]

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_next_state, dtype=torch.float32),
            torch.tensor(y_future, dtype=torch.float32),
            torch.tensor(y_stage, dtype=torch.long)
        )
