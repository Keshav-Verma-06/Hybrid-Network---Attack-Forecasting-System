import torch
import torch.nn as nn

class GRUWorldModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, num_stages=6):
        super().__init__()
        
        # Feature projection (Page 12 layout)
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2-layer GRU encoder
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-task Heads
        self.next_state_head = nn.Linear(hidden_dim, input_dim)
        self.future_risk_head = nn.Linear(hidden_dim, 3)       # For K=1, 3, 5
        self.attack_stage_head = nn.Linear(hidden_dim, num_stages)

    def forward(self, x):
        # x shape: (batch, seq_len=30, input_dim)
        x_proj = torch.relu(self.feature_proj(x))
        
        gru_out, _ = self.gru(x_proj)
        
        # Extract latent state z_t from the final timestep
        z_t = gru_out[:, -1, :] # shape: (batch, hidden_dim)
        
        # Multi-task predictions
        next_state_pred = self.next_state_head(z_t)
        future_risk_logits = self.future_risk_head(z_t)
        attack_stage_logits = self.attack_stage_head(z_t)
        
        # Return z_t as well, required for Phase 5 (OCSVM Hybrid)
        return next_state_pred, future_risk_logits, attack_stage_logits, z_t
