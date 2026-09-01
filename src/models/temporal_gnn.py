import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class TemporalGATGRU(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, gat_layers=2, heads=4, dropout=0.2):
        super().__init__()
        
        # Phase 6 PDF Specs: 2 GAT layers, 4 heads, 64 hidden dim
        self.gat1 = GATConv(node_in_dim, hidden_dim, heads=heads, edge_dim=edge_in_dim, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_in_dim, dropout=dropout)
        
        # Temporal Encoder: GRU, 128 hidden units
        self.gru = nn.GRU(hidden_dim, 128, num_layers=1, batch_first=True)
        
        # Forecast Head
        self.risk_head = nn.Linear(128, 1)

    def forward(self, graph_sequence):
        """Processes a list of PyG Data objects representing a time sequence."""
        batch_size = len(graph_sequence)
        
        graph_embeddings = []
        for seq in graph_sequence: # Iterate over batch
            seq_embs = []
            for t_graph in seq: # Iterate over time windows in sequence
                # Skip empty graphs
                if t_graph.x.size(0) == 0:
                    seq_embs.append(torch.zeros(64, device=t_graph.x.device))
                    continue
                    
                x = t_graph.x
                edge_index = t_graph.edge_index
                edge_attr = t_graph.edge_attr
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device) # Single graph pool
                
                x = F.elu(self.gat1(x, edge_index, edge_attr))
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.gat2(x, edge_index, edge_attr)
                
                g_emb = global_mean_pool(x, batch).squeeze(0)
                seq_embs.append(g_emb)
                
            graph_embeddings.append(torch.stack(seq_embs))
            
        # Shape: (Batch, Seq_Len, GAT_Hidden_Dim)
        gru_in = torch.stack(graph_embeddings)
        
        gru_out, _ = self.gru(gru_in)
        z_t = gru_out[:, -1, :] # Take final timestep
        
        risk_logits = self.risk_head(z_t)
        return risk_logits, z_t
