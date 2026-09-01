import os
import sys
import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.temporal_gnn import TemporalGATGRU

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphSequenceDataset(Dataset):
    def __init__(self, graphs, seq_len=10):
        self.graphs = graphs
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.graphs) - self.seq_len
        
    def __getitem__(self, idx):
        # Return sequence of graphs and the target of the NEXT window
        seq = self.graphs[idx : idx + self.seq_len]
        target = self.graphs[idx + self.seq_len].y
        return seq, target

def custom_collate(batch):
    """Custom collate to handle sequences of PyG Data objects."""
    sequences = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    return sequences, targets

def main():
    GRAPH_DIR = 'data/processed/graphs'
    MODEL_DIR = 'models/gnn'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    graph_path = os.path.join(GRAPH_DIR, 'temporal_graphs.pt')
    if not os.path.exists(graph_path):
        logging.error("temporal_graphs.pt not found. Run build_graphs.py first!")
        return
        
    logging.info("Loading Graph Sequence...")
    all_graphs = torch.load(graph_path, weights_only=False)
    
    # We use a shorter sequence (10) for Graph processing to save memory
    dataset = GraphSequenceDataset(all_graphs, seq_len=10)
    
    # Train/Test Split (80/20)
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    # Keep batch size small (16) to prevent Graph OOM
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=custom_collate)
    
    # Initialize Temporal GNN
    node_dim = all_graphs[0].x.shape[1]
    edge_dim = all_graphs[0].edge_attr.shape[1]
    
    model = TemporalGATGRU(node_in_dim=node_dim, edge_in_dim=edge_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Heavy penalty for missing attacks (due to imbalance)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    
    EPOCHS = 10
    logging.info("Starting Temporal GAT-GRU Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for seq_batch, targets in train_loader:
            targets = targets.to(device)
            optimizer.zero_grad()
            
            # Move graphs to device
            for seq in seq_batch:
                for g in seq:
                    g = g.to(device)
                    
            logits, _ = model(seq_batch)
            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for seq_batch, targets in test_loader:
                targets = targets.to(device)
                logits, _ = model(seq_batch)
                
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_tgts.extend(targets.cpu().numpy())
                
        f1 = f1_score(all_tgts, all_preds, zero_division=0)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Graph Forecast F1: {f1:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'gat_gru_model.pth'))
    logging.info("Phase 6 GNN Model Saved Successfully!")

if __name__ == '__main__':
    main()
