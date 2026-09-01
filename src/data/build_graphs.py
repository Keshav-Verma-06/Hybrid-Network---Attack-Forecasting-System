import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_temporal_graphs(parquet_path, out_dir, window_size='30s', max_rows=100000):
    """Builds a sequence of PyG Data (graphs) from raw network flows."""
    logging.info(f"Loading raw flows from {parquet_path} (Limiting to {max_rows} rows for demo speed)...")
    
    # Load raw flows (we need IPs for nodes)
    try:
        df = pd.read_parquet(parquet_path).head(max_rows)
    except FileNotFoundError:
        logging.error("Raw parquet not found. Ensure processed_cicids2017.parquet exists.")
        return
        
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
    df = df.sort_values('Timestamp')
    
    # Fallback column names if CIC-IDS format varies
    src_ip_col = 'Source IP' if 'Source IP' in df.columns else df.columns[1]
    dst_ip_col = 'Destination IP' if 'Destination IP' in df.columns else df.columns[3]
    
    graphs = []
    
    # Group by time windows
    grouped = df.groupby(pd.Grouper(key='Timestamp', freq=window_size))
    
    for time_idx, (name, group) in enumerate(grouped):
        if len(group) == 0:
            continue
            
        # 1. Node Extraction
        unique_ips = pd.concat([group[src_ip_col], group[dst_ip_col]]).unique()
        ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
        
        # Node Features (Page 14): [in_degree, out_degree, flow_count]
        node_features = np.zeros((len(unique_ips), 3), dtype=np.float32)
        
        # 2. Edge Extraction
        src_indices = group[src_ip_col].map(ip_to_idx).values
        dst_indices = group[dst_ip_col].map(ip_to_idx).values
        edge_index = torch.tensor(np.vstack((src_indices, dst_indices)), dtype=torch.long)
        
        # Populate Node Features
        for src, dst in zip(src_indices, dst_indices):
            node_features[src][1] += 1 # out_degree
            node_features[dst][0] += 1 # in_degree
            node_features[src][2] += 1 # flow_count
            
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # 3. Edge Features (Page 15)
        # Using [Protocol, Flow Duration, Total Fwd Packets] as proxies
        cols = ['Protocol', 'Flow Duration', 'Total Fwd Packets']
        avail_cols = [c for c in cols if c in group.columns]
        
        if avail_cols:
            edge_attr = torch.tensor(group[avail_cols].fillna(0).values, dtype=torch.float32)
        else:
            edge_attr = torch.ones((len(group), 1), dtype=torch.float32)
            
        # 4. Graph Label (Is there an attack in this window?)
        is_attack = 1 if (group['Label'] != 'BENIGN').any() else 0
        y = torch.tensor([is_attack], dtype=torch.float32)
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graphs.append(graph)
        
    logging.info(f"Successfully generated {len(graphs)} temporal graphs.")
    
    os.makedirs(out_dir, exist_ok=True)
    torch.save(graphs, os.path.join(out_dir, 'temporal_graphs.pt'))
    logging.info(f"Saved to {out_dir}/temporal_graphs.pt")

if __name__ == '__main__':
    IN_PATH = "data/interim/cleaned/processed_cicids2017.parquet"
    OUT_DIR = "data/processed/graphs"
    build_temporal_graphs(IN_PATH, OUT_DIR)
