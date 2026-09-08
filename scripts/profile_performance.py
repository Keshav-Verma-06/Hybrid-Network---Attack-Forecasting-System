"""
scripts/profile_performance.py
------------------------------
Profiles throughput, latency, and memory usage of the Hybrid model
under high-load scenarios (e.g., simulating 10k windows/sec).
"""

import sys
import time
import psutil
import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.gru_world_model import GRUWorldModel
from src.models.fusion_model import RiskFusionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def profile():
    DATA_DIR    = ROOT / "data" / "processed" / "windows"
    SCALER_PATH = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
    GRU_PATH    = ROOT / "models" / "gru" / "gru_world_model.pth"
    FUSION_PFX  = str(ROOT / "models" / "gru" / "fusion")
    METRICS_DIR = ROOT / "outputs" / "metrics"
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading models...")
    scaler = joblib.load(SCALER_PATH)
    input_dim = scaler.n_features_in_
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUWorldModel(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    model.load_state_dict(torch.load(GRU_PATH, map_location=device))
    model.eval()

    engine = RiskFusionEngine()
    engine.load(FUSION_PFX)

    # Generate mock high-volume traffic (e.g. 50,000 windows)
    # Shape: (N, 30, D)
    N_SAMPLES = 50000
    SEQ_LEN = 30
    BATCH_SIZE = 1024

    logging.info(f"Generating {N_SAMPLES} synthetic windows of shape ({SEQ_LEN}, {input_dim}) for profiling...")
    x_mock = torch.randn(N_SAMPLES, SEQ_LEN, input_dim, dtype=torch.float32)

    logging.info("Starting throughput test...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    start_time = time.time()
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, N_SAMPLES, BATCH_SIZE):
            batch = x_mock[i:i+BATCH_SIZE].to(device)
            next_pred, fut_logits, stage_logits, z_t = model(batch)
            probs = torch.sigmoid(fut_logits[:, 0]).cpu().numpy()
            
            # Note: We need a mock target or use the last step of input for state_error profiling
            last_real = batch[:, -1, :].cpu().numpy()
            next_np = next_pred.cpu().numpy()
            err = np.mean((next_np - last_real) ** 2, axis=1)
            
            risk, a, r = engine.predict_risk(probs, z_t.cpu().numpy(), err)

    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 * 1024)

    duration = end_time - start_time
    throughput = N_SAMPLES / duration
    latency_ms = (duration / N_SAMPLES) * 1000
    mem_used = mem_after - mem_before

    results = {
        "n_samples_tested": N_SAMPLES,
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "total_duration_sec": float(duration),
        "throughput_windows_per_sec": float(throughput),
        "avg_latency_per_window_ms": float(latency_ms),
        "memory_consumed_mb": float(mem_used)
    }

    print("\n" + "="*50)
    print("PERFORMANCE PROFILING RESULTS")
    print("="*50)
    print(f"Device     : {results['device']}")
    print(f"Throughput : {results['throughput_windows_per_sec']:.2f} windows/sec")
    print(f"Latency    : {results['avg_latency_per_window_ms']:.4f} ms/window")
    print(f"Memory Diff: +{results['memory_consumed_mb']:.2f} MB")
    print("="*50)

    out_path = METRICS_DIR / "performance_profile.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved profile to {out_path}")

if __name__ == "__main__":
    profile()
