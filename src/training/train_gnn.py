"""
src/training/train_gnn.py
--------------------------
⚠️  STATUS: STUB — GNN path disabled pending fixes

KNOWN ISSUES (to be resolved before enabling):
  1. Temporal data leakage: `torch.utils.data.random_split` shuffles sequences,
     breaking the temporal ordering required for causal evaluation.
     Fix: implement a chronological split identical to create_windows.py.
  2. Device handling bug: graphs are assigned to device inside the inner loop
     (`g = g.to(device)`) but the assignment is not persisted — the original
     reference in the list is unchanged. Fix: `seq[i] = seq[i].to(device)`.
  3. The TemporalGATGRU model requires `torch_geometric` which significantly
     increases the environment footprint. The GRU World Model achieves better
     results with a smaller dependency footprint.
  4. No evaluation save: results are only logged, never written to JSON/CSV.

HOW TO RE-ENABLE:
  1. Fix the chronological split — use 80% / 20% time-ordered split.
  2. Fix the device assignment loop.
  3. Add post-training evaluation with tuned threshold and save to
     outputs/metrics/gnn_metrics.json.
  4. Test that `from torch_geometric.data import Data` imports cleanly
     in the deployment environment.

DO NOT run this script in the current state — it will produce misleading results.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    raise NotImplementedError(
        "\n\n"
        "  train_gnn.py is currently DISABLED.\n"
        "  See the module docstring for known issues and re-enable instructions.\n"
        "  The GRU World Model (train_gru.py + train_hybrid.py) is the active production path.\n"
    )


if __name__ == "__main__":
    main()
