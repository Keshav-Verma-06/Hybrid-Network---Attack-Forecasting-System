"""
src/inference/rollout.py
------------------------
Recursive multi-step state rollout for the GRU World Model.

Instead of the 3 parallel heads (K=1, K=3, K=5) which are trained to jump
directly to future risk levels, this module feeds predicted next-states back
into the GRU as actual inputs — giving a true autoregressive forecast.

Usage:
    from src.inference.rollout import recursive_rollout
    timeline = recursive_rollout(model, engine, seed_seq, n_steps=15)
    # returns list of dicts: [{step, p_k1, risk, a_ocsvm, r_error, z_t}, ...]
"""

import torch
import numpy as np
from typing import Optional


@torch.no_grad()
def recursive_rollout(
    model,
    engine,
    seed_sequence: torch.Tensor,   # shape (1, 30, input_dim)  — the "now" context
    n_steps: int = 15,
    device: str = "cpu",
) -> list[dict]:
    """
    Autoregressively roll out the GRU World Model for n_steps future windows.

    At each step t:
      1. Run the GRU on the current 30-window context -> next_state_pred, z_t
      2. Compute hybrid risk from (p_k1, z_t, state_error)
      3. Slide the context window: drop oldest, append next_state_pred
      4. Repeat

    Parameters
    ----------
    model         : GRUWorldModel (eval mode expected)
    engine        : RiskFusionEngine (already fitted)
    seed_sequence : torch.Tensor (1, 30, input_dim) — the last 30 real windows
    n_steps       : How many steps to roll out
    device        : 'cpu' or 'cuda'

    Returns
    -------
    list of dicts with keys:
      step          - 1-indexed step number
      timestamp_offset_s - step * 30 (seconds ahead)
      p_k1          - GRU P(attack) head K=1
      p_k3          - GRU P(attack) head K=3
      p_k5          - GRU P(attack) head K=5
      risk          - Fused hybrid risk score
      a_ocsvm       - OCSVM anomaly component
      r_error       - Reconstruction error component
      is_autoregressive - True for steps > 1 (using predicted states)
    """
    model = model.to(device)
    model.eval()

    context = seed_sequence.clone().to(device)  # (1, 30, D)
    results = []

    for step in range(1, n_steps + 1):
        # ── Forward pass ────────────────────────────────────────────────────
        next_pred, fut_logits, stage_logits, z_t = model(context)

        # Probabilities
        probs = torch.sigmoid(fut_logits[0]).cpu().numpy()   # (3,)
        p_k1, p_k3, p_k5 = float(probs[0]), float(probs[1]), float(probs[2])

        # State reconstruction error (vs. the last real window in context)
        last_real = context[0, -1, :].cpu().numpy()
        next_np   = next_pred[0].detach().cpu().numpy()
        state_err = float(np.mean((next_np - last_real) ** 2))

        # Hybrid risk
        z_np = z_t.detach().cpu().numpy()
        risk_arr, a_arr, r_arr = engine.predict_risk(
            np.array([p_k1]), z_np, np.array([state_err])
        )
        risk    = float(risk_arr[0])
        a_ocsvm = float(a_arr[0])
        r_error = float(r_arr[0])

        results.append({
            "step":               step,
            "timestamp_offset_s": step * 30,
            "p_k1":               p_k1,
            "p_k3":               p_k3,
            "p_k5":               p_k5,
            "risk":               risk,
            "a_ocsvm":            a_ocsvm,
            "r_error":            r_error,
            "is_autoregressive":  step > 1,
        })

        # ── Slide context window ─────────────────────────────────────────────
        # Append next_pred as new "observed" window
        next_tensor = next_pred.detach().unsqueeze(1)    # (1, 1, D)
        context = torch.cat([context[:, 1:, :], next_tensor], dim=1)  # (1, 30, D)

    return results
