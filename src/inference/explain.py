"""
src/inference/explain.py  (enhanced)
-------------------------------------
Explainability utilities for the GRU World Model.

New in this version:
  - GRURiskWrapper: wraps GRU for Captum (unchanged, kept for compatibility)
  - explain_prediction(): feature-level attributions (summed over time) — top-N
  - explain_temporal(): per-TIMESTEP attributions — returns (30, n_features) matrix
    for heatmap display showing which windows in history drove the alert

Both functions use Captum IntegratedGradients.
"""

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import numpy as np


class GRURiskWrapper(nn.Module):
    """Wraps the GRU model to return ONLY the K=1 risk logit for Captum."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, fut_logits, _, _ = self.model(x)
        return fut_logits[:, 0].unsqueeze(1)  # K=1 future risk logit


def explain_prediction(
    model,
    input_seq: torch.Tensor,
    feature_names: list[str],
    device: str = "cpu",
    top_n: int = 10,
) -> list[dict]:
    """
    Feature-level attributions (summed across the 30-step time dimension).

    Returns list of {Feature, Contribution} dicts sorted by |attribution|.
    """
    wrapper = GRURiskWrapper(model).to(device)
    wrapper.eval()

    ig = IntegratedGradients(wrapper)
    inp = input_seq.to(device).float().requires_grad_(True)

    attributions, _ = ig.attribute(inp, target=0, return_convergence_delta=True)

    # Sum across time axis → (n_features,)
    attr_sum = attributions.squeeze(0).sum(dim=0).detach().cpu().numpy()

    importance = [
        {"Feature": name, "Contribution": float(attr_sum[i])}
        for i, name in enumerate(feature_names)
    ]
    importance.sort(key=lambda x: abs(x["Contribution"]), reverse=True)
    return importance[:top_n]


def explain_temporal(
    model,
    input_seq: torch.Tensor,
    feature_names: list[str],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-TIMESTEP attributions for temporal heatmap display.

    Returns
    -------
    attr_matrix : np.ndarray (seq_len, n_features) — raw per-step per-feature attributions
    attr_by_step: np.ndarray (seq_len,)             — |attribution| summed across features per step
    """
    wrapper = GRURiskWrapper(model).to(device)
    wrapper.eval()

    ig = IntegratedGradients(wrapper)
    inp = input_seq.to(device).float().requires_grad_(True)

    attributions, _ = ig.attribute(inp, target=0, return_convergence_delta=True)

    # Shape: (1, seq_len, n_features) → (seq_len, n_features)
    attr_matrix = attributions.squeeze(0).detach().cpu().numpy()

    # Per-step importance: sum |attribution| across feature axis
    attr_by_step = np.abs(attr_matrix).sum(axis=1)

    return attr_matrix, attr_by_step
