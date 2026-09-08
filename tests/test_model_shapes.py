"""
tests/test_model_shapes.py
---------------------------
Pytest tests for model forward-pass output shapes.

Validates:
  - GRUWorldModel produces correct output tensor shapes for various input dims
  - RiskFusionEngine.predict_risk() returns correct output shapes
  - GRURiskWrapper (used by Captum) produces a scalar output per batch item
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

from src.models.gru_world_model import GRUWorldModel
from src.models.fusion_model import RiskFusionEngine
from src.inference.explain import GRURiskWrapper


# ── Fixtures ──────────────────────────────────────────────────────────────────

INPUT_DIM  = 30
HIDDEN_DIM = 64
SEQ_LEN    = 30
BATCH_SIZE = 8


@pytest.fixture(scope="module")
def gru_model():
    m = GRUWorldModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_batch():
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)


@pytest.fixture(scope="module")
def trained_fusion():
    """Create a fitted RiskFusionEngine using synthetic data."""
    np.random.seed(42)
    benign_z  = np.random.randn(200, HIDDEN_DIM).astype(np.float32)
    all_z     = np.random.randn(400, HIDDEN_DIM).astype(np.float32)
    all_errors= np.random.rand(400).astype(np.float32)

    engine = RiskFusionEngine()
    engine.fit(benign_z, all_z, all_errors)
    return engine


# ── GRUWorldModel shape tests ─────────────────────────────────────────────────

def test_gru_next_state_shape(gru_model, dummy_batch):
    with torch.no_grad():
        next_state, _, _, _ = gru_model(dummy_batch)
    assert next_state.shape == (BATCH_SIZE, INPUT_DIM), (
        f"next_state shape {next_state.shape} != ({BATCH_SIZE}, {INPUT_DIM})"
    )


def test_gru_future_logits_shape(gru_model, dummy_batch):
    with torch.no_grad():
        _, fut_logits, _, _ = gru_model(dummy_batch)
    assert fut_logits.shape == (BATCH_SIZE, 3), (
        f"fut_logits shape {fut_logits.shape} != ({BATCH_SIZE}, 3)"
    )


def test_gru_stage_logits_shape(gru_model, dummy_batch):
    with torch.no_grad():
        _, _, stage_logits, _ = gru_model(dummy_batch)
    assert stage_logits.shape == (BATCH_SIZE, 6), (
        f"stage_logits shape {stage_logits.shape} != ({BATCH_SIZE}, 6)"
    )


def test_gru_latent_state_shape(gru_model, dummy_batch):
    with torch.no_grad():
        _, _, _, z_t = gru_model(dummy_batch)
    assert z_t.shape == (BATCH_SIZE, HIDDEN_DIM), (
        f"z_t shape {z_t.shape} != ({BATCH_SIZE}, {HIDDEN_DIM})"
    )


def test_gru_different_input_dims():
    """GRU should handle various input dims without error."""
    for dim in [10, 20, 50]:
        m = GRUWorldModel(input_dim=dim, hidden_dim=32, num_layers=1)
        m.eval()
        x = torch.randn(4, SEQ_LEN, dim)
        with torch.no_grad():
            out = m(x)
        assert len(out) == 4, "GRU should return exactly 4 outputs"


def test_gru_batch_size_one(gru_model):
    x = torch.randn(1, SEQ_LEN, INPUT_DIM)
    with torch.no_grad():
        next_state, fut_logits, stage_logits, z_t = gru_model(x)
    assert next_state.shape[0]   == 1
    assert fut_logits.shape[0]   == 1
    assert stage_logits.shape[0] == 1
    assert z_t.shape[0]          == 1


# ── RiskFusionEngine shape tests ──────────────────────────────────────────────

def test_fusion_risk_output_shape(trained_fusion):
    np.random.seed(1)
    z_t     = np.random.randn(16, HIDDEN_DIM).astype(np.float32)
    p_fore  = np.random.rand(16).astype(np.float32)
    err     = np.random.rand(16).astype(np.float32)

    risk, a_ocsvm, r_err = trained_fusion.predict_risk(p_fore, z_t, err)
    assert risk.shape    == (16,), f"Risk shape {risk.shape} != (16,)"
    assert a_ocsvm.shape == (16,), f"a_ocsvm shape {a_ocsvm.shape} != (16,)"
    assert r_err.shape   == (16,), f"r_err shape {r_err.shape} != (16,)"


def test_fusion_risk_bounded(trained_fusion):
    np.random.seed(2)
    z_t    = np.random.randn(32, HIDDEN_DIM).astype(np.float32)
    p_fore = np.random.rand(32).astype(np.float32)
    err    = np.random.rand(32).astype(np.float32)

    risk, a_ocsvm, _ = trained_fusion.predict_risk(p_fore, z_t, err)
    assert np.all(risk >= 0.0),    "Risk scores must be >= 0"
    assert np.all(a_ocsvm >= 0.0), "OCSVM scores must be >= 0"
    assert np.all(a_ocsvm <= 1.0), "OCSVM scores must be <= 1"


def test_fusion_single_sample(trained_fusion):
    z_t   = np.random.randn(1, HIDDEN_DIM).astype(np.float32)
    p_f   = np.array([0.7], dtype=np.float32)
    err   = np.array([0.05], dtype=np.float32)

    risk, _, _ = trained_fusion.predict_risk(p_f, z_t, err)
    assert risk.shape == (1,)


# ── GRURiskWrapper (Captum integration) ───────────────────────────────────────

def test_gru_risk_wrapper_output_shape(gru_model, dummy_batch):
    wrapper = GRURiskWrapper(gru_model)
    wrapper.eval()
    with torch.no_grad():
        out = wrapper(dummy_batch)
    assert out.shape == (BATCH_SIZE, 1), (
        f"Wrapper output shape {out.shape} != ({BATCH_SIZE}, 1)"
    )


def test_gru_risk_wrapper_gradient_flow(gru_model, dummy_batch):
    """Integrated Gradients requires gradient flow through the wrapper."""
    wrapper = GRURiskWrapper(gru_model)
    x = dummy_batch.clone().requires_grad_(True)
    out = wrapper(x)
    out.sum().backward()
    assert x.grad is not None, "Gradients should flow through GRURiskWrapper"
