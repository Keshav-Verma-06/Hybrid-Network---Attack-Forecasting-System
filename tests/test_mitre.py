"""
tests/test_mitre.py
--------------------
Pytest tests for the MitreAttackMapper.

Validates:
  - All 5 attack stages produce valid mapped technique dicts
  - Required keys are always present
  - Benign / low-risk traffic is classified correctly
  - Technique IDs match the YAML configuration
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from src.inference.mitre_mapper import MitreAttackMapper


@pytest.fixture(scope="module")
def mapper():
    return MitreAttackMapper()


REQUIRED_KEYS = {
    "predicted_stage",
    "confidence",
    "mapped_technique",
    "technique_id",
    "tactic",
    "evidence",
    "conditions_met",
    "conditions_total",
    "data_driven",
}


# ── Key structure ─────────────────────────────────────────────────────────────

def test_output_has_all_required_keys(mapper):
    result = mapper.map_to_mitre("Reconnaissance", 0.9, {})
    assert REQUIRED_KEYS == set(result.keys()), (
        f"Missing keys: {REQUIRED_KEYS - set(result.keys())}"
    )


def test_confidence_is_percentage_string(mapper):
    result = mapper.map_to_mitre("Reconnaissance", 0.75, {})
    assert result["confidence"].endswith("%"), "Confidence should end with '%'"
    pct = float(result["confidence"].replace("%", ""))
    assert 0 <= pct <= 100, "Confidence percentage out of range"


def test_evidence_is_list(mapper):
    result = mapper.map_to_mitre("Reconnaissance", 0.9, {})
    assert isinstance(result["evidence"], list)


# ── Benign / low-risk classification ─────────────────────────────────────────

def test_low_risk_classified_as_benign(mapper):
    result = mapper.map_to_mitre("Reconnaissance", 0.05, {})
    assert result["predicted_stage"] == "Benign"
    assert result["technique_id"] == "N/A"


def test_benign_stage_returns_benign(mapper):
    result = mapper.map_to_mitre("Benign", 0.9, {})
    assert result["predicted_stage"] == "Benign"


# ── Technique ID checks ───────────────────────────────────────────────────────

EXPECTED_TECHNIQUE_IDS = {
    "Reconnaissance":       "T1046",
    "Initial Access":       "T1110",
    "Lateral Movement":     "T1021",
    "Command and Control":  "T1071",
    "Exfiltration":         "T1567",
}


@pytest.mark.parametrize("stage, expected_id", EXPECTED_TECHNIQUE_IDS.items())
def test_technique_id_correct(mapper, stage, expected_id):
    result = mapper.map_to_mitre(stage, 0.9, {})
    assert result["technique_id"] == expected_id, (
        f"Stage '{stage}': expected {expected_id}, got {result['technique_id']}"
    )


# ── Evidence triggers ─────────────────────────────────────────────────────────

def test_high_syn_ratio_generates_evidence(mapper):
    features = {"syn_to_ack_ratio": 0.95}
    result = mapper.map_to_mitre("Reconnaissance", 0.9, features)
    syn_evidence = [e for e in result["evidence"] if "SYN" in e]
    assert len(syn_evidence) > 0, "Expected SYN ratio evidence for syn_to_ack_ratio=0.95"


def test_port_scan_generates_evidence(mapper):
    features = {"unique_destination_ports": 50}
    result = mapper.map_to_mitre("Reconnaissance", 0.9, features)
    port_evidence = [e for e in result["evidence"] if "port" in e.lower() or "scan" in e.lower()]
    assert len(port_evidence) > 0, "Expected port scanning evidence for 50 destination ports"


def test_low_syn_ratio_no_syn_evidence(mapper):
    features = {"syn_to_ack_ratio": 0.1}
    result = mapper.map_to_mitre("Reconnaissance", 0.9, features)
    syn_evidence = [e for e in result["evidence"] if "SYN" in e]
    assert len(syn_evidence) == 0, "Should not generate SYN evidence for low ratio"


# ── Run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    m = MitreAttackMapper()
    print("\n" + "=" * 60)
    print("MITRE ATT&CK MAPPER — STANDALONE TEST")
    print("=" * 60)

    result = m.map_to_mitre(
        "Reconnaissance",
        0.84,
        {"unique_destination_ports": 42, "syn_to_ack_ratio": 0.91},
    )
    print(f"Predicted stage: {result['predicted_stage']}")
    print(f"Confidence:      {result['confidence']}")
    print(f"Technique:       {result['technique_id']} — {result['mapped_technique']}")
    print(f"Evidence:        {result['evidence']}")
    print("=" * 60)
