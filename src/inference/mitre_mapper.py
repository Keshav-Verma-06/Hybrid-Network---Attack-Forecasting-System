"""
src/inference/mitre_mapper.py  (fully YAML-driven)
----------------------------------------------------
Maps GRU-predicted attack stages to MITRE ATT&CK techniques by
*evaluating the actual conditions from configs/mitre_rules.yaml*
against real window feature values.

YAML condition syntax (operators: >, <, >=, <=, ==, !=):
    conditions:
      syn_to_ack_ratio: "> 0.70"
      unique_destination_ports: "> 20"

Each condition that evaluates True becomes an evidence string.
The stage with the most satisfied conditions wins the final mapping
(overriding the GRU stage if evidence is stronger elsewhere).
"""

import re
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_OPS  = {">": "__gt__", "<": "__lt__", ">=": "__ge__", "<=": "__le__",
         "==": "__eq__", "!=": "__ne__"}

_CONDITION_LABELS = {
    "syn_to_ack_ratio":          "High SYN-to-ACK ratio",
    "unique_destination_ports":  "Port scanning behaviour",
    "flow_count":                "High flow volume",
    "failed_connection_ratio":   "High failed-connection ratio",
    "destination_fanin":         "High destination fan-in (lateral spread)",
    "outbound_bytes_ratio":      "High outbound-to-inbound bytes ratio",
}


def _parse_condition(condition_str: str):
    """
    Parse a condition string like '> 0.70' into (operator_fn, threshold).
    Returns a callable: val -> bool
    """
    condition_str = str(condition_str).strip()
    for op_str, op_fn in _OPS.items():
        if condition_str.startswith(op_str):
            threshold = float(condition_str[len(op_str):].strip())
            return lambda val, _fn=op_fn, _t=threshold: getattr(float(val), _fn)(_t)
    raise ValueError(f"Cannot parse condition: {condition_str!r}")


class MitreAttackMapper:
    def __init__(self, rules_path=None):
        if rules_path is None:
            rules_path = _ROOT / "configs" / "mitre_rules.yaml"
        rules_path = Path(rules_path)
        if not rules_path.exists():
            raise FileNotFoundError(f"MITRE rules not found at {rules_path}")
        with open(rules_path, "r") as f:
            raw = yaml.safe_load(f)
        self.rules = raw
        # Pre-compile condition callables
        self._compiled: dict[str, dict] = {}
        for stage_key, rule in raw.items():
            compiled_conds = {}
            for feat, cond_str in (rule.get("conditions") or {}).items():
                try:
                    compiled_conds[feat] = (_parse_condition(cond_str), str(cond_str))
                except ValueError:
                    pass
            self._compiled[stage_key] = {
                "conditions":      compiled_conds,
                "tactic":          rule.get("tactic", stage_key),
                "technique_id":    rule.get("technique_id", "N/A"),
                "technique_name":  rule.get("technique_name", "Unknown"),
            }

    # ── Public API ─────────────────────────────────────────────────────────────

    def map_to_mitre(
        self,
        gru_predicted_stage: str,
        hybrid_risk: float,
        current_features: dict,
    ) -> dict:
        """
        Map predicted stage → MITRE technique, fully driven by YAML conditions.

        Algorithm:
          1. Evaluate ALL stage conditions against current_features.
          2. Score each stage by (n_conditions_met / total_conditions).
          3. If GRU stage has any evidence → use it.
             Else → use highest-scoring stage with ≥1 condition met.
             Else → Benign / Unknown Threat.
          4. Build evidence list from satisfied conditions.

        Returns dict with keys:
          predicted_stage, confidence, mapped_technique, technique_id,
          tactic, evidence, conditions_met, conditions_total, data_driven
        """
        result = {
            "predicted_stage":  gru_predicted_stage,
            "confidence":       f"{hybrid_risk * 100:.1f}%",
            "mapped_technique": "Unknown Threat",
            "technique_id":     "TBD",
            "tactic":           "Unknown",
            "evidence":         [],
            "conditions_met":   0,
            "conditions_total": 0,
            "data_driven":      False,
        }

        # Benign / low risk short-circuit
        stage_key = gru_predicted_stage.lower().replace(" ", "_")
        if stage_key == "benign" or hybrid_risk < 0.10:
            result.update({
                "predicted_stage":  "Benign",
                "mapped_technique": "Normal Background Traffic",
                "technique_id":     "N/A",
                "tactic":           "None",
                "evidence":         ["Traffic metrics fall within normal operational baselines."],
            })
            return result

        # ── Evaluate all stages ────────────────────────────────────────────────
        stage_scores: dict[str, dict] = {}
        for sk, rule in self._compiled.items():
            met, evidence, total = [], [], len(rule["conditions"])
            for feat, (check_fn, cond_str) in rule["conditions"].items():
                feat_val = current_features.get(feat)
                if feat_val is None:
                    continue
                try:
                    passed = check_fn(feat_val)
                except Exception:
                    passed = False
                if passed:
                    label = _CONDITION_LABELS.get(feat, feat.replace("_", " ").title())
                    evidence.append(f"{label}: {feat_val:.3g} {cond_str}")
                    met.append(feat)
            stage_scores[sk] = {
                "n_met":    len(met),
                "total":    total,
                "score":    len(met) / max(total, 1),
                "evidence": evidence,
                "rule":     rule,
            }

        # ── Choose stage ──────────────────────────────────────────────────────
        # Prefer GRU-predicted stage if it has evidence
        gru_info = stage_scores.get(stage_key, {})
        if gru_info.get("n_met", 0) > 0:
            chosen_key  = stage_key
            chosen_info = gru_info
        else:
            # Fall back to highest-scoring stage with ≥1 condition met
            candidates = {k: v for k, v in stage_scores.items() if v["n_met"] > 0}
            if candidates:
                chosen_key  = max(candidates, key=lambda k: candidates[k]["score"])
                chosen_info = candidates[chosen_key]
                # Override predicted stage with data-driven choice
                result["predicted_stage"] = chosen_key.replace("_", " ").title()
                result["data_driven"]     = True
            else:
                # No conditions satisfied — trust GRU label alone
                chosen_key  = stage_key
                chosen_info = stage_scores.get(stage_key, {})

        rule = chosen_info.get("rule", {})
        result.update({
            "mapped_technique": rule.get("technique_name", "Unknown Threat"),
            "technique_id":     rule.get("technique_id",   "TBD"),
            "tactic":           rule.get("tactic",         "Unknown"),
            "evidence":         chosen_info.get("evidence", []),
            "conditions_met":   chosen_info.get("n_met",  0),
            "conditions_total": chosen_info.get("total",  0),
        })

        # Append fallback evidence if no conditions were met
        if not result["evidence"]:
            result["evidence"].append(
                f"GRU model predicted stage '{result['predicted_stage']}' "
                f"with risk score {hybrid_risk:.1%} — no individual feature threshold exceeded."
            )

        return result
