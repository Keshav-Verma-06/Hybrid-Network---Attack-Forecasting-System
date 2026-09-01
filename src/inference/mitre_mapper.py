import yaml
import os

class MitreAttackMapper:
    def __init__(self, rules_path="configs/mitre_rules.yaml"):
        if not os.path.exists(rules_path):
            rules_path = os.path.join(os.path.dirname(__file__), '../../configs/mitre_rules.yaml')
        with open(rules_path, 'r') as f:
            self.rules = yaml.safe_load(f)

    def map_to_mitre(self, gru_predicted_stage, hybrid_risk, current_features):
        stage_key = gru_predicted_stage.lower().replace(" ", "_")
        
        result = {
            "predicted_stage": gru_predicted_stage,
            "confidence": f"{hybrid_risk * 100:.1f}%",
            "mapped_technique": "Unknown Threat",
            "technique_id": "TBD",
            "evidence": []
        }
        
        # FIX: Handle Benign traffic explicitly
        if stage_key == "benign" or hybrid_risk < 0.10:
            result["predicted_stage"] = "Benign"
            result["mapped_technique"] = "Normal Background Traffic"
            result["technique_id"] = "N/A"
            result["evidence"] = ["Traffic metrics fall within normal operational baselines."]
            return result
            
        if stage_key in self.rules:
            rule = self.rules[stage_key]
            result["mapped_technique"] = rule["technique_name"]
            result["technique_id"] = rule["technique_id"]
            
            if "syn_to_ack_ratio" in current_features and current_features["syn_to_ack_ratio"] > 0.7:
                result["evidence"].append(f"High SYN-to-ACK ratio: {current_features['syn_to_ack_ratio']:.2f}")
            if "unique_destination_ports" in current_features and current_features["unique_destination_ports"] > 20:
                result["evidence"].append(f"Port scanning behavior: {current_features['unique_destination_ports']} ports contacted")
            if "flow_count" in current_features and current_features["flow_count"] > 100:
                result["evidence"].append(f"High flow volume detected: {current_features['flow_count']}")
                
        return result
