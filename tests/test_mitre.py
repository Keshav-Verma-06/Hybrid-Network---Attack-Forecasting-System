import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.mitre_mapper import MitreAttackMapper

def run_mitre_test():
    print("\n" + "="*60)
    print("🛡️  PHASE 7: MITRE ATT&CK MAPPER TEST")
    print("="*60)
    
    # Initialize the mapper
    mapper = MitreAttackMapper(rules_path="configs/mitre_rules.yaml")
    
    # ---------------------------------------------------------
    # SIMULATE AN ATTACK SCENARIO (As requested on Page 17)
    # ---------------------------------------------------------
    gru_predicted_stage = "Reconnaissance"
    hybrid_risk_score = 0.84  # 84% confidence from our Fusion Engine
    
    # The current 30-second window features triggering the alert
    current_features = {
        "unique_destination_ports": 42,
        "syn_to_ack_ratio": 0.91,
        "destination_fanin": 2
    }
    
    print("\n[+] Simulating GRU Hybrid Output...")
    print(f"    -> Forecasted Stage: {gru_predicted_stage}")
    print(f"    -> Hybrid Risk:      {hybrid_risk_score}")
    
    print("\n[+] Processing through MITRE Rules Engine...")
    result = mapper.map_to_mitre(gru_predicted_stage, hybrid_risk_score, current_features)
    
    # ---------------------------------------------------------
    # EXACT OUTPUT FORMAT FROM PAGE 17 OF PDF
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print(f"Predicted stage: {result['predicted_stage']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nMapped technique:")
    print(f"{result['technique_id']} - {result['mapped_technique']}")
    print(f"\nEvidence:")
    if result['evidence']:
        for evidence in result['evidence']:
            print(f" - {evidence}")
    else:
        print(" - No specific feature thresholds exceeded.")
    print("-" * 60 + "\n")

if __name__ == '__main__':
    run_mitre_test()
