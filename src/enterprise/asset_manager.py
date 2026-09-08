"""
src/enterprise/asset_manager.py
-------------------------------
Manages asset criticality and network-zone context for enterprise deployment.
Allows dynamic adjustment of risk scores based on asset importance.
"""

import json
from pathlib import Path
from typing import Dict

_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_FILE = _ROOT / "configs" / "assets.json"

class AssetManager:
    def __init__(self, config_path: Path = _ASSETS_FILE):
        self.config_path = config_path
        self.assets: Dict[str, dict] = {}
        self.load()

    def load(self):
        """Load asset definitions from config file."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.assets = json.load(f)
        else:
            # Default fallback mock data
            self.assets = {
                "192.168.1.100": {"zone": "DMZ", "criticality": 3, "role": "Web Server"},
                "10.0.0.50": {"zone": "Internal", "criticality": 4, "role": "Database Server"},
                "10.0.0.200": {"zone": "SCADA", "criticality": 5, "role": "ICS Controller"},
                "default": {"zone": "Unknown", "criticality": 1, "role": "Endpoint"}
            }
            self.save()

    def save(self):
        """Save current asset definitions to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self.assets, f, indent=2)

    def get_asset_info(self, ip_address: str) -> dict:
        """Retrieve zone, criticality, and role for an IP."""
        return self.assets.get(ip_address, self.assets["default"])

    def adjust_risk(self, base_risk: float, ip_address: str) -> float:
        """
        Adjust risk based on asset criticality.
        Multiplier formula: 1.0 + (criticality - 1) * 0.1
        Max multiplier (crit 5) = 1.4x. Max risk is capped at 1.0.
        """
        info = self.get_asset_info(ip_address)
        crit = info["criticality"]
        multiplier = 1.0 + (crit - 1) * 0.1
        adjusted = base_risk * multiplier
        return min(adjusted, 1.0)
