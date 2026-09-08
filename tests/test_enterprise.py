import pytest
import os
import sqlite3
import json
from pathlib import Path
from src.enterprise.asset_manager import AssetManager
from src.enterprise.siem_notifier import SIEMNotifier
from src.enterprise.audit_logger import AuditLogger

@pytest.fixture
def mock_db_path(tmp_path):
    return tmp_path / "mock_audit.db"

@pytest.fixture
def mock_asset_path(tmp_path):
    p = tmp_path / "assets.json"
    data = {
        "10.0.0.1": {"zone": "Internal", "criticality": 3, "role": "App"},
        "default": {"zone": "Unknown", "criticality": 1, "role": "Endpoint"}
    }
    with open(p, "w") as f:
        json.dump(data, f)
    return p

def test_asset_manager_loads_and_adjusts(mock_asset_path):
    manager = AssetManager(config_path=mock_asset_path)
    info = manager.get_asset_info("10.0.0.1")
    assert info["zone"] == "Internal"
    assert info["criticality"] == 3
    
    # 1.0 + (3 - 1) * 0.1 = 1.2
    adjusted = manager.adjust_risk(0.5, "10.0.0.1")
    assert pytest.approx(adjusted) == 0.6
    
    # Check default
    default_info = manager.get_asset_info("192.168.99.99")
    assert default_info["zone"] == "Unknown"
    
def test_siem_notifier_format():
    siem = SIEMNotifier()
    alert = {
        "risk": 0.9,
        "predicted_stage": "Exfiltration",
        "mapped_technique": "Data Transfer Size Limits",
        "technique_id": "T1030",
        "target_ip": "10.0.0.1",
        "asset_zone": "Internal",
        "evidence": ["High flow volume"]
    }
    formatted = siem.format_alert(alert)
    data = json.loads(formatted)
    assert data["severity"] == "CRITICAL"
    assert data["risk_score"] == 0.9
    assert data["affected_asset"] == "10.0.0.1"

def test_audit_logger(mock_db_path):
    logger = AuditLogger(db_path=mock_db_path)
    
    # Log event
    logger.log_event("Login", "admin", "Successful login")
    logs = logger.get_recent_logs()
    assert len(logs) == 1
    assert logs[0]["event_type"] == "Login"
    
    # Log alert
    logger.log_alert({"risk": 0.85, "target_ip": "10.0.0.1"})
    alerts = logger.get_recent_alerts()
    assert len(alerts) == 1
    assert alerts[0]["risk_score"] == 0.85
    
    # Test retention policy (no deletion because they are new)
    d_logs, d_alerts = logger.enforce_retention_policy(90)
    assert d_logs == 0
    assert d_alerts == 0
