"""
src/enterprise/audit_logger.py
------------------------------
SQLite-backed Audit Logger for enterprise deployments.
Records security alerts, system changes, and enforces retention policies.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _ROOT / "data" / "enterprise_audit.db"

class AuditLogger:
    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("AuditLogger")
        self._init_db()

    def _init_db(self):
        """Initialise SQLite tables if they do not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Audit Logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user TEXT NOT NULL,
                details TEXT NOT NULL
            )
        ''')
        
        # Alerts History table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                risk_score REAL NOT NULL,
                target_ip TEXT NOT NULL,
                mitre_stage TEXT,
                mitre_technique TEXT,
                mitre_id TEXT,
                resolved INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()

    def log_event(self, event_type: str, user: str, details: str):
        """Log a generic system event or admin action."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO audit_logs (timestamp, event_type, user, details) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), event_type, user, details)
        )
        conn.commit()
        conn.close()

    def log_alert(self, alert_data: dict):
        """Log a high-confidence threat forecast alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO alert_history 
               (timestamp, risk_score, target_ip, mitre_stage, mitre_technique, mitre_id) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                alert_data.get("risk", 0.0),
                alert_data.get("target_ip", "Unknown"),
                alert_data.get("predicted_stage", "Unknown"),
                alert_data.get("mapped_technique", "Unknown"),
                alert_data.get("technique_id", "Unknown")
            )
        )
        conn.commit()
        conn.close()

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Fetch recent alerts for the dashboard."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alert_history ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        
        conn.close()
        return results
        
    def get_recent_logs(self, limit: int = 50) -> List[Dict]:
        """Fetch recent audit logs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        
        conn.close()
        return results

    def enforce_retention_policy(self, days_to_keep: int = 90):
        """Delete logs and alerts older than specified days."""
        cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM audit_logs WHERE timestamp < ?", (cutoff_date,))
        logs_deleted = cursor.rowcount
        
        cursor.execute("DELETE FROM alert_history WHERE timestamp < ?", (cutoff_date,))
        alerts_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Retention policy enforced (>{days_to_keep} days). Deleted {logs_deleted} logs, {alerts_deleted} alerts.")
        return logs_deleted, alerts_deleted
