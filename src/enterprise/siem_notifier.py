"""
src/enterprise/siem_notifier.py
-------------------------------
Handles formatting and transmitting high-confidence alerts to external
SIEM and SOAR systems via Webhook or Syslog.
Supports generic JSON formats.
"""

import json
import logging
import requests
import socket
from datetime import datetime
from typing import Dict, Optional

class SIEMNotifier:
    def __init__(self, webhook_url: Optional[str] = None, syslog_host: Optional[str] = None, syslog_port: int = 514):
        self.webhook_url = webhook_url
        self.syslog_host = syslog_host
        self.syslog_port = syslog_port
        self.logger = logging.getLogger("SIEMNotifier")

    def format_alert(self, alert_data: dict) -> str:
        """Formats alert payload into a standardized JSON string."""
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "NetSentinel_Threat_Forecast",
            "severity": "CRITICAL" if alert_data.get("risk", 0) > 0.8 else "HIGH",
            "risk_score": alert_data.get("risk", 0),
            "predicted_stage": alert_data.get("predicted_stage", "Unknown"),
            "mitre_technique": alert_data.get("mapped_technique", "Unknown"),
            "mitre_id": alert_data.get("technique_id", "Unknown"),
            "affected_asset": alert_data.get("target_ip", "Unknown"),
            "asset_zone": alert_data.get("asset_zone", "Unknown"),
            "evidence": alert_data.get("evidence", [])
        }
        return json.dumps(payload)

    def send_webhook(self, payload_str: str) -> bool:
        """Sends JSON payload via HTTP POST Webhook."""
        if not self.webhook_url:
            self.logger.warning("SIEM Webhook URL not configured.")
            return False
        
        try:
            # Mock behavior if webhook URL is dummy
            if self.webhook_url.startswith("http://mock"):
                self.logger.info("MOCK SIEM Webhook Sent: %s", payload_str)
                return True

            response = requests.post(
                self.webhook_url, 
                data=payload_str, 
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            response.raise_for_status()
            self.logger.info("SIEM Webhook transmitted successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send SIEM Webhook: {e}")
            return False

    def send_syslog(self, payload_str: str) -> bool:
        """Sends JSON payload via UDP Syslog."""
        if not self.syslog_host:
            self.logger.warning("Syslog Host not configured.")
            return False
        
        try:
            # Mock behavior
            if self.syslog_host == "mock":
                self.logger.info("MOCK Syslog Sent: %s", payload_str)
                return True

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Standard syslog format: <PRI>TIMESTAMP HOSTNAME APP-NAME MESSAGE
            syslog_msg = f"<14>{datetime.utcnow().strftime('%b %d %H:%M:%S')} NetSentinel {payload_str}".encode('utf-8')
            sock.sendto(syslog_msg, (self.syslog_host, self.syslog_port))
            sock.close()
            self.logger.info("Syslog transmitted successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Syslog: {e}")
            return False

    def emit(self, alert_data: dict) -> dict:
        """Emits to all configured channels."""
        payload_str = self.format_alert(alert_data)
        success = {"webhook": False, "syslog": False}
        
        if self.webhook_url:
            success["webhook"] = self.send_webhook(payload_str)
        if self.syslog_host:
            success["syslog"] = self.send_syslog(payload_str)
            
        return success
