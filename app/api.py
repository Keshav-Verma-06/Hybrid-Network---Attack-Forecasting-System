"""NetSentinel web API.

FastAPI adapter around the existing ingestion, forecasting, explainability,
and enterprise modules. The browser client never loads model files directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest_csv import csv_to_windows, preprocess_for_inference, validate_csv
from src.enterprise.asset_manager import AssetManager
from src.enterprise.audit_logger import AuditLogger
from src.inference.mitre_mapper import MitreAttackMapper
from src.inference.rollout import recursive_rollout
from src.models.fusion_model import RiskFusionEngine
from src.models.gru_world_model import GRUWorldModel

app = FastAPI(title="NetSentinel API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

SESSION: dict[str, Any] = {"loaded": False, "current_idx": 30}
MODELS: dict[str, Any] = {}
EXCLUDE = {"Timestamp", "Label", "attack_stage", "future_attack_1", "future_attack_3", "future_attack_5"}
STAGE_NAMES = ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement", "Command and Control", "Exfiltration"]


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _record(row: pd.Series) -> dict[str, Any]:
    return {str(k): _json(v) for k, v in row.to_dict().items()}


def _load_models() -> dict[str, Any]:
    if MODELS:
        return MODELS
    scaler_path = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
    gru_path = ROOT / "models" / "gru" / "gru_world_model.pth"
    fusion_prefix = str(ROOT / "models" / "gru" / "fusion")
    if not scaler_path.exists() or not gru_path.exists():
        raise HTTPException(503, "Model artifacts are missing. Run the training pipeline first.")
    scaler = joblib.load(scaler_path)
    model = GRUWorldModel(input_dim=scaler.n_features_in_, hidden_dim=128, num_layers=2)
    model.load_state_dict(torch.load(gru_path, map_location="cpu"))
    model.eval()
    engine = RiskFusionEngine()
    engine.load(fusion_prefix)
    threshold = 0.05
    threshold_path = ROOT / "outputs" / "metrics" / "optimal_threshold.json"
    if threshold_path.exists():
        threshold = float(json.loads(threshold_path.read_text()).get("threshold", threshold))
    MODELS.update({"scaler": scaler, "model": model, "engine": engine, "mapper": MitreAttackMapper(), "assets": AssetManager(), "threshold": threshold})
    return MODELS


def _set_session(df: pd.DataFrame, scaled: np.ndarray, feature_names: list[str], source: str) -> None:
    attack_indices = df.index[df.get("current_attack_state", pd.Series(0, index=df.index)).eq(1)] if "current_attack_state" in df else []
    first_attack = int(attack_indices[0]) if len(attack_indices) else 30
    SESSION.update({"loaded": True, "df": df, "scaled": scaled, "feature_names": feature_names, "source": source, "current_idx": max(30, first_attack - 5)})


def _require_session() -> None:
    if not SESSION.get("loaded"):
        raise HTTPException(400, "Load a CSV, PCAP, or offline demo first.")


def _forecast() -> dict[str, Any]:
    _require_session()
    if SESSION["current_idx"] < 30:
        raise HTTPException(400, "At least 30 history windows are required for forecasting.")
    models = _load_models()
    df = SESSION["df"]
    idx = SESSION["current_idx"]
    # Reuse the exact aligned feature order produced during ingestion. The raw
    # upload may contain valid extra aggregate columns that are not model inputs.
    features = SESSION["feature_names"]
    context = df.iloc[idx - 29: idx + 1]
    x_tensor = torch.tensor(models["scaler"].transform(context[features].fillna(0).values), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_pred, fut_logits, stage_logits, z_t = models["model"](x_tensor)
        probs = torch.sigmoid(fut_logits[0]).numpy()
        state_error = float(torch.mean((next_pred[0] - x_tensor[0, -1]) ** 2).item())
        risk, anomaly, reconstruction = models["engine"].predict_risk(np.array([probs[0]]), z_t.numpy(), np.array([state_error]))
    current_features = context.iloc[-1][features].to_dict()
    stage = STAGE_NAMES[int(torch.argmax(stage_logits[0]).item())]
    risk_value = float(risk[0])
    mitre = models["mapper"].map_to_mitre(stage, risk_value, current_features)
    timeline = recursive_rollout(models["model"], models["engine"], x_tensor, n_steps=10)
    target_ip = "10.0.0.50"
    asset = models["assets"].get_asset_info(target_ip)
    adjusted = models["assets"].adjust_risk(risk_value, target_ip)
    return {"risk": adjusted, "base_risk": risk_value, "threshold": models["threshold"], "alert": adjusted > models["threshold"], "probabilities": {"30s": float(probs[0]), "90s": float(probs[1]), "150s": float(probs[2])}, "anomaly": float(anomaly[0]), "reconstruction_error": float(reconstruction[0]), "stage": stage, "mitre": mitre, "asset": {"ip": target_ip, **asset}, "timeline": [{k: _json(v) for k, v in point.items()} for point in timeline], "tensor": x_tensor, "features": features}


def _public_forecast(result: dict[str, Any]) -> dict[str, Any]:
    """Remove inference-only tensors before returning a JSON response."""
    return {key: value for key, value in result.items() if key not in {"tensor", "features"}}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"api": "online", "model": (ROOT / "models" / "gru" / "gru_world_model.pth").exists(), "offline_mode": True, "loaded": SESSION.get("loaded", False)}


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    _require_session()
    current = SESSION["df"].iloc[SESSION["current_idx"]]
    try:
        forecast = _public_forecast(_forecast())
        risk = round(forecast["risk"] * 100)
        forecasts = {k: round(v * 100) for k, v in forecast["probabilities"].items()}
    except HTTPException:
        forecast, risk, forecasts = None, 0, {"30s": 0, "90s": 0, "150s": 0}
    return {"source": SESSION["source"], "window": SESSION["current_idx"], "window_count": len(SESSION["df"]), "risk": risk, "forecasts": forecasts, "current": _record(current), "forecast": forecast}


@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        raw = pd.read_csv(file.file, encoding="latin1", low_memory=False)
        report = validate_csv(raw)
        if not report["ok"]:
            return {"ok": False, "report": report}
        windows = csv_to_windows(raw, label_col=report["label_col"])
        if len(windows) < 31:
            return {"ok": False, "report": {**report, "errors": [f"Only {len(windows)} windows produced; need at least 31."]}}
        scaled, names = preprocess_for_inference(windows)
        _set_session(windows, scaled, names, file.filename or "uploaded.csv")
        return {"ok": True, "report": report, "windows": len(windows), "overview": overview()}
    except Exception as exc:
        raise HTTPException(400, f"CSV processing failed: {exc}") from exc


@app.post("/api/upload/demo")
def upload_demo() -> dict[str, Any]:
    path = ROOT / "data" / "processed" / "windows" / "test_known_windows_30s.parquet"
    if not path.exists():
        raise HTTPException(404, "Offline demo data is not available.")
    try:
        try:
            df = pd.read_parquet(path)
        except Exception:
            # Some environments ship a pyarrow build that cannot read the
            # nested parquet metadata; Polars is already a project dependency.
            import polars as pl
            df = pl.read_parquet(path).to_pandas()
        scaled, names = preprocess_for_inference(df)
        _set_session(df, scaled, names, "offline demo")
        return {"ok": True, "overview": overview()}
    except Exception as exc:
        raise HTTPException(500, f"Demo loading failed: {exc}") from exc


@app.post("/api/upload/pcap")
async def upload_pcap(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        from src.data.ingest_pcap import pcap_bytes_to_windows
        windows = pcap_bytes_to_windows(await file.read())
        if len(windows) < 31:
            raise HTTPException(400, f"Only {len(windows)} windows produced; need at least 31.")
        scaled, names = preprocess_for_inference(windows)
        _set_session(windows, scaled, names, file.filename or "capture.pcap")
        return {"ok": True, "windows": len(windows), "overview": overview()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"PCAP processing failed: {exc}") from exc


@app.post("/api/navigation/{direction}")
def navigate(direction: str) -> dict[str, Any]:
    _require_session()
    step = {"previous": -1, "next": 1, "fast": 10}.get(direction)
    if step is None:
        raise HTTPException(400, "Unknown navigation direction.")
    SESSION["current_idx"] = max(30, min(len(SESSION["df"]) - 1, SESSION["current_idx"] + step))
    return overview()


@app.get("/api/state")
def state() -> dict[str, Any]:
    _require_session()
    row = SESSION["df"].iloc[SESSION["current_idx"]]
    values = SESSION["scaled"][SESSION["current_idx"]]
    top = sorted(zip(SESSION["feature_names"], values), key=lambda item: abs(item[1]), reverse=True)[:8]
    return {"window": SESSION["current_idx"], "count": len(SESSION["df"]), "current": _record(row), "features": [{"name": n, "value": _json(v)} for n, v in top]}


@app.get("/api/forecast")
def forecast() -> dict[str, Any]:
    result = _forecast()
    result.pop("tensor", None)
    return result


@app.get("/api/explainability")
def explainability() -> dict[str, Any]:
    try:
        from src.inference.explain import explain_prediction, explain_temporal
    except ImportError as exc:
        raise HTTPException(503, "Explainability requires the Captum dependency.") from exc
    result = _forecast()
    models = _load_models()
    top = explain_prediction(models["model"], result["tensor"], result["features"], top_n=10)
    matrix, by_step = explain_temporal(models["model"], result["tensor"], result["features"])
    return {"top_features": top, "temporal": matrix[:, [result["features"].index(item["Feature"]) for item in top]].tolist(), "temporal_totals": by_step.tolist(), "feature_names": [item["Feature"] for item in top]}


@app.get("/api/benchmarks")
def benchmarks() -> dict[str, Any]:
    metrics = ROOT / "outputs" / "metrics"
    def load_json(name: str) -> Any:
        path = metrics / name
        return json.loads(path.read_text()) if path.exists() else {}
    def load_csv(name: str) -> list[dict[str, Any]]:
        path = metrics / name
        return pd.read_csv(path).replace({np.nan: None}).to_dict("records") if path.exists() else []
    return {"gru": load_json("gru_metrics.json"), "hybrid": load_json("hybrid_metrics.json"), "baselines": load_csv("baseline_metrics_table.csv"), "attacks": load_csv("per_attack_evaluation.csv"), "stages": load_csv("per_stage_evaluation.csv"), "lead_time": load_json("lead_time_results.json"), "holdout": load_json("holdout_family_evaluation.json")}


@app.get("/api/admin")
def admin() -> dict[str, Any]:
    logger = AuditLogger()
    metrics = ROOT / "outputs" / "metrics"
    def read(name: str) -> Any:
        path = metrics / name
        return json.loads(path.read_text()) if path.exists() else {}
    return {"alerts": logger.get_recent_alerts(100), "logs": logger.get_recent_logs(100), "drift": read("drift_monitoring.json"), "robustness": read("adversarial_robustness.json")}
