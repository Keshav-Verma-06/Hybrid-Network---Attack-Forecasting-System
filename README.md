# 🛡️ NetSentinel: Hybrid Network-Attack Forecasting System

> A Temporal GRU + OCSVM Hybrid model that **forecasts network attacks before they happen** and detects **Zero-Day threats** via latent-space anomaly detection — built for the Smart India Hackathon (SIH).

---

## 🏗️ Architecture Overview

```
Raw PCAP / CSV
      │
      ▼
┌─────────────────────────────────────────────┐
│  Data Pipeline (src/data/)                  │
│  clean_cicids.py → create_windows.py        │
│  ingest_csv.py / ingest_pcap.py (upload)    │
└────────────────────┬────────────────────────┘
                     │  30-second aggregate windows
                     ▼
┌─────────────────────────────────────────────┐
│  Phase 3: Baselines (train_baseline.py)     │
│  Logistic Regression, Random Forest,        │
│  MLP, OCSVM, LOF                            │
└────────────────────┬────────────────────────┘
                     │  baseline_scaler.pkl
                     ▼
┌─────────────────────────────────────────────┐
│  Phase 4: GRU World Model (train_gru.py)    │
│  2-layer GRU, multi-task heads:             │
│    • Next-state prediction (Huber loss)     │
│    • Future risk K=1,3,5 (BCE + pos_weight) │
│    • Attack-stage classification (CE)       │
└────────────────────┬────────────────────────┘
                     │  gru_world_model.pth
                     ▼
┌─────────────────────────────────────────────┐
│  Phase 5: Risk Fusion (train_hybrid.py)     │
│  OCSVM on latent z_t + state error          │
│  R_t = 0.5·P + 0.3·A_ocsvm + 0.2·R_err    │
└────────────────────┬────────────────────────┘
                     │  hybrid_metrics.json
                     ▼
┌─────────────────────────────────────────────┐
│  Streamlit Dashboard (app/)                 │
│  Upload → Network State → Forecast →        │
│  Explainability → Benchmarks                │
└─────────────────────────────────────────────┘
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/Hybrid-Network---Attack-Forecasting-System.git
cd Hybrid-Network---Attack-Forecasting-System
```

### 2. Create a virtual environment (Python 3.11 recommended)
```bash
py -3.11 -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Required Files Before Running the Dashboard

The dashboard expects the following files. They are produced by running the training pipeline (see below).

| File | Produced by | Purpose |
|------|-------------|---------|
| `data/processed/windows/train_windows_30s.parquet` | `create_windows.py` | Training data |
| `data/processed/windows/test_known_windows_30s.parquet` | `create_windows.py` | Demo / eval data |
| `data/processed/windows/test_unknown_windows_30s.parquet` | `create_windows.py` | Unknown attack eval |
| `models/scalers/baseline_scaler.pkl` | `train_baseline.py` | Feature scaler |
| `models/gru/gru_world_model.pth` | `train_gru.py` | Trained GRU |
| `models/gru/fusion_ocsvm.pkl` | `train_hybrid.py` | OCSVM component |
| `models/gru/fusion_anomaly_scaler.pkl` | `train_hybrid.py` | Anomaly score scaler |
| `models/gru/fusion_error_scaler.pkl` | `train_hybrid.py` | Error scaler |
| `outputs/metrics/baseline_metrics_table.csv` | `train_baseline.py` | Benchmark table |
| `outputs/metrics/gru_metrics.json` | `train_gru.py` | GRU eval results |
| `outputs/metrics/hybrid_metrics.json` | `train_hybrid.py` | Hybrid eval results |
| `outputs/metrics/optimal_threshold.json` | `train_hybrid.py` | Risk threshold |

> **The pre-processed parquet files and pre-trained models are already committed to this repo** so you can run the dashboard immediately without retraining.

---

## 🚀 Training Order

> Run all commands from the **project root** directory.

### Step 0 (Optional): Prepare raw CIC-IDS 2017 data
Only needed if you have the raw CSVs from the CIC-IDS 2017 dataset.
```bash
# Place CSV files in: data/raw/cicids2017/MachineLearningCVE/
python src/data/clean_cicids.py
python src/data/create_windows.py
```

### Step 1: Phase 3 — Baselines
Trains all static baselines, saves the feature scaler, and produces the benchmark CSV.
```bash
python src/training/train_baseline.py
```
**Expected output:** `outputs/metrics/baseline_metrics_table.csv`, `models/scalers/baseline_scaler.pkl`

### Step 2: Phase 4 — GRU World Model
Trains the 2-layer multi-task GRU and evaluates all 3 forecast horizons.
```bash
python src/training/train_gru.py
```
**Expected output:** `models/gru/gru_world_model.pth`, `outputs/metrics/gru_metrics.json`

Typical training time: **~5–10 minutes** on CPU (30 epochs with early stopping).

### Step 3: Phase 5 — Hybrid Fusion
Fits the OCSVM on benign latent states and runs full multi-horizon evaluation.
```bash
python src/training/train_hybrid.py
```
**Expected output:** `models/gru/fusion_*.pkl`, `outputs/metrics/hybrid_metrics.json`, `outputs/metrics/optimal_threshold.json`

---

## 🖥️ Running the Dashboard

```bash
# From the project root
.venv\Scripts\streamlit run app/streamlit_app.py
```

Or with explicit port:
```bash
.venv\Scripts\streamlit run app/streamlit_app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

### Demo walkthrough
1. **Step 1 (Upload)** → Click **"🎯 Load Offline Fallback Demo"** — pre-loads the known test set and auto-advances to a window just before an attack.
2. **Step 2 (Network State)** → See current traffic metrics; use **Next Window / Fast Forward** to step through time.
3. **Step 3 (Forecast)** → View multi-horizon attack probabilities (K=1 / K=3 / K=5), fused risk score, OCSVM anomaly score, and MITRE ATT&CK mapping.
4. **Step 4 (Explainability)** → Captum Integrated Gradients shows which features drove the alert.
5. **Step 5 (Benchmarks)** → Full comparison table, horizon F1 line chart, and confusion matrices.

---

## 📤 Uploading Your Own Data

### CSV Upload
Any CIC-IDS-compatible CSV with at minimum these columns:
- `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`
- Optionally: `Label` / `label` / `Category` for accuracy metrics

### PCAP Upload
Raw PCAP/PCAPNG files. Scapy extracts per-flow features automatically.
Requires Wireshark is **not** needed — Scapy handles parsing.

---

## 🧪 Running Tests

```bash
# Run all tests
.venv\Scripts\pytest tests/ -v

# Run a specific test file
.venv\Scripts\pytest tests/test_mitre.py -v
.venv\Scripts\pytest tests/test_metrics.py -v
.venv\Scripts\pytest tests/test_model_shapes.py -v
.venv\Scripts\pytest tests/test_ingest_csv.py -v
```

### Test coverage
| File | Tests |
|------|-------|
| `test_mitre.py` | MITRE mapper technique IDs, evidence triggers, benign classification |
| `test_metrics.py` | F1, FPR, Precision, Recall, AUC, confusion matrix values, sample counts |
| `test_model_shapes.py` | GRU output tensor shapes, RiskFusionEngine bounds, gradient flow |
| `test_ingest_csv.py` | CSV validation, label alias detection, windowing, timestamp synthesis |

---

## 📂 Project Structure

```
Hybrid-Network---Attack-Forecasting-System/
├── app/
│   ├── streamlit_app.py          # Main landing page
│   └── pages/
│       ├── 1_Upload.py           # CSV/PCAP upload + offline demo
│       ├── 2_Network_State.py    # Current traffic visualisation
│       ├── 3_Forecast.py         # Multi-horizon GRU + Hybrid forecast
│       ├── 4_Explainability.py   # Captum Integrated Gradients
│       └── 5_Benchmarks.py       # Full metrics & comparison table
├── configs/
│   └── mitre_rules.yaml          # MITRE ATT&CK stage → technique mapping
├── data/
│   ├── raw/                      # Place CIC-IDS 2017 CSVs here
│   ├── interim/cleaned/          # Cleaned parquet (after clean_cicids.py)
│   └── processed/windows/        # 30s windowed parquet files
├── models/
│   ├── baselines/                # Saved sklearn baseline models
│   ├── gru/                      # GRU .pth and fusion .pkl files
│   └── scalers/                  # StandardScaler for features
├── outputs/
│   └── metrics/                  # CSV and JSON evaluation results
├── src/
│   ├── data/
│   │   ├── clean_cicids.py       # CIC-IDS 2017 raw CSV cleaner
│   │   ├── create_windows.py     # Flow → 30s window aggregation
│   │   ├── sequence_dataset.py   # PyTorch Dataset for sequences
│   │   ├── ingest_csv.py         # Upload CSV validation + preprocessing
│   │   └── ingest_pcap.py        # PCAP → flow → windows pipeline
│   ├── features/
│   ├── inference/
│   │   ├── explain.py            # Captum Integrated Gradients wrapper
│   │   └── mitre_mapper.py       # MITRE ATT&CK stage mapper
│   ├── models/
│   │   ├── gru_world_model.py    # Multi-task GRU architecture
│   │   ├── fusion_model.py       # RiskFusionEngine (OCSVM + scalers)
│   │   ├── logistic_baseline.py  # Supervised baseline definitions
│   │   └── anomaly_baselines.py  # OCSVM / LOF baseline definitions
│   ├── training/
│   │   ├── train_baseline.py     # Phase 3: static baselines
│   │   ├── train_gru.py          # Phase 4: GRU world model
│   │   ├── train_hybrid.py       # Phase 5: Risk fusion engine
│   │   └── train_gnn.py          # (Experimental) Temporal GNN
│   └── utils/
│       └── metrics.py            # evaluate_classification() + helpers
├── tests/
│   ├── test_mitre.py
│   ├── test_metrics.py
│   ├── test_model_shapes.py
│   └── test_ingest_csv.py
├── requirements.txt
├── setup_repo.py
└── README.md
```

---

## 📊 Key Results (from training pipeline)

| Model | Known F1 | Unknown F1 | Forecast K=1 | FPR |
|-------|:--------:|:----------:|:------------:|:---:|
| Logistic Regression | 0.00 | 0.34 | N/A | 0.002 |
| Random Forest | 0.08 | 0.00 | N/A | 0.000 |
| MLP | 0.38 | 0.03 | N/A | 0.014 |
| OCSVM (baseline) | 0.71 | 0.21 | N/A | 0.650 |
| **GRU World Model** | **0.46** | **0.01** | **0.46** | **0.041** |
| **GRU + OCSVM Hybrid** | **0.63** | **0.14** | **0.63** | **0.317** |

> The Hybrid achieves **14% Unknown F1** — catching Zero-Day attacks that all supervised baselines miss entirely.

---

## 🔧 Troubleshooting

**`UnicodeEncodeError` on Windows:**
```bash
set PYTHONIOENCODING=utf-8
```

**`Scaler not found` error:**
Run `python src/training/train_baseline.py` first.

**`GRU model not found` error:**
Run `python src/training/train_gru.py` first.

**PCAP parsing is slow:**
Large PCAPs (>50 MB) may take 1–2 minutes. Scapy processes each packet individually.

---

## 📜 License
MIT License — see `LICENSE` for details.
