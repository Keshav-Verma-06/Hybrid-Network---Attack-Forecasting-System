import pandas as pd
import numpy as np
import os
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- CONFIGURATION ---
WINDOW_SIZE = '30s'
# As per page 7, hold out these specific attacks for the Unknown-Attack baseline
HELD_OUT_ATTACKS = ['DoS slowloris', 'DoS Slowhttptest', 'Bot']

# Map CIC-IDS2017 labels to MITRE ATT&CK Stages (Page 9)
ATTACK_STAGE_MAP = {
    'BENIGN': 'Benign',
    'PortScan': 'Reconnaissance',
    'FTP-Patator': 'Initial Access',
    'SSH-Patator': 'Initial Access',
    'Web Attack - Brute Force': 'Initial Access',
    'Web Attack - XSS': 'Initial Access',
    'Web Attack - Sql Injection': 'Initial Access',
    'Infiltration': 'Lateral Movement',
    'Bot': 'Command and Control',
    'DoS Hulk': 'Exfiltration',
    'DoS GoldenEye': 'Exfiltration',
    'DoS slowloris': 'Exfiltration',
    'DoS Slowhttptest': 'Exfiltration',
    'DDoS': 'Exfiltration',
    'Heartbleed': 'Exfiltration'
}

# 30-45 Core flow features (Page 8)
CORE_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'SYN Flag Count', 'ACK Flag Count', 'RST Flag Count', 'FIN Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Packet Length Mean',
    'Packet Length Std', 'Active Mean', 'Idle Mean', 'Destination Port', 'Protocol'
]


def load_and_prepare_data(filepath):
    """Loads cleaned Parquet and ensures standard datetime indexing."""
    logging.info("Loading data from %s", filepath)
    df = pd.read_parquet(filepath)

    # Ensure Timestamp is a datetime object
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=True)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Strip spaces from column names to safely match CORE_FEATURES
    df.columns = df.columns.str.strip()

    logging.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def split_and_impute(df, output_dir):
    """Splits into Train, Known-Test, and Unknown-Test; imputes only from Train medians."""
    logging.info("Splitting dataset into Known and Unknown attacks...")

    # 1. Separate the held-out unknown attacks
    unknown_mask = df["Label"].isin(HELD_OUT_ATTACKS)
    unknown_df = df[unknown_mask].copy()
    known_df = df[~unknown_mask].copy()

    # 2. Chronological split for Known data (80% Train, 20% Known-Test)
    split_idx = int(len(known_df) * 0.8)
    train_df = known_df.iloc[:split_idx].copy()
    known_test_df = known_df.iloc[split_idx:].copy()

    # Mix unknown attacks with a benign background from test split
    benign_test = known_test_df[known_test_df["Label"] == "BENIGN"].sample(frac=0.5, random_state=42)
    unknown_test_df = pd.concat([unknown_df, benign_test]).sort_values("Timestamp")

    logging.info("Splits: Train=%d  KnownTest=%d  UnknownTest=%d",
                 len(train_df), len(known_test_df), len(unknown_test_df))

    # 3. Imputation: Fit ONLY on training data to prevent leakage
    logging.info("Calculating medians on training data only...")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    medians = train_df[numeric_cols].median()

    # Save medians for inference pipeline later
    scalers_dir = os.path.join("models", "scalers")
    os.makedirs(scalers_dir, exist_ok=True)
    joblib.dump(medians, os.path.join(scalers_dir, "train_medians.pkl"))
    logging.info("Saved training medians to %s/train_medians.pkl", scalers_dir)

    train_df[numeric_cols] = train_df[numeric_cols].fillna(medians)
    known_test_df[numeric_cols] = known_test_df[numeric_cols].fillna(medians)
    unknown_test_df[numeric_cols] = unknown_test_df[numeric_cols].fillna(medians)

    return train_df, known_test_df, unknown_test_df


def aggregate_windows(df):
    """Aggregates flow-level data into fixed time-window network states."""
    df = df.set_index("Timestamp")

    # Keep only available core features to avoid KeyError
    available_features = [f for f in CORE_FEATURES if f in df.columns]

    # Define aggregation logic
    agg_dict = {f: "mean" for f in available_features}
    agg_dict["Label"] = lambda x: x.mode()[0] if not x.empty else "BENIGN"

    # Summing flags for derived ratios
    for col in ["SYN Flag Count", "ACK Flag Count", "RST Flag Count"]:
        if col in df.columns:
            agg_dict[col] = "sum"

    # Unique IP/Port counts
    for col in ["Source IP", "Destination IP", "Destination Port"]:
        if col in df.columns:
            agg_dict[col] = "nunique"

    # Perform Resampling
    windows = df.resample(WINDOW_SIZE).agg(agg_dict).dropna(subset=["Label"])

    # Derived State Features (Page 8-9)
    windows["flow_count"] = df["Label"].resample(WINDOW_SIZE).count()

    if "SYN Flag Count" in windows.columns and "ACK Flag Count" in windows.columns:
        windows["syn_to_ack_ratio"] = (
            windows["SYN Flag Count"] / (windows["ACK Flag Count"] + 1e-5)
        )

    # Rename IP/Port unique counts for clarity
    windows.rename(columns={
        "Source IP": "unique_source_ips",
        "Destination IP": "unique_destination_ips",
        "Destination Port": "unique_destination_ports",
    }, inplace=True, errors="ignore")

    return windows.reset_index()


def generate_forecast_labels(windows):
    """Generates future prediction targets for K=1, 3, and 5 windows."""
    logging.info("Generating target labels and forecast horizons...")

    # 1. Current attack state
    windows["current_attack_state"] = (windows["Label"] != "BENIGN").astype(int)

    # 2. Attack stage
    windows["attack_stage"] = windows["Label"].map(ATTACK_STAGE_MAP).fillna("Benign")

    # 3. Future attack states using rolling max on reversed series
    reversed_state = windows["current_attack_state"].iloc[::-1]

    for k in [1, 3, 5]:
        windows[f"future_attack_{k}"] = (
            reversed_state.rolling(window=k, min_periods=1).max()
            .iloc[::-1]
            .shift(-1)
            .fillna(0)
            .astype(int)
        )

    return windows


def pipeline_runner(input_path, output_dir):
    df = load_and_prepare_data(input_path)
    train_df, known_test_df, unknown_test_df = split_and_impute(df, output_dir)

    datasets = {
        "train_windows_30s": train_df,
        "test_known_windows_30s": known_test_df,
        "test_unknown_windows_30s": unknown_test_df,
    }

    for name, data in datasets.items():
        logging.info("Aggregating %s...", name)
        windowed_data = aggregate_windows(data)
        windowed_data = generate_forecast_labels(windowed_data)

        out_file = os.path.join(output_dir, f"{name}.parquet")
        windowed_data.to_parquet(out_file, index=False)
        logging.info("Saved %s -> shape %s to %s", name, windowed_data.shape, out_file)


if __name__ == "__main__":
    # Run from project root: python src/data/create_windows.py
    # Switch to sample file for rapid testing:
    #   INPUT_FILE = "data/interim/cleaned/cicids2017_sample.parquet"
    INPUT_FILE = "data/interim/cleaned/processed_cicids2017.parquet"
    OUTPUT_DIR = "data/processed/windows"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipeline_runner(INPUT_FILE, OUTPUT_DIR)
