import pandas as pd
import numpy as np
import glob
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Map CIC-IDS2017 day filenames to real calendar dates (week of Jul 3-7, 2017)
FILENAME_DATE_MAP = {
    "monday":    "2017-07-03",
    "tuesday":   "2017-07-04",
    "wednesday": "2017-07-05",
    "thursday":  "2017-07-06",
    "friday":    "2017-07-07",
}


def clean_cicids2017_csv(input_dir, output_dir):
    """
    Cleans CIC-IDS2017 CSV files as per Phase 2 requirements, saves as Parquet.
    NOTE: Median imputation is intentionally deferred to training-time only
    to prevent data leakage (mandated by plan).
    """
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_files:
        logging.error("No CSV files found in %s. Please place raw data there.", input_dir)
        return

    df_list = []
    # Sort files so Monday < Tuesday < ... < Friday for correct chronology
    all_files = sorted(all_files, key=lambda f: list(FILENAME_DATE_MAP.keys()).index(
        next((d for d in FILENAME_DATE_MAP if d in os.path.basename(f).lower()), "monday")
    ))
    for file in all_files:
        logging.info("Loading %s...", file)
        df = pd.read_csv(file, encoding="latin1", low_memory=False)
        # Assign a day-anchor timestamp based on filename (MachineLearningCSV has no Timestamp)
        day_key = next((d for d in FILENAME_DATE_MAP if d in os.path.basename(file).lower()), None)
        if day_key:
            base_ts = pd.Timestamp(FILENAME_DATE_MAP[day_key] + " 08:00:00")
            # Spread rows evenly across an 8-hour work window per file
            n = len(df)
            delta_ms = int((8 * 3600 * 1000) / max(n, 1))
            df["Timestamp"] = [base_ts + pd.Timedelta(milliseconds=i * delta_ms) for i in range(n)]
            logging.info("Assigned timestamps: %s to %s (%d rows)",
                         df["Timestamp"].iloc[0], df["Timestamp"].iloc[-1], n)
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    logging.info("Combined Data Shape: %s", str(full_df.shape))

    # 1. Strip whitespace from column names
    full_df.columns = full_df.columns.str.strip()
    logging.info("Stripped column names.")

    # 2. Replace Inf/-Inf with NaN (median imputation deferred to training step)
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_cols = full_df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if not nan_cols.empty:
        logging.info("NaN counts per column after inf replacement:\n%s", nan_cols)

    # 3. Document and remove duplicates
    duplicates_count = full_df.duplicated().sum()
    if duplicates_count > 0:
        full_df.drop_duplicates(inplace=True)
    logging.info("Removed %d duplicate rows. Remaining: %d", duplicates_count, len(full_df))

    # 4. Standardize labels (fix CIC-IDS2017 encoding artifacts — latin1 mis-decoded em-dashes)
    if "Label" in full_df.columns:
        full_df["Label"] = full_df["Label"].str.strip()
        # Use regex to catch all encoding variants of "Web Attack - X" (â, Â, spaces, etc.)
        full_df["Label"] = full_df["Label"].apply(lambda x: re.sub(
            r"Web Attack\s*[\x80-\xff\s]+Brute Force", "Web Attack - Brute Force", str(x)
        ))
        full_df["Label"] = full_df["Label"].apply(lambda x: re.sub(
            r"Web Attack\s*[\x80-\xff\s]+XSS", "Web Attack - XSS", str(x)
        ))
        full_df["Label"] = full_df["Label"].apply(lambda x: re.sub(
            r"Web Attack\s*[\x80-\xff\s]+Sql Injection", "Web Attack - Sql Injection", str(x)
        ))
        # Also fix any remaining double-space variants
        full_df["Label"] = full_df["Label"].replace({
            "Web Attack  Brute Force": "Web Attack - Brute Force",
            "Web Attack  XSS": "Web Attack - XSS",
            "Web Attack  Sql Injection": "Web Attack - Sql Injection",
        })
        logging.info("Label distribution:\n%s", full_df["Label"].value_counts())

    # 5. Save 10k-row sample for rapid testing
    os.makedirs(output_dir, exist_ok=True)
    sample_df = full_df.sample(n=min(10000, len(full_df)), random_state=42)
    sample_path = os.path.join(output_dir, "cicids2017_sample.parquet")
    sample_df.to_parquet(sample_path, index=False)
    logging.info("Saved 10k-row sample to %s", sample_path)

    # 6. Save full cleaned data as Parquet
    final_path = os.path.join(output_dir, "processed_cicids2017.parquet")
    full_df.to_parquet(final_path, index=False)
    logging.info("Saved full cleaned dataset to %s", final_path)
    logging.info("Pipeline complete.")


if __name__ == "__main__":
    # Run from project root: python src/data/clean_cicids.py
    RAW_DATA_DIR = "data/raw/cicids2017/MachineLearningCVE"
    CLEANED_DATA_DIR = "data/interim/cleaned"
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    clean_cicids2017_csv(RAW_DATA_DIR, CLEANED_DATA_DIR)
