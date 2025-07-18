import os
import pandas as pd

# ─────────────────────────────────────────
# Timestamp Handling
# ─────────────────────────────────────────

def handle_timestamp(df: pd.DataFrame, timestamp_col="timestamp", resample_rule=None):
    """
    Parse timestamp, set index, and optionally resample only numeric columns.
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df.set_index(timestamp_col, inplace=True)

    # Preserve non-numeric columns (e.g., equipment_id)
    non_numeric_cols = df.select_dtypes(exclude="number")
    numeric_df = df.select_dtypes(include="number")

    # Add temporal features to numeric data
    numeric_df["hour"] = df.index.hour
    numeric_df["day"] = df.index.day
    numeric_df["weekday"] = df.index.weekday
    numeric_df["is_weekend"] = numeric_df["weekday"].isin([5, 6]).astype(int)

    if resample_rule:
        numeric_df = numeric_df.resample(resample_rule).mean().interpolate()

    # Merge non-numeric columns back (optional or drop if not useful)
    numeric_df = numeric_df.reset_index()
    if not non_numeric_cols.empty:
        non_numeric_cols = non_numeric_cols.reset_index()
        numeric_df = pd.merge(numeric_df, non_numeric_cols, on=timestamp_col, how="left")

    return numeric_df


# ─────────────────────────────────────────
# Missing Data Handling
# ─────────────────────────────────────────

def fill_missing_values(df: pd.DataFrame, method="ffill"):
    """
    Fill missing values using specified method: 'ffill', 'bfill', 'median', 'mean'.
    """
    if method == "ffill":
        return df.ffill()
    elif method == "bfill":
        return df.bfill()
    elif method == "median":
        return df.fillna(df.median(numeric_only=True))
    elif method == "mean":
        return df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("Unsupported fill method.")

def drop_missing_rows(df: pd.DataFrame):
    """Drop rows with any missing values."""
    return df.dropna()

# ─────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────

def preprocess_pipeline(df: pd.DataFrame, resample_rule=None, fill_method="ffill"):
    df = handle_timestamp(df, resample_rule=resample_rule)
    df = fill_missing_values(df, method=fill_method)
    return df

# ─────────────────────────────────────────
# Process Multiple Equipment Files
# ─────────────────────────────────────────

def process_all_equipment(input_folder="synthetic_oilrig_data", output_folder="processed_data",
                           resample_rule=None, fill_method="ffill"):
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    print("[INFO] Found files:", files)

    for file in files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)

        df_processed = preprocess_pipeline(df, resample_rule=resample_rule, fill_method=fill_method)

        out_path = os.path.join(output_folder, file)
        df_processed.to_csv(out_path, index=False)
        print(f"[INFO] Processed and saved: {out_path}")

# Example usage
if __name__ == "__main__":
    process_all_equipment(
        input_folder="data/synthetic_oilrig_data/",
        output_folder="data/processed_oilrig_data",
        resample_rule="5min",
        fill_method="median"
    )
