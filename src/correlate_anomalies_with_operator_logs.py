import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Settings
log_path = "../data/operator_logs/operator_logs_text.txt"
anomaly_folder = "../anomaly_outputs"
output_file = "../correlation_outputs/correlation_logs.csv"
time_tolerance = timedelta(minutes=240)
similarity_threshold = 0.0  # Adjust as needed

def load_processed_files(folder: str = anomaly_folder) -> Dict[str, pd.DataFrame]:
    """Return dict of {filename: DataFrame} for all CSVs in *folder*."""
    return {
        f.name: pd.read_csv(f, parse_dates=["timestamp"])
        for f in Path(folder).glob("*.csv")
    }

# Load anomaly datasets
datasets = load_processed_files()

# Parse operator logs
with open(log_path, 'r') as f:
    raw_logs = f.readlines()

parsed_logs = []
timestamp_regex = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

for line in raw_logs:
    timestamp_match = timestamp_regex.search(line)
    if timestamp_match:
        timestamp_str = timestamp_match.group()
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except:
            continue
        remaining_text = line[timestamp_match.end():].strip()
        equipment_match = re.search(
            r'(blowout[_ ]preventer|drill[_ ]pipe|mud[_ ]pump|rotary[_ ]table)',
            remaining_text,
            re.IGNORECASE
        )
        equipment_type = equipment_match.group(1).replace(" ", "_").lower() if equipment_match else "unknown"
        parsed_logs.append({
            "timestamp": timestamp,
            "equipment_type": equipment_type,
            "log_text": remaining_text
        })

logs_df = pd.DataFrame(parsed_logs)
logs_df['equipment_type'] = logs_df['equipment_type'].str.lower()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for logs
logs_df['embedding'] = model.encode(logs_df['log_text'].tolist(), show_progress_bar=True).tolist()

df_correlations = []

# Process each anomaly file
for name, df in datasets.items():
    print(name)
    match = re.match(r"^(.*?)_[^_]+\.csv$", name)
    equipment_name = match.group(1).lower() if match else "unknown"
    df['equipment_type'] = equipment_name

    anomaly_col = [col for col in df.columns if "anomaly" in col.lower()]
    anomalies = df[df[anomaly_col[0]] == 1] if anomaly_col else df.copy()

    if anomalies.empty:
        continue

    # Add anomaly text column
    anomalies['anomaly_text'] = anomalies.apply(
        lambda row: f"Anomaly detected in {row['equipment_type']} at {row['timestamp']}",
        axis=1
    )

    # Generate embeddings for anomalies
    anomalies['embedding'] = model.encode(anomalies['anomaly_text'].tolist(), show_progress_bar=True).tolist()

    correlation_results = []

    for _, anomaly_row in anomalies.iterrows():
        a_time = anomaly_row['timestamp']
        a_eq = anomaly_row['equipment_type']
        a_emb = np.array(anomaly_row['embedding'])

        # Filter logs within time window
        time_window_logs = logs_df[
            (logs_df['timestamp'] >= a_time - time_tolerance) &
            (logs_df['timestamp'] <= a_time + time_tolerance)
        ]

        for _, log_row in time_window_logs.iterrows():
            l_emb = np.array(log_row['embedding'])
            similarity = cosine_similarity([a_emb], [l_emb])[0][0]

            if similarity >= similarity_threshold:
                row = {
                    "anomaly_time": a_time,
                    "equipment_type": equipment_name,
                    "anomaly_text": anomaly_row['anomaly_text'],
                    "log_time": log_row['timestamp'],
                    "log_text": log_row['log_text'],
                    "similarity": round(similarity, 3)
                }

                # Include all sensor columns except metadata
                exclude_cols = {'timestamp', 'equipment_type', 'embedding', anomaly_col[0], 'anomaly_text'}
                sensor_cols = [col for col in anomaly_row.index if col not in exclude_cols]
                for sensor in sensor_cols:
                    row[f"sensor_{sensor}"] = anomaly_row[sensor]

                correlation_results.append(row)
    print(len(correlation_results))
    # Get top 20 most similar results
    if correlation_results:
        correlation_df = pd.DataFrame(correlation_results)
        correlation_df = correlation_df.sort_values(by="similarity", ascending=False).head(20)
        df_correlations.append(correlation_df)

# Save final result
if df_correlations:
    df_final = pd.concat(df_correlations, ignore_index=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"✅ Correlation file written to: {output_file}")
else:
    print("⚠️ No correlations found.")
