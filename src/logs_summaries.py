import openai
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm

# Load OpenAI API key
def load_api_key(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["openai"]["api_key"]

# Format equipment-level prompt
def generate_equipment_prompt(equipment_type, group_df):
    context_lines = []
    for idx, row in group_df.iterrows():
        context_lines.append(ff"""
You are a reliability engineer tasked with summarizing anomalies detected on the equipment: **{equipment_type}**.

From the data below, extract and compile the following structured insights:

Context:
{context}

Please generate a single-row summary with these fields:

- Equipment: Standardized equipment name (e.g., "Blowout Preventer")
- Total Anomalies Observed: Count of total anomalies in the context
- Most Recent Anomaly Time: Timestamp of the latest anomaly
- Recent Anomaly Description: Text summary of the most recent anomaly
- Diagnosis & Likely Root Cause: Root cause hypotheses based on anomaly–log correlation
- Relevant Log: Log text that aligns with or supports the anomaly
- Actionable Review: Concise, practical recommendations (e.g., inspections, checks, tests)

Output the result in **tabular form** (one row per equipment) with columns exactly matching the above headers.
"""
    return prompt

# Call OpenAI API
def get_summary(prompt, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "Summary generation failed."

def main():
    openai.api_key = load_api_key()

    input_csv = "correlation_outputs/correlation_logs.csv"
    output_csv = "summaries/equipment_type_summaries.csv"

    df = pd.read_csv(input_csv)

    # Ensure required columns exist
    required_cols = {"equipment_type", "anomaly_time", "anomaly_text", "log_time", "log_text", "similarity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    summary_rows = []

    for equipment_type, group in tqdm(df.groupby("equipment_type"), desc="Summarizing per equipment"):
        prompt = generate_equipment_prompt(equipment_type, group)
        summary = get_summary(prompt)
        summary_rows.append({"equipment_type": equipment_type, "summary": summary})

    # Save to CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_csv, index=False)
    print(f" Equipment-wise summaries saved to: {output_csv}")

if __name__ == "__main__":
    main()
