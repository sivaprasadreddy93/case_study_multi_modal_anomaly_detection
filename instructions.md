
# Getting Started with the Anomaly Detection Pipeline

## Setup Instructions

To set up this repository:

1. **Install Python 3.10**
   - Download and install from [python.org](https://www.python.org/downloads/release/python-3100/)

2. **Install Required Libraries**
   - From the root folder, run:
     ```bash
     pip install -r requirements.txt
     ```

3. **Activate Your Python Environment**
   - Ensure your virtual environment is activated before running any scripts.

---

## Exploratory Data Analysis (EDA)

To perform initial data exploration:

- Open the notebook `EDA.ipynb` located in the `notebooks/` folder.
- Run through the notebook to analyze the synthetic dataset.

---

## Running the Pipeline

Once your environment is ready, execute the following scripts in order from the root folder.

### 1. `data_generation.py`
- **Purpose**: Generates synthetic sensor data and operator logs.
- **Output**: CSV files in `data/synthetic_oilrig_data/` and `operator_logs/`
- **Command**:
  ```bash
  python src/data_generation.py
  ```

### 2. `preprocess_data.py`
- **Purpose**: Cleans and transforms the generated data.
- **Output**: Processed CSV files saved in `processed_oil_rig_data/`
- **Command**:
  ```bash
  python src/preprocess_data.py
  ```

### 3. `anomaly_detection.py`
- **Purpose**: Applies anomaly detection models to the processed data.
- **Output**: Prediction results saved in the `anomaly_output/` folder.
- **Command**:
  ```bash
  python src/anomaly_detection.py
  ```

### 4. `correlate_anomalies_with_operator_logs.py`
- **Purpose**: Correlates detected anomalies with operator logs.
- **Output**: Correlation results saved in `correlation_outputs/`
- **Command**:
  ```bash
  python src/correlate_anomalies_with_operator_logs.py
  ```

### 5. `logs_summaries.py`
- **Purpose**: Generates summaries of correlated logs using GPT models.
- **Output**: Summarized logs saved in the appropriate folder.
- **Command**:
  ```bash
  python src/logs_summaries.py
  ```

---

## Launching the Streamlit App

After executing all pipeline scripts, launch the web application:

```bash
streamlit run app/app.py
```

This will open the Streamlit dashboard in your web browser.
