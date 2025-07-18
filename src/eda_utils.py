import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from IPython.display import display, Markdown

# ────────────────────────────────────────────────────────
# EDA Utilities
# ────────────────────────────────────────────────────────

def basic_summary(df: pd.DataFrame):
    """
    Display a well-formatted summary of the DataFrame:
    - Shape of the dataset
    - Data types + missing values (combined)
    - Descriptive statistics (excluding timestamp)
    """

    # Section: Shape
    display(Markdown("###  Dataset Shape"))
    shape_df = pd.DataFrame({
        "Metric": ["Rows", "Columns"],
        "Value": [df.shape[0], df.shape[1]]
    })
    display(shape_df)

    # Section: Data types + missing values
    display(Markdown("### Data Types & Missing Values"))
    types = df.dtypes.astype(str)
    missing = df.isnull().sum()
    non_nulls = df.notnull().sum()
    summary = pd.DataFrame({
        "Data Type": types,
        "Non-Null Count": non_nulls,
        "Missing Count": missing,
        "Missing (%)": (missing / len(df) * 100).round(2)
    })
    display(summary)

    # Section: Descriptive statistics
    display(Markdown("### Descriptive Statistics (Numerical Columns Only)"))
    num_cols = df.select_dtypes(include=['number']).columns
    desc = df[num_cols].describe().transpose()
    desc["missing"] = df[num_cols].isnull().sum()
    display(desc.round(3))




def plot_null_heatmap(df: pd.DataFrame):
    """
    Visualize missing data using a heatmap.
    """
    print("\n[INFO] Visualizing missing values")
    msno.heatmap(df)
    plt.title("Missing Data Heatmap")
    plt.show()

def plot_correlation(df: pd.DataFrame):
    """
    Plot heatmap of sensor correlations.
    """
    sensor_cols = [col for col in df.columns if col not in ['timestamp', 'equipment_id']]
    corr = df[sensor_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Sensor Correlation Matrix")
    plt.show()

def plot_time_series(df: pd.DataFrame):
    """
    Plot time series of each sensor.
    """
    sensor_cols = [col for col in df.columns if col not in ['timestamp', 'equipment_id']]
    plt.figure(figsize=(14, 2 * len(sensor_cols)))
    for i, col in enumerate(sensor_cols):
        plt.subplot(len(sensor_cols), 1, i + 1)
        plt.plot(df['timestamp'], df[col], label=col)
        plt.title(col)
        plt.xlabel("Timestamp")
        plt.ylabel(col)
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_histograms(df: pd.DataFrame, sensor_cols: list = None, bins=50):
    """
    Plot histograms of sensor values to understand distributions.
    """
    if sensor_cols is None:
        sensor_cols = [col for col in df.columns if col not in ['timestamp', 'equipment_id']]

    n = len(sensor_cols)
    ncols = 2
    nrows = (n + 1) // 2
    plt.figure(figsize=(14, 5 * nrows))

    for i, col in enumerate(sensor_cols):
        plt.subplot(nrows, ncols, i + 1)
        sns.histplot(df[col], kde=True, bins=bins)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_boxplots(df: pd.DataFrame, sensor_cols: list = None):
    """
    Plot boxplots for sensor columns to identify outliers.
    """
    if sensor_cols is None:
        sensor_cols = [col for col in df.columns if col not in ['timestamp', 'equipment_id']]

    plt.figure(figsize=(12, len(sensor_cols) * 1.5))
    sns.boxplot(data=df[sensor_cols], orient='h')
    plt.title("Boxplot of Sensor Readings")
    plt.grid(True)
    plt.show()

def plot_pairwise_relationships(df: pd.DataFrame, sensor_cols: list = None):
    """
    Plot pairwise relationships between sensor readings.
    Useful to detect clusters or nonlinear relationships.
    """
    if sensor_cols is None:
        sensor_cols = [col for col in df.columns if col not in ['timestamp', 'equipment_id']]

    sns.pairplot(df[sensor_cols].dropna())
    plt.suptitle("Pairwise Sensor Relationships", y=1.02)
    plt.show()

def plot_rolling_mean(df: pd.DataFrame, col: str, window: int = 60):
    """
    Plot rolling mean for a single sensor to identify trends.
    """
    df_sorted = df.sort_values("timestamp")
    plt.figure(figsize=(12, 4))
    plt.plot(df_sorted["timestamp"], df_sorted[col], label="Raw")
    plt.plot(df_sorted["timestamp"], df_sorted[col].rolling(window=window).mean(), label=f"{window}-point Rolling Mean", color="orange")
    plt.title(f"{col} - Raw vs Rolling Mean")
    plt.xlabel("Time")
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_time_heatmap(df: pd.DataFrame, col: str):
    """
    Visualize how sensor readings evolve over time with a heatmap.
    Good for spotting long-term drift or sudden anomalies.
    """
    df = df.set_index("timestamp")
    reshaped = df[[col]].copy()
    reshaped["hour"] = reshaped.index.hour
    reshaped["day"] = reshaped.index.date
    pivot = reshaped.pivot_table(index="day", columns="hour", values=col)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="viridis", linewidths=0.05)
    plt.title(f"Heatmap of {col} by Day/Hour")
    plt.xlabel("Hour")
    plt.ylabel("Day")
    plt.show()

def plot_threshold_breaches(df: pd.DataFrame, sensor_specs: dict):
    """
    Plot bar chart showing threshold breaches for each sensor.
    """
    breach_counts = {}
    for sensor, spec in sensor_specs.items():
        if sensor in df.columns:
            count = ((df[sensor] < spec.min_ok) | (df[sensor] > spec.max_ok)).sum()
            breach_counts[sensor] = count

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(breach_counts.keys()), y=list(breach_counts.values()))
    plt.title("Sensor Threshold Breaches")
    plt.ylabel("Breach Count")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
