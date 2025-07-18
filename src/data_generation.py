import numpy as np
import pandas as pd
import json
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List


# ──────────────────────────────────────────────────────
# Configuration Loader
# ──────────────────────────────────────────────────────

def load_config(config_path: str = "./config/config.json") -> Dict:
    """
    Load the simulation configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────
# Sensor Specification Data Structure
# ──────────────────────────────────────────────────────

@dataclass
class SensorSpec:
    """
    A dataclass that stores the characteristics of a sensor.

    Attributes:
        mean (float): Mean value for generating normal distribution.
        std (float): Standard deviation for noise in sensor.
        min_ok (float): Minimum acceptable value.
        max_ok (float): Maximum acceptable value.
        units (str): Measurement unit.
    """
    mean: float
    std: float
    min_ok: float
    max_ok: float
    units: str


# ──────────────────────────────────────────────────────
# Anomaly Injection Utilities
# ──────────────────────────────────────────────────────

def _inject_gradual(series: pd.Series, start: int, length: int, slope: float):
    """
    Injects a gradual drift anomaly into the given time series.

    Args:
        series (pd.Series): The sensor time series.
        start (int): Start index for drift.
        length (int): Number of points to inject drift over.
        slope (float): Amount of change per time step.
    """
    drift = np.linspace(0, slope * length, length)
    series.loc[series.index[start:start + length]] += drift


def _inject_spike(series: pd.Series, idx: int, magnitude: float, width: int = 5):
    """
    Injects a sudden spike anomaly into the given time series.

    Args:
        series (pd.Series): The sensor time series.
        idx (int): Start index of spike.
        magnitude (float): Amount of spike.
        width (int): Width of spike in number of data points.
    """
    series.loc[series.index[idx:idx + width]] += magnitude


# ──────────────────────────────────────────────────────
# Operator Log Template and Generator
# ──────────────────────────────────────────────────────

_LOG_TEMPLATES = [
    "noticed unusual readings on {equip}",
    "manual inspection initiated for {equip}",
    "investigating equipment condition on {equip}",
    "scheduled maintenance triggered for {equip}",
    "alert received from monitoring system for {equip}",
    "operator flagged concern in {equip}"
]

_OPERATORS = ["Anita R", "Ravi K", "Suresh M", "Leela D", "Kumar V"]
_SHIFTS = ["A", "B", "C"]

def _make_log(ts, equip, sensor_row: pd.Series = None):
    """
    Create a synthetic log message simulating operator feedback.

    Args:
        ts (datetime): Timestamp of the event.
        equip (str): Equipment name.
        sensor_row (pd.Series): Row of sensor data to log.

    Returns:
        Dict: Dictionary representing a log entry.
    """
    log_entry = {
        "log_time": ts,
        "equipment_id": equip,
        "operator_name": random.choice(_OPERATORS),
        "shift": random.choice(_SHIFTS),
        "text": random.choice(_LOG_TEMPLATES).format(equip=equip)
    }

    if sensor_row is not None:
        for col in sensor_row.index:
            if col not in ["timestamp", "equipment_id"]:
                log_entry[col] = sensor_row[col]

    return log_entry


# ──────────────────────────────────────────────────────
# Base Simulator Class (for common logic)
# ──────────────────────────────────────────────────────

class BaseEquipmentSimulator:
    name: str
    sensors: Dict[str, SensorSpec]

    def __init__(self, config: Dict):
        self.start = pd.to_datetime(config["global"]["start_date"])
        self.end = pd.to_datetime(config["global"]["end_date"])
        self.freq = config["global"]["frequency"].replace("T", "min")
        self.missing_frac = config["global"]["missing_fraction"]
        self.rng = np.random.default_rng(config["global"]["seed"])
        self.logs: List[Dict] = []

    def generate(self) -> pd.DataFrame:
        idx = pd.date_range(self.start, self.end, freq=self.freq, inclusive="left")
        data = {
            "timestamp": idx,
            "equipment_id": [self.name] * len(idx)
        }
        for chan, spec in self.sensors.items():
            data[chan] = self.rng.normal(spec.mean, spec.std, len(idx))

        df = pd.DataFrame(data)
        self._make_missing(df)
        self.inject_anomalies(df)
        return df

    def _make_missing(self, df: pd.DataFrame):
        n_rows = len(df)
        n_missing = int(n_rows * len(self.sensors) * self.missing_frac)
        idx_rows = self.rng.integers(0, n_rows, n_missing)
        idx_cols = self.rng.choice(list(self.sensors.keys()), n_missing)
        for r, c in zip(idx_rows, idx_cols):
            df.at[r, c] = np.nan

    def save_csv(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        df = self.generate()
        csv_path = outdir / f"{self.name.lower().replace(' ', '_')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[✓] {self.name:20} → {len(df):,} rows")

    def inject_anomalies(self, df: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement this method.")


# ─────────── Equipment Simulators ─────────── #

class MudPumpSimulator(BaseEquipmentSimulator):
    name = "Mud Pump"
    sensors = {
        "pressure_psi": SensorSpec(120, 4, 110, 130, "psi"),
        "flow_rate_lpm": SensorSpec(90, 3, 80, 100, "L/min"),
        "pump_temp_C": SensorSpec(58, 2, 50, 65, "C")
    }

    def inject_anomalies(self, df):
        idx = df.index
        idx1 = self.rng.integers(1000, len(idx) - 1000)
        _inject_spike(df["pressure_psi"], idx1, -35)
        self.logs.append(_make_log(df.at[idx1, "timestamp"], self.name, df.loc[idx1]))

        idx2 = self.rng.integers(2000, len(idx) - 3000)
        _inject_gradual(df["pump_temp_C"], idx2, 1500, 0.02)
        self.logs.append(_make_log(df.at[idx2, "timestamp"], self.name, df.loc[idx2]))

        idx3 = self.rng.integers(5000, len(idx) - 500)
        _inject_spike(df["flow_rate_lpm"], idx3, -35, 20)
        self.logs.append(_make_log(df.at[idx3, "timestamp"], self.name, df.loc[idx3]))


class CentrifugalPumpSimulator(BaseEquipmentSimulator):
    name = "Centrifugal Pump"
    sensors = {
        "vibration_mmps": SensorSpec(3.0, 0.4, 0, 5, "mm/s RMS"),
        "bearing_temp_C": SensorSpec(52, 2, 45, 60, "C"),
        "motor_current_A": SensorSpec(60, 4, 50, 70, "A")
    }

    def inject_anomalies(self, df):
        idx1 = self.rng.integers(2000, len(df) - 2000)
        _inject_spike(df["vibration_mmps"], idx1, 6)
        self.logs.append(_make_log(df.at[idx1, "timestamp"], self.name, df.loc[idx1]))

        idx2 = self.rng.integers(3000, len(df) - 5000)
        _inject_gradual(df["bearing_temp_C"], idx2, 2000, 0.02)
        self.logs.append(_make_log(df.at[idx2, "timestamp"], self.name, df.loc[idx2]))

        idx3 = self.rng.integers(6000, len(df) - 600)
        _inject_spike(df["motor_current_A"], idx3, 25, 10)
        self.logs.append(_make_log(df.at[idx3, "timestamp"], self.name, df.loc[idx3]))


class BOPSimulator(BaseEquipmentSimulator):
    name = "Blowout Preventer"
    sensors = {
        "hydraulic_pressure_psi": SensorSpec(2500, 60, 1500, 3000, "psi"),
        "accumulator_pressure_psi": SensorSpec(3000, 30, 2900, 3100, "psi"),
    }

    def inject_anomalies(self, df):
        idx = self.rng.integers(8000, len(df) - 5000)
        _inject_gradual(df["hydraulic_pressure_psi"], idx, 3000, -0.4 / 60)
        self.logs.append(_make_log(df.at[idx, "timestamp"], self.name, df.loc[idx]))


class ShaleShakerSimulator(BaseEquipmentSimulator):
    name = "Shale Shaker"
    sensors = {
        "vibration_rpm": SensorSpec(2000, 40, 1800, 2200, "RPM"),
        "motor_load_pct": SensorSpec(68, 4, 60, 75, "%"),
        "throughput_lpm": SensorSpec(350, 10, 300, 400, "L/min")
    }

    def inject_anomalies(self, df):
        idx = self.rng.integers(3000, len(df) - 3000)
        _inject_spike(df["vibration_rpm"], idx, 600)
        _inject_spike(df["motor_load_pct"], idx, 20)
        self.logs.append(_make_log(df.at[idx, "timestamp"], self.name, df.loc[idx]))

        idx2 = self.rng.integers(9000, len(df) - 900)
        _inject_spike(df["throughput_lpm"], idx2, -150, 30)
        self.logs.append(_make_log(df.at[idx2, "timestamp"], self.name, df.loc[idx2]))


class DieselGeneratorSimulator(BaseEquipmentSimulator):
    name = "Diesel Generator"
    sensors = {
        "load_pct": SensorSpec(70, 5, 60, 80, "%"),
        "engine_temp_C": SensorSpec(85, 3, 75, 95, "C"),
        "fuel_level_pct": SensorSpec(90, 0.02, 0, 100, "%"),
        "oil_pressure_psi": SensorSpec(42, 1.5, 35, 50, "psi")
    }

    def inject_anomalies(self, df):
        n = len(df)
        df["fuel_level_pct"] = df["fuel_level_pct"].iloc[0] - np.arange(n) * 0.015

        idx1 = self.rng.integers(15000, n - 1500)
        _inject_spike(df["fuel_level_pct"], idx1, -25, 60)
        self.logs.append(_make_log(df.at[idx1, "timestamp"], self.name, df.loc[idx1]))

        idx2 = self.rng.integers(20000, n - 600)
        _inject_spike(df["load_pct"], idx2, 25, 30)
        _inject_spike(df["engine_temp_C"], idx2, 20, 30)
        self.logs.append(_make_log(df.at[idx2, "timestamp"], self.name, df.loc[idx2]))

        idx3 = self.rng.integers(25000, n - 250)
        _inject_spike(df["oil_pressure_psi"], idx3, -20, 20)
        self.logs.append(_make_log(df.at[idx3, "timestamp"], self.name, df.loc[idx3]))


# ─────────── Main ─────────── #
def main():
    config = load_config()
    out_dir = Path(config["global"]["output_dir"])
    operator_logs_dir = Path(config["global"]["operator_logs_dir"])
    simulators = [
        MudPumpSimulator(config),
        CentrifugalPumpSimulator(config),
        BOPSimulator(config),
        ShaleShakerSimulator(config),
        DieselGeneratorSimulator(config),
    ]

    all_logs = []
    for sim in simulators:
        sim.save_csv(out_dir)
        all_logs.extend(sim.logs)

    # Save structured log CSV
    log_df = pd.DataFrame(all_logs)
    log_df.to_csv(operator_logs_dir / "synthetic_operator_logs.csv", index=False)

    # Save human-readable log text
    with open(operator_logs_dir / "operator_logs_text.txt", "w", encoding="utf-8") as f:
        for _, row in log_df.iterrows():
            sensors_info = ", ".join(f"{k}: {v:.2f}" for k, v in row.items()
                                     if k not in ["log_time", "equipment_id", "shift", "operator_name", "text"]
                                     and pd.notnull(v))
            f.write(f"{row['log_time']} | Equipment: {row['equipment_id']} | Shift: {row['shift']} | "
                    f"Operator: {row['operator_name']} | {row['text']} | Sensors: {sensors_info}\n")

if __name__ == "__main__":
    main()
