import streamlit as st
import pandas as pd
import os
import glob

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Oil Rig Monitoring Dashboard",
    page_icon=":oil_drum:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS Styling ----------------
st.markdown("""
    <style>
        #MainMenu, footer {visibility: hidden;}
        h1, h2 {
            font-family: 'Segoe UI', sans-serif;
        }
        @media (max-width: 768px) {
            h1 { font-size: 20px !important; }
            h2 { font-size: 16px !important; }
        }
        .sidebar .sidebar-content {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)


# ---------- Custom CSS: Sticky Header + No Scroll + No Top Padding ----------
st.markdown(
    """
    <style>
    /* Remove top margin/padding from Streamlit default layout */
    .block-container {
        padding-top: 0rem !important;
    }

    /* Sticky Header Styling */
    .sticky-header {
        position: sticky;
        top: 0;
        background-color: white;
        padding: 16px 0;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        z-index: 999;
        border-bottom: 1px solid #ccc;
        margin: 0;
    }

    /* Remove vertical scrolling */
    html, body, [class*="main"] {
        overflow-y: hidden !important;
        height: 100vh !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Sticky Header ----------
st.markdown("<div class='sticky-header'>🛢 Real-Time Oil Rig Monitoring & Anomaly Insights</div>", unsafe_allow_html=True)

# ---------------- Utility Functions ----------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def normalize_name(path):
    return os.path.basename(path).replace(".csv", "").replace("_", " ").title()

def clean_dataframe(df):
    df.columns = df.columns.str.strip()
    return df.dropna(how="all")

# ---------------- File Directory Setup ----------------
EQUIPMENT_DIR = "data/synthetic_oilrig_data"
ANOMALY_DIR = "anomaly_outputs"
CORRELATION_DIR = "correlation_outputs"
SUMMARY_PATH = "summaries/equipment_anomaly_summary.csv"

equipment_map = {normalize_name(f): f for f in glob.glob(os.path.join(EQUIPMENT_DIR, "*.csv"))}
anomaly_map = {normalize_name(f): f for f in glob.glob(os.path.join(ANOMALY_DIR, "*.csv"))}
correlation_map = {normalize_name(f): f for f in glob.glob(os.path.join(CORRELATION_DIR, "*.csv"))}

# ---------------- Sidebar Navigation ----------------
view = st.sidebar.radio(
    "Select View",
    ["Equipment Logs", "Anomaly Detection", "Correlations", "Anomaly Summary"]
)

# ---------------- Equipment Logs View ----------------
if view == "Equipment Logs":
    st.markdown("<h2 style='font-size:18px; color:black;'>Equipment Logs</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("### Log Filters")

    if equipment_map:
        eq_choice = st.sidebar.selectbox("Select Equipment File", list(equipment_map.keys()))
        eq_df = clean_dataframe(load_csv(equipment_map[eq_choice]))

        # Time filter
        datetime_cols = [col for col in eq_df.columns if "time" in col.lower() or "date" in col.lower()]
        if datetime_cols:
            time_col = datetime_cols[0]
            eq_df[time_col] = pd.to_datetime(eq_df[time_col], errors="coerce")
            eq_df = eq_df.dropna(subset=[time_col])
            min_date, max_date = eq_df[time_col].min(), eq_df[time_col].max()
            date_range = st.sidebar.date_input("Select Date Range", (min_date.date(), max_date.date()))
            eq_df = eq_df[(eq_df[time_col].dt.date >= date_range[0]) & (eq_df[time_col].dt.date <= date_range[1])]

        st.sidebar.markdown("### Numeric Filters")
        for col in eq_df.select_dtypes(include="number").columns:
            min_val, max_val = float(eq_df[col].min()), float(eq_df[col].max())
            step = (max_val - min_val) / 100 if max_val != min_val else 1.0
            selected_range = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val), step=step)
            eq_df = eq_df[(eq_df[col] >= selected_range[0]) & (eq_df[col] <= selected_range[1])]

        st.dataframe(eq_df, use_container_width=True)
    else:
        st.warning("No equipment log files found.")

# ---------------- Anomaly Detection View ----------------
elif view == "Anomaly Detection":
    st.markdown("<h2 style='font-size:18px; color:black;'>Anomaly Detection Results</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("### Anomaly Filters")

    if anomaly_map:
        anom_choice = st.sidebar.selectbox("Select Anomaly File", list(anomaly_map.keys()))
        anom_df = clean_dataframe(load_csv(anomaly_map[anom_choice]))

        equip_col = next((col for col in anom_df.columns if "equipment" in col.lower()), None)
        if equip_col:
            selected_equipment = st.sidebar.selectbox("Select Equipment", sorted(anom_df[equip_col].dropna().unique()))
            anom_df = anom_df[anom_df[equip_col] == selected_equipment]

        date_col = next((col for col in anom_df.columns if "time" in col.lower() or "date" in col.lower()), None)
        if date_col:
            anom_df[date_col] = pd.to_datetime(anom_df[date_col], errors="coerce")
            anom_df = anom_df.dropna(subset=[date_col])
            min_date, max_date = anom_df[date_col].min(), anom_df[date_col].max()
            date_range = st.sidebar.date_input("Select Date Range", (min_date.date(), max_date.date()))
            anom_df = anom_df[(anom_df[date_col].dt.date >= date_range[0]) & (anom_df[date_col].dt.date <= date_range[1])]

        st.sidebar.markdown("### Numeric Filters")
        for col in anom_df.select_dtypes(include="number").columns:
            min_val, max_val = float(anom_df[col].min()), float(anom_df[col].max())
            step = (max_val - min_val) / 100 if max_val != min_val else 1.0
            selected_range = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val), step=step)
            anom_df = anom_df[(anom_df[col] >= selected_range[0]) & (anom_df[col] <= selected_range[1])]

        st.dataframe(anom_df, use_container_width=True)
    else:
        st.warning("No anomaly detection files found.")

# ---------------- Correlation View ----------------
elif view == "Correlations":
    st.markdown("<h2 style='font-size:18px; color:black;'>Correlation Results</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("### Correlation Filters")

    if correlation_map:
        corr_df = pd.concat([clean_dataframe(load_csv(f)) for f in correlation_map.values()], ignore_index=True)

        equip_col = next((col for col in corr_df.columns if "equipment" in col.lower()), None)
        if equip_col:
            selected_equipment = st.sidebar.selectbox("Select Equipment", sorted(corr_df[equip_col].dropna().unique()))
            corr_df = corr_df[corr_df[equip_col] == selected_equipment]

        st.dataframe(corr_df, use_container_width=True)
    else:
        st.warning("No correlation results found.")

# ---------------- Anomaly Summary View ----------------
elif view == "Anomaly Summary":
    st.markdown("<h2 style='font-size:18px; color:black;'>Equipment Anomaly Summary</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("### Summary Filters")

    try:
        summary_df = clean_dataframe(load_csv(SUMMARY_PATH))
        equipment_types = summary_df["Equipment"].dropna().unique()
        selected_equipment = st.sidebar.selectbox("Select Equipment", ["All"] + list(equipment_types))

        if selected_equipment != "All":
            summary_df = summary_df[summary_df["Equipment"] == selected_equipment]

        for _, row in summary_df.iterrows():
            st.markdown("---")
            st.markdown(f"**Equipment:** {row['Equipment']}")
            st.markdown(f"- **Total Anomalies Observed:** {int(row['Total Anomalies Observed'])}")
            st.markdown(f"- **Most Recent Anomaly Time:** {row['Most Recent Anomaly Time']}")
            st.markdown(f"- **Recent Anomaly Description:** {row['Recent Anomaly Description']}")
            st.markdown(f"- **Diagnosis & Likely Root Cause:** {row['Diagnosis & Likely Root Cause']}")
            st.markdown(f"- **Relevant Log:** {row['Relevant Log']}")
            st.markdown(f"- **Actionable Review:** {row['Actionable Review']}")
    except Exception as e:
        st.error(f"Error loading anomaly summary: {e}")
