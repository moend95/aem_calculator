import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# ============================================================================
# STREAMLIT APP: AEM Electrolyzer H2 Production Calculator
# ============================================================================

st.set_page_config(page_title="AEM H2 Calculator", layout="wide")
st.title("🔬 AEM Electrolyzer H2 Production Calculator")
st.markdown("---")

# ============================================================================
# 1. LOAD MEASUREMENT DATA (Reference data from efficiency_curve_0-100)
# ============================================================================

@st.cache_data
def load_reference_data():
    """Loads the reference measurement data of the electrolyzer"""
    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple possible paths (local vs. cloud deployment)
        possible_paths = [
            # Local development path
            os.path.join(os.path.dirname(script_dir), "aem_enapter_tests", "electrolyser_el21_test4.csv"),
            # Same directory (if CSV is uploaded next to the script)
            os.path.join(script_dir, "electrolyser_el21_test4.csv"),
            # Streamlit Cloud path (if in root)
            os.path.join(script_dir, "..", "aem_enapter_tests", "electrolyser_el21_test4.csv"),
            # Current directory
            "electrolyser_el21_test4.csv",
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError(f"Could not find electrolyser_el21_test4.csv in any of the expected locations: {possible_paths}")
        
        df_raw = pd.read_csv(csv_path, sep=',', decimal='.')
        
        # Spaltennamen ändern
        df_raw = df_raw.rename(columns={
            'PSU_out_power': 'Power True',
            'HASS_in_a': 'Current [A]',
            'PSU_in_v': 'Voltage [V]',
            'h2_flow': 'h2 flow [NL/h]',
            'production_rate': 'production rate [%]',
            'stack': 'stack state'
        })
        
        df = df_raw[[
            'Current [A]',
            'Voltage [V]',
            'h2 flow [NL/h]',
        ]].copy()
        
        df['Power [W]'] = df['Current [A]'] * df['Voltage [V]']
        
        # Filter: h2 flow != 0
        df = df[df['h2 flow [NL/h]'] != 0.0]
        
        # Sortieren nach Power
        df = df.sort_values(by='Power [W]', ascending=True).reset_index(drop=True)
        
        # H2 in kg/h umrechnen
        df['h2 flow [kg/h]'] = df['h2 flow [NL/h]'] / 12120
        
        # Effizienz berechnen
        df['efficiency lhv'] = (df['h2 flow [kg/h]'] * 33.33) / (df['Power [W]'] / 1000)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading reference data: {e}")
        return None

# Load reference data (with Session State for single loading)
if 'df_reference' not in st.session_state:
    st.session_state.df_reference = load_reference_data()

df_reference = st.session_state.df_reference

if df_reference is not None:
    st.sidebar.success(f"✅ Reference data loaded: {len(df_reference)} data points")
    
    # Show statistics
    with st.sidebar.expander("📊 Reference Data Info"):
        st.write(f"**Power Range:** {df_reference['Power [W]'].min():.0f} - {df_reference['Power [W]'].max():.0f} W")
        st.write(f"**Current Range:** {df_reference['Current [A]'].min():.1f} - {df_reference['Current [A]'].max():.1f} A")
        st.write(f"**Voltage Range:** {df_reference['Voltage [V]'].min():.1f} - {df_reference['Voltage [V]'].max():.1f} V")
        st.write(f"**H2 Flow Range:** {df_reference['h2 flow [kg/h]'].min():.4f} - {df_reference['h2 flow [kg/h]'].max():.4f} kg/h")
else:
    st.error("❌ Reference data could not be loaded!")
    st.stop()

# ============================================================================
# 2. POWER LOAD CURVE: upload or use test data
# ============================================================================

st.header("📂 1. Power Load Curve")

data_choice = st.radio(
    "Choose input source:",
    ("Use test data (pv_load_curve_3days.csv)", "Upload my CSV file")
)


@st.cache_data
def load_test_load_curve():
    """Load the bundled test load curve CSV if available."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "pv_load_curve_3days.csv"),
            os.path.join(os.path.dirname(script_dir), "pv_load_curve_3days.csv"),
            "pv_load_curve_3days.csv",
        ]
        csv_path = None
        for p in possible_paths:
            if os.path.exists(p):
                csv_path = p
                break
        if csv_path is None:
            raise FileNotFoundError("pv_load_curve_3days.csv not found in expected locations.")
        df = pd.read_csv(csv_path, sep=',', decimal='.')
        return df
    except Exception as e:
        st.error(f"Error loading test load curve: {e}")
        return None


df_load = None
power_col = None

if data_choice.startswith("Use test"):
    df_load = load_test_load_curve()
    if df_load is not None:
        st.success(f"✅ Test data loaded: {len(df_load)} time steps")
        with st.expander("🔍 Data Preview (test data)"):
            st.dataframe(df_load.head(10))
    else:
        st.error("❌ Test data could not be loaded. You can try uploading a file.")

else:
    uploaded_file = st.file_uploader(
        "CSV file with power load curve (columns: timestamp, power_W or Power [W])",
        type=['csv']
    )
    if uploaded_file is not None:
        df_load = pd.read_csv(uploaded_file, sep=',', decimal='.')
        st.success(f"✅ File loaded: {len(df_load)} time steps")
        with st.expander("🔍 Data Preview"):
            st.dataframe(df_load.head(10))

# If we have a dataframe, determine power column and continue
if df_load is not None:
    if 'power_W' in df_load.columns:
        power_col = 'power_W'
    elif 'Power [W]' in df_load.columns:
        power_col = 'Power [W]'
    else:
        st.error("❌ Column 'power_W' or 'Power [W]' not found in the selected data!")
        st.stop()

    # ========================================================================
    # 3. CALCULATION: Interpolation
    # ========================================================================
    
    st.header("⚙️ 2. H2 Production Calculation")
    
    # Maximum power limit
    MAX_POWER_W = 1400
    
    # ========================================================================
    # 3. CALCULATION: Interpolation
    # ========================================================================
    
    st.header("⚙️ 2. H2 Production Calculation")
    
    # Maximum power limit
    MAX_POWER_W = 1400
    
    # Interpolationsfunktionen erstellen
    interp_current = interp1d(
        df_reference['Power [W]'], 
        df_reference['Current [A]'],
        kind='linear',
        fill_value='extrapolate'
    )
    
    interp_voltage = interp1d(
        df_reference['Power [W]'], 
        df_reference['Voltage [V]'],
        kind='linear',
        fill_value='extrapolate'
    )
    
    interp_h2_flow_kg = interp1d(
        df_reference['Power [W]'], 
        df_reference['h2 flow [kg/h]'],
        kind='linear',
        fill_value='extrapolate'
    )
    
    interp_efficiency = interp1d(
        df_reference['Power [W]'], 
        df_reference['efficiency lhv'],
        kind='linear',
        fill_value='extrapolate'
    )
    
    # Calculate for each time step
    df_result = df_load.copy()
    
    # Calculate excess energy
    df_result['Excess Energy [W]'] = df_result[power_col].apply(
        lambda x: max(0, x - MAX_POWER_W)
    )
    
    # Usable power (maximum 1400 W)
    df_result['Usable Power [W]'] = df_result[power_col].apply(
        lambda x: min(x, MAX_POWER_W)
    )
    
    # Interpolation - BUT: if 0 W → set everything to 0
    df_result['Current [A]'] = df_result['Usable Power [W]'].apply(
        lambda x: 0.0 if x == 0 else interp_current(x)
    )
    df_result['Voltage [V]'] = df_result['Usable Power [W]'].apply(
        lambda x: 0.0 if x == 0 else interp_voltage(x)
    )
    df_result['h2 flow [kg/h]'] = df_result['Usable Power [W]'].apply(
        lambda x: 0.0 if x == 0 else interp_h2_flow_kg(x)
    )
    df_result['h2 flow [NL/h]'] = df_result['h2 flow [kg/h]'] * 12120
    df_result['efficiency lhv'] = df_result['Usable Power [W]'].apply(
        lambda x: 0.0 if x == 0 else interp_efficiency(x)
    )
    
    st.success("✅ Calculation completed!")
    
    # ========================================================================
    # 4. BALANCE CALCULATION
    # ========================================================================
    
    # Time interval in hours (15 min = 0.25 h)
    time_interval_h = 0.25
    
    # H2 production: NL/h × 0.25 h = NL per time step
    total_h2_NL = (df_result['h2 flow [NL/h]'] * time_interval_h).sum()
    total_h2_kg = total_h2_NL / 12120
    
    # Used electricity: W × 0.25 h / 1000 = kWh
    total_energy_used_kWh = (df_result['Usable Power [W]'] * time_interval_h / 1000).sum()
    
    # Excess energy: W × 0.25 h / 1000 = kWh
    total_energy_excess_kWh = (df_result['Excess Energy [W]'] * time_interval_h / 1000).sum()
    
    # Total available energy
    total_energy_available_kWh = (df_result[power_col] * time_interval_h / 1000).sum()
    
    # ========================================================================
    # 5. DISPLAY RESULTS
    # ========================================================================
    
    st.header("📊 3. Results & Balance")
    
    # Balance box
    st.subheader("🔋 Energy Balance for Observation Period")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Produced H2", f"{total_h2_NL:.1f} NL", help=f"≈ {total_h2_kg:.3f} kg")
    
    with col2:
        st.metric("Used Electricity", f"{total_energy_used_kWh:.2f} kWh")
    
    with col3:
        st.metric("Excess Energy", f"{total_energy_excess_kWh:.2f} kWh")
    
    with col4:
        usage_percent = (total_energy_used_kWh / total_energy_available_kWh * 100) if total_energy_available_kWh > 0 else 0
        st.metric("Utilization Rate", f"{usage_percent:.1f} %")
    
    # Average values
    st.subheader("📈 Average Values")
    col1, col2 = st.columns(2)
    
    with col1:
        # Only consider time steps with production
        active_steps = df_result[df_result['h2 flow [NL/h]'] > 0]
        if len(active_steps) > 0:
            avg_efficiency = active_steps['efficiency lhv'].mean() * 100
            st.metric("Avg. Efficiency (active)", f"{avg_efficiency:.1f} %")
        else:
            st.metric("Avg. Efficiency (active)", "N/A")
    
    with col2:
        avg_power = df_result['Usable Power [W]'].mean()
        st.metric("Avg. Power", f"{avg_power:.0f} W")
    
    # Table
    st.subheader("📋 Detailed Results Table")
    
    # Column selection for better overview
    display_columns = [
        power_col,
        'Usable Power [W]',
        'Excess Energy [W]',
        'Current [A]',
        'Voltage [V]',
        'h2 flow [NL/h]',
        'h2 flow [kg/h]',
        'efficiency lhv'
    ]
    
    # Only show columns that exist
    display_columns = [col for col in display_columns if col in df_result.columns]
    
    st.dataframe(df_result[display_columns], use_container_width=True)
    
    # Download
    csv = df_result.to_csv(index=False, sep=',', decimal='.')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="h2_production_results.csv",
        mime="text/csv"
    )
    
    # ========================================================================
    # 6. VISUALIZATIONS
    # ========================================================================
    
    st.header("📈 4. Visualizations")
    
    # Prepare X-axis
    if 'timestamp' in df_result.columns:
        x_data = pd.to_datetime(df_result['timestamp'])
        xlabel = 'Time'
    else:
        x_data = df_result.index
        xlabel = 'Time Step'
    
    # Plot 1: Power and excess
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.fill_between(x_data, 0, df_result['Usable Power [W]'], 
                      color='green', alpha=0.3, label='Used Power')
    ax1.fill_between(x_data, df_result['Usable Power [W]'], 
                      df_result[power_col], 
                      color='red', alpha=0.3, label='Excess Energy')
    ax1.plot(x_data, df_result[power_col], linewidth=1.5, color='orange', label='Available Power')
    ax1.axhline(y=MAX_POWER_W, color='red', linestyle='--', linewidth=2, label=f'Max. Limit ({MAX_POWER_W} W)')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('Power [W]', fontsize=12)
    ax1.set_title('Power Distribution: Used vs. Excess', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)
    
    # Plot 2: H2 production over time
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(x_data, df_result['h2 flow [NL/h]'], linewidth=2, color='blue')
    ax2.fill_between(x_data, 0, df_result['h2 flow [NL/h]'], color='blue', alpha=0.2)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('H2 Flow [NL/h]', fontsize=12)
    ax2.set_title('H2 Production over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)
    
    # Plot 3: Efficiency over time
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ax3.plot(x_data, df_result['efficiency lhv'] * 100, linewidth=2, color='green')
    ax3.set_ylabel('Efficiency LHV [%]', fontsize=12)
    ax3.set_xlabel(xlabel)
    ax3.set_title('Efficiency over Time', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)

else:
    st.info("👆 Please upload a CSV file with the power load curve to begin.")

# ============================================================================
# SIDEBAR: Info
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Info")
st.sidebar.markdown("""
**Version:** 1.0  
**Author:** Moritz  
**Date:** 2025-11-19

This tool calculates the H2 production of an AEM electrolyzer 
based on real measurement data and an input power load curve.
""")
