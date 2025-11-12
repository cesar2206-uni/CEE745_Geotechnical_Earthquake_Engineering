import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

#===================================================================================================
# Functions
#===================================================================================================

#---------------------------------------------------------------------------------------------------
# Function 1: ASCE 7-10 Design Response Spectrum
#---------------------------------------------------------------------------------------------------
def asce7_10_design_spectrum(S_s, S_1, F_a, F_v, T_L, max_period=7):
    """
    Calculates the design acceleration spectra based on ASCE 7-10.

    Parameters:
    - S_s: Spectral acceleration at short periods
    - S_1: Spectral acceleration at 1-second period
    - F_a: Site coefficient at short periods
    - F_v: Site coefficient at 1-second period
    - T_L: Long-period transition period
    - max_period: Maximum period for the plot (default 7s)

    Returns:
    - periods: Array of period values
    - S_a: Array of spectral acceleration values
    - params: Dictionary containing calculated parameters
    """
    # Calculate intermediate values
    S_MS = F_a * S_s
    S_M1 = F_v * S_1
    S_DS = (2/3) * S_MS
    S_D1 = (2/3) * S_M1
    T_0 = 0.2 * S_D1 / S_DS
    T_S = S_D1 / S_DS

    # Generate period array with step 0.05 and include key points
    periods = np.arange(0, max_period + 0.05, 0.05)
    key_points = [0.025, T_0, T_S, T_L]
    # Add key points within the range and sort
    for point in key_points:
        if 0 <= point <= max_period and point not in periods:
            periods = np.append(periods, point)
    periods = np.sort(periods)
    
    # Initialize spectral accelerations
    S_a = np.zeros_like(periods)
    
    # Calculate S_a for each period
    for i, T in enumerate(periods):
        if T < T_0:
            S_a[i] = S_DS * (0.4 + 0.6 * (T / T_0))
        elif T <= T_S:
            S_a[i] = S_DS
        elif T <= T_L:
            S_a[i] = S_D1 / T
        else:
            S_a[i] = (S_D1 * T_L) / (T ** 2)
    
    params = {
        'S_MS': S_MS, 'S_M1': S_M1, 'S_DS': S_DS, 'S_D1': S_D1,
        'T_0': T_0, 'T_S': T_S, 'T_L': T_L
    }
    
    return periods, S_a, params

#---------------------------------------------------------------------------------------------------
# Function 2: ASCE 7-10 MCE Response Spectrum
#---------------------------------------------------------------------------------------------------
def asce7_10_mce_spectrum(S_s, S_1, F_a, F_v, T_L, max_period=7):
    """
    Calculates the MCE acceleration spectra based on ASCE 7-10 (Design spectrum × 1.5).

    Parameters:
    - S_s: Spectral acceleration at short periods
    - S_1: Spectral acceleration at 1-second period
    - F_a: Site coefficient at short periods
    - F_v: Site coefficient at 1-second period
    - T_L: Long-period transition period
    - max_period: Maximum period for the plot (default 7s)

    Returns:
    - periods: Array of period values
    - S_a_mce: Array of MCE spectral acceleration values
    - params: Dictionary containing calculated parameters
    """
    periods, S_a_design, params = asce7_10_design_spectrum(S_s, S_1, F_a, F_v, T_L, max_period)
    S_a_mce = 1.5 * S_a_design
    
    return periods, S_a_mce, params

#---------------------------------------------------------------------------------------------------
# Function 3: ASCE 7-22 Design Response Spectrum
#---------------------------------------------------------------------------------------------------
def asce7_22_design_spectrum(S_DS, S_D1, T_L, max_period=7):
    """
    Calculates the design acceleration spectra based on ASCE 7-22.

    Parameters:
    - S_DS: Design spectral response acceleration parameter at short periods
    - S_D1: Design spectral response acceleration parameter at 1-s period
    - T_L: Long-period transition period
    - max_period: Maximum period for the plot (default 7s)

    Returns:
    - periods: Array of period values
    - S_a: Array of spectral acceleration values
    - params: Dictionary containing calculated parameters
    """
    # Calculate S_MS and S_M1 according to ASCE 7-22
    S_MS = 1.5 * S_DS
    S_M1 = 1.5 * S_D1
    
    # Calculate T_0 and T_S
    T_0 = 0.2 * S_D1 / S_DS
    T_S = S_D1 / S_DS
    
    # Generate period array with step 0.05 and include key points
    periods = np.arange(0, max_period + 0.05, 0.05)
    key_points = [0.025, T_0, T_S, T_L]
    
    # Add key points within the range and sort
    for point in key_points:
        if 0 <= point <= max_period and point not in periods:
            periods = np.append(periods, point)
    periods = np.sort(periods)
    
    # Initialize spectral accelerations
    S_a = np.zeros_like(periods)
    
    # Calculate S_a for each period according to ASCE 7-22
    for i, T in enumerate(periods):
        if T < T_0:
            S_a[i] = S_DS * (0.4 + 0.6 * (T / T_0))
        elif T <= T_S:
            S_a[i] = S_DS
        elif T <= T_L:
            S_a[i] = S_D1 / T
        else:
            S_a[i] = (S_D1 * T_L) / (T ** 2)
    
    params = {
        'S_MS': S_MS, 'S_M1': S_M1, 'S_DS': S_DS, 'S_D1': S_D1,
        'T_0': T_0, 'T_S': T_S, 'T_L': T_L
    }
    
    return periods, S_a, params

#---------------------------------------------------------------------------------------------------
# Function 4: ASCE 7-22 MCE Response Spectrum
#---------------------------------------------------------------------------------------------------
def asce7_22_mce_spectrum(S_DS, S_D1, T_L, max_period=7):
    """
    Calculates the MCE acceleration spectra based on ASCE 7-22 (Design spectrum × 1.5).

    Parameters:
    - S_DS: Design spectral response acceleration parameter at short periods
    - S_D1: Design spectral response acceleration parameter at 1-s period
    - T_L: Long-period transition period
    - max_period: Maximum period for the plot (default 7s)

    Returns:
    - periods: Array of period values
    - S_a_mce: Array of MCE spectral acceleration values
    - params: Dictionary containing calculated parameters
    """
    periods, S_a_design, params = asce7_22_design_spectrum(S_DS, S_D1, T_L, max_period)
    S_a_mce = 1.5 * S_a_design
    
    return periods, S_a_mce, params

# Function to create SVG download
def get_svg_download_link(fig, filename):
    """Convert matplotlib figure to SVG and create download link"""
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    svg_data = buf.getvalue()
    buf.close()
    
    b64 = base64.b64encode(svg_data).decode()
    href = f'<a href="data:image/svg+xml;base64,{b64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 4px;">Download as SVG</a>'
    return href

#===================================================================================================
# Streamlit code
#===================================================================================================

st.set_page_config(page_title="ASCE 7 Response Spectra Comparison", layout="wide")

st.title("ASCE 7 Response Spectra Comparison Tool")
st.markdown("""
This tool compares design and MCE response spectra based on **ASCE 7-10** and **ASCE 7-22** standards.
""")

import base64

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Common parameters
max_period = st.sidebar.slider("Maximum Period (s)", min_value=5.0, max_value=10.0, value=7.0, step=0.5)
T_L = st.sidebar.slider("Long-period Transition Period, T_L (s)", min_value=4.0, max_value=12.0, value=6.0, step=0.5)

# CSV Upload Section
st.sidebar.header("Upload Multi-Period Spectra")
uploaded_file = st.sidebar.file_uploader("Upload CSV with multi-period spectra", type=['csv'], 
                                         help="Upload a CSV file with columns: 'Period (s)', 'Design Response Spectra (g)', 'MCE Response spectra'")

# Initialize custom spectra data
multi_period_design_periods = None
multi_period_design_sa = None
multi_period_mce_periods = None
multi_period_mce_sa = None

# Process uploaded CSV file
if uploaded_file is not None:
    try:
        # Read the CSV file
        multi_period_df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_columns = ['Period (s)', 'Design Response Spectra (g)', 'MCE Response spectra']
        if all(col in multi_period_df.columns for col in required_columns):
            # Extract data
            multi_period_design_periods = multi_period_df['Period (s)'].values
            multi_period_design_sa = multi_period_df['Design Response Spectra (g)'].values
            multi_period_mce_periods = multi_period_df['Period (s)'].values
            multi_period_mce_sa = multi_period_df['MCE Response spectra'].values
            
            st.sidebar.success("Multi-period spectra loaded successfully!")
            
            # Display preview of uploaded data
            with st.sidebar.expander("Preview uploaded data"):
                st.dataframe(multi_period_df.head(10))
        else:
            st.sidebar.error("CSV file must contain columns: 'Period (s)', 'Design Response Spectra (g)', 'MCE Response spectra'")
            
    except Exception as e:
        st.sidebar.error(f"Error reading CSV file: {str(e)}")

# Create tabs for different code versions
tab1, tab2 = st.tabs(["ASCE 7-10 Parameters", "ASCE 7-22 Parameters"])

with tab1:
    st.header("ASCE 7-10 Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        S_s = st.number_input("S_s - MCE_R Spectral Response at 0.2s (g)", min_value=0.1, max_value=3.0, value=1.501, step=0.1, key="s_s_10")
        S_1 = st.number_input("S_1 - MCE_R Spectral Response at 1.0s (g)", min_value=0.1, max_value=1.5, value=0.548, step=0.1, key="s_1_10")
    
    with col2:
        F_a = st.number_input("F_a - Site Coefficient at 0.2s", min_value=0.8, max_value=2.5, value=1.0, step=0.1, key="f_a_10")
        F_v = st.number_input("F_v - Site Coefficient at 1.0s", min_value=0.8, max_value=3.5, value=1.5, step=0.1, key="f_v_10")

with tab2:
    st.header("ASCE 7-22 Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        S_DS_22 = st.number_input("S_DS - Design Spectral Response at 0.2s (g)", min_value=0.1, max_value=2.0, value=1.15, step=0.1, key="s_ds_22")
    
    with col2:
        S_D1_22 = st.number_input("S_D1 - Design Spectral Response at 1.0s (g)", min_value=0.1, max_value=1.0, value=0.91, step=0.1, key="s_d1_22")

# Calculate spectra
periods_10_design, sa_10_design, params_10 = asce7_10_design_spectrum(S_s, S_1, F_a, F_v, T_L, max_period)
periods_10_mce, sa_10_mce, _ = asce7_10_mce_spectrum(S_s, S_1, F_a, F_v, T_L, max_period)
periods_22_design, sa_22_design, params_22 = asce7_22_design_spectrum(S_DS_22, S_D1_22, T_L, max_period)
periods_22_mce, sa_22_mce, _ = asce7_22_mce_spectrum(S_DS_22, S_D1_22, T_L, max_period)

# Separate plots for Design and MCE spectra
st.header("Design Response Spectra Comparison")

# Create Design Spectra plot
fig_design, ax_design = plt.subplots(figsize=(8, 5))
ax_design.plot(periods_10_design, sa_10_design, 'b-', linewidth=2, label='ASCE 7-10 Design')
ax_design.plot(periods_22_design, sa_22_design, 'r-', linewidth=2, label='ASCE 7-22 Design')

# Add multi-period design spectrum if uploaded
if multi_period_design_periods is not None and multi_period_design_sa is not None:
    ax_design.plot(multi_period_design_periods, multi_period_design_sa, 'g-', linewidth=2, label='ASCE 7-22 Multi-period Design')

ax_design.set_xlabel('Period, T (s)', fontsize=11)
ax_design.set_ylabel('Spectral Acceleration, Sa (g)', fontsize=11)
ax_design.set_title('Design Response Spectra Comparison', fontsize=12, fontweight='bold')
ax_design.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax_design.legend(fontsize=10)
ax_design.set_xlim(0, max_period)
ax_design.set_ylim(bottom=0)

plt.tight_layout()
st.pyplot(fig_design)

# SVG download for Design Spectra
st.markdown(get_svg_download_link(fig_design, "design_spectra_comparison.svg"), unsafe_allow_html=True)

st.header("MCE Response Spectra Comparison")

# Create MCE Spectra plot
fig_mce, ax_mce = plt.subplots(figsize=(8, 5))
ax_mce.plot(periods_10_mce, sa_10_mce, 'b--', linewidth=2, label='ASCE 7-10 MCE')
ax_mce.plot(periods_22_mce, sa_22_mce, 'r--', linewidth=2, label='ASCE 7-22 MCE')

# Add multi-period MCE spectrum if uploaded
if multi_period_mce_periods is not None and multi_period_mce_sa is not None:
    ax_mce.plot(multi_period_mce_periods, multi_period_mce_sa, 'g--', linewidth=2, label='ASCE 7-22 Multi-period MCE')

ax_mce.set_xlabel('Period, T (s)', fontsize=11)
ax_mce.set_ylabel('Spectral Acceleration, Sa (g)', fontsize=11)
ax_mce.set_title('MCE Response Spectra Comparison', fontsize=12, fontweight='bold')
ax_mce.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax_mce.legend(fontsize=10)
ax_mce.set_xlim(0, max_period)
ax_mce.set_ylim(bottom=0)

plt.tight_layout()
st.pyplot(fig_mce)

# SVG download for MCE Spectra
st.markdown(get_svg_download_link(fig_mce, "mce_spectra_comparison.svg"), unsafe_allow_html=True)

# Display calculated parameters
st.header("Calculated Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("ASCE 7-10 Parameters")
    st.write(f"S_MS = {params_10['S_MS']:.3f}g")
    st.write(f"S_M1 = {params_10['S_M1']:.3f}g")
    st.write(f"S_DS = {params_10['S_DS']:.3f}g")
    st.write(f"S_D1 = {params_10['S_D1']:.3f}g")
    st.write(f"T_0 = {params_10['T_0']:.3f}s")
    st.write(f"T_S = {params_10['T_S']:.3f}s")

with col2:
    st.subheader("ASCE 7-22 Parameters")
    st.write(f"S_MS = {params_22['S_MS']:.3f}g")
    st.write(f"S_M1 = {params_22['S_M1']:.3f}g")
    st.write(f"S_DS = {params_22['S_DS']:.3f}g")
    st.write(f"S_D1 = {params_22['S_D1']:.3f}g")
    st.write(f"T_0 = {params_22['T_0']:.3f}s")
    st.write(f"T_S = {params_22['T_S']:.3f}s")

# Data download section
st.header("Download Data")

# Create DataFrames for download
df_10_design = pd.DataFrame({
    'Period_s': periods_10_design,
    'Spectral_Acceleration_g': sa_10_design,
    'Spectrum_Type': 'ASCE_7_10_Design'
})

df_10_mce = pd.DataFrame({
    'Period_s': periods_10_mce,
    'Spectral_Acceleration_g': sa_10_mce,
    'Spectrum_Type': 'ASCE_7_10_MCE'
})

df_22_design = pd.DataFrame({
    'Period_s': periods_22_design,
    'Spectral_Acceleration_g': sa_22_design,
    'Spectrum_Type': 'ASCE_7_22_Design'
})

df_22_mce = pd.DataFrame({
    'Period_s': periods_22_mce,
    'Spectral_Acceleration_g': sa_22_mce,
    'Spectrum_Type': 'ASCE_7_22_MCE'
})

# Add multi-period spectra to download if available
if multi_period_design_periods is not None and multi_period_design_sa is not None:
    df_multi_period_design = pd.DataFrame({
        'Period_s': multi_period_design_periods,
        'Spectral_Acceleration_g': multi_period_design_sa,
        'Spectrum_Type': 'ASCE_7_22_Multi-period_Design'
    })
    df_multi_period_mce = pd.DataFrame({
        'Period_s': multi_period_mce_periods,
        'Spectral_Acceleration_g': multi_period_mce_sa,
        'Spectrum_Type': 'ASCE_7_22_Multi-period_MCE'
    })
    # Combine all data including multi-period spectra
    df_combined = pd.concat([df_10_design, df_10_mce, df_22_design, df_22_mce, df_multi_period_design, df_multi_period_mce], ignore_index=True)
else:
    # Combine all data without multi-period spectra
    df_combined = pd.concat([df_10_design, df_10_mce, df_22_design, df_22_mce], ignore_index=True)

# Download button
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_combined)

st.download_button(
    label="Download All Spectra Data as CSV",
    data=csv,
    file_name="asce_spectra_comparison.csv",
    mime="text/csv",
)

# Additional information
st.header("About")
st.markdown("""
- **Design Response Spectrum**: The earthquake ground motion for design purposes
- **MCE Response Spectrum**: Maximum Considered Earthquake spectrum (Design × 1.5)
- **ASCE 7-10**: Uses site coefficients F_a and F_v with MCE parameters S_s and S_1
- **ASCE 7-22**: Uses direct design parameters S_DS and S_D1
- **T_L**: Long-period transition period (site-specific)
- **ASCE 7-22 Multi-period Spectra**: Upload a CSV file with columns: 'Period (s)', 'Design Response Spectra (g)', 'MCE Response spectra' to compare with ASCE spectra
""")