
import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#===================================================================================================
# Functions
#===================================================================================================

#---------------------------------------------------------------------------------------------------
# Read JSON File of USGS page https://earthquake.usgs.gov/hazards/interactive/
#---------------------------------------------------------------------------------------------------

def process_hazard_data(response_data):
    """Process hazard curve data from JSON response"""
    hazard_curve_data = []
    
    period_map = {
        'PGA': 0.0, 'SA0P1': 0.1, 'SA0P2': 0.2, 'SA0P3': 0.3,
        'SA0P5': 0.5, 'SA0P75': 0.75, 'SA1P0': 1.0, 'SA2P0': 2.0,
        'SA3P0': 3.0, 'SA4P0': 4.0, 'SA5P0': 5.0
    }
    
    for resp in response_data:
        metadata = resp['metadata']
        imt_value = metadata['imt']['value']
        imt_display = metadata['imt']['display']
        
        period = period_map.get(imt_value, imt_value)
        
        for comp_data in resp['data']:
            if comp_data['component'] == 'Total':
                xvalues = metadata['xvalues']
                yvalues = comp_data['yvalues']
                
                for gm, freq in zip(xvalues, yvalues):
                    hazard_curve_data.append({
                        'Spectral_Period_s': period,
                        'IMT_Display': imt_display,
                        'IMT_Value': imt_value,
                        'Ground_Motion_g': gm,
                        'Annual_Frequency_Exceedance': freq
                    })
                break
    
    return pd.DataFrame(hazard_curve_data)

#---------------------------------------------------------------------------------------------------
# Process information of UHS
#---------------------------------------------------------------------------------------------------

def process_uhs_data(response_data):
    """Process uniform hazard spectrum data from JSON response"""
    target_frequencies = [1/2475, 1/1000, 1/975, 1/475, 1/100, 1/50, 1/10]
    return_periods = [1/freq for freq in target_frequencies]
    
    uhs_data = []
    
    period_map = {
        'PGA': 0.0, 'SA0P1': 0.1, 'SA0P2': 0.2, 'SA0P3': 0.3,
        'SA0P5': 0.5, 'SA0P75': 0.75, 'SA1P0': 1.0, 'SA2P0': 2.0,
        'SA3P0': 3.0, 'SA4P0': 4.0, 'SA5P0': 5.0
    }
    
    for resp in response_data:
        metadata = resp['metadata']
        imt_value = metadata['imt']['value']
        imt_display = metadata['imt']['display']
        
        if imt_value in period_map:
            period = period_map[imt_value]
            
            for comp_data in resp['data']:
                if comp_data['component'] == 'Total':
                    xvalues = metadata['xvalues']
                    yvalues = comp_data['yvalues']
                    
                    x = np.array(xvalues)
                    y = np.array(yvalues)
                    mask = y > 0
                    
                    if np.any(mask):
                        sort_idx = np.argsort(x[mask])
                        x_sorted = x[mask][sort_idx]
                        y_sorted = y[mask][sort_idx]
                        
                        for target_freq, rp in zip(target_frequencies, return_periods):
                            if target_freq <= max(y_sorted) and target_freq >= min(y_sorted):
                                log_x = np.log(x_sorted)
                                log_y = np.log(y_sorted)
                                f = interpolate.interp1d(log_y, log_x, kind='linear', 
                                                       fill_value='extrapolate')
                                spectral_accel = np.exp(f(np.log(target_freq)))
                            else:
                                spectral_accel = np.nan
                            
                            uhs_data.append({
                                'Spectral_Period_s': period,
                                'IMT_Display': imt_display,
                                'IMT_Value': imt_value,
                                'Return_Period_yr': rp,
                                'Annual_Frequency': target_freq,
                                'Spectral_Acceleration_g': spectral_accel
                            })
                    break
    
    return pd.DataFrame(uhs_data)

#---------------------------------------------------------------------------------------------------
# Plot information functions
#---------------------------------------------------------------------------------------------------

def plot_hazard_curves(response_data):
    """Create hazard curves plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(response_data)))
    
    for i, resp in enumerate(response_data):
        metadata = resp['metadata']
        imt_display = metadata['imt']['display']
        
        for comp_data in resp['data']:
            if comp_data['component'] == 'Total':
                xvalues = metadata['xvalues']
                yvalues = comp_data['yvalues']
                
                x = np.array(xvalues)
                y = np.array(yvalues)
                mask = y > 0
                if np.any(mask):
                    ax.loglog(x[mask], y[mask],'o-', markersize=4,color=colors[i], linewidth=1.5, label=imt_display)
                break
    
    ax.set_xlabel('Ground Motion (g)', fontsize=12)
    ax.set_ylabel('Annual Frequency of Exceedance', fontsize=12)
    ax.set_title('Hazard Curves', fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-13, 1e0)
    
    return fig


def plot_uniform_hazard_spectrum(df_uhs, response_data):
    """Create uniform hazard spectrum plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare periods for plotting
    period_map = {
        'PGA': 0.0, 'SA0P1': 0.1, 'SA0P2': 0.2, 'SA0P3': 0.3,
        'SA0P5': 0.5, 'SA0P75': 0.75, 'SA1P0': 1.0, 'SA2P0': 2.0,
        'SA3P0': 3.0, 'SA4P0': 4.0, 'SA5P0': 5.0
    }
    
    periods = []
    for resp in response_data:
        imt_value = resp['metadata']['imt']['value']
        if imt_value in period_map:
            periods.append(period_map[imt_value])
    
    # Plot each return period
    return_periods = df_uhs['Return_Period_yr'].unique()
    colors_uhs = plt.cm.plasma(np.linspace(0, 1, len(return_periods)))
    
    for i, rp in enumerate(sorted(return_periods)):
        rp_data = []
        periods_sorted = []
        
        for period in sorted(periods):
            period_rp_data = df_uhs[
                (df_uhs['Spectral_Period_s'] == period) & 
                (df_uhs['Return_Period_yr'] == rp)
            ]
            if not period_rp_data.empty and not pd.isna(period_rp_data['Spectral_Acceleration_g'].iloc[0]):
                rp_data.append(period_rp_data['Spectral_Acceleration_g'].iloc[0])
                periods_sorted.append(period)
        
        if rp_data:
            ax.plot(periods_sorted, rp_data, color=colors_uhs[i], linewidth=2, 
                    marker='o', markersize=4, label=f'RP={rp:.0f} yr')
    
    ax.set_xlabel('Spectral Period (s)', fontsize=12)
    ax.set_ylabel('Spectral Acceleration (g)', fontsize=12)
    ax.set_title('Uniform Hazard Response Spectrum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5)
    ax.set_ylim(bottom=0)
    
    return fig

def get_location_info(response_data):
    """Extract location information from response data"""
    if not response_data:
        return {}
    
    metadata = response_data[0]['metadata']
    return {
        'latitude': metadata['latitude'],
        'longitude': metadata['longitude'],
        'vs30': metadata['vs30']['display'],
        'region': metadata['region']['display'],
        'hazard_model': metadata['edition']['display']
    }


#===================================================================================================
# Streamlit code
#===================================================================================================

st.set_page_config(
    page_title="Seismic Hazard Analysis",
    page_icon="🌋",
    layout="wide"
)

# Title and description
st.markdown('<h1 class="main-header">🌋 Seismic Hazard Analysis Tool</h1>', unsafe_allow_html=True)
st.write("Upload a USGS seismic hazard JSON file to visualize hazard curves and uniform hazard spectrum.")

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.subheader("📁 Upload JSON File")
uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        # Load and parse the JSON data
        data = json.load(uploaded_file)
        response_data = data['response']
        
        # Display basic information
        st.success("✅ File successfully loaded!")
        
        # Get location info
        location_info = get_location_info(response_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Location", f"{location_info['latitude']}, {location_info['longitude']}")
        with col2:
            st.metric("VS30", location_info['vs30'])
        with col3:
            st.metric("Region", location_info['region'])
        
        # Process the data using functions
        df_hazard = process_hazard_data(response_data)
        df_uhs = process_uhs_data(response_data)
        
        # Create plots section
        st.subheader("📊 Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Hazard Curves**")
            fig1 = plot_hazard_curves(response_data)
            st.pyplot(fig1)
        
        with col2:
            st.write("**Uniform Hazard Response Spectrum**")
            fig2 = plot_uniform_hazard_spectrum(df_uhs, response_data)
            st.pyplot(fig2)
        
        # Download section
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.subheader("📥 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Convert hazard data to CSV
            csv_hazard = df_hazard.to_csv(index=False)
            st.download_button(
                label="📊 Download Hazard Curves Data (CSV)",
                data=csv_hazard,
                file_name="hazard_curves_data.csv",
                mime="text/csv",
                help="Download the annual frequency of exceedance vs ground motion data"
            )
            
            # Show hazard data info
            st.write(f"**Hazard Data Info:**")
            st.write(f"- Records: {len(df_hazard)}")
            st.write(f"- Periods: {len(df_hazard['Spectral_Period_s'].unique())}")
        
        with col2:
            # Convert UHS data to CSV
            csv_uhs = df_uhs.to_csv(index=False)
            st.download_button(
                label="📈 Download Uniform Hazard Spectrum (CSV)",
                data=csv_uhs,
                file_name="uniform_hazard_spectrum_data.csv",
                mime="text/csv",
                help="Download the spectral period vs spectral acceleration data"
            )
            
            # Show UHS data info
            st.write(f"**UHS Data Info:**")
            st.write(f"- Records: {len(df_uhs)}")
            st.write(f"- Return periods: {len(df_uhs['Return_Period_yr'].unique())}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data preview
        st.subheader("🔍 Data Preview")
        
        tab1, tab2 = st.tabs(["Hazard Curves Data", "Uniform Hazard Spectrum Data"])
        
        with tab1:
            st.dataframe(df_hazard.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(df_uhs.head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please make sure you're uploading a valid USGS seismic hazard JSON file.")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a JSON file to get started.")
    
    st.markdown("""
    ### 📋 Expected JSON Format
    The app expects a USGS seismic hazard JSON file with the following structure:
    
    ```json
    {
        "status": "success",
        "response": [
            {
                "metadata": {
                    "imt": {"value": "PGA", "display": "Peak Ground Acceleration"},
                    "latitude": 39.54,
                    "longitude": -119.813,
                    "vs30": {"display": "760 m/s"},
                    "region": {"display": "Western US"},
                    "xvalues": [0.0025, 0.0045, ...],
                    ...
                },
                "data": [
                    {
                        "component": "Total",
                        "yvalues": [0.51195643, 0.35387709, ...]
                    }
                ]
            }
        ]
    }
    ```
    
    ### 🎯 How to Use
    1. Upload a USGS seismic hazard JSON file
    2. View the interactive plots
    3. Download the data as CSV files for further analysis
    """)
