import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#===================================================================================================
# Functions
#===================================================================================================

#---------------------------------------------------------------------------------------------------
# Function to calculate layer thickness
#---------------------------------------------------------------------------------------------------
def calculate_thickness(df):
    """Calculate thickness for each layer"""
    df['Thickness (ft)'] = df['Final Depth (ft)'] - df['Initial Depth (ft)']
    return df

#---------------------------------------------------------------------------------------------------
# Function to calculate Vs30
#---------------------------------------------------------------------------------------------------
def calculate_vs30(df):
    """Calculate Vs30 using the formula: Vs30 = total_thickness / sum(thickness_i / Vs_i)"""
    total_thickness = df['Thickness (ft)'].sum()
    sum_di_vsi = (df['Thickness (ft)'] / df['Shear Wave Velocity (ft/s)']).sum()
    vs30 = total_thickness / sum_di_vsi
    return vs30, total_thickness

#---------------------------------------------------------------------------------------------------
# Function to determine site class
#---------------------------------------------------------------------------------------------------
def determine_site_class(vs30):
    """Determine site class based on Vs30 value"""
    if vs30 > 5000:
        return "A", "Hard rock"
    elif 3000 < vs30 <= 5000:
        return "B", "Medium hard rock"
    elif 2100 < vs30 <= 3000:
        return "B/C", "Soft rock"
    elif 1450 < vs30 <= 2100:
        return "C", "Very dense sand or hard clay"
    elif 1000 < vs30 <= 1450:
        return "C/D", "Dense sand or very stiff clay"
    elif 700 < vs30 <= 1000:
        return "D", "Medium dense sand or stiff clay"
    elif 500 < vs30 <= 700:
        return "D/E", "Loose sand or medium stiff clay"
    elif vs30 <= 500:
        return "E", "Very loose sand or soft clay"
    else:
        return "Unknown", "Unknown"

#---------------------------------------------------------------------------------------------------
# Function to calculate fundamental period
#---------------------------------------------------------------------------------------------------
def calculate_fundamental_period(vs30, total_thickness):
    """Calculate fundamental period T = 4 * total_thickness / Vs30"""
    T = 4 * total_thickness / vs30
    return T

#---------------------------------------------------------------------------------------------------
# Function to create shear wave velocity profile plot
#---------------------------------------------------------------------------------------------------
def create_vs_profile_plot(df, vs30):
    """Create a plot of shear wave velocity profile using step function"""
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Sort dataframe by depth
    df = df.sort_values('Initial Depth (ft)').reset_index(drop=True)
    
    # Prepare data for step plot
    depths = [df.iloc[0]['Initial Depth (ft)']]  # Start with top of first layer
    vs_values = [df.iloc[0]['Shear Wave Velocity (ft/s)']]
    
    for _, row in df.iterrows():
        depths.append(row['Final Depth (ft)'])
        vs_values.append(row['Shear Wave Velocity (ft/s)'])
    
    # Create step plot - 'post' gives the right-aligned step (like in the image)
    ax.step(vs_values, depths, where='post', color='blue', linewidth=2)
    
    # Fill the area under the curve for better visualization
    #ax.fill_betweenx(depths, 0, vs_values, step='post', alpha=0.3, color='blue')
    
    # Add Vs30 vertical line
    ax.axvline(x=vs30, color='red', linestyle='--', linewidth=2, label=f'Vs30 = {vs30:.0f} ft/s')
    
    ax.set_xlabel('Shear Wave Velocity (ft/s)')
    ax.set_ylabel('Depth (ft)')
    ax.set_xlim(0,)
    ax.set_ylim(0, 100)
    ax.set_title('Shear Wave Velocity Profile')
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig
#---------------------------------------------------------------------------------------------------
# Function to process uploaded CSV data
#---------------------------------------------------------------------------------------------------
def process_uploaded_csv(uploaded_file):
    """Process uploaded CSV file and convert to the required format"""
    try:
        # Read the CSV file
        df_uploaded = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        if 'Depth_ft' not in df_uploaded.columns or 'Vs_ft_s' not in df_uploaded.columns:
            st.error("CSV file must contain 'Depth_ft' and 'Vs_ft_s' columns")
            return None
        
        # Create the dynamic table format
        depths = df_uploaded['Depth_ft'].tolist()
        vs_values = df_uploaded['Vs_ft_s'].tolist()
        
        # Create pairs of initial and final depths
        processed_data = []
        for i in range(len(depths) - 1):
            initial_depth = depths[i]
            final_depth = depths[i + 1]
            vs = vs_values[i]  # Use the Vs value for the current depth
            
            processed_data.append({
                'Initial Depth (ft)': initial_depth,
                'Final Depth (ft)': final_depth,
                'Shear Wave Velocity (ft/s)': vs
            })
        
        # Add the last layer (from last depth to 100 ft if needed)
        if depths[-1] < 100:
            processed_data.append({
                'Initial Depth (ft)': depths[-1],
                'Final Depth (ft)': 100,
                'Shear Wave Velocity (ft/s)': vs_values[-1]
            })
        
        return pd.DataFrame(processed_data)
    
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None

#---------------------------------------------------------------------------------------------------
# Function to create sample CSV
#---------------------------------------------------------------------------------------------------
def create_sample_csv():
    """Create a sample CSV file for download"""
    sample_data = {
        'Depth_ft': [0, 18.3268, 18.3268, 27.4902, 27.4902, 65.6726, 65.6726, 100],
        'Vs_ft_s': [485.4692, 485.4692, 888.7041, 888.7041, 595.4397, 595.4397, 1078.1, 1078.1]
    }
    df_sample = pd.DataFrame(sample_data)
    return df_sample.to_csv(index=False)
#===================================================================================================
# Streamlit code
#===================================================================================================

st.title("Site Classification and Shear Wave Velocity Analysis")
st.markdown("""
This app calculates the average shear wave velocity for the upper 100 ft of soil profile (Vs30), 
determines the site class according to AASHTO LRFD Bridge Design Specifications, 
and calculates the fundamental period of the soil profile.
""")


# Initialize session state for dataframe
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
        'Initial Depth (ft)': [0, 10, 25, 40, 60],
        'Final Depth (ft)': [10, 25, 40, 60, 100],
        'Shear Wave Velocity (ft/s)': [400, 450, 600, 800, 1000]
    })

st.header("Input Soil Layers Data")

# CSV Upload Section
st.subheader("Upload CSV File")
st.markdown("""
**CSV Format Requirements:**
- Must contain two columns: `Depth_ft` and `Vs_ft_s`
- Depth values should be in feet
- Shear wave velocity values should be in ft/s
- The last depth should be 100 ft for complete analysis
""")

# Create sample CSV for download
sample_csv = create_sample_csv()
st.download_button(
    label="Download Sample CSV Format",
    data=sample_csv,
    file_name="sample_vs_data.csv",
    mime="text/csv"
)

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    processed_df = process_uploaded_csv(uploaded_file)
    if processed_df is not None:
        st.session_state.df = processed_df
        st.success("CSV file processed successfully! The table below has been updated.")


# Instructions
st.markdown("""
**Instructions:**
- Enter the soil layer data for the upper 100 ft
- Ensure the final depth of the last layer is 100 ft
- Initial and final depths should be in feet
- Shear wave velocity should be in ft/s
""")

# Editable dataframe
edited_df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Initial Depth (ft)": st.column_config.NumberColumn(
            "Initial Depth (ft)",
            help="Initial depth of layer in feet",
            format="%.4f",
            min_value=0.0
        ),
        "Final Depth (ft)": st.column_config.NumberColumn(
            "Final Depth (ft)", 
            help="Final depth of layer in feet",
            format="%.4f",
            min_value=0.0
        ),
        "Shear Wave Velocity (ft/s)": st.column_config.NumberColumn(
            "Shear Wave Velocity (ft/s)",
            help="Shear wave velocity in ft/s", 
            format="%.4f",
            min_value=0.0
        )
    }
)


# Update session state
st.session_state.df = edited_df

# Calculate thickness
df_with_thickness = calculate_thickness(st.session_state.df.copy())

# Display thickness column
st.subheader("Layer Thickness Calculation")
st.dataframe(df_with_thickness, use_container_width=True)

# Check if total depth is 100 ft
total_depth = df_with_thickness['Final Depth (ft)'].max()
if total_depth != 100:
    st.warning(f"Total depth is {total_depth} ft. For accurate Vs30 calculation, the profile should extend to 100 ft.")

# Calculate Vs30 and other parameters
if st.button("Calculate Vs30 and Site Classification"):
    try:
        vs30, total_thickness = calculate_vs30(df_with_thickness)
        site_class, site_description = determine_site_class(vs30)
        fundamental_period = calculate_fundamental_period(vs30, total_thickness)
        
        # Display results
        st.header("Results")
        
        # Summary text
        st.success(f"""
        **Site Classification Summary:**
        
        The site has an average shear wave velocity for the upper 100 ft of the soil profile of **{vs30:.0f} ft/s**. 
        This corresponds to **Class {site_class} ({site_description})** based on the AASHTO LRFD Bridge Design Specifications, 
        and a fundamental period of **{fundamental_period:.2f} s**.
        """)
        
        # Create and display plot
        st.subheader("Shear Wave Velocity Profile")
        fig = create_vs_profile_plot(df_with_thickness, vs30)
        st.pyplot(fig)
        
        # Detailed calculations
        st.subheader("Detailed Calculations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Vs30", f"{vs30:.0f} ft/s")
        
        with col2:
            st.metric("Site Class", f"Class {site_class}")
        
        with col3:
            st.metric("Fundamental Period", f"{fundamental_period:.2f} s")
            
    except Exception as e:
        st.error(f"Error in calculations: {str(e)}")
        st.info("Please check your input data for errors (e.g., zero shear wave velocity values).")

# Display site class reference table
st.header("Site Class Reference Table")
st.markdown("""
The following table shows the site class definitions according to AASHTO LRFD Bridge Design Specifications:
""")

site_class_df = pd.DataFrame({
    'Site Class': ['A', 'B', 'B/C', 'C', 'C/D', 'D', 'D/E', 'E'],
    'Soil Type and Profile': [
        'Hard rock',
        'Medium hard rock', 
        'Soft rock',
        'Very dense sand or hard clay',
        'Dense sand or very stiff clay',
        'Medium dense sand or stiff clay',
        'Loose sand or medium stiff clay',
        'Very loose sand or soft clay'
    ],
    'Shear Wave Velocity Range (ft/s)': [
        '> 5,000',
        '3,000 - 5,000',
        '2,100 - 3,000',
        '1,450 - 2,100',
        '1,000 - 1,450',
        '700 - 1,000',
        '500 - 700',
        '≤ 500'
    ]
})

st.dataframe(site_class_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("*Based on AASHTO LRFD Bridge Design Specifications*")