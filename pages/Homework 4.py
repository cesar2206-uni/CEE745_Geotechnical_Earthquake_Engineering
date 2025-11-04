import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

#===================================================================================================
# Functions
#===================================================================================================

#---------------------------------------------------------------------------------------------------
# Uniform Undamped Soil in rigid rock
#---------------------------------------------------------------------------------------------------
def uniform_undamped(x):
    return 1 / np.abs(np.cos(x))

#---------------------------------------------------------------------------------------------------
# Uniform Damped Soil in rigid rock
#---------------------------------------------------------------------------------------------------
def uniform_damped(x, zeta):
    return 1 / np.sqrt(np.cos(x)**2 + (zeta * x)**2)

#---------------------------------------------------------------------------------------------------
# Shear Stress and Bedrock Acceleration Transfer Function
#---------------------------------------------------------------------------------------------------
def shear_stress_bedrock(x):
    return np.abs(np.sin(x/2)) / np.abs(x * np.cos(x))

#===================================================================================================
# Streamlit code
#===================================================================================================

st.title("Transfer Function Analysis")

# Create dimensionless frequency array
x = np.linspace(0.01, np.pi * 4, 1000)  # Start from 0.01 to avoid division by zero

# Case selection
case = st.selectbox(
    "Select Transfer Function Case",
    [
        "Uniform Undamped Soil in Rigid Rock",
        "Uniform Damped Soil in Rigid Rock", 
        "Shear Stress and Bedrock Acceleration"
    ]
)

# Initialize session state for comparison cases
if 'comparison_cases' not in st.session_state:
    st.session_state.comparison_cases = []

# Main plot section
st.subheader(f"{case}")

# Display the appropriate formula
if case == "Uniform Undamped Soil in Rigid Rock":
    st.latex(r"|F(\omega)| = \frac{1}{|\cos(kH)|}")
elif case == "Uniform Damped Soil in Rigid Rock":
    st.latex(r"|F(\omega)| = \frac{1}{\sqrt{\cos^2(kH) + (\zeta kH)^2}}")
else:
    st.latex(r"|F(\omega)| = \rho H \frac{|\sin\left(\frac{kH}{2}\right)|}{|kH \cos(kH)|}")

# Create main plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot based on selected case
if case == "Uniform Undamped Soil in Rigid Rock":
    F = uniform_undamped(x)
    ax.plot(x, F, 'b-', linewidth=2, label='Undamped')
    
elif case == "Uniform Damped Soil in Rigid Rock":
    # Damping ratio selection for main plot
    zeta_main = st.slider("Select Damping Ratio ζ for main plot", 0.01, 0.50, 0.05, 0.01)
    F = uniform_damped(x, zeta_main)
    ax.plot(x, F, 'r-', linewidth=2, label=f'ζ = {zeta_main:.2f}')
    
else:  # Shear Stress case
    F = shear_stress_bedrock(x)
    ax.plot(x, F, 'purple', linewidth=2, label='Shear Stress')

# Set x-ticks as multiples of π
pi_ticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2, 3*np.pi, 7*np.pi/2, 4*np.pi])
pi_labels = ['0', 'π/2', 'π', '3π/2', '2π', '5π/2', '3π', '7π/2', '4π']

ax.set_xticks(pi_ticks)
ax.set_xticklabels(pi_labels)

ax.set_xlabel('Dimensionless Frequency, $\\frac{\omega H}{V_s}$')
ax.set_xlim(0, np.pi * 4)
ax.grid(True, alpha=0.3)
ax.legend()

# Set appropriate y-limits based on case
if case == "Shear Stress and Bedrock Acceleration":
    ax.set_ylim(0, 12)
    ax.set_ylabel(r'$|F(\omega)| / \rho H$')
else:
    ax.set_ylim(0, 12)
    ax.set_ylabel('$|F(\\omega)|$')

st.pyplot(fig)

# Comparison section
st.subheader("Comparison with Other Cases")

# Add current case to comparison
if st.button("Add Current Case to Comparison"):
    case_info = {
        'case': case,
        'zeta': zeta_main if case == "Uniform Damped Soil in Rigid Rock" else None
    }
    st.session_state.comparison_cases.append(case_info)
    st.success(f"Added {case} to comparison")

# Clear comparison button
if st.button("Clear All Comparisons"):
    st.session_state.comparison_cases = []
    st.success("Comparison cleared")

# Plot comparison if we have cases to compare
if st.session_state.comparison_cases:
    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, case_info in enumerate(st.session_state.comparison_cases):
        color = colors[i % len(colors)]
        
        if case_info['case'] == "Uniform Undamped Soil in Rigid Rock":
            F_comp = uniform_undamped(x)
            label = 'Undamped'
        elif case_info['case'] == "Uniform Damped Soil in Rigid Rock":
            F_comp = uniform_damped(x, case_info['zeta'])
            label = f'Damped ζ={case_info["zeta"]:.2f}'
        else:  # Shear Stress
            F_comp = shear_stress_bedrock(x)
            label = 'Shear Stress (pH = 1)'
        
        ax_comp.plot(x, F_comp, color=color, linewidth=2, label=label)
    
    # Set x-ticks as multiples of π for comparison plot too
    ax_comp.set_xticks(pi_ticks)
    ax_comp.set_xticklabels(pi_labels)
    
    ax_comp.set_xlabel('Dimensionless Frequency, $\\frac{\omega H}{V_s}$')
    ax_comp.set_ylabel('$|F(\\omega)|$')
    ax_comp.set_ylim(0, 12)
    ax_comp.set_xlim(0, np.pi * 4)
    ax_comp.grid(True, alpha=0.3)
    ax_comp.legend()
    ax_comp.set_title('Comparison of Transfer Functions')
    
    st.pyplot(fig_comp)
    
    # Show comparison cases
    st.write("**Current Comparison Cases:**")
    for i, case_info in enumerate(st.session_state.comparison_cases):
        st.write(f"{i+1}. {case_info['case']}" + 
                (f" (ζ={case_info['zeta']:.2f})" if case_info['zeta'] is not None else ""))
else:
    st.info("Add cases to comparison to see them plotted together")

# Individual case controls in sidebar
st.sidebar.subheader("Case Controls")
if st.sidebar.button("Add Damped Case (ζ=0.05)"):
    st.session_state.comparison_cases.append({
        'case': "Uniform Damped Soil in Rigid Rock",
        'zeta': 0.05
    })

if st.sidebar.button("Add Damped Case (ζ=0.10)"):
    st.session_state.comparison_cases.append({
        'case': "Uniform Damped Soil in Rigid Rock", 
        'zeta': 0.10
    })

if st.sidebar.button("Add Undamped Case"):
    st.session_state.comparison_cases.append({
        'case': "Uniform Undamped Soil in Rigid Rock",
        'zeta': None
    })

if st.sidebar.button("Add Shear Stress Case"):
    st.session_state.comparison_cases.append({
        'case': "Shear Stress and Bedrock Acceleration",
        'zeta': None
    })