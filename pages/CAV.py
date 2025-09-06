import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#===================================================================================================
# Functions
#===================================================================================================
#---------------------------------------------------------------------------------------------------
# Read *.ACC Files
#---------------------------------------------------------------------------------------------------

def read_acc(uploaded_file, scalefactor=None):
    """
    Reads a .acc file from an uploaded file-like object (e.g., Streamlit file)

    Parameters
    ----------
    uploaded_file : file-like
        File object from st.file_uploader or similar
    scalefactor : float, optional
        Scale factor for acceleration values. Default is 1.0.

    Returns
    -------
    acc : np.ndarray
        Array of acceleration values.
    dt : float
        Time step between samples.
    """
    try:
        if not scalefactor:
            scalefactor = 1.0

        # Read file as text
        content = uploaded_file.read().decode("utf-8").splitlines()

        # Line 2: N and dt
        n, dt = content[1].split()
        n = int(n)
        dt = float(dt)

        # Remaining lines = accelerations
        acc = np.array([float(val.strip()) * scalefactor for val in content[2:]])

        if len(acc) != n:
            print(f"Warning: expected {n} rows, but found {len(acc)}")
            acc = acc[:n]

        return acc, dt
    
    except Exception as e:
        print(f"Error reading .acc file: {e}")
        return None, None

#---------------------------------------------------------------------------------------------------
# CALCULATES CUMULATIVE ABSOLUTE VELOCITY - EPRI (1988)
#---------------------------------------------------------------------------------------------------
# Input: accel in m/s**2
def get_CAV(accel, dt):
	CAV = np.cumsum(np.abs(accel))*dt
	return CAV, CAV[-1]

#---------------------------------------------------------------------------------------------------
# Plot Acceleration and Cummulative Velocities
#---------------------------------------------------------------------------------------------------

def plot_accel_CAV(accel, gmtime, dt):
    """
	Parameter
	=========
	gmtime: Array of time
	accel: Array of accelerations. 
	
	Returns
	=======
	plot of acceleration, velocities and displacements vs time
	"""
    # Transforming to units
    CAV_array, CAV_max = get_CAV(accel, dt)
    CAV_array = CAV_array * 100  # from m/s to cm/s
    CAV_max = CAV_max * 100  # from m/s to cm/s
    accel = accel / 9.81  # from m/s2 to g

	# Creating the subplots (3 rows, 1 column)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

    # Plot Acceleration
    axs[0].plot(gmtime, accel, 'k')
    axs[0].set_ylabel('Acceleration (g)')
    #axs[0].legend(loc="upper right")
    axs[0].grid(which='both', color='lightgray')

    # Plot Velocity
    axs[1].plot(gmtime, CAV_array, 'b', label="$CAV_{max}$"+ " = {:.2f} cm/s".format(CAV_max))
    axs[1].set_ylabel('Cummulative Absolute Velocity (cm/s)')
    axs[1].legend(loc="lower right")
    axs[1].grid(which='both', color='lightgray')
    #plt.tight_layout()
    st.pyplot(fig)
	
    plt.close(fig)


#===================================================================================================
# Streamlit code
#===================================================================================================

st.title("Commulative Absolute Velocity from .ACC Files")
st.write("Upload a .ACC file to obtain Cummulative Absolute Velocities (CAV)")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a file", type=["acc"])

with col2:
    units = st.selectbox("Select units:", ["m/s²", "g", "cm/s²"])


if uploaded_file is not None:
	filename = uploaded_file.name
	st.success(f"✅ File uploaded: {filename}")

	if units == "g":
		scaling = 9.81
	elif units == "m/s²":
		scaling = 1
	elif units == "cm/s²":
		scaling = 100

    # Handle .AT2 files
	if filename.endswith(".acc"):
		accel, dt = read_acc(uploaded_file, scaling)
		gmtime = np.linspace(dt,dt*len(accel),len(accel))
	
    # Plot the array of acceleration and CAV
	plot_accel_CAV(accel, gmtime, dt)

else:
	st.info("Please upload a file to continue. ")



