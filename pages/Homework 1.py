import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#===================================================================================================
# Functions
#===================================================================================================

#---------------------------------------------------------------------------------------------------
# Read *.AT2 (Code from gmimtools)
#---------------------------------------------------------------------------------------------------

def read_AT2(file,scaling,Nskip=4):
	"""
	Parameter
	=========
	file: File name including extension ".AT2"
	scaling: A scaling factor, e.g., to convert from g to m/s**2
	Nskip: Number of rows to skip (default = 4)
	
	Returns
	=======
	accel: Array of accelerations. 
	dt: Time step.
	"""
	data = pd.read_csv(file,delimiter=',',header=None,skiprows=Nskip-1,nrows=1,engine='python').values[0]
	dt   = data[1].split('DT=')[1]
	
	try:
		dt = float(dt.split('SEC')[0])
	except:
		pass
	
	data  = pd.read_csv(file,sep='\\s+',engine='python',header=None,skiprows=Nskip)
	data  = data.to_numpy()
	
	[h_dim, w_dim]= np.shape(data)
	
	accel = data.reshape(h_dim*w_dim,1)
	accel = np.asarray(accel, dtype = np.float64)
	accel = np.concatenate(accel)
	accel = accel[~np.isnan(accel)]
	accel = scaling*accel

	return accel, dt

#---------------------------------------------------------------------------------------------------
# Create a plot of acceleration, velocity and displacements
#---------------------------------------------------------------------------------------------------

def plot_time_histories(gmtime, accel):
    """
	Parameter
	=========
	gmtime: Array of time
	accel: Array of accelerations. 
	
	Returns
	=======
	plot of acceleration vs time
	"""
	
    # Computing velociies and displacements
    vel = integrate.cumulative_trapezoid(accel, gmtime, initial = 0)
    dis = integrate.cumulative_trapezoid(vel, gmtime, initial = 0)
    
    # Obtaining the PGA of the accel array
    PGA = max(abs(accel))
    PGV = max(abs(vel))
    PGD = max(abs(dis))
	
    # Creating the subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    # Plot Acceleration
    axs[0].plot(gmtime, accel, 'k', label="PGA = {:.2f} $m/s^2$ ({:.2f} g) ".format(PGA, PGA/9.81))
    axs[0].set_ylabel('Acceleration ($m/s^2$)')
    axs[0].legend(loc="upper right")
    axs[0].grid(which='both', color='lightgray')

    # Plot Velocity
    axs[1].plot(gmtime, vel, 'b', label="PGV = {:.2f} m/s".format(PGV))
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend(loc="upper right")
    axs[1].grid(which='both', color='lightgray')

    # Plot Displacement
    axs[2].plot(gmtime, dis, 'r', label="PGD = {:.2f} m".format(PGD))
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Displacement (m)')
    axs[2].legend(loc="upper right")
    axs[2].grid(which='both', color='lightgray')

    st.pyplot(fig)
	
    plt.close(fig)

#===================================================================================================
# Streamlit code
#===================================================================================================

st.title("PGA, PGV, PGD from .AT2 Files")
st.write("Upload a .AT2 or .ACC file to explore the data and obtain parameters")

uploaded_file = st.file_uploader("Choose a file", type = ["AT2", "ACC"])

if uploaded_file is not None:
	filename = uploaded_file.name
	st.success(f"✅ File uploaded: {filename}")
	
    # Handle .AT2 files
	if filename.endswith(".AT2"):
		accel, dt = read_AT2(uploaded_file, 9.81)
		gmtime = np.linspace(dt,dt*len(accel),len(accel))
	
    # Plot the array of acceleration
	plot_time_histories(gmtime, accel)
		
else:
	st.info("Please upload a file to continue. ")






