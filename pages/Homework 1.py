import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#===================================================================================================
# Functions
#===================================================================================================

#---------------------------------------------------------------------------------------------------
# Read *.AT2 (Code from gmimtools) - Have problems with DT2 and VT2 files - use read_AT2_v2 instead
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
# Read *.AT2 (version 2) @author: Daniel Hutabarat - UC Berkeley, 2017
#---------------------------------------------------------------------------------------------------

def read_AT2_v2(uploaded_file, scalefactor=None):
    '''
    This function process acceleration history for NGA data file (.AT2 format)
    to a single column value and return the total number of data points and 
    time iterval of the recording.
    Parameters:
    ------------
    filepath : string (location and name of the file)
    scalefactor : float (Optional) - multiplier factor that is applied to each
                  component in acceleration array.
    
    Output:
    ------------
    desc: Description of the earthquake (e.g., name, year, etc)
    npts: total number of recorded points (acceleration data)
    dt: time interval of recorded points
    time: array (n x 1) - time array, same length with npts
    inp_acc: array (n x 1) - acceleration array, same length with time
             unit usually in (g) unless stated as other.
    
    Example: (plot time vs acceleration)
    filepath = os.path.join(os.getcwd(),'motion_1')
    desc, npts, dt, time, inp_acc = processNGAfile (filepath)
    plt.plot(time,inp_acc)
        
    '''    
    try:
        if not scalefactor:
            scalefactor = 1.0
        
        # Read file content as text
        content = uploaded_file.read().decode("utf-8").splitlines()

        counter = 0
        desc, row4Val, acc_data = "", "", []

        for x in content:
            if counter == 1:
                desc = x
            elif counter == 3:
                row4Val = x
                if row4Val[0] == 'N':
                    val = row4Val.split()
                    npts = float(val[(val.index('NPTS='))+1].rstrip(','))
                    dt = float(val[(val.index('DT='))+1])
                else:
                    val = row4Val.split()
                    npts = float(val[0])
                    dt = float(val[1])
            elif counter > 3:
                data = str(x).split()
                for value in data:
                    a = float(value) * scalefactor
                    acc_data.append(a)
            counter += 1

        inp_acc = np.asarray(acc_data)
        return inp_acc, dt
    
    except Exception as e:
        print("processMotion FAILED!: ", e)
        return None, None


#---------------------------------------------------------------------------------------------------
# Base Line Correction
#---------------------------------------------------------------------------------------------------

def baseline_correction(gmtime, accel_array):
	"""
	Parameter
	=========
	gmtime: Array of time
	accel: Array of accelerations. 

	Returns
	=======
	accel_c: Arrray of acceleations corrected for baseline drift
	vel_c: Array of velocities corrected for baseline drift
	dis_c: Array of displacements corrected for baseline drift
	"""

	## Polynomial fitting for baseline correction
	tf = gmtime[-1]
	t = gmtime
	vel = integrate.cumulative_trapezoid(accel_array, gmtime, initial = 0)

	aa = [[tf**3/3,tf**4/8,tf**5/15,tf**6/24],
      [tf**4/8,tf**5/20,tf**6/36,tf**7/56],
      [tf**5/15,tf**6/36,tf**7/63,tf**8/96],
      [tf**6/24,tf**7/56,tf**8/96,tf**9/144]]
	aa = np.array(aa)

	b1 = integrate.trapezoid(vel*t,t)
	b2 = integrate.trapezoid(vel*t**2/2,t)
	b3 = integrate.trapezoid(vel*t**3/3,t)
	b4 = integrate.trapezoid(vel*t**4/4,t)
	bb = np.array([b1,b2,b3,b4])

	C = np.linalg.solve(aa,bb)

	acc_c = accel_array - (C[0]+C[1]*t+C[2]*t**2+C[3]*t**3)
	vel_c = integrate.cumulative_trapezoid(acc_c,t,initial=0)
	dis_c = integrate.cumulative_trapezoid(vel_c,t,initial=0)

	return acc_c, vel_c, dis_c	


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
	plot of acceleration, velocities and displacements vs time
	"""
	
    # Computing velociies and displacements with baseline correction
    accel, vel, dis = baseline_correction(gmtime, accel)


	# Transforming to units
    accel = accel / 9.81  # from m/s2 to g
    vel = vel * 100  # from m/s to cm/s
    dis = dis * 100  # from m to cm
	
	# Obtaining the PGA of the accel array
    PGA = max(abs(accel))
    PGV = max(abs(vel))
    PGD = max(abs(dis))
	
    # Creating the subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    # Plot Acceleration
    axs[0].plot(gmtime, accel, 'k', label="PGA = {:.2f} g ".format(PGA))
    axs[0].set_ylabel('Acceleration (g)')
    axs[0].legend(loc="upper right")
    axs[0].grid(which='both', color='lightgray')

    # Plot Velocity
    axs[1].plot(gmtime, vel, 'b', label="PGV = {:.2f} cm/s".format(PGV))
    axs[1].set_ylabel('Velocity (cm/s)')
    axs[1].legend(loc="upper right")
    axs[1].grid(which='both', color='lightgray')

    # Plot Displacement
    axs[2].plot(gmtime, dis, 'r', label="PGD = {:.2f} cm".format(PGD))
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Displacement (cm)')
    axs[2].legend(loc="upper right")
    axs[2].grid(which='both', color='lightgray')

    st.pyplot(fig)
	
    plt.close(fig)

#---------------------------------------------------------------------------------------------------
# Create a plot of acceleration, velocity and displacements and add the other two time histories
#---------------------------------------------------------------------------------------------------

def plot_time_histories_check(gmtime, accel, vel_data, dis_data):
    """
	Parameter
	=========
	gmtime: Array of time
	accel: Array of accelerations. 
	
	Returns
	=======
	plot of acceleration vs time
	"""
	
    # Computing velociies and displacements with baseline correction
    accel, vel, dis = baseline_correction(gmtime, accel)

	# Transforming to units
    accel = accel / 9.81  # from m/s2 to g
    vel = vel * 100  # from m/s to cm/s
    dis = dis * 100  # from m to cm
	
	# Obtaining the PGA of the accel array
    PGA = max(abs(accel))
    PGV = max(abs(vel))
    PGD = max(abs(dis))
	
    PGV_from_data = max(abs(vel_data))
    PGD_from_data = max(abs(dis_data))
	
    # Creating the subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    # Plot Acceleration
    axs[0].plot(gmtime, accel, 'k', label="AT2 Data")
    axs[0].set_ylabel('Acceleration (g)')
    axs[0].legend(loc="upper right")
    axs[0].grid(which='both', color='lightgray')

    # Plot Velocity
    axs[1].plot(gmtime, vel, 'b', label="Calculated")
    axs[1].plot(gmtime, vel_data, "--", color = 'yellow', label = "VT2 Data")
    axs[1].set_ylabel('Velocity (cm/s)')
    axs[1].legend(loc="upper right")
    axs[1].grid(which='both', color='lightgray')

    # Plot Displacement
    axs[2].plot(gmtime, dis, 'r', label="Calculated")
    axs[2].plot(gmtime, dis_data, "--", color = 'lightblue', label = "DT2 Data")
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Displacement (cm)')
    axs[2].legend(loc="upper right")
    axs[2].grid(which='both', color='lightgray')

    st.pyplot(fig)
    st.write("- The PGV calculated from the acceleration time history is {:.2f} cm/s and the PGV from the data is {:.2f} cm/s".format(PGV, PGV_from_data))
    st.write("- The PGD calculated from the acceleration time history is {:.2f} cm and the PGD from the data is {:.2f} cm".format(PGD, PGD_from_data))
    plt.close(fig)


#===================================================================================================
# Streamlit code
#===================================================================================================

st.title("PGA, PGV, PGD from .AT2 Files")
st.write("Upload a .AT2 or .ACC file to explore the data and obtain parameters")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a file", type=["AT2"])

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
	if filename.endswith(".AT2"):
		accel, dt = read_AT2_v2(uploaded_file, scaling)
		gmtime = np.linspace(dt,dt*len(accel),len(accel))
	
    # Plot the array of acceleration
	plot_time_histories(gmtime, accel)

	check_extra = st.checkbox(
    "Check with .VT2 and .DT2 files for velocity and displacement time histories.")
	if check_extra:
		# --- Velocity ---
		col3, col4 = st.columns(2)
		with col3:
			vel_file = st.file_uploader("Upload Velocity File", type=["VT2"])
		with col4:
			vel_units = st.selectbox("Velocity Units", ["m/s", "cm/s"])

		# --- Displacement ---
		col5, col6 = st.columns(2)
		with col5:
			disp_file = st.file_uploader("Upload Displacement File", type=["DT2"])
		with col6:
			disp_units = st.selectbox("Displacement Units", ["m", "cm"])

		if vel_file is not None and disp_file is not None:
			# Read velocity file
			if vel_units == "m/s":
				vel_scaling = 1/100
			elif vel_units == "cm/s":
				vel_scaling = 1

			vel_vector, dt_vel = read_AT2_v2(vel_file, vel_scaling)
			print(accel)
			gmtime_vel = np.linspace(dt_vel, dt_vel*len(vel_vector), len(vel_vector))

			# Read displacement file
			if disp_units == "m":
				disp_scaling = 1/100
			elif disp_units == "cm":
				disp_scaling = 1

			disp_vector, dt_disp = read_AT2_v2(disp_file, disp_scaling)
			gmtime_disp = np.linspace(dt_disp, dt_disp*len(disp_vector), len(disp_vector))

			# Plot velocity and displacement
			plot_time_histories_check(gmtime, accel, vel_vector, disp_vector)

else:
	st.info("Please upload a file to continue. ")



