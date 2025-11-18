import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#===================================================================================================
# Functions
#===================================================================================================

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
# Base Line Correction (Polynomial Fitting)
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
# Response Spectrum Calculation  -@author: Loic Viens
#---------------------------------------------------------------------------------------------------

def RS_function(data, delta, T, xi, Resp_type):
    """
    - Function to compute the response spectra of time series using the Duhamel integral technique.

    Inputs:
    - data: acceleration data in the time domain
    - delta: Sampling rate of the time-series (in Hz)
    - T: Output period range in second, Example (if delta>=20 Hz): T = np.concatenate((np.arange(.1, 1, .01), np.arange(1, 20, .1)))
    - xi: Damping factor (Standard: 5% -> 0.05)
    - Resp_type: Response type, choose between:
                    - 'SA'  : Acceleration Spectra
                    - 'PSA' : Pseudo-acceleration Spectra
                    - 'SV'  : Velocity Spectra
                    - 'PSV' : Pseudo-velocity Spectra
                    - 'SD'  : Displacement Spectra

    Output:
        - Response spectra in the unit specified by 'Resp_type'
    """
    dt = 1/delta 
    w = 2*np.pi/T 
    
    mass = 1 #  constant mass (=1)
    c = 2*xi*w*mass
    wd = w*np.sqrt(1-xi**2)
    p1 = -mass*data
    
    # predefine output matrices
    S = np.zeros(len(T))
    D1 = S
    for j in np.arange(len(T)):
        # Duhamel time domain matrix form
        I0 = 1/w[j]**2*(1-np.exp(-xi*w[j]*dt)*(xi*w[j]/wd[j]*np.sin(wd[j]*dt)+np.cos(wd[j]*dt)))
        J0 = 1/w[j]**2*(xi*w[j]+np.exp(-xi*w[j]*dt)*(-xi*w[j]*np.cos(wd[j]*dt)+wd[j]*np.sin(wd[j]*dt)))
        
        AA = [[np.exp(-xi*w[j]*dt)*(np.cos(wd[j]*dt)+xi*w[j]/wd[j]*np.sin(wd[j]*dt)) , np.exp(-xi*w[j]*dt)*np.sin(wd[j]*dt)/wd[j] ] , 
               [-w[j]**2*np.exp(-xi*w[j]*dt)*np.sin(wd[j]*dt)/wd[j] ,np.exp(-xi*w[j]*dt)*(np.cos(wd[j]*dt)-xi*w[j]/wd[j]*np.sin(wd[j]*dt)) ]]
        BB = [[I0*(1+xi/w[j]/dt)+J0/w[j]**2/dt-1/w[j]**2 , -xi/w[j]/dt*I0-J0/w[j]**2/dt+1/w[j]**2 ] ,
            [J0-(xi*w[j]+1/dt)*I0, I0/dt] ]
        
        u1 = np.zeros(len(data))
        udre1 = np.zeros(len(data));
        for xx in range(1,len(data),1) :
    
            u1[xx] = AA[0][0]*u1[xx-1] + AA[0][1]*udre1[xx-1] + BB[0][0]*p1[xx-1] + BB[0][1]*p1[xx]
            udre1[xx] = AA[1][0]*u1[xx-1]+AA[1][1]*udre1[xx-1] + BB[1][0]*p1[xx-1]+BB[1][1]*p1[xx]
       
        if Resp_type == 'SA':
            udd1 = -(w[j]**2*u1+c[j]*udre1)-data  # calculate acceleration
            S[j] = np.max(np.abs(udd1+data))
        elif Resp_type == 'PSA':
            D1[j] = np.max(np.abs(u1))
            S[j] = D1[j]*w[j]**2
        elif Resp_type == 'SV':
            S[j] = np.max(np.abs(udre1))
        elif Resp_type == 'PSV':
            D1[j] = np.max(np.abs(u1))
            S[j] = D1[j]*w[j]
        elif Resp_type == 'SD':
            S[j] = np.max(np.abs(u1)) 
    return S


#---------------------------------------------------------------------------------------------------
# Response Spectrum Plotting
#---------------------------------------------------------------------------------------------------

def plot_response_spectrum(accel_array, dt, T, xi, Resp_type):
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
    gmtime = np.linspace(dt, dt * len(accel_array), len(accel_array))
    accel, vel, dis = baseline_correction(gmtime, accel_array)

    # Compute response spectrum
    delta = 1/dt
    RS = RS_function(accel, delta, T, xi, Resp_type)

    # Plot response spectrum
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.set_xlabel('Period (s)')

    if Resp_type == 'SA':
        ax.plot(T, RS / 9.81, color='blue', lw=2) # Convert to g
        ax.set_ylabel('Spectral Acceleration (' + r'g' + ')')
    elif Resp_type == 'PSA':
        ax.plot(T, RS / 9.81, color='blue', lw=2) # Convert to g
        ax.set_ylabel('Pseudo-Spectral Acceleration (' + r'g' + ')')
    elif Resp_type == 'SV':
        ax.plot(T, RS * 100, color='blue', lw=2) # Convert to cm/s
        ax.set_ylabel('Spectral Velocity (' + r'cm/s' + ')')
    elif Resp_type == 'PSV':
        ax.plot(T, RS * 100, color='blue', lw=2) # Convert to cm/s
        ax.set_ylabel('Pseudo-Spectral Velocity (' + r'cm/s' + ')')
    elif Resp_type == 'SD':
        ax.plot(T, RS * 100, color='blue', lw=2) # Convert to cm/s
        ax.set_ylabel('Spectral Displacement (' + r'cm' + ')')

    ax.grid(which='both', color='lightgray')
    ax.set_xlim(0, )
    ax.set_ylim(0,)
    st.pyplot(fig)
    plt.close(fig)

#---------------------------------------------------------------------------------------------------
# Multiple Response Spectrum Plotting
#---------------------------------------------------------------------------------------------------

def plot_multiple_response_spectra(spectra_params, units):
    """
    Plot multiple response spectra on a single figure.

    Parameters
    ----------
    spectra_params : list of dicts
        Each dict must have:
        {
            "file": uploaded_file,
            "name": str (filename),
            "Tmin": float,
            "Tmax": float,
            "xi": float,
            "Resp_type": str ("SA", "PSA", "SV", "PSV", "SD")
        }
    units : str
        Acceleration units: "m/s²", "g", or "cm/s²"
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for params in spectra_params:
        filename = params["name"]
        Tmax, xi, Resp_type = params["Tmax"], params["xi"], params["Resp_type"]
        T = T =np.concatenate((np.array([0.01]), np.arange(0.05, Tmax, 0.05))) 

        # Scaling factor
        if units == "g":
            scaling = 9.81
        elif units == "m/s²":
            scaling = 1
        elif units == "cm/s²":
            scaling = 100

        # Read .AT2 file
        accel, dt = read_AT2_v2(params["file"], scaling)

        # Baseline correction
        gmtime = np.linspace(dt, dt * len(accel), len(accel))
        accel, vel, dis = baseline_correction(gmtime, accel)

        # Compute response spectrum
        delta = 1 / dt
        RS = RS_function(accel, delta, T, xi, Resp_type)

        # Plot depending on response type
        if Resp_type in ["SA", "PSA"]:
            ax.plot(T, RS / 9.81, lw=2, label=f"{filename} ({Resp_type}, ξ={xi})")
            ax.set_ylabel("Response Acceleration, g")
        elif Resp_type in ["SV", "PSV"]:
            ax.plot(T, RS * 100, lw=2, label=f"{filename} ({Resp_type}, ξ={xi})")
            ax.set_ylabel("Response Velocity, cm/s")
        elif Resp_type == "SD":
            ax.plot(T, RS * 100, lw=2, label=f"{filename} ({Resp_type}, ξ={xi})")
            ax.set_ylabel("Response Displacement, cm")

    # Labels (generic)
    ax.set_xlabel("Period (s)")
    
    ax.grid(which="both", color="lightgray")
    ax.legend()
    ax.set_xlim(0, )
    ax.set_ylim(0,)
    st.pyplot(fig)
    plt.close(fig)

#===================================================================================================
# Streamlit code
#===================================================================================================

st.title("Response Spectra from .AT2 Files")
st.write("Upload a .AT2 file to explore the data and obtain parameters")

col1, col2, col3 = st.columns(3)

with col1:
    uploaded_file = st.file_uploader("Choose a file", type=["AT2"])

with col2:
    units = st.selectbox("Select units:", ["m/s²", "g", "cm/s²"])
with col3:
	scale_accel = st.number_input(f"Acceleration Scale", value=1.00, step=0.05)
	
if uploaded_file is not None:
    filename = uploaded_file.name
    st.success(f"✅ File uploaded: {filename}. BaseLine correction applied.")

    if units == "g":
        scaling = 9.81
    elif units == "m/s²":
        scaling = 1
    elif units == "cm/s²":
        scaling = 100

    # Handle .AT2 files
    if filename.endswith(".AT2"):
        accel, dt = read_AT2_v2(uploaded_file, scaling)
        accel = accel * scale_accel

    # Plot the array of acceleration
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        Tmax = st.number_input("T max (s)", value=4.05, step=0.05)
    with col2:
        xi = st.number_input("Damping ξ", value=0.05, step=0.01, format="%.2f")
    with col3:
        Resp_type = st.selectbox(
            "Response type",
            ["SA", "PSA", "SV", "PSV", "SD"],
            index=0
        )

    # Compute T array
    #T = np.concatenate((np.arange(Tmin, 1, .01), np.arange(1, Tmax, .1)))
    T =np.concatenate((np.arange(0.01, 0.05, .01), np.arange(0.05, Tmax, 0.05))) 
    plot_response_spectrum(accel, dt, T, xi, Resp_type)

# Comparing multiple files
st.header("Compare Multiple Response Spectra")
col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader("Choose multiple files", type=["AT2"], accept_multiple_files=True)

with col2:
    units_multiple = st.selectbox("Select units: ", ["m/s²", "g", "cm/s²"])

spectra_params = []

if uploaded_files is not None:
    for i, uploaded_file_i in enumerate(uploaded_files):
        st.markdown(f"### File {i+1}: {uploaded_file_i.name}")

        # Parameters for this file
        col1, col2, col3= st.columns([1, 1, 1])
        with col1:
            Tmax_i = st.number_input(f"T max (s)", value=4.05, step=0.05, key=f"Tmax_{i}")
        with col2:
            xi_i = st.number_input(f"Damping ξ", value=0.05, step=0.01, format="%.2f", key=f"xi_{i}")
        with col3:
            Resp_type_i = st.selectbox(
                f"Response type",
                ["SA", "PSA", "SV", "PSV", "SD"],
                index=0,
                key=f"Resp_{i}"
            )

        spectra_params.append({
            "file": uploaded_file_i,
            "name": uploaded_file_i.name,
            "Tmax": Tmax_i,
            "xi": xi_i,
            "Resp_type": Resp_type_i
        })

if spectra_params:
    plot_multiple_response_spectra(spectra_params, units_multiple)
