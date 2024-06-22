#
#   Integrated Aircraft Navigation
#
#   Project Template 
#
#   Department for Flight Guidance and Air Traffic
#   Institute of Aeronautics and Astronautics
#   TU Berlin
#

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from ephcal import *
from llh2ecef import *
from ecef2llh import *
from ecef2enu import *
from gnss_routines import compute_svpos_svclock_compensate
from gnss_routines import compute_pos_ecef


plt.close('all')

# GPS value for speed of light
vlight   = 299792458.0    

# --------------------------------------------------------
# Load Inertial Data
# --------------------------------------------------------
mat = io.loadmat("ins_proj.mat")
t_ins = mat["t_ins"]
r_llh_ins = mat["r_llh_ins"]
rpy_ins = mat["rpy_ins"]

# Compute the INS positions in ECEF
r_ecef_ins = np.zeros((3, len(t_ins)))
for jj in range(0, len(t_ins)) :
    
    # Compute INS position in ECEF
    r_ecef_ins[:,[jj]] = llh2ecef(r_llh_ins[:,[jj]]);


# --------------------------------------------------------
# Load GNSS Data
# --------------------------------------------------------
mat = io.loadmat("gps_proj_scen1.mat")
t_gps = mat["t_gps"]
svid_gps = mat["svid_gps"]
pr_gps = mat["pr_gps"]
adr_gps = mat["adr_gps"]
ephem = mat["ephem"]

# --------------------------------------------------------
# Load GNSS Reference Data
# --------------------------------------------------------
mat = io.loadmat("gps_ref.mat")
r_ref_gps = mat["r_ref_gps"]


# -----------------------------------------------------------------------
# Initialize the Kalman Filter - Loose Coupling
# -----------------------------------------------------------------------

# --- INSERT YOUR CODE ---

# --------------------------------------------------------
# Output variables
# --------------------------------------------------------

# Length of the time array
N = t_gps.shape[0]

# Output arrays
gnsstime = np.array([])
lat = np.array([])
lon = np.array([])
h = np.array([]) 
llh_all = np.zeros((4,0))

numsvused = np.zeros((N,1));

r_ecef_gps =  np.zeros((3,len(t_ins)))
r_llh_gps =  np.zeros((3,len(t_ins)))
r_ecef_ins_corrected =  np.zeros((3,len(t_ins)))
r_llh_ins_corrected =  np.zeros((3,len(t_ins)))


# --------------------------------------------------------
# Go through all GPS data and compute trajectory
# --------------------------------------------------------
ii = 0
while ii < N :
    
    # Extract the GPS measurment data for this time epoch
    gpstime = t_gps[ii,0]
    index_gps = (t_gps[:,0] == gpstime).nonzero()[0]
    svid = np.int32(svid_gps[index_gps,0])
    pr = (pr_gps[index_gps,0])
    
    # Find the corresponding INS measurment
    index_ins = (t_ins[:,0] == gpstime).nonzero()[0]
    
    
    (sv_ecef, sv_clock) = compute_svpos_svclock_compensate( gpstime, svid, pr, ephem )
    
    if svid.size >= 4 :
        (r_ecef_gps[:,index_ins], user_clock) = compute_pos_ecef( gpstime, pr, sv_ecef, sv_clock )
            
        gnsstime = np.append(gnsstime,gpstime)
        r_llh_gps[:,index_ins] =  ecef2llh(r_ecef_gps[:,index_ins])

    else :
        print('Not enough satellites at time = %f' % (gpstime))
    
    # Compensate the pseuforanges for the staellite clock errors
    pr = pr + sv_clock*vlight    
     
    # --------------------------------------------------------------
    #   Tight coupling
    # --------------------------------------------------------------
    
    
    # --- INSERT YOUR CODE ---
    
      
    # --------------------------------------------------------
    # Store information for plotting
    # --------------------------------------------------------
   
    # --- INSERT YOUR CODE ---
    
    
    # Update the index
    ii = index_gps[-1] + 1
   

    
# --------------------------------------------------------
# Output
# --------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(r_llh_ins[1, :]*180.0/np.pi, r_llh_ins[0, :]*180.0/np.pi, 'b')
ax.plot(r_llh_gps[1, :]*180.0/np.pi, r_llh_gps[0, :]*180.0/np.pi, 'r')
ax.legend(['INS', 'GPS'])
ax.grid()
ax.set_title('Ground Tracks with Inertial and GPS')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.axis("equal")


plt.tight_layout();
plt.show()



