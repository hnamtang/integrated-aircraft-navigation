#
#   Integrated Aircraft Navigation
#
#   Project Template
#
#   Department for Flight Guidance and Air Traffic
#   Institute of Aeronautics and Astronautics
#   TU Berlin
#

import sys

sys.path.insert(
    1, "../template_and_functions"
)  # add path to custom functions for import

import os

data_path = os.path.abspath("../data")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from ecef2enu import ecef2enu
from ecef2llh import ecef2llh
from ephcal import ephcal
from gnss_routines import compute_pos_ecef, compute_svpos_svclock_compensate
from llh2ecef import llh2ecef

plt.close("all")

# GPS value for speed of light
V_LIGHT = 299792458.0

# --------------------------------------------------------
# Load Inertial Data
# --------------------------------------------------------
mat = io.loadmat(os.path.join(data_path, "ins_proj.mat"))
t_ins = mat["t_ins"]
r_llh_ins = mat["r_llh_ins"]
rpy_ins = mat["rpy_ins"]

# Compute the INS positions in ECEF
r_ecef_ins = np.zeros((3, len(t_ins)))
for jj in range(0, len(t_ins)):

    # Compute INS position in ECEF
    r_ecef_ins[:, [jj]] = llh2ecef(r_llh_ins[:, [jj]])


# --------------------------------------------------------
# Load GNSS Data
# --------------------------------------------------------
mat = io.loadmat(os.path.join(data_path, "gps_proj_scen1.mat"))
t_gps = mat["t_gps"]
svid_gps = mat["svid_gps"]
pr_gps = mat["pr_gps"]
adr_gps = mat["adr_gps"]
ephem = mat["ephem"]

# --------------------------------------------------------
# Load GNSS Reference Data
# --------------------------------------------------------
mat = io.loadmat(os.path.join(data_path, "gps_ref.mat"))
r_ref_gps = mat["r_ref_gps"]


# -----------------------------------------------------------------------
# Initialize the Kalman Filter - Loose Coupling
# -----------------------------------------------------------------------

# --- INSERT YOUR CODE ---

# --------------------------------------------------------
# Output variables
# --------------------------------------------------------

# Length of the time array
N = len(t_gps)

# Output arrays
# gnsstime = np.array([])
gnsstime = []
lat = np.array([])
lon = np.array([])
h = np.array([])
llh_all = np.zeros((4, 0))

numsvused = np.zeros((N, 1))

r_ecef_gps = np.zeros((3, len(t_ins)))
r_llh_gps = np.zeros((3, len(t_ins)))
r_ecef_ins_corrected = np.zeros((3, len(t_ins)))
r_llh_ins_corrected = np.zeros((3, len(t_ins)))


# --------------------------------------------------------
# Go through all GPS data and compute trajectory
# --------------------------------------------------------
ii = 0
while ii < N:

    # Extract the GPS measurment data for this time epoch
    gpstime = t_gps[ii, 0]
    index_gps = (t_gps[:, 0] == gpstime).nonzero()[0]
    svid = np.int32(svid_gps[index_gps, 0])
    pr = pr_gps[index_gps, 0]

    # Find the corresponding INS measurment
    index_ins = (t_ins[:, 0] == gpstime).nonzero()[0]

    (sv_ecef, sv_clock) = compute_svpos_svclock_compensate(gpstime, svid, pr, ephem)

    # if svid.size >= 4:
    if len(svid) >= 4:
        (r_ecef_gps[:, index_ins], user_clock) = compute_pos_ecef(
            gpstime, pr, sv_ecef, sv_clock
        )

        # gnsstime = np.append(gnsstime, gpstime)
        gnsstime.append(gpstime)
        r_llh_gps[:, index_ins] = ecef2llh(r_ecef_gps[:, index_ins])

    else:
        print("Not enough satellites at time = %f" % (gpstime))

    # Compensate the pseuforanges for the staellite clock errors
    pr = pr + sv_clock * V_LIGHT

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

gnsstime = np.asarray(gnsstime, dtype=t_gps.dtype)

# --------------------------------------------------------
# Output
# --------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(r_llh_ins[1, :] * 180.0 / np.pi, r_llh_ins[0, :] * 180.0 / np.pi, "b")
ax.plot(r_llh_gps[1, :] * 180.0 / np.pi, r_llh_gps[0, :] * 180.0 / np.pi, "r")
ax.legend(["INS", "GPS"])
ax.grid()
ax.set_title("Ground Tracks with Inertial and GPS")
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.axis("equal")


plt.tight_layout()
plt.show()
