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
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..', 'template_and_functions')
sys.path.insert(0, parent_dir)

#sys.path.insert(
#    1, "../template_and_functions"
#)  # add path to custom functions for import

data_path = os.path.abspath("../integrated-aircraft-navigation_losse_coupling/data")  # path to measurement data

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from ecef2enu import ecef2enu
from ecef2llh import ecef2llh
from ephcal import ephcal
from gnss_routines import compute_pos_ecef, compute_svpos_svclock_compensate
from llh2ecef import llh2ecef

plt.close("all")

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
# Load GNSS Reference Data
# --------------------------------------------------------
mat = io.loadmat(os.path.join(data_path, "gps_ref.mat"))
r_ref_gps = mat["r_ref_gps"]


# --------------------------------------------------------
# Load GNSS Data
# --------------------------------------------------------
mat = io.loadmat(os.path.join(data_path, "gps_proj_scen1.mat"))
t_gps = mat["t_gps"]
svid_gps = mat["svid_gps"]
pr_gps = mat["pr_gps"]
adr_gps = mat["adr_gps"]
ephem = mat["ephem"]


# -----------------------------------------------------------------------
# Initialize the Kalman Filter - Loose Coupling
# -----------------------------------------------------------------------
v_ins = np.array([[0], [-100]])
b_ins = np.zeros((2, 1))
#Time Interval
dt_ins = t_ins[1, 0] - t_ins[0, 0]

N_STATES = 8

# required number of available satellites
# 3 SVs for position estimation + 1 SV for clock error + 1 SV for integrity check
N_SV_REQUIRED = 3

# Initialize measurement matrix
H = np.zeros(shape=(N_SV_REQUIRED, N_STATES), dtype=float)
H[:, -2] = 1.0

# Setup state transition matrix
# State vector is INS error vector
N_STATES = 8  # dx, dx_dot, dy, dy_dot, dz, dz_dot, db, db_dot (b: user clock error)
Phi = np.eye(N_STATES, dtype=float)
Phi[[0, 2, 4, 6], [1, 3, 5, 7]] = dt_ins

sigma_w = 100

    # Setup process noise
Q = np.zeros(shape=(N_STATES, N_STATES), dtype=float)
q = (sigma_w**2) * np.array(
        [[dt_ins**3 / 3.0, dt_ins**2 / 2.0], [dt_ins**2 / 2.0, dt_ins]]
    )
Q[:2, :2] = Q[2:4, 2:4] = Q[4:6, 4:6] = q
Sf = 2e-19 / 2.0  # approximation (Lecture 5, p. 59)
Sg = 2 * np.pi**2 * 2e-20  # approximation (Lecture 5, p. 59)
Q[6:, 6:] = np.array(
        [
            [Sf * dt_ins + Sg * dt_ins**3 / 3.0, Sg * dt_ins**2 / 2.0],
            [Sg * dt_ins**2 / 2.0, Sg * dt_ins],
        ]
    )

I = np.eye(N_STATES)

# Initialize error covariance
r_unc = 1000
v_unc = 100
b_unc = 1e-1

# Initialize error covariance
# P_pre = (100**2) * np.eye(N_STATES)
P_pre = np.diag(
        [r_unc**2, v_unc**2, r_unc**2, v_unc**2, r_unc**2, v_unc**2, b_unc**2, b_unc**2]
    )

x_est = np.zeros(shape=(N_STATES, len(t_ins)), dtype=float)
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

r_ecef_gps = np.zeros((3, len(t_ins)))
r_llh_gps = np.zeros((3, len(t_ins)))
r_ecef_ins_corrected = np.zeros((3, len(t_ins)))
r_llh_ins_corrected = np.zeros((3, len(t_ins)))

R = (0.5**2) * np.eye(N_SV_REQUIRED)
x_pre = x_est
r_hist = np.zeros((3, len(t_ins)), dtype=float)
r_ins = r_ecef_ins
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

    # --------------------------------------------------------
    # Kalman Filter iteration
    # --------------------------------------------------------

    # --- INSERT YOUR CODE ---

    z = r_ins - r_ecef_gps[:N_SV_REQUIRED, [ii]]

    K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + R)

    z_pred = H @ x_pre
    x_est = x_pre + K @ (z - z_pred)
    
    r_ins = r_ins + x_est[:3]
    b_ins = b_ins + x_est[6:]

    x_est = np.zeros(shape=(N_STATES, len(t_ins)), dtype=float)

    P_est = (np.eye(N_STATES) - K @ H) @ P_pre

    x_pre = Phi @ x_est

    P_pre = Phi @ P_est @ Phi.T + Q

    r_ecef_ins_corrected[:, index_ins] = (
        r_ecef_ins[:, index_ins] - x_est[[[0, 2, 4]], index_ins].T
        )
    # --------------------------------------------------------
    # Store information for plotting
    # --------------------------------------------------------
    r_llh_ins_corrected[:, index_ins] = ecef2llh(r_ecef_ins_corrected[:, index_ins])


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


cvtlat = 1852 * 60
cvtlon = 1852 * 60 * np.cos(39 * np.pi / 180)


plt.tight_layout()
plt.show()
