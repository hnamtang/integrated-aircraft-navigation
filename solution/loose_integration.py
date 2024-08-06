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

data_path = os.path.abspath("../data")  # path to measurement data

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
N_INS = len(t_ins)
N_STATES = 6


# Initialize measurement matrix
H = np.zeros((3, 6))
H[0, 0] = 1
H[1, 2] = 1
H[2, 4] = 1

# Setup state transition matrix
# State vector is INS error vector
N_STATES = 6  # dx, dx_dot, dy, dy_dot, dz, dz_dot, db, db_dot (b: user clock error)
Phi = np.eye(N_STATES)
Phi[[0, 2, 4], [1, 3, 5]] = dt_ins

sigma_w = 100

    # Setup process noise
Q = np.zeros(shape=(N_STATES, N_STATES), dtype=float)
q = (sigma_w**2) * np.array(
        [[dt_ins**3 / 3.0, dt_ins**2 / 2.0], [dt_ins**2 / 2.0, dt_ins]]
    )
Q[:2, :2] = Q[2:4, 2:4] = Q[4:6, 4:6] = q

I = np.eye(N_STATES)

# Initialize error covariance
r_unc = 1000
v_unc = 100
b_unc = 1e-1

# Initialize error covariance
# P_pre = (100**2) * np.eye(N_STATES)
P_pre = np.diag(
        [r_unc**2, v_unc**2, r_unc**2, v_unc**2, r_unc**2, v_unc**2]
    )

x_est = np.zeros((6))
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

residuals = []

r_ecef_gps = np.zeros((3, len(t_ins)))
r_llh_gps = np.zeros((3, len(t_ins)))
r_enu_gps = np.zeros_like(r_ecef_gps)

r_ecef_ins_corrected = np.zeros((3, len(t_ins)))
r_llh_ins_corrected = np.zeros((3, len(t_ins)))
r_enu_ins_corrected = np.zeros_like(r_ecef_ins_corrected)

R = (0.5**2) * np.eye(3)
x_pre = x_est
r_hist = np.zeros((3, len(t_ins)), dtype=float)
r_ins = r_ecef_ins
# --------------------------------------------------------
# Go through all GPS data and compute trajectory
# --------------------------------------------------------

# Initialize innovation vector
#resi = np.zeros(shape=(N_SV_REQUIRED, N_INS), dtype=float)

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
        r_enu_gps[:, index_ins] = ecef2enu(
            r_ecef_gps[:, index_ins],  # ECEF coordinates of points to be converted
            r_ecef_gps[:, [0]],  # ENU origin in ECEF
            r_llh_gps[:, [0]],  # ENU origin in LLH
        )

    else:
        print("Not enough satellites at time = %f" % (gpstime))
        r_ecef_gps[:, index_ins[0]] = np.NaN
        r_llh_gps[:, index_ins[0]] = np.NaN

    # --------------------------------------------------------
    # Kalman Filter iteration
    # --------------------------------------------------------
    if ii != 0:
        dz = np.full((3), np.NaN)
        if len(svid) >= 4:
            z = r_ecef_ins[:, index_ins[0]] - r_ecef_gps[:, index_ins[0]]

            K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + R)

            dz = z - H @ x_pre

            x_est = x_pre + K @ dz

            P_est = (I - K @ H) @ P_pre

            x_pre = Phi @ x_est

            P_pre = Phi @ P_est @ Phi.T + Q
        else:
            H_deadrec = np.zeros((3, 6))
            H_deadrec[0, 1] = 1
            H_deadrec[1, 3] = 1
            H_deadrec[2, 5] = 1

            sigma_ins = 9.81/1000 * (t_gps[ii] - t_gps[0])**2
            R_ins = sigma_ins * np.eye(3)

            z = -(r_ecef_ins[:, index_ins[0]] - r_ecef_ins[:, index_ins[0]-1]) / dt_ins

            K = P_pre @ H_deadrec.T @ np.linalg.inv(H_deadrec @ P_pre @ H_deadrec.T + R_ins)

            dz = z - H_deadrec @ x_pre

            x_est = x_pre + K @ dz
            P_est = (I - K @ H_deadrec) @ P_pre
            x_pre = Phi @ x_est
            P_pre = Phi @ P_est @ Phi.T + Q

            dz[:] = np.NaN

        residuals.append(dz)
    else:
        x_pre[::2] = r_ecef_ins[:,index_ins[0]] - r_ecef_gps[:,index_ins[0]]
        x_est = x_pre
        residuals.append(np.full(3, np.NaN))

    # --------------------------------------------------------
    # Store information for plotting
    # --------------------------------------------------------
    r_ecef_ins_corrected[:, index_ins[0]] = r_ecef_ins[:, index_ins[0]] - x_est[::2]
    r_llh_ins_corrected[:, index_ins] = ecef2llh(r_ecef_ins_corrected[:, index_ins])
    r_enu_ins_corrected[:, index_ins] = ecef2enu(r_ecef_ins_corrected[:, index_ins], r_ecef_ins_corrected[:, [0]], r_llh_ins_corrected[:, [0]])

    # Update the index
    ii = index_gps[-1] + 1

gnsstime = np.asarray(gnsstime, dtype=t_gps.dtype)

# --------------------------------------------------------
# Output
# --------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(r_llh_ins[1, :] * 180.0 / np.pi, r_llh_ins[0, :] * 180.0 / np.pi, "b")
ax.plot(r_llh_gps[1, :] * 180.0 / np.pi, r_llh_gps[0, :] * 180.0 / np.pi, "r")
ax.plot(r_llh_ins_corrected[1, :] * 180.0 / np.pi, r_llh_ins_corrected[0, :] * 180.0 / np.pi, linestyle="--", color="g")
ax.legend(["INS", "GPS", "Loose Coopling"])
ax.grid()
ax.set_title("Ground Tracks with Inertial and GPS")
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.axis("equal")


cvtlat = 1852 * 60
cvtlon = 1852 * 60 * np.cos(39 * np.pi / 180)

plt.tight_layout()

# Height vs. elapsed time plot
fig, ax2 = plt.subplots()
ax2.plot((t_ins - t_ins[0]), r_llh_ins[2, :], linewidth=1.5, linestyle="-", color="b", label="INS")
ax2.plot((t_ins - t_ins[0]), r_llh_gps[2, :], linewidth=1.5, linestyle="-", color="r", label="GPS")
ax2.plot(
        (t_ins - t_ins[0]),
        r_llh_ins_corrected[2, :],
        # linewidth=3,
        linewidth=1.5,
        linestyle="--",
        color="g",
        label="INS/GNSS Loose Coupling",
    )
ax2.legend(loc="lower right")
ax2.grid()
ax2.set_title("Height with Inertial and GPS")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Height [m]")
plt.tight_layout()

# Error plot
# TODO: compute and include the covariance in this plot
fig, ax4 = plt.subplots()
ax4.plot(
    (t_ins - t_ins[0]),
    (r_enu_ins_corrected - r_enu_gps).T,
    linewidth=1.5,
    linestyle="-",
)
ax4.legend(["East", "North", "Up"], loc="lower right")
ax4.grid()
ax4.set_title("Difference between GPS and INS/GPS Solution")
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("Error in ENU [m]")
plt.tight_layout()

fig, ax5 = plt.subplots(3, 1, sharex=True)
ax5[0].plot((t_ins - t_ins[0])[5:],
            [np.linalg.norm(x) for x in residuals[5:]])
ax5[0].set_title("Residual Norm [m]")
ax5[0].grid()

res_enu = np.zeros((len(residuals), 3))
for i in range(len(residuals)):
    res_enu[i, :] = r_enu_ins_corrected[:, i] @ residuals[i]

ax5[1].plot(t_ins - t_ins[0],
            [np.linalg.norm(x[:2]) for x in np.rollaxis(res_enu, 0)])
ax5[1].set_title("Norm Horizontal residual [m]")
ax5[1].grid()

ax5[2].plot(t_ins - t_ins[0], res_enu[:, 2])
ax5[2].set_title("Vertical residual [m]")
ax5[2].grid()

# Benchmark
_, ax6 = plt.subplots(nrows=3, ncols=1, sharex=True)
ax6[0].plot(
        (t_ins - t_ins[0]),
        r_llh_ins_corrected[1, :] - r_ref_gps[1, :],
        linewidth=1.5,
        color="b",
        label="longitude",
    )
ax6[0].grid()
ax6[0].set_title("Position Estimation Error")
ax6[0].set_ylabel("Longitude [deg]")
ax6[1].plot(
        (t_ins - t_ins[0]),
        r_llh_ins_corrected[0, :] - r_ref_gps[0, :],
        linewidth=1.5,
        color="b",
        label="latitude",
    )
ax6[1].grid()
ax6[1].set_ylabel("Latitude [deg]")
ax6[2].plot(
        (t_ins - t_ins[0]),
        r_llh_ins_corrected[2, :] - r_ref_gps[2, :],
        linewidth=1.5,
        color="b",
        label="height",
    )
ax6[2].grid()
ax6[2].set_xlabel("Time [s]")
ax6[2].set_ylabel("Height [m]")


plt.show()
