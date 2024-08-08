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

data_path = os.path.abspath("../data")  # path to measurement data

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from scipy.stats import chi2
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
dt = 1

N_STATES = 8  # dx, dx_dot, dy, dy_dot, dz, dz_dot, dt, dt_dot (b: user clock error)
Phi = np.eye(N_STATES)
Phi[[0, 2, 4, 6], [1, 3, 5, 7]] = dt

# --------------------------------------------------------
# Output variables
# --------------------------------------------------------

# Length of the time array
N = len(t_gps)
residuals = []
Ss = []
Ts = []
svid_ban = []
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
r_enu_ins_corrected = np.zeros((3, len(t_ins)))

sigma_q = 1

# Setup process noise
Q = np.zeros(shape=(N_STATES, N_STATES), dtype=float)
q = (sigma_q ** 2) * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
Q[:2, :2] = Q[2:4, 2:4] = Q[4:6, 4:6] = q
Sf = 2e-19 / 2.0  # approximation (Lecture 5, p. 59)
Sg = 2 * np.pi ** 2 * 2e-20  # approximation (Lecture 5, p. 59)
Q[6:, 6:] = np.array([[Sf * dt + Sg * dt ** 3 / 3.0, Sg * dt ** 2 / 2.0],[Sg * dt ** 2 / 2.0, Sg * dt],])

I = np.eye(N_STATES)

r_unc = 1000
v_unc = 100
b_unc = 1e-1

x_est = np.zeros((8))
x_pre = x_est
P_pre = np.eye(8) * 10
#P_pre = np.diag([r_unc ** 2, v_unc ** 2, r_unc ** 2, v_unc ** 2, r_unc ** 2, v_unc ** 2, b_unc ** 2, b_unc ** 2])
# --------------------------------------------------------
# Go through all GPS data and compute trajectory
# --------------------------------------------------------
ii = 0
j = 0
while ii < N:

    # Extract the GPS measurment data for this time epoch
    gpstime = t_gps[ii, 0]
    index_gps = (t_gps[:, 0] == gpstime).nonzero()[0]
    svid = np.int32(svid_gps[index_gps, 0])
    for s in svid_ban:
        index_gps = index_gps[svid != s]
        svid = svid[svid != s]

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
        r_ecef_gps[:, index_ins[0]] = np.NaN
        r_llh_gps[:, index_ins[0]] = np.NaN

    # Compensate the pseudoranges for the satellite clock errors
    pr = pr + sv_clock * V_LIGHT

    # --------------------------------------------------------------
    #   Tight coupling
    # --------------------------------------------------------------

    if(ii != 0):
        while True:
            sv_vectors = np.zeros((svid.size, 3))
            pr_ins = np.zeros_like(pr)

            for i, sv_pos in enumerate(np.rollaxis(sv_ecef, 1)):
                sv_vectors[i, :] = r_ecef_ins[:, i] - sv_pos
                pr_ins[i] = np.linalg.norm(sv_vectors[i, :])

            H = np.zeros((svid.size, 8))
            H[:, 0] = sv_vectors[:, 0] / pr_ins
            H[:, 2] = sv_vectors[:, 1] / pr_ins
            H[:, 4] = sv_vectors[:, 2] / pr_ins
            H[:, 6] = 1

            R = np.identity(svid.size) * 4

            z = pr_ins - pr

            dz = z - (H @ x_pre)
            print(dz)
            S_k = H@P_pre@H.T + R
            S_inv = np.linalg.inv(S_k)

            P_FA = 1e-5
            P_MD = 1e-3
            U = np.linalg.cholesky(S_inv)

            s = U @ dz
            s_sq = s.dot(s)

            T = chi2.ppf(1 - P_FA, len(z))

            settling_time = 1000000
            if(s_sq > T and j > settling_time):
                print(f"Loss of Integrity at timestamp {j}: {s_sq=}/{T=}")
                print(s)
                index = np.argmax(np.abs(s))
                svid_ban.append(svid[index])
                print(f"Ban satellite {svid_ban} ({index}th available) with norm error of {np.max(np.abs(s))}")
                pr = np.delete(pr, index)
                svid = np.delete(svid, index)
                sv_ecef = np.delete(sv_ecef, index, axis=1)
                continue
            break

        K = P_pre @ H.T @ S_inv

        x_est = x_pre + K @ dz

        P_est = (I - K @ H) @ P_pre

        x_pre = Phi @ x_est
        P_pre = Phi @ P_est @ Phi.T + Q

        residuals.append(dz)
        Ss.append(s_sq)
        Ts.append(T)

    else:
        x_pre[:6:2] = 0
        x_est = x_pre
        residuals.append(np.full(3, np.NaN))
        Ss.append(np.NaN)
        Ts.append(np.NaN)


    # --------------------------------------------------------
    # Store information for plotting
    # --------------------------------------------------------

    r_ecef_ins_corrected[:, index_ins[0]] = r_ecef_ins[:, index_ins[0]] - x_est[:6:2]
    r_llh_ins_corrected[:, index_ins] = ecef2llh(r_ecef_ins_corrected[:, index_ins])
    r_enu_ins_corrected[:, index_ins] = ecef2enu(r_ecef_ins_corrected[:, index_ins], r_ecef_ins_corrected[:, [0]],
                                                 r_llh_ins_corrected[:, [0]])

    # Update the index
    ii = ((t_gps[:, 0] == gpstime).nonzero()[0])[-1] + 1
    j += 1

gnsstime = np.asarray(gnsstime, dtype=t_gps.dtype)

# --------------------------------------------------------
# Output
# --------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(r_llh_ins[1, :] * 180.0 / np.pi, r_llh_ins[0, :] * 180.0 / np.pi, "b")
ax.plot(r_llh_gps[1, :] * 180.0 / np.pi, r_llh_gps[0, :] * 180.0 / np.pi, "r")
ax.plot(r_llh_ins_corrected[1, :] * 180.0 / np.pi, r_llh_ins_corrected[0, :] * 180.0 / np.pi, linestyle="--", color="g")
ax.legend(["INS", "GPS", "Thight Integration"])
ax.grid()
ax.set_title("Ground Tracks with Inertial and GPS")
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.axis("equal")


plt.tight_layout()
plt.show()
