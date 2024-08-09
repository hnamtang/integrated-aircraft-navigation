#
#   Integrated Aircraft Navigation
#
#   INS/GNSS Tight Integration
#
#   Department for Flight Guidance and Air Traffic
#   Institute of Aeronautics and Astronautics
#   TU Berlin
#

import sys

import numpy as np

sys.path.insert(
    1, "../template_and_functions"
)  # add custom function path to PYTHONPATH for import

from ecef2enu import ecef2enu
from ecef2llh import ecef2llh
from ephcal import ephcal
from gnss_routines import compute_pos_ecef, compute_svpos_svclock_compensate
from llh2ecef import llh2ecef

from read_measurement_data import read_measurement_data
from show_plots import show_plots

# GPS value for speed of light
V_LIGHT = 299792458.0


# --------------------------------------------------------
# Load measurement data
# --------------------------------------------------------

# Inertial data
t_ins, r_llh_ins, r_ecef_ins, N_INS, dt_ins = read_measurement_data("ins_proj.mat")

# GNSS data
t_gps, svid_gps, pr_gps, adr_gps, ephem, N_GPS = read_measurement_data(
    "gps_proj_scen1.mat"
)
# t_gps, svid_gps, pr_gps, adr_gps, ephem, N_GPS = read_measurement_data(
#    "gps_proj_scen2.mat"
# )
# t_gps, svid_gps, pr_gps, adr_gps, ephem, N_GPS = read_measurement_data("gps_proj_scen3.mat")

# GNSS reference data
r_ref_gps = read_measurement_data("gps_ref.mat")


# --------------------------------------------------------
# IMU data
# --------------------------------------------------------

# Gyroscope
b_gyro = 1.0 / 3600  # [deg / s]
ARW = 0.5 / 60  # [deg / sqrt(s)]
sigma_gyro = ARW * np.sqrt(dt_ins)

# Accelerometer
b_accel = 9.80665 * 1e-3  # [m / s**2]
VRW = 0.2 / 60  # [(m/s) / sqrt(s)]
sigma_vel = VRW * np.sqrt(dt_ins)


# --------------------------------------------------------
# Initialize Kalman Filter - Tight Coupling
# --------------------------------------------------------

# Setup state transition matrix
# State vector is INS error vector
N_STATES = 8  # dx, dx_dot, dy, dy_dot, dz, dz_dot, db, db_dot (b: user clock error)
Phi = np.eye(N_STATES)
Phi[[0, 2, 4, 6], [1, 3, 5, 7]] = dt_ins


# Setup process noise
# TODO: What is the reasonable value for sigma_w?
sigma_w = 50.0
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

# Initialize error vector
x_pre = np.zeros(shape=(N_STATES, 1), dtype=float)

# Initialize error covariance
# TODO: What are the reasonable values for r_unc, v_unc, b_unc?
r_unc = 100
v_unc = 10
b_unc = 1e-2
P_pre = np.diag(
    [r_unc**2, v_unc**2, r_unc**2, v_unc**2, r_unc**2, v_unc**2, b_unc**2, b_unc**2]
)

x_est = np.zeros(shape=(N_STATES, N_INS), dtype=float)


# --------------------------------------------------------
# Output variables
# --------------------------------------------------------

# Output arrays
gnsstime = np.array([])
# lat = np.array([])
# lon = np.array([])
# h = np.array([])
# llh_all = np.zeros(shape=(4, 0))

numsvused = np.zeros(
    shape=(1, N_INS), dtype=int
)  # storage for number of available satellites

r_ecef_gps = np.nan * np.ones(shape=(3, N_INS))
r_llh_gps = np.nan * np.ones_like(r_ecef_gps)
r_enu_gps = np.nan * np.ones_like(r_ecef_gps)

r_ecef_ins_corrected = np.zeros(shape=(3, N_INS))
r_llh_ins_corrected = np.zeros_like(r_ecef_ins_corrected)
r_enu_ins_corrected = np.zeros_like(r_ecef_ins_corrected)


# --------------------------------------------------------
# Go through all GPS data and compute trajectory
# --------------------------------------------------------

# Required number of available satellites
N_SV_REQUIRED = 4

# Initialize innovation vector
resi = np.zeros(shape=(N_SV_REQUIRED, N_INS))
resi_normalized = np.zeros(shape=(1, N_INS))

# Iteration
ii = 0
while ii < N_GPS:
    # Extract GPS measurement data for this time epoch
    gpstime = t_gps[ii, 0]
    index_gps = (t_gps[:, 0] == gpstime).nonzero()[0]
    svid = np.int32(
        svid_gps[index_gps, 0]
    )  # available satellites at current time epoch

    pr = pr_gps[index_gps, 0]

    # Find the corresponding INS measurement
    index_ins = (t_ins[:, 0] == gpstime).nonzero()[0]

    numsvused[0, index_ins] = len(index_gps)

    (sv_ecef, sv_clock) = compute_svpos_svclock_compensate(gpstime, svid, pr, ephem)

    # --------------------------------------------------------------
    # Tight coupling
    # --------------------------------------------------------------

    if svid.size >= N_SV_REQUIRED:
        (r_ecef_gps[:, index_ins], user_clock) = compute_pos_ecef(
            gpstime, pr, sv_ecef, sv_clock
        )

        gnsstime = np.append(gnsstime, gpstime)
        r_llh_gps[:, index_ins] = ecef2llh(r_ecef_gps[:, index_ins])
        r_enu_gps[:, index_ins] = ecef2enu(
            r_ecef_gps[:, index_ins],  # ECEF coordinates of points to be converted
            r_ecef_gps[:, [0]],  # ENU origin in ECEF
            r_llh_gps[:, [0]],  # ENU origin in LLH
        )

        # Synthesize ranges from the inertial (INS)
        # and set up measurement matrix and measurement noise
        rho_ins = np.zeros(shape=(N_SV_REQUIRED, 1))

        # sigma_v = 0.6  # Paper DLR
        sigma_v = 5
        R = (sigma_v**2) * np.eye(N_SV_REQUIRED)

        H = np.zeros(shape=(N_SV_REQUIRED, N_STATES), dtype=float)
        H[:, -2] = 1.0

        for kk in range(N_SV_REQUIRED):
            rho_ins[kk, 0] = np.linalg.norm(
                r_ecef_ins[:, index_ins] - sv_ecef[:, [kk]]
            )  # pseudorange from INS position to location of each available SV
            H[kk, [[0, 2, 4]]] = (r_ecef_ins[:, index_ins] - sv_ecef[:, [kk]]).T
            H[kk, [[0, 2, 4]]] /= rho_ins[kk, 0]

        # Compensate the pseudoranges for the satellite clock error
        pr += (sv_clock + x_est[-2, index_ins[0]]) * V_LIGHT
        pr = pr.reshape(numsvused[0, index_ins[0]], 1)
    else:
        print(f"Not enough satellites at time = {gpstime}")

        # Synthesize ranges from the inertial (INS)
        # and set up measurement matrix and measurement noise
        rho_ins = np.zeros(shape=(numsvused[0, index_ins[0]], 1))

        sigma_v = 100.0
        R = (sigma_v**2) * np.eye(numsvused[0, index_ins[0]])

        H = np.zeros(shape=(numsvused[0, index_ins[0]], N_STATES), dtype=float)
        H[:, -2] = 1.0

        for kk in range(numsvused[0, index_ins[0]]):
            rho_ins[kk, 0] = np.linalg.norm(
                r_ecef_ins[:, index_ins] - sv_ecef[:, [kk]]
            )  # pseudorange from INS position to location of each available SV
            H[kk, [[0, 2, 4]]] = (r_ecef_ins[:, index_ins] - sv_ecef[:, [kk]]).T
            H[kk, [[0, 2, 4]]] /= rho_ins[kk, 0]

        # Compensate the pseudoranges for the satellite clock error
        pr += sv_clock * V_LIGHT
        pr = pr.reshape(numsvused[0, index_ins[0]], 1)

    # Form Kalman filter measurement vector
    # Difference between distances caluclated by INS position
    # and GPS pseudorange measurement
    z = rho_ins - pr[:N_SV_REQUIRED, [0]]

    # ---------------------------
    # Update
    # ---------------------------

    # (1) Compute Kalman gain
    cov_resi = H @ P_pre @ H.T + R
    cov_resi_inv = np.linalg.inv(cov_resi)
    K = P_pre @ H.T @ cov_resi_inv

    # Save the innovations/residuals
    if svid.size >= N_SV_REQUIRED:
        resi[:, index_ins] = z - H @ x_pre
        resi_normalized[0, index_ins] = np.sqrt(
            resi[:, index_ins].T @ cov_resi_inv @ resi[:, index_ins]
        )
    else:
        resi[: numsvused[0, index_ins[0]], index_ins] = z - H @ x_pre
        resi[numsvused[0, index_ins[0]] :, index_ins] = np.nan

    resi_normalized[0, index_ins] = np.sqrt(
        resi[: numsvused[0, index_ins[0]], index_ins].T
        @ cov_resi_inv
        @ resi[: numsvused[0, index_ins[0]], index_ins]
    )

    # (2) Update estimate with measurement
    if svid.size >= N_SV_REQUIRED:
        x_est[:, index_ins] = x_pre + K @ resi[:, index_ins]
    else:
        x_est[:, index_ins] = x_pre + K @ resi[: numsvused[0, index_ins[0]], index_ins]

    # (3) Update the error covariance
    # P = (I - K @ H) @ P_pre
    P = (I - K @ H) @ P_pre @ (I - K @ H).T + K @ R @ K.T

    # ---------------------------
    # Predict
    # ---------------------------

    # (4) Project the state ahead
    x_pre = Phi @ x_est[:, index_ins]

    # (5) Project the error covariance ahead
    P_pre = Phi @ P @ Phi.T + Q

    # --------------------------------------------------------------
    # Remove the drift term (CKF)
    # --------------------------------------------------------------

    r_ecef_ins_corrected[:, index_ins] = (
        r_ecef_ins[:, index_ins] - x_est[[[0, 2, 4]], index_ins].T
    )

    # --------------------------------------------------------------
    # Store information for plotting
    # --------------------------------------------------------------

    r_llh_ins_corrected[:, index_ins] = ecef2llh(r_ecef_ins_corrected[:, index_ins])
    r_enu_ins_corrected[:, index_ins] = ecef2enu(
        r_ecef_ins_corrected[:, index_ins],
        r_ecef_gps[:, [0]],  # ENU origin in ECEF
        r_llh_gps[:, [0]],  # ENU origin in LLH
    )

    # Update the index
    ii = index_gps[-1] + 1


# --------------------------------------------------------
# Plots
# --------------------------------------------------------

show_plots(
    t_ins,
    r_llh_ins,
    r_llh_gps,
    r_ref_gps,
    r_llh_ins_corrected,
    r_enu_gps,
    r_enu_ins_corrected,
    resi,
    resi_normalized,
    numsvused,
    True,
)
