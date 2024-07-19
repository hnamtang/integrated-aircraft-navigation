import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

sys.path.insert(
    1, "../template_and_functions"
)  # add custom function path to PYTHONPATH for import

from ecef2enu import ecef2enu
from ecef2llh import ecef2llh
from ephcal import ephcal
from gnss_routines import compute_pos_ecef, compute_svpos_svclock_compensate
from llh2ecef import llh2ecef


def ins_gnss_tight(ins_file_path, gps_file_path, plot=True):
    # GPS value for speed of light
    V_LIGHT = 299792458.0

    # --------------------------------------------------------
    # Load Inertial Data
    # --------------------------------------------------------
    mat = loadmat(ins_file_path)
    t_ins = mat["t_ins"]
    r_llh_ins = mat["r_llh_ins"]
    # rpy_ins = mat["rpy_ins"]

    # Compute the INS positions in ECEF
    N_INS = len(t_ins)
    r_ecef_ins = np.zeros(shape=(3, N_INS))
    for jj in range(0, len(t_ins)):
        # Compute INS position in ECEF
        r_ecef_ins[:, [jj]] = llh2ecef(r_llh_ins[:, [jj]])

    # --------------------------------------------------------
    # Load GNSS Data
    # --------------------------------------------------------
    mat = loadmat(gps_file_path)
    t_gps = mat["t_gps"]
    svid_gps = mat["svid_gps"]
    pr_gps = mat["pr_gps"]
    adr_gps = mat["adr_gps"]
    ephem = mat["ephem"]

    # --------------------------------------------------------
    # Load GNSS Reference Data
    # --------------------------------------------------------
    mat = loadmat(os.path.join(os.path.abspath("../data"), "gps_ref.mat"))
    r_ref_gps = mat["r_ref_gps"]

    # -----------------------------------------------------------------------
    # Initialize the Kalman Filter - Loose Coupling
    # -----------------------------------------------------------------------
    dt_ins = t_ins[1, 0] - t_ins[0, 0]

    # IMU data: gyroscope
    b_gyro = 1.0 / 3600  # [deg / s]
    ARW = 0.5 / 60  # [deg / sqrt(s)]
    sigma_gyro = ARW * np.sqrt(dt_ins)

    # IMU data: accelerometer
    b_accel = 9.80665 * 1e-3  # [m / s**2]
    VRW = 0.2 / 60  # [(m/s) / sqrt(s)]
    sigma_accel = VRW * np.sqrt(dt_ins)

    # Setup state transition matrix
    Phi = np.eye(6, dtype=float)
    Phi[0, 1] = Phi[2, 3] = Phi[4, 5] = dt_ins

    # Setup process noise
    sigma_w = 1e-3
    Q = (sigma_w**2) * np.array(
        [
            [(dt_ins**3) / 3.0, (dt_ins**2) / 2.0, 0, 0, 0, 0],
            [(dt_ins**2) / 2.0, dt_ins, 0, 0, 0, 0],
            [0, 0, (dt_ins**3) / 3.0, (dt_ins**2) / 2.0, 0, 0],
            [0, 0, (dt_ins**2) / 2.0, dt_ins, 0, 0],
            [0, 0, 0, 0, (dt_ins**3) / 3.0, (dt_ins**2) / 2.0],
            [0, 0, 0, 0, (dt_ins**2) / 2.0, dt_ins],
        ],
        dtype=float,
    )

    I = np.eye(6)

    # Initialize error vector and error covariance
    x_pre = np.zeros(shape=(6, 1), dtype=float)
    P_pre = 1000 * np.eye(6)
    x_est = np.zeros(shape=(6, N_INS), dtype=float)

    # --- INSERT YOUR CODE ---

    # --------------------------------------------------------
    # Output variables
    # --------------------------------------------------------

    # Length of the time array
    N = len(t_gps)

    # Output arrays
    gnsstime = np.array([])
    lat = np.array([])
    lon = np.array([])
    h = np.array([])
    llh_all = np.zeros(shape=(4, 0))

    numsvused = np.zeros(shape=(N, 1))

    r_ecef_gps = np.zeros(shape=(3, N_INS))
    r_llh_gps = np.zeros(shape=(3, N_INS))
    r_ecef_ins_corrected = np.zeros(shape=(3, N_INS))
    r_llh_ins_corrected = np.zeros(shape=(3, N_INS))

    # --------------------------------------------------------
    # Go through all GPS data and compute trajectory
    # --------------------------------------------------------
    N_SV_REQUIRED = 4  # required number of available satellites
    H = np.zeros(shape=(N_SV_REQUIRED, 6), dtype=float)  # initialize measurement matrix
    # TODO: Check measurement noise (see note)
    R = np.eye(N_SV_REQUIRED) * sigma_accel**2  # Setup measurement noise
    resi = np.zeros(
        shape=(N_SV_REQUIRED, N_INS), dtype=float
    )  # initialize innovation vector

    ii = 0
    while ii < N:

        # Extract the GPS measurment data for this time epoch
        gpstime = t_gps[ii, 0]
        index_gps = (t_gps[:, 0] == gpstime).nonzero()[0]
        svid = np.int32(
            svid_gps[index_gps, 0]
        )  # available satellites at current time epoch

        pr = pr_gps[index_gps, 0]

        # Find the corresponding INS measurment
        index_ins = (t_ins[:, 0] == gpstime).nonzero()[0]

        (sv_ecef, sv_clock) = compute_svpos_svclock_compensate(gpstime, svid, pr, ephem)

        if svid.size >= N_SV_REQUIRED:
            (r_ecef_gps[:, index_ins], user_clock) = compute_pos_ecef(
                gpstime, pr, sv_ecef, sv_clock
            )

            gnsstime = np.append(gnsstime, gpstime)
            r_llh_gps[:, index_ins] = ecef2llh(r_ecef_gps[:, index_ins])
        else:
            print(f"Not enough satellites at time = {gpstime}")

        # Compensate the pseuforanges for the staellite clock errors
        pr += sv_clock * V_LIGHT
        pr = pr.reshape(len(pr), 1)

        # --------------------------------------------------------------
        #   Tight coupling
        # --------------------------------------------------------------

        # Form the measurement from the inertial (INS)
        # and setup measurement matrix
        rho_ins = np.zeros(shape=(N_SV_REQUIRED, 1), dtype=float)
        for kk in range(N_SV_REQUIRED):
            rho_ins[kk, 0] = np.linalg.norm(
                r_ecef_ins[:, index_ins] - sv_ecef[:, [kk]]
            )  # pseudorange from INS position to location of each available SV

            H[kk, [0]] = (r_ecef_ins[0, index_ins] - sv_ecef[0, [kk]]) / rho_ins[kk, 0]
            H[kk, [2]] = (r_ecef_ins[1, index_ins] - sv_ecef[1, [kk]]) / rho_ins[kk, 0]
            H[kk, [4]] = (r_ecef_ins[2, index_ins] - sv_ecef[2, [kk]]) / rho_ins[kk, 0]

        # Form the Kalman filter measurement vector
        # Difference between distances calculated by INS position
        # and GPS pseudorange measurement
        z = rho_ins - pr[: len(rho_ins), [0]]

        # -----------------------
        # Update
        # -----------------------

        # (1) Compute the Kalman gain
        K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + R)

        # Save the innovations/residuals
        resi[:, index_ins] = z - H @ x_pre

        # (2) Update estimate with measurement
        x_est[:, index_ins] = x_pre + K @ resi[:, index_ins]

        # (3) Update the error covariance
        P = (I - K @ H) @ P_pre

        # -----------------------
        # Predict
        # -----------------------

        # (4) Project the state ahead
        x_pre = Phi @ x_est[:, index_ins]

        # (5) Project the error covariance ahead
        P_pre = Phi @ P @ Phi.T + Q

        # -----------------------
        # Remove the drift term
        # -----------------------

        r_ecef_ins_corrected[:, index_ins] = (
            r_ecef_ins[:, index_ins] - x_est[[[0, 2, 4]], index_ins].T
        )

        # --- INSERT YOUR CODE ---

        # --------------------------------------------------------
        # Store information for plotting
        # --------------------------------------------------------
        r_llh_ins_corrected[:, index_ins] = ecef2llh(r_ecef_ins_corrected[:, index_ins])

        # --- INSERT YOUR CODE ---

        # Update the index
        ii = index_gps[-1] + 1

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if plot:
        fig1, ax1 = plt.subplots()
        ax1.plot(
            r_llh_ins[1, :] * 180.0 / np.pi,
            r_llh_ins[0, :] * 180.0 / np.pi,
            linewidth=1.5,
            linestyle="-",
            color="b",
            label="INS",
        )
        ax1.plot(
            r_llh_gps[1, :] * 180.0 / np.pi,
            r_llh_gps[0, :] * 180.0 / np.pi,
            linewidth=1.5,
            linestyle="-",
            color="r",
            label="GPS",
        )
        ax1.plot(
            r_llh_ins_corrected[1, :] * 180.0 / np.pi,
            r_llh_ins_corrected[0, :] * 180.0 / np.pi,
            linestyle="--",
            linewidth=1.5,
            color="g",
            label="INS corrected",
        )
        # ax1.plot(
        #    r_ref_gps[1, :] * 180.0 / np.pi,
        #    r_ref_gps[0, :] * 180.0 / np.pi,
        #    linestyle="--",
        #    linewidth=3,
        #    color="k",
        #    label="GPS reference",
        # )
        ax1.legend(loc="best")
        ax1.grid()
        ax1.set_title("Ground Tracks with Inertial and GPS")
        ax1.set_xlabel("Longitude [deg]")
        ax1.set_ylabel("Latitude [deg]")
        ax1.axis("equal")
        plt.tight_layout()

        # fig2, ax2 = plt.subplots()
        # ax2.plot(
        #    r_ecef_ins[0, :],
        #    r_ecef_ins[1, :],
        #    color="b",
        #    label="INS",
        # )
        # ax2.plot(
        #    r_ecef_gps[0, :],
        #    r_ecef_gps[1, :],
        #    color="r",
        #    label="GPS",
        # )
        # ax2.plot(
        #    r_ecef_ins_corrected[0, :],
        #    r_ecef_ins_corrected[1, :],
        #    color="g",
        #    label="INS corrected",
        # )
        ## ax1.legend(["INS", "GPS", "INS corrected"], loc="best")
        # ax2.legend(loc="best")
        # ax2.grid()
        # ax2.set_title("Ground Tracks with Inertial and GPS in ECEF Coordinate System")
        # ax2.set_xlabel("x")
        # ax2.set_ylabel("y")
        # ax2.axis("equal")
        # plt.tight_layout()
        #
        # fig3, ax3 = plt.subplots()
        # ax3.plot(t_ins, resi.T)
        # ax3.legend([r"$\rho_1$", r"$\rho_2$", r"$\rho_3$", r"$\rho_4$"], loc="best")
        # ax3.grid()
        # ax3.set_title("Residual")
        # ax3.set_xlabel("Time [s]")
        # ax3.set_ylabel("Residual")
        # plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    data_path = os.path.abspath("../data")
    ins_file_path = os.path.join(data_path, "ins_proj.mat")
    gps_file_path = os.path.join(data_path, "gps_proj_scen3.mat")

    ins_gnss_tight(ins_file_path, gps_file_path, plot=True)
