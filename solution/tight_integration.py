import sys

import numpy as np

sys.path.insert(
    1, "../template_and_functions"
)  # add custom function path to PYTHONPATH for import

from ecef2enu import ecef2enu
from ecef2llh import ecef2llh
from gnss_routines import compute_pos_ecef, compute_svpos_svclock_compensate


def tight_integration(
    t_ins,
    r_llh_ins,
    r_ecef_ins,
    t_gps,
    svid_gps,
    pr_gps,
    adr_gps,
    ephem,
    b_gyro,
    ARW,
    b_accel,
    VRW,
    sigma_w,
    r_unc,
    v_unc,
    b_unc,
):
    # GPS value for speed of light
    V_LIGHT = 299792458.0

    N_INS = len(t_ins)
    N_GPS = len(t_gps)
    dt_ins = t_ins[1, 0] - t_ins[0, 0]

    # IMU data: gyroscope
    sigma_gyro = ARW * np.sqrt(dt_ins)

    # IMU data: accelerometer
    sigma_vel = VRW * np.sqrt(dt_ins)

    # -----------------------------------------------------------------------
    # Initialize the Kalman Filter - Tight Coupling
    # -----------------------------------------------------------------------

    # Setup state transition matrix
    # State vector is INS error vector
    N_STATES = 8  # dx, dx_dot, dy, dy_dot, dz, dz_dot, db, db_dot (b: user clock error)
    Phi = np.eye(N_STATES, dtype=float)
    Phi[[0, 2, 4, 6], [1, 3, 5, 7]] = dt_ins

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

    # Initialize error vector
    x_pre = np.zeros(shape=(N_STATES, 1), dtype=float)

    # Initialize error covariance
    # P_pre = (100**2) * np.eye(N_STATES)
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
        shape=(N_INS, 1), dtype=int
    )  # storage for number of available satellites

    r_ecef_gps = np.zeros(shape=(3, N_INS))
    r_llh_gps = np.zeros_like(r_ecef_gps)
    r_enu_gps = np.zeros_like(r_ecef_gps)

    r_ecef_ins_corrected = np.zeros(shape=(3, N_INS))
    r_llh_ins_corrected = np.zeros_like(r_ecef_ins_corrected)
    r_enu_ins_corrected = np.zeros_like(r_ecef_ins_corrected)

    # --------------------------------------------------------
    # Go through all GPS data and compute trajectory
    # --------------------------------------------------------

    # required number of available satellites
    # 3 SVs for position estimation + 1 SV for clock error + 1 SV for integrity check
    N_SV_REQUIRED = 5

    # Initialize measurement matrix
    H = np.zeros(shape=(N_SV_REQUIRED, N_STATES), dtype=float)
    H[:, -2] = 1.0

    # TODO: Check measurement noise (see note)
    # TODO: R_INS, R_GPS -> R = scipy.linalg.block_diag(R_INS, R_GPS)
    # R = (sigma_vel**2) * np.eye(N_SV_REQUIRED)
    R = (0.5**2) * np.eye(N_SV_REQUIRED)

    # Initialize innovation vector
    resi = np.zeros(shape=(N_SV_REQUIRED, N_INS), dtype=float)

    # Iteration
    ii = 0
    while ii < N_GPS:
        # Extract the GPS measurement data for this time epoch
        gpstime = t_gps[ii, 0]
        index_gps = (t_gps[:, 0] == gpstime).nonzero()[0]
        svid = np.int32(
            svid_gps[index_gps, 0]
        )  # available satellites at current time epoch

        pr = pr_gps[index_gps, 0]

        # Find the corresponding INS measurement
        index_ins = (t_ins[:, 0] == gpstime).nonzero()[0]

        numsvused[index_ins, 0] = len(index_gps)

        (sv_ecef, sv_clock) = compute_svpos_svclock_compensate(gpstime, svid, pr, ephem)

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
        else:
            print(f"Not enough satellites at time = {gpstime}")

        # Compensate the pseudoranges for the satellite clock error
        pr += sv_clock * V_LIGHT
        pr = pr.reshape(numsvused[index_ins[0], 0], 1)

        # --------------------------------------------------------------
        #   Tight coupling
        # --------------------------------------------------------------

        # Synthesize ranges from the inertial (INS)
        # and setup measurement matrix
        rho_ins = np.zeros(shape=(N_SV_REQUIRED, 1), dtype=float)
        H_PDOP = np.zeros(shape=(N_SV_REQUIRED, 3), dtype=float)
        # OPTIMIZE: pick the best satellites (low dilution of precision (= min. singular value of G))
        for kk in range(N_SV_REQUIRED):
            rho_ins[kk, 0] = np.linalg.norm(
                r_ecef_ins[:, index_ins] - sv_ecef[:, [kk]]
            )  # pseudorange from INS position to location of each available SV
            H[kk, [[0, 2, 4]]] = (r_ecef_ins[:, index_ins] - sv_ecef[:, [kk]]).T
            H[kk, [[0, 2, 4]]] /= rho_ins[kk, 0]

            # Calculate H for PDOP
            r_llh = ecef2enu(
                r_ecef_ins[:, index_ins], r_ecef_gps[:, [0]], r_llh_gps[:, [0]]
            )
            sv_llh = ecef2enu(sv_ecef[:, [kk]], r_ecef_gps[:, [0]], r_llh_gps[:, [0]])
            rho_llh = np.linalg.norm(r_llh - sv_llh)
            H_PDOP[kk, :] = (r_llh - sv_llh).T / rho_llh

        # Calculate PDOP (note: H must be expressed in ENU coordinate system)
        # H_PDOP = H[:, [0, 2, 4]]
        G = np.linalg.inv(H_PDOP.T @ H_PDOP)  # DOP matrix
        PDOP = np.sqrt(np.trace(G))
        print(f"Position Dilution of Precision: PDOP = {PDOP:.3f}")

        # From the Kalman filter measurement vector
        # Difference between distances calculated by INS position
        # and GPS pseudorange measurement
        z = rho_ins - pr[:N_SV_REQUIRED, [0]]

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
        r_enu_ins_corrected[:, index_ins] = ecef2enu(
            r_ecef_ins_corrected[:, index_ins],
            r_ecef_ins_corrected[:, [0]],  # ENU origin in ECEF
            r_llh_ins_corrected[:, [0]],  # ENU origin in LLH
        )

        # --- INSERT YOUR CODE ---

        # Update the index
        ii = index_gps[-1] + 1

    return (
        r_ecef_ins_corrected,
        r_llh_ins_corrected,
        r_enu_ins_corrected,
        r_llh_gps,
        r_enu_gps,
        resi,
        numsvused,
    )


# TODO: Edit code to handle the case when #available SVs < N_SV_REQUIRED
# TODO: Pick 4 in all available GPS satellites which results in gut geometry (low Dilution of Precision (DOP))
