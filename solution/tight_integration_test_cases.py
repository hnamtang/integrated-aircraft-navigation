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
from tight_integration import tight_integration

# --------------------------------------------------------
# Load Inertial Data
# --------------------------------------------------------
t_ins, r_llh_ins, r_ecef_ins = read_measurement_data("ins_proj.mat")

# --------------------------------------------------------
# Load GNSS Data
# --------------------------------------------------------
t_gps, svid_gps, pr_gps, adr_gps, ephem = read_measurement_data("gps_proj_scen1.mat")
# t_gps, svid_gps, pr_gps, adr_gps, ephem = read_measurement_data("gps_proj_scen2.mat")
# t_gps, svid_gps, pr_gps, adr_gps, ephem = read_measurement_data("gps_proj_scen3.mat")

# --------------------------------------------------------
# Load GNSS Reference Data
# --------------------------------------------------------
r_ref_gps = read_measurement_data("gps_ref.mat")

# IMU data: gyroscope
b_gyro = 1.0 / 3600  # [deg / s]
ARW = 0.5 / 60  # [deg / sqrt(s)]

# IMU data: accelerometer
b_accel = 9.80665 * 1e-3  # [m / s**2]
VRW = 0.2 / 60  # [(m/s) / sqrt(s)]

# Setup process noise
sigma_w = 100.0

# Initialize error covariance
r_unc = 1000
v_unc = 100
b_unc = 1e-1

# --------------------------------------------------------
# INS/GNSS Tight Integration
# --------------------------------------------------------
(
    r_ecef_ins_corrected,
    r_llh_ins_corrected,
    r_enu_ins_corrected,
    r_llh_gps,
    r_enu_gps,
    resi,
    numsvused,
) = tight_integration(
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
)
# --------------------------------------------------------
# Output - Plot
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
    numsvused,
)
