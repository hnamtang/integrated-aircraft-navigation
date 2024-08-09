import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(
    1, "../template_and_functions"
)  # add custom function path to PYTHONPATH for import

from llh2ecef import llh2ecef


def read_measurement_data(filename):
    if ".mat" not in filename:
        filename += ".mat"

    data_path = os.path.abspath("../data")

    if "ins" in filename:
        # Load inertial data
        data = loadmat(os.path.join(data_path, filename))
        t_ins = data["t_ins"]
        r_llh_ins = data["r_llh_ins"]

        # Compute the INS positions in ECEF
        dt_ins = t_ins[1, 0] - t_ins[0, 0]
        N_INS = len(t_ins)
        r_ecef_ins = np.zeros(shape=(3, N_INS))
        for jj in range(N_INS):
            r_ecef_ins[:, [jj]] = llh2ecef(r_llh_ins[:, [jj]])
        return t_ins, r_llh_ins, r_ecef_ins, N_INS, dt_ins

    elif "scen" in filename:
        # Load GNSS data
        data = loadmat(os.path.join(data_path, filename))
        t_gps = data["t_gps"]
        svid_gps = data["svid_gps"]
        pr_gps = data["pr_gps"]
        adr_gps = data["adr_gps"]
        ephem = data["ephem"]
        N_GPS = len(t_gps)
        return t_gps, svid_gps, pr_gps, adr_gps, ephem, N_GPS

    elif "ref" in filename:
        # Load GNSS reference data
        data = loadmat(os.path.join(data_path, filename))
        r_ref_gps = data["r_ref_gps"]  # in LLH coordinate system
        return r_ref_gps

    else:
        print("File name invalid.")
