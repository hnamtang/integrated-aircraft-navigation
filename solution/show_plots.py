import numpy as np
from matplotlib import pyplot as plt


def show_plots(
    t_ins,
    r_llh_ins,
    r_llh_gps,
    r_ref_gps,
    r_llh_ins_corrected,
    r_enu_gps,
    r_enu_ins_corrected,
    resi,
    numsvused,
):
    # Latitude vs. longitude plot
    _, ax1 = plt.subplots()
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
        # linewidth=3,
        linewidth=1.5,
        color="g",
        label="INS/GNSS tight integration",
    )
    ax1.legend(loc="lower right")
    ax1.grid()
    ax1.set_title("Ground Tracks")
    ax1.set_xlabel("Longitude [deg]")
    ax1.set_ylabel("Latitude [deg]")
    ax1.axis("equal")
    plt.tight_layout()

    # Height vs. elapsed time plot
    _, ax2 = plt.subplots()
    ax2.plot(
        t_ins, r_llh_ins[2, :], linewidth=1.5, linestyle="-", color="b", label="INS"
    )
    ax2.plot(
        t_ins, r_llh_gps[2, :], linewidth=1.5, linestyle="-", color="r", label="GPS"
    )
    ax2.plot(
        t_ins,
        r_llh_ins_corrected[2, :],
        # linewidth=3,
        linewidth=1.5,
        linestyle="--",
        color="g",
        label="INS/GNSS tight integration",
    )
    ax2.legend(loc="lower right")
    ax2.grid()
    ax2.set_title("Height with Inertial and GPS")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Height [m]")
    plt.tight_layout()

    # Number of available GPS satellites as a function of time
    _, ax3 = plt.subplots()
    ax3.plot(t_ins, numsvused, linewidth=1.5, linestyle="-", color="b")
    ax3.grid()
    ax3.set_title("Number of available GPS satellites")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Number of available satellites")
    plt.tight_layout()

    # Error plot
    # TODO: compute and include the covariance in this plot
    _, ax4 = plt.subplots()
    ax4.plot(
        t_ins,
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

    # Filter innovations plot
    _, ax5 = plt.subplots()
    ax5.plot(t_ins, resi.T)
    ax5.legend(
        [r"$\Delta\rho_1$", r"$\Delta\rho_2$", r"$\Delta\rho_3$", r"$\Delta\rho_4$"],
        loc="lower right",
    )
    ax5.grid()
    ax5.set_title("Residual")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Residual")
    ax5.set_ylim([-10, 10])
    plt.tight_layout()

    # Benchmark
    _, ax6 = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax6[0].plot(
        t_ins,
        r_llh_ins_corrected[1, :] - r_ref_gps[1, :],
        linewidth=1.5,
        color="b",
        label="longitude",
    )
    ax6[0].grid()
    ax6[0].set_title("Position Estimation Error")
    ax6[0].set_ylabel("Longitude [deg]")
    ax6[1].plot(
        t_ins,
        r_llh_ins_corrected[0, :] - r_ref_gps[0, :],
        linewidth=1.5,
        color="b",
        label="latitude",
    )
    ax6[1].grid()
    ax6[1].set_ylabel("Latitude [deg]")
    ax6[2].plot(
        t_ins,
        r_llh_ins_corrected[2, :] - r_ref_gps[2, :],
        linewidth=1.5,
        color="b",
        label="height",
    )
    ax6[2].grid()
    ax6[2].set_xlabel("Time [s]")
    ax6[2].set_ylabel("Height [m]")

    plt.show()
