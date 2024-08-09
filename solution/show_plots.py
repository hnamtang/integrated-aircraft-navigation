import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse


def show_plots(
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
    save_figure=False,
):
    # Latitude vs. longitude plot
    fig_size = (12, 6)
    figs = {}
    axs = {}
    fig_names = {}

    figs[0], axs[0] = plt.subplots(figsize=fig_size)
    axs[0].plot(
        r_ref_gps[1, :] * 180.0 / np.pi,
        r_ref_gps[0, :] * 180.0 / np.pi,
        linewidth=3,
        linestyle="--",
        color="k",
        label="GNSS reference",
    )
    axs[0].plot(
        r_llh_ins[1, :] * 180.0 / np.pi,
        r_llh_ins[0, :] * 180.0 / np.pi,
        linewidth=1.5,
        linestyle="-",
        color="b",
        label="INS",
    )
    axs[0].plot(
        r_llh_gps[1, :] * 180.0 / np.pi,
        r_llh_gps[0, :] * 180.0 / np.pi,
        linewidth=1.5,
        linestyle="-",
        color="r",
        label="GNSS",
    )
    axs[0].plot(
        r_llh_ins_corrected[1, :] * 180.0 / np.pi,
        r_llh_ins_corrected[0, :] * 180.0 / np.pi,
        # linewidth=3,
        linewidth=1.5,
        linestyle="--",
        color="g",
        label="GNSS/INS tight integration",
    )
    # covariance_ellipse(
    #    ax1,
    #    r_llh_ins_corrected[1, :] * 180.0 / np.pi,
    #    r_llh_ins_corrected[0, :] * 180.0 / np.pi,
    #    n_std=2.0,
    #    facecolor="firebrick",
    #    alpha=0.5,
    #    zorder=0,
    #    label="Covariance",
    # )
    axs[0].legend(loc="lower right")
    axs[0].grid()
    axs[0].set_title("Ground Tracks")
    axs[0].set_xlabel("Longitude [deg]")
    axs[0].set_ylabel("Latitude [deg]")
    axs[0].axis("equal")
    fig_names[0] = "plot1_ground_track"
    plt.tight_layout()

    # Height vs. elapsed time plot
    figs[1], axs[1] = plt.subplots(figsize=fig_size)
    axs[1].plot(
        t_ins,
        r_ref_gps[2, :],
        linewidth=3,
        linestyle="--",
        color="k",
        label="GNSS reference",
    )
    axs[1].plot(
        t_ins, r_llh_ins[2, :], linewidth=1.5, linestyle="-", color="b", label="INS"
    )
    axs[1].plot(
        t_ins, r_llh_gps[2, :], linewidth=1.5, linestyle="-", color="r", label="GNSS"
    )
    axs[1].plot(
        t_ins,
        r_llh_ins_corrected[2, :],
        # linewidth=3,
        linewidth=1.5,
        linestyle="--",
        color="g",
        label="INS/GNSS tight integration",
    )
    axs[1].legend(loc="lower right")
    axs[1].grid()
    axs[1].set_title("Height with Inertial and GNSS")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Height [m]")
    fig_names[1] = "plot2_height_vs_time"
    plt.tight_layout()

    # Number of available GPS satellites as a function of time
    figs[2], axs[2] = plt.subplots(figsize=fig_size)
    axs[2].plot(t_ins, numsvused.T, linewidth=1.5, linestyle="-", color="b")
    axs[2].grid()
    axs[2].set_title("Number of available GPS satellites")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Number of available satellites")
    fig_names[2] = "plot3_available_sat"
    plt.tight_layout()

    # Error plot
    # TODO: compute and include the covariance in this plot
    figs[3], axs[3] = plt.subplots(figsize=fig_size)
    axs[3].plot(
        t_ins,
        (r_enu_ins_corrected - r_enu_gps).T,
        linewidth=1.5,
        linestyle="-",
    )
    axs[3].legend(["East", "North", "Up"], loc="lower right")
    axs[3].grid()
    axs[3].set_title("Difference between GNSS and GNSS/INS Solution")
    # axs[3].set_xlabel("Time [s]")
    # axs[3].set_ylabel("Error in ENU [m]")
    # axs[3].set_xlim([t_ins[0], t_ins[-1]])
    axs[3].set_xlabel("East")
    axs[3].set_ylabel("North")
    fig_names[3] = "plot4_difference_GNSS_GNSSINS_ENU"
    plt.tight_layout()

    # Filter innovations plot
    figs[4], axs[4] = plt.subplots(figsize=fig_size)
    axs[4].plot(t_ins, resi.T, linewidth=1.5)
    axs[4].legend(
        [r"$\Delta\rho_1$", r"$\Delta\rho_2$", r"$\Delta\rho_3$", r"$\Delta\rho_4$"],
        loc="lower right",
    )
    axs[4].grid()
    axs[4].set_title("Residual")
    axs[4].set_xlabel("Time [s]")
    axs[4].set_ylabel("Residual")
    axs[4].set_ylim([-10, 10])
    fig_names[4] = "plot5_residual"
    plt.tight_layout()

    # Mahalanobis plot
    figs[5], axs[5] = plt.subplots(figsize=fig_size)
    axs[5].plot(t_ins, resi_normalized.T, linewidth=1.5)
    axs[5].grid()
    axs[5].set_title("Mahalanobis Distance")
    axs[5].set_xlabel("Time [s]")
    axs[5].set_ylabel("Residual")
    axs[5].set_ylim([0, 0.3])
    fig_names[5] = "plot6_mahalanobis"
    plt.tight_layout()

    # Benchmark
    figs[6], axs[6] = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=fig_size)
    axs[6][0].plot(
        t_ins,
        r_llh_ins_corrected[1, :] - r_ref_gps[1, :],
        linewidth=1.5,
        color="b",
        label="longitude",
    )
    axs[6][0].grid()
    axs[6][0].set_title("Position Estimation Error")
    axs[6][0].set_ylabel("Longitude [deg]")
    axs[6][1].plot(
        t_ins,
        r_llh_ins_corrected[0, :] - r_ref_gps[0, :],
        linewidth=1.5,
        color="b",
        label="latitude",
    )
    axs[6][1].grid()
    axs[6][1].set_ylabel("Latitude [deg]")
    axs[6][2].plot(
        t_ins,
        r_llh_ins_corrected[2, :] - r_ref_gps[2, :],
        linewidth=1.5,
        color="b",
        label="height",
    )
    axs[6][2].grid()
    axs[6][2].set_xlabel("Time [s]")
    axs[6][2].set_ylabel("Height [m]")
    fig_names[6] = "plot7_r_llh_vs_time"

    # Save figures
    if save_figure:
        for fig_idx, name in fig_names.items():
            figs[fig_idx].savefig(
                "./img/" + name + ".eps",
                format="eps",
                dpi=600,
                bbox_inches="tight",
            )
            figs[fig_idx].savefig(
                "./img/" + name + ".png",
                format="png",
                dpi=600,
                bbox_inches="tight",
            )
    else:
        plt.show()


def covariance_ellipse(ax, x, y, n_std=2.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size.")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this 2D dataset
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the standard deviation of x from the square root of the variance and
    # multiplying with the given number of standard deviations
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # Calculating the standard deviation of y
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
