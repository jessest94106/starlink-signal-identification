import numpy as np
import matplotlib.pyplot as plt

def plot_waterfall(sparse_waterfall, rel_vel_axis, time_axis=None,
                   out_name=None, min_db=-136, max_db=20):
    """
    Plot a waterfall from a reconstructed sparse waterfall array.

    Parameters
    ----------
    sparse_waterfall : np.ndarray
        2D array of shape [num_time_frames, num_velocity_bins].
        This should match the saved sparseSTFT.T format from the original code.
    rel_vel_axis : np.ndarray
        1D array of relative velocity values (m/s), length = num_velocity_bins.
    time_axis : np.ndarray or None
        1D array of time values (s), length = num_time_frames.
        If None, frame index is used.
    out_name : str or None
        If given, save the figure to this file. Otherwise just show it.
    min_db, max_db : float
        Display range for the color scale in dB.
    """

    # Convert to dB
    Sxx_db = 10.0 * np.log10(np.abs(sparse_waterfall) + 1e-15)

    # Default time axis if not provided
    if time_axis is None:
        time_axis = np.arange(sparse_waterfall.shape[0])

    extent = [rel_vel_axis[0], rel_vel_axis[-1], time_axis[0], time_axis[-1]]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        Sxx_db,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        interpolation="bilinear",
        vmin=min_db,
        vmax=max_db,
    )
    fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_xlabel("Relative Velocity (m/s)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Waterfall")
    fig.tight_layout()

    if out_name is not None:
        fig.savefig(out_name, dpi=200)
        plt.close(fig)
    else:
        plt.show()

waterfall = np.load("r0_waterfall_CF_12.325GHz.npy")
rel_vel = np.load("rel_vel_r0_waterfall_CF_12.325GHz.npy")
time_axis = np.load("time_r0_waterfall_CF_12.325GHz.npy", allow_pickle=True)

plot_waterfall(waterfall, rel_vel, time_axis)
