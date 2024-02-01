import os
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram (initial_errors, optimal_errors, n_bins=20, vis_folder=None):
    """
    Plots the error histogram of the initial and optimal trajectories compared to the real trajectory. Input shape [N_traj, N_frames].
    """
    os.makedirs(vis_folder, exist_ok=True)
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    mm = 1/25.4

    num_traj = initial_errors.shape[0]
    # Iterate over all trajectories
    for traj_i in range(num_traj):
        fig, ax = plt.subplots(figsize=(88*mm, 60*mm))

        ax.hist(initial_errors[traj_i], bins=n_bins, alpha=0.5, label='Initial Errors')
        ax.hist(optimal_errors[traj_i], bins=n_bins, alpha=0.5, label='Optimal Errors')

        ### Visualization settings
        ax.set_xlabel("Distance Error (m)")
        ax.set_ylabel("Number of Occurences (-)")
        ax.ticklabel_format(axis="x", style='sci', scilimits=(0,0))
        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(loc='lower center', bbox_to_anchor=[0.5, -0.35], ncol=2, fancybox=True, shadow=True)
        fig.savefig(f"{vis_folder}/error_histogram_{traj_i}.png", dpi=300, bbox_inches='tight', pad_inches=1*mm)

        plt.close()

def plot_displacement (sim_markers, real_markers, dt=0.01, vis_folder=None):
    """
    Plots the displacements of the simulated and real markers for all trajectories in the input. Input shape [N_traj, N_frames, N_markers, 3]. Mean of the last 4 markers is taken to be plotted (for SoPRA this should be the bottom markers).
    """
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    mm = 1/25.4

    num_traj = sim_markers.shape[0]
    num_frames = sim_markers.shape[1]
    # Iterate over all trajectories
    for traj_i in range(num_traj):
        fig, ax = plt.subplots(ncols=1, figsize=(180*mm, 60*mm))
        times = np.linspace(0, (num_frames-1) * dt, num_frames)

        ax.plot(times, sim_markers[traj_i, :, :, -1].mean(1), linestyle='-', label="Simulated Z Axis")
        ax.plot(times, real_markers[traj_i, :, :, -1].mean(1), linestyle='--', label="Real Z Axis")


        ### Visualization settings
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.set_xlim(-dt, (num_frames-1) * dt)
        ax.ticklabel_format(axis="y", style='sci', scilimits=(0,0))
        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(loc='lower center', bbox_to_anchor=[0.5, -0.35], ncol=2, fancybox=True, shadow=True)
        plt.subplots_adjust(wspace=0.3)
        fig.savefig(f"{vis_folder}/displacement_trajectory_{traj_i}.png", dpi=300, bbox_inches='tight', pad_inches=1*mm)
        plt.close()



def plot_displacement_xy (sim_markers, real_markers, dt=0.01, vis_folder=None):
    """
    Plots the displacements of the simulated and real markers for all trajectories in the input in XY plane with time shown in color. Input shape [N_traj, N_frames, N_markers, 3]. Mean of the last 4 markers is taken to be plotted (for SoPRA this should be the bottom markers).
    """
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    mm = 1/25.4

    num_traj = sim_markers.shape[0]
    num_frames = sim_markers.shape[1]
    for traj_i in range(num_traj):
        fig, ax = plt.subplots(figsize=(88*mm, 60*mm))
        times = np.linspace(0, (num_frames-1) * dt, num_frames)

        scatter1 = ax.scatter(sim_markers[traj_i, :, -4:, 0].mean(1), sim_markers[traj_i, :, -4:, 1].mean(1), alpha= 0.5, s=20.0, c=times, cmap='winter')
        scatter2 = ax.scatter(real_markers[traj_i, :, -4:, 0].mean(1), real_markers[traj_i, :, -4:, 1].mean(1), alpha= 0.5, s=20.0 , c=times, cmap='autumn', marker='x')

        ### Visualization settings
        ax.set_xlabel("X-Displacement (m)")
        ax.set_ylabel("Y-Displacement (m)")
        ax.grid()
        ax.set_axisbelow(True)
        plt.colorbar(scatter1, ax=ax, label='Simulated Markers over Time (s)', pad=3*mm)
        plt.colorbar(scatter2, ax=ax, label='Real Markers over Time (s)', pad=1*mm)
        plt.axis('equal')

        fig.savefig(f"{vis_folder}/displacementXY_trajectory_{traj_i}.png", dpi=300, bbox_inches='tight', pad_inches=1*mm)
        plt.close()


def plot_sim2real_errors (marker_errors_mean, marker_errors_std, dt=0.01, vis_folder=None):
    """
    Plots the displacements of the simulated and real markers for all trajectories in the input in XY plane with time shown in color. Input shape [N_traj, N_frames, N_markers, 3]. Mean of the last 4 markers is taken to be plotted (for SoPRA this should be the bottom markers).
    """
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    mm = 1/25.4

    num_traj = marker_errors_mean.shape[0]
    num_frames = marker_errors_mean.shape[1]
    for traj_i in range(num_traj):
        fig, ax = plt.subplots(figsize=(88*mm, 60*mm))
        times = np.linspace(0, (num_frames-1) * dt, num_frames)

        ax.plot(times, marker_errors_mean[traj_i])
        plt.fill_between(times, (marker_errors_mean[traj_i]-marker_errors_std[traj_i]), (marker_errors_mean[traj_i]+marker_errors_std[traj_i]), color='b', alpha=.1)

        ### Visualization settings
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance Error (m)")
        ax.set_xlim(-dt, (num_frames-1) * dt)
        ax.ticklabel_format(axis="y", style='sci', scilimits=(0,0))
        ax.grid()
        ax.set_axisbelow(True)
        fig.savefig(f"{vis_folder}/sim2real_error_trajectory_{traj_i}.png", dpi=300, bbox_inches='tight', pad_inches=1*mm)

def error_computation (sim_markers, real_markers, dt, vis_folder, verbose=False):
    """
    Compute various error metrics between the simulated and real marker locations.
    """
    # Error shapes [chambers, timesteps]
    marker_errors_mean = np.linalg.norm(real_markers - sim_markers, axis=-1).mean(axis=-1)
    marker_errors_std = np.linalg.norm(real_markers - sim_markers, axis=-1).std(axis=-1)

    # Error per chamber
    if verbose:
        for traj_i in range(sim_markers.shape[0]):
            print(f"Trajectory {traj_i} Time-Averaged Marker Error: \t{1000*marker_errors_mean[traj_i].mean():.4f}mm +- {1000*marker_errors_std[traj_i].mean():.4f}mm")

    ### Visualization
    os.makedirs(vis_folder, exist_ok=True)

    # Plot mean tip marker sim2real trajectories 
    plot_displacement(sim_markers, real_markers, dt=dt, vis_folder=vis_folder)

    # Plot mean tip marker sim2real trajectories in projected XY plane
    plot_displacement_xy(sim_markers, real_markers, dt=dt, vis_folder=vis_folder)

    # Plot errors
    plot_sim2real_errors(marker_errors_mean, marker_errors_std, dt=dt, vis_folder=vis_folder)

    return marker_errors_mean, marker_errors_std

def plot_trajectory(
    vis_1d_folder,
    figsize,
    transformed_markers,
    sim_markers,
    res_markers,
    test_data_idx,
    real_frames,
    dt,
):
    mm = 1.5 / 25.4
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    times = np.linspace(0, real_frames * dt, real_frames + 1)[:-1]
    offset = transformed_markers[:real_frames, :, -1].mean(1)[0]
    ax.plot(
        times,
        sim_markers[:real_frames, :, -1].mean(1)-offset,
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Simulation",
        linewidth=2.0,
    )
    ax.plot(
        times,
        res_markers[:real_frames, :, -1].mean(1)-offset,
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Residual Physics",
        linewidth=2.0,
    )
    ax.plot(
        times,
        transformed_markers[:real_frames, :, -1].mean(1)-offset,
        linestyle="--",
        marker="o",
        markersize=0.5,
        label=f"Real",
        linewidth=2.0,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=[0.5, -0.35],
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    # ax.set_title(f"z axis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (m)")
    ax.set_xlim(0, real_frames * dt)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=[0.5, -0.35],
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    fig.savefig(
        f"{vis_1d_folder}/beam_{test_data_idx}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=1 * mm,
    )
    fig.savefig(
        f"{vis_1d_folder}/beam_{test_data_idx}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=1 * mm,
    )
    plt.close()


def plot_forces_norm(
    vis_1d_folder,
    data_idx,
    figsize,
    predicted_residual_force_norms,
    ground_truth_residual_force_norms,
    dt,
):
    mm = 1.5 / 25.4
    fig, ax = plt.subplots(figsize=figsize)
    real_frames = predicted_residual_force_norms.shape[0]
    times = np.linspace(0, real_frames * dt, real_frames + 1)[:-1]
    ax.plot(
        times,
        predicted_residual_force_norms,
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Predicted Residual Force Norm",
        linewidth=2.0,
    )
    ax.plot(
        times,
        ground_truth_residual_force_norms,
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Ground Truth Residual Force Norm",
        linewidth=2.0,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual Force Norm (N)")
    ax.set_xlim(0, real_frames * dt)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    plt.savefig(
        f"{vis_1d_folder}/residual_force_norm_{data_idx}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=1 * mm,
    )
    plt.close()
    