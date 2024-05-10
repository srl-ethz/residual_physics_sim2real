

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})

mm = 1 / 25.4
figsize = (40 * mm, 32 * mm)

### Load Data
times = np.arange(0, 10, 0.01)[:-2]

qs_sim, qs_res, qs_gt, qs_dd = [], [], [], []
trajectories = np.arange(0, 10)
# trajectories = np.arange(180, 200)
for i in trajectories:
    # qs_sim.append(np.load(f"trajectory_data/sim_markers_{i}.npy"))

    # qs_sim.append(np.load(f"trajectory_data/sysid_markers_{i}.npy"))
    # qs_res.append(np.load(f"trajectory_data/res_markers_{i}.npy"))
    # qs_gt.append(np.load(f"trajectory_data/real_markers_{i}.npy"))
    # qs_dd.append(np.load(f"trajectory_data/dd_markers_{i}.npy")[:-1])

    qs_sim.append(np.load(f"trajectory_data_10s/sysid_markers_{i}.npy"))
    qs_res.append(np.load(f"trajectory_data_10s/res_markers_{i}.npy"))
    qs_gt.append(np.load(f"trajectory_data_10s/real_markers_{i}.npy"))
    qs_dd.append(np.load(f"trajectory_data_10s/dd_markers_{i}.npy")[:-1])
qs_sim = np.stack(qs_sim, axis=0)
qs_res = np.stack(qs_res, axis=0)
qs_gt  = np.stack(qs_gt, axis=0)
qs_dd  = np.stack(qs_dd, axis=0)


# qs_sim shape [n_trajectories, n_timesteps, n_markers, 3]
frame_errors_sim = np.linalg.norm(qs_sim - qs_gt, axis=-1).mean(axis=-1)
frame_errors_res = np.linalg.norm(qs_res - qs_gt, axis=-1).mean(axis=-1)
frame_errors_dd = np.linalg.norm(qs_dd - qs_gt, axis=-1).mean(axis=-1)


fig, ax = plt.subplots(figsize=figsize)
line_width = 1

frame_mean_sim = np.mean(frame_errors_sim, axis=0)
frame_std_sim = np.std(frame_errors_sim, axis=0)
frame_mean_res = np.mean(frame_errors_res, axis=0)
frame_std_res = np.std(frame_errors_res, axis=0)
frame_mean_dd = np.mean(frame_errors_dd, axis=0)
frame_std_dd = np.std(frame_errors_dd, axis=0)


# from scipy.signal import savgol_filter

# polyorder = 1
# windowsize = 201
# frame_mean_sim = savgol_filter(frame_mean_sim, windowsize, polyorder)
# frame_mean_res = savgol_filter(frame_mean_res, windowsize, polyorder)
# frame_mean_dd = savgol_filter(frame_mean_dd, windowsize, polyorder)

from scipy.signal import medfilt

kernel_size = 101
frame_mean_sim = medfilt(frame_mean_sim, kernel_size)
frame_mean_res = medfilt(frame_mean_res, kernel_size)
frame_mean_dd = medfilt(frame_mean_dd, kernel_size)

frame_std_sim = medfilt(frame_std_sim, kernel_size)
frame_std_res = medfilt(frame_std_res, kernel_size)
frame_std_dd = medfilt(frame_std_dd, kernel_size)


ax.plot(times, frame_mean_sim, linewidth=line_width, label="SysID", c='tab:blue', zorder=2)
ax.fill_between(
    times,
    frame_mean_sim - frame_std_sim,
    frame_mean_sim + frame_std_sim,
    alpha=0.3,
    facecolor='tab:blue',
)

ax.plot(times, frame_mean_res, linewidth=line_width, label="ResPhys", c='tab:orange', zorder=3)
ax.fill_between(
    times,
    frame_mean_res - frame_std_res,
    frame_mean_res + frame_std_res,
    alpha=0.3,
    facecolor='tab:orange',
)

ax.plot(times, frame_mean_dd, linewidth=line_width, label="Data", c='tab:red', zorder=1)
ax.fill_between(
    times,
    frame_mean_dd - frame_std_dd,
    frame_mean_dd + frame_std_dd,
    alpha=0.3,
    facecolor='tab:red',
)

# ax.vlines(5.0, -10, 10, linestyles='dashed', linewidth=1, color='k', alpha=0.7, zorder=1)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Distance Error (m)")
ax.grid()
ax.set_axisbelow(True)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(loc="lower center", bbox_to_anchor=[0.5, -0.6], fontsize=6, ncol=3, fancybox=True, handlelength=0.5, columnspacing=1.0, handletextpad=0.3)
ax.set_xlim(times.min(), times.max()+0.02)
ax.set_ylim(
    min((frame_mean_sim - frame_std_sim).min(), (frame_mean_res - frame_std_res).min()),
    max((frame_mean_sim + frame_std_sim).max(), (frame_mean_res + frame_std_res).max())
)
# ax.set_title("Error distribution of test trajectories")


fig.savefig("fig8_generalizationerror.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
fig.savefig("fig8_generalizationerror.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()