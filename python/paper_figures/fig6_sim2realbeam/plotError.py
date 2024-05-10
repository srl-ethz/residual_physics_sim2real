

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
times = np.arange(0, 1.4, 0.01)

qs_sim, qs_res, qs_gt = [], [], []
trajectories = [2, 7, 9, 11, 14, 16]
for i in trajectories:
    qs_sim.append(np.load(f"trajectory_data/sim_markers_{i}.npy"))
    qs_res.append(np.load(f"trajectory_data/res_markers_{i}.npy"))
    qs_gt.append(np.load(f"trajectory_data/real_markers_{i}.npy")[:-10])
qs_sim = np.stack(qs_sim, axis=0)
qs_res = np.stack(qs_res, axis=0)
qs_gt  = np.stack(qs_gt, axis=0)

# qs_sim shape [n_trajectories, n_timesteps, n_markers, 3]
frame_errors_sim = np.linalg.norm(qs_sim - qs_gt, axis=-1).mean(axis=-1)
frame_errors_res = np.linalg.norm(qs_res - qs_gt, axis=-1).mean(axis=-1)


fig, ax = plt.subplots(figsize=figsize)
line_width = 1

frame_mean_sim = np.mean(frame_errors_sim, axis=0)
frame_std_sim = np.std(frame_errors_sim, axis=0)
frame_mean_res = np.mean(frame_errors_res, axis=0)
frame_std_res = np.std(frame_errors_res, axis=0)

ax.plot(times, frame_mean_sim, linewidth=line_width, label="SysID", c='tab:blue')
ax.fill_between(
    times,
    frame_mean_sim - frame_std_sim,
    frame_mean_sim + frame_std_sim,
    alpha=0.3,
    facecolor='tab:blue',
)

ax.plot(times, frame_mean_res, linewidth=line_width, label="ResPhys", c='tab:orange')
ax.fill_between(
    times,
    frame_mean_res - frame_std_res,
    frame_mean_res + frame_std_res,
    alpha=0.3,
    facecolor='tab:orange',
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Distance Error (m)")
ax.grid()
ax.set_axisbelow(True)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(
    loc = "upper right",
    fontsize='small',
    # fancybox=True,
)
ax.set_xlim(times.min(), times.max())
# ax.set_title("Error distribution of test trajectories")


fig.savefig("fig6_sim2realbeam_error.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
fig.savefig("fig6_sim2realbeam_error.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()