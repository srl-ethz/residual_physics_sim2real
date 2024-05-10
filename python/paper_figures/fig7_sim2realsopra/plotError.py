

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
times = np.arange(0, 10, 0.01)[:-1]

qs_sim, qs_res, qs_gt, qs_dd = [], [], [], []
trajectories = np.arange(180, 200)
for i in trajectories:
    qs_sim.append(np.load(f"trajectory_data/sim_markers_{i}.npy"))
    qs_res.append(np.load(f"trajectory_data/res_markers_{i}.npy"))
    qs_gt.append(np.load(f"trajectory_data/real_markers_{i}.npy")[:-1])
    qs_dd.append(np.load(f"trajectory_data/dd_markers_{i}.npy"))
qs_sim = np.stack(qs_sim, axis=0)
qs_res = np.stack(qs_res, axis=0)
qs_gt  = np.stack(qs_gt, axis=0)
qs_dd  = np.stack(qs_dd, axis=0)

start, end = 0, 500
times = times[start:end]
qs_sim = qs_sim[:, start:end]
qs_res = qs_res[:, start:end]
qs_gt  = qs_gt[:, start:end]
qs_dd = qs_dd[:, start:end]

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

ax.plot(times, frame_mean_dd, linewidth=line_width, label="Datadriven", c='tab:red')
ax.fill_between(
    times,
    frame_mean_dd - frame_std_dd,
    frame_mean_dd + frame_std_dd,
    alpha=0.3,
    facecolor='tab:red',
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Distance Error (m)")
ax.grid()
ax.set_axisbelow(True)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(loc="lower center", bbox_to_anchor=[0.5, -0.6], fontsize=6, ncol=4, fancybox=True, handlelength=0.5, columnspacing=1.0, handletextpad=0.3)
ax.set_xlim(times.min(), times.max()+0.02)
# ax.set_title("Error distribution of test trajectories")


fig.savefig("fig7_sim2realsopra_error.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
fig.savefig("fig7_sim2realsopra_error.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()