

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})

mm = 1 / 25.4
figsize = (40 * mm, 40 * mm)

### Load Data
times = np.arange(0, 10, 0.01)[:-1]
qs_sim = np.load("trajectory_data/sim_markers_187.npy").mean(axis=1)
qs_res = np.load("trajectory_data/res_markers_187.npy").mean(axis=1)
qs_gt  = np.load("trajectory_data/real_markers_187.npy")[:-1].mean(axis=1)

start, end = 0, 300
times = times[start:end]
qs_sim = qs_sim[start:end]
qs_res = qs_res[start:end]
qs_gt  = qs_gt[start:end]

line_width = 1

fig, ax = plt.subplots(figsize=figsize)

line_width = 1
offsetX = qs_gt[0, 0]
offsetY = qs_gt[0, 1]
ax.scatter(qs_sim[:, 0]-offsetX, qs_sim[:, 1]-offsetY, linestyle="-", linewidth=line_width, marker='o', s=1, label="SysID", zorder=3)
ax.scatter(qs_res[:, 0]-offsetX, qs_res[:, 1]-offsetY, linestyle="-", linewidth=line_width, marker='o', s=1, label="ResPhys", zorder=2)
ax.scatter(qs_gt[:, 0]-offsetX, qs_gt[:, 1]-offsetY, linestyle="--", linewidth=line_width, marker='o', s=1, label="Target", zorder=1)
ax.legend(loc="lower center", bbox_to_anchor=[0.5, -0.6], fontsize=6, ncol=3, fancybox=True, handlelength=0.5, columnspacing=1.0, handletextpad=0.3)

ax.set_xlabel("Displacement X (m)")
ax.set_ylabel("Displacement Y (m)")
#ax.set_xlim(times.min(), times.max())
#ax.set_ylim(qs_gt.min()-offset-2e-3, qs_gt.max()-offset+1e-3)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax.get_xaxis().get_offset_text().set_position((1.2, 0.0))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.grid()
ax.set_axisbelow(True)


fig.savefig("fig7_sim2realsopraXYspace.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
fig.savefig("fig7_sim2realsopraXYspace.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()