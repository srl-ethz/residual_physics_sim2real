

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
idx = 9
qs_sim = np.load(f"trajectory_data/sim_markers_{idx}.npy").mean(axis=1)
qs_res = np.load(f"trajectory_data/res_markers_{idx}.npy").mean(axis=1)
qs_gt  = np.load(f"trajectory_data/real_markers_{idx}.npy")[:-1].mean(axis=1)

start, end = 0, 500
times = times[start:end]
qs_sim = qs_sim[start:end]
qs_res = qs_res[start:end]
qs_gt  = qs_gt[start:end]

line_width = 1

fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
axs = [fig.add_subplot(211), fig.add_subplot(212)]
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
ax.set_ylabel("Displacement (m)", labelpad=8)
plt.subplots_adjust(hspace=0.6)

offset = qs_gt[0, 0]
axs[0].plot(times, qs_sim[:, 0]-offset, linestyle="-", label="SysID", linewidth=line_width, zorder=1)
axs[0].plot(times, qs_res[:, 0]-offset, linestyle="-", label="ResPhys", linewidth=line_width, zorder=2)
axs[0].plot(times, qs_gt[:, 0]-offset, linestyle="--", label="Target", linewidth=0.7*line_width, zorder=3)

axs[0].set_title("x-axis", fontsize=7)
axs[0].set_xlim(times.min(), times.max()+0.02)
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[0].tick_params(axis='x', which='both', labelbottom=False)
axs[0].tick_params(axis='y', which='minor', left=False)
axs[0].grid()
axs[0].set_axisbelow(True)

offset = qs_gt[0, 1]
axs[1].plot(times, qs_sim[:, 1]-offset, linestyle="-", label="SysID", linewidth=line_width, zorder=1)
axs[1].plot(times, qs_res[:, 1]-offset, linestyle="-", label="ResPhys", linewidth=line_width, zorder=2)
axs[1].plot(times, qs_gt[:, 1]-offset, linestyle="--", label="Target", linewidth=0.7*line_width, zorder=3)

axs[1].set_title("y-axis", fontsize=7)
axs[1].set_xlabel("Time (s)")
axs[1].set_xlim(times.min(), times.max()+0.02)
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[1].grid()
axs[1].grid(axis='y', which='minor')
axs[1].set_axisbelow(True)
axs[1].legend(loc="lower center", bbox_to_anchor=[0.5, -1.3], fontsize=6, ncol=3, fancybox=True, handlelength=0.5, columnspacing=1.0, handletextpad=0.3)


fig.savefig("fig7_sim2realsopraXY.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
fig.savefig("fig7_sim2realsopraXY.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()