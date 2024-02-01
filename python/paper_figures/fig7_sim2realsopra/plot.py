

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
qs_sim = np.load("trajectory_data/sim_markers_187.npy").mean(axis=1)
qs_res = np.load("trajectory_data/res_markers_187.npy").mean(axis=1)
qs_gt  = np.load("trajectory_data/real_markers_187.npy")[:-1].mean(axis=1)

start, end = 0, 500
times = times[start:end]
qs_sim = qs_sim[start:end]
qs_res = qs_res[start:end]
qs_gt  = qs_gt[start:end]


fig, ax = plt.subplots(figsize=figsize)

line_width = 1
offset = qs_gt[0, 1]
ax.plot(times, qs_sim[:, 1]-offset, linestyle="-", label="SysID", linewidth=line_width, zorder=1)
ax.plot(times, qs_res[:, 1]-offset, linestyle="-", label="ResPhys", linewidth=line_width, zorder=2)
ax.plot(times, qs_gt[:, 1]-offset, linestyle="--", label="Target", linewidth=0.8*line_width, zorder=3)
ax.legend(loc="lower center", bbox_to_anchor=[0.5, -0.6], fontsize=6, ncol=3, fancybox=True, handlelength=0.5, columnspacing=1.0, handletextpad=0.3)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.set_xlim(times.min(), times.max()+0.02)
#ax.set_ylim(-2e-2, 2e-2)
#ax.set_ylim(qs_gt.min()-offset-2e-3, qs_gt.max()-offset+1e-3)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#ax.minorticks_on()
#ax.xaxis.set_tick_params(which='minor', bottom=False)
from matplotlib.ticker import AutoMinorLocator
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.grid(True, which='both')
ax.set_axisbelow(True)


fig.savefig("fig7_sim2realsopra.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
fig.savefig("fig7_sim2realsopra.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()