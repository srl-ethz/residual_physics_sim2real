

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
times = np.arange(0, 1.5, 0.01)
qs_sim = np.load("qs_sim_14.npy")[...,2].mean(axis=1)
qs_res = np.load("qs_res_14.npy")[...,2].mean(axis=1)
qs_gt  = np.load("qs_ground_truth_14.npy")[...,2].mean(axis=1)

fig, ax = plt.subplots(figsize=figsize)

offset = qs_gt[0]
line_width = 1

ax.plot(times, (qs_sim-offset), linestyle="-", label="Simulation", linewidth=line_width)
ax.plot(times, (qs_res-offset), linestyle="-", label="ResPhys", linewidth=line_width)
ax.plot(times, (qs_gt-offset), linestyle="--", label="Target", linewidth=line_width)
ax.legend(loc="lower right", ncol=1, fancybox=True, fontsize=6)
#, bbox_to_anchor=[0.5, -0.35]

ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.set_xlim(times.min(), times.max())
ax.set_ylim(qs_gt.min()-offset-2e-3, qs_gt.max()-offset+1e-3)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.grid()
ax.set_axisbelow(True)


fig.savefig("fig5_simbeamtrajectory.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
fig.savefig("fig5_simbeamtrajectory.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()