

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})


youngs_modulus = np.arange(50e3, 2e6, 10e3) / 1e6
sim_error = np.load("grid_search_loss.npy")

mm = 1 / 25.4
#figsize = (40 * mm, 32 * mm)
figsize = (88 * mm, 32 * mm)
fig, ax = plt.subplots(figsize=figsize)

ax.plot(youngs_modulus, sim_error, linestyle="--", marker="o", markersize=1, linewidth=0.5)
ax.scatter(youngs_modulus[np.argmin(sim_error)], np.min(sim_error), marker="x", color="tab:red", s=20, linewidths=1, zorder=2)


### Plot zoomed in plot
# ax2 = plt.axes([.4, .375, .5, .5])
# ax2.plot(youngs_modulus, sim_error, linestyle="--", marker="o", markersize=1, linewidth=0.5)
# plt.setp(ax2, xticks=[], yticks=[])


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

zoom_range = [52, 100]
#axins = inset_axes(ax, width=0.72, height=0.48, bbox_to_anchor=[1, .9], bbox_transform=ax.transAxes)
axins = inset_axes(ax, width=1.8, height=0.48, bbox_to_anchor=[1, .9], bbox_transform=ax.transAxes)
axins.plot(youngs_modulus[zoom_range[0]:zoom_range[1]], sim_error[zoom_range[0]:zoom_range[1]], linestyle="--", marker="o", markersize=1, linewidth=0.5)
axins.scatter(youngs_modulus[np.argmin(sim_error)], np.min(sim_error), marker="x", color="tab:red", s=20, linewidths=1, zorder=2)
#axins.set_yscale("log")
#axins.set_xlim(youngs_modulus[zoom_range[0]], youngs_modulus[zoom_range[1]])
#axins.set_ylim(sim_error[zoom_range[0]:zoom_range[1]].min(), sim_error[zoom_range[0]:zoom_range[1]].max())
axins.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--', linewidth=0.5)
axins.grid()


ax.set_xlabel("Young's Modulus (MPa)")
ax.set_ylabel("Distance Error (m)")
ax.set_xlim(youngs_modulus.min(), youngs_modulus.max())
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_yscale("log")
#ax.grid()
#ax.set_axisbelow(True)


fig.savefig("fig5_gridsearch.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
fig.savefig("fig5_gridsearch.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()