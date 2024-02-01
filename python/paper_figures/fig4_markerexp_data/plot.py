### Create a plot from generated marker ablation data, same visualization style as in sim2real_beam_model/_visualization.py plot_sim2real_errors

# Requires to have first run `python marker_ablation.py`

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})

### Load data
n_markers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 32, 64, 128]
marker_errors = np.loadtxt("marker_errors.csv", delimiter=",")

### Plot data
mm = 1/25.4
figsize = (88*mm, 30*mm)

fig, ax = plt.subplots(figsize=figsize)

ax.plot(n_markers, np.mean(marker_errors, axis=1), marker='x', markersize=5)
plot_percentile = 90
plt.fill_between(
    n_markers, 
    np.percentile(marker_errors, plot_percentile, axis=1),
    np.percentile(marker_errors, 100-plot_percentile, axis=1),
    color='b', alpha=.1
)

### Visualization settings
ax.set_xlabel("Number of Markers (-)")
ax.set_ylabel("Distance Error (m)")
ax.set_xlim([1, len(marker_errors)])
ax.set_xticks(n_markers)
ax.set_yscale('log')
#ax.ticklabel_format(axis="y", style='sci', scilimits=(0,0))
ax.grid()
ax.set_axisbelow(True)
fig.savefig("marker_ablation_error.png", dpi=300, bbox_inches='tight', pad_inches=1*mm)
fig.savefig("marker_ablation_error.pdf", bbox_inches='tight', pad_inches=1*mm)
plt.close()


### Create a box plot of the data
fig, ax = plt.subplots(figsize=figsize)

#ax.plot(n_markers, np.median(marker_errors, axis=1), marker='o', markersize=2, linewidth=1)
#marker_choices = np.arange(0, len(marker_errors))
marker_choices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17], dtype=int)
positions = np.arange(1, len(marker_choices)+1)
positions[-4:] += 1
positions[-3:] += 1
positions[-2:] += 1
positions[-1] += 1
ax.boxplot(marker_errors[marker_choices].T, positions=positions, showfliers=False)
#ax.boxplot(marker_errors.T, positions=n_markers, showfliers=False)

### Visualization settings
ax.set_xlabel("Number of Markers (-)")
ax.set_ylabel("Distance Error (m)")
ax.set_xticks(positions, np.array(n_markers)[marker_choices])
ax.set_yscale('log')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

ax.set_axisbelow(True)
fig.savefig("marker_ablation_error_boxplot.png", dpi=300, bbox_inches='tight', pad_inches=1*mm)
fig.savefig("marker_ablation_error_boxplot.pdf", bbox_inches='tight', pad_inches=1*mm)
plt.close()