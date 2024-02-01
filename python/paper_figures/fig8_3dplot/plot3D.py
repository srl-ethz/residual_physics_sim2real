

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})

plt.rcParams['axes.labelpad'] = -10


mm = 1 / 25.4
figsize = (40 * mm, 40 * mm)

### Load Data
times = np.arange(0, 10, 0.01)[:-1]
qs_sim = 1e2 * np.load("sim_markers_187.npy")[:,-1]
qs_res = 1e2 * np.load("res_markers_187.npy")[:,-1]
qs_gt  = 1e2 * np.load("real_markers_187.npy")[:-1, -1]

start, end = 190, 251
times = times[start:end]
qs_sim = qs_sim[start:end]
qs_res = qs_res[start:end]
qs_gt  = qs_gt[start:end]

fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(projection='3d')

line_width = 0.5
offsetX = qs_gt[0, 0]
offsetY = qs_gt[0, 1]
offsetZ = qs_gt[0, 2]

disp_sim = [qs_sim[:, 0]-offsetX, qs_sim[:, 1]-offsetY, qs_sim[:, 2]-offsetZ]
disp_res = [qs_res[:, 0]-offsetX, qs_res[:, 1]-offsetY, qs_res[:, 2]-offsetZ]
disp_gt  = [qs_gt[:, 0]-offsetX, qs_gt[:, 1]-offsetY, qs_gt[:, 2]-offsetZ]

skip = 20
ax.scatter(disp_sim[0][::skip], disp_sim[1][::skip], disp_sim[2][::skip], marker='o', s=1, label="SysID", zorder=1, depthshade=False)
ax.scatter(disp_res[0][::skip], disp_res[1][::skip], disp_res[2][::skip], marker='o', s=1, label="ResPhys", zorder=2, depthshade=False)
ax.scatter(disp_gt[0][::skip], disp_gt[1][::skip], disp_gt[2][::skip], marker='o', s=1, label="Target", zorder=3, c='tab:green', depthshade=False)

# ax.scatter(qs_sim[:, 0]-offsetX, qs_sim[:, 1]-offsetY, zs=qs_sim[:, 2]-offsetZ, linestyle="-", linewidth=line_width, marker='o', s=1, label="SysID", zorder=3)
# ax.scatter(qs_res[:, 0]-offsetX, qs_res[:, 1]-offsetY, zs=qs_res[:, 2]-offsetZ, linestyle="-", linewidth=line_width, marker='o', s=1, label="ResPhys", zorder=2)
# ax.scatter(qs_gt[:, 0]-offsetX, qs_gt[:, 1]-offsetY, zs=qs_gt[:, 2]-offsetZ, linestyle="--", linewidth=line_width, marker='o', s=1, label="Target", zorder=1)

skip = 1
ax.plot(disp_sim[0][::skip], disp_sim[1][::skip], disp_sim[2][::skip], linestyle="-", linewidth=line_width, zorder=3)
ax.plot(disp_res[0][::skip], disp_res[1][::skip], disp_res[2][::skip], linestyle="-", linewidth=line_width, zorder=1)
ax.plot(disp_gt[0][::skip], disp_gt[1][::skip], disp_gt[2][::skip], linestyle="--", linewidth=line_width, zorder=2, c='tab:green')

ax.legend(loc="lower center", bbox_to_anchor=[0.5, -0.28], fontsize=6, ncol=3, fancybox=True, handlelength=0.5, columnspacing=1.0, handletextpad=0.3)


plt.locator_params(nbins=2)
ax.tick_params(axis='both', which='major', pad=-4)
ax.tick_params(axis='z', which='major', pad=-4)
ax.yaxis.labelpad = -6
ax.zaxis.labelpad = -10
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")
ax.set_xlim(-2, 5.2)
ax.set_ylim(-4.8, 2.4)
ax.set_zlim(-2, 5)
ax.grid()
ax.set_axisbelow(True)
ax.view_init(elev=25, azim=-65)
#ax.dist = 20


fig.savefig("fig8_sim2realsopra3d.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
fig.savefig("fig8_sim2realsopra3d.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()