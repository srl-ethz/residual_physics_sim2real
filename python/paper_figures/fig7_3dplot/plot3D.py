

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})

plt.rcParams['axes.labelpad'] = -10


mm = 1 / 25.4
figsize = (50 * mm, 42 * mm)

### Load Data
times = np.arange(0, 10, 0.01)[:-1]
idx = 9
# qs_sim = 1e2 * np.load(f"trajectory_data/sim_markers_{idx}.npy")[:,-1]
qs_sim = 1e2 * np.load(f"trajectory_data/sysid_markers_{idx}.npy")[:,-1]
qs_res = 1e2 * np.load(f"trajectory_data/res_markers_{idx}.npy")[:,-1]
qs_gt  = 1e2 * np.load(f"trajectory_data/real_markers_{idx}.npy")[:-1, -1]
qs_dd = 1e2 * np.load(f"trajectory_data/dd_markers_{idx}.npy")[:,-1]
offsetX = qs_gt[0, 0]
offsetY = qs_gt[0, 1]
offsetZ = qs_gt[0, 2]


start, end = 152, 200
times = times[start:end]
qs_sim = qs_sim[start:end]
qs_res = qs_res[start:end]
qs_gt  = qs_gt[start:end]
qs_dd = qs_dd[start:end]

fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(projection='3d')

line_width = 0.5
disp_sim = [qs_sim[:, 0]-offsetX, qs_sim[:, 1]-offsetY, qs_sim[:, 2]-offsetZ]
disp_res = [qs_res[:, 0]-offsetX, qs_res[:, 1]-offsetY, qs_res[:, 2]-offsetZ]
disp_gt  = [qs_gt[:, 0]-offsetX, qs_gt[:, 1]-offsetY, qs_gt[:, 2]-offsetZ]
disp_dd = [qs_dd[:, 0]-offsetX, qs_dd[:, 1]-offsetY, qs_dd[:, 2]-offsetZ]

skip = 15
ax.scatter(disp_sim[0][::skip], disp_sim[1][::skip], disp_sim[2][::skip], marker='o', s=1, label="SysID", zorder=1, depthshade=False)
ax.scatter(disp_dd[0][::skip], disp_dd[1][::skip], disp_dd[2][::skip], marker='o', s=1, label="Data", zorder=2, c='tab:red', depthshade=False)
ax.scatter(disp_res[0][::skip], disp_res[1][::skip], disp_res[2][::skip], marker='o', s=1, label="ResPhys", zorder=3, depthshade=False)
ax.scatter(disp_gt[0][::skip], disp_gt[1][::skip], disp_gt[2][::skip], marker='o', s=1, label="Target", zorder=4, c='tab:green', depthshade=False)

# ax.scatter(qs_sim[:, 0]-offsetX, qs_sim[:, 1]-offsetY, zs=qs_sim[:, 2]-offsetZ, linestyle="-", linewidth=line_width, marker='o', s=1, label="SysID", zorder=3)
# ax.scatter(qs_res[:, 0]-offsetX, qs_res[:, 1]-offsetY, zs=qs_res[:, 2]-offsetZ, linestyle="-", linewidth=line_width, marker='o', s=1, label="ResPhys", zorder=2)
# ax.scatter(qs_gt[:, 0]-offsetX, qs_gt[:, 1]-offsetY, zs=qs_gt[:, 2]-offsetZ, linestyle="--", linewidth=line_width, marker='o', s=1, label="Target", zorder=1)

skip = 1
ax.plot(disp_sim[0][::skip], disp_sim[1][::skip], disp_sim[2][::skip], linestyle="-", linewidth=line_width, zorder=1)
ax.plot(disp_dd[0][::skip], disp_dd[1][::skip], disp_dd[2][::skip], linestyle="-", linewidth=line_width, zorder=2, c='tab:red')
ax.plot(disp_res[0][::skip], disp_res[1][::skip], disp_res[2][::skip], linestyle="-", linewidth=line_width, zorder=3)
ax.plot(disp_gt[0][::skip], disp_gt[1][::skip], disp_gt[2][::skip], linestyle="--", linewidth=line_width, zorder=4, c='tab:green')

ax.legend(loc="lower center", bbox_to_anchor=[0.5, -0.35], fontsize=6, ncol=2, fancybox=True, handlelength=0.5, columnspacing=1.0, handletextpad=0.3)


plt.locator_params(nbins=2)
ax.tick_params(axis='both', which='major', pad=-5)
ax.tick_params(axis='z', which='major', pad=-1)
ax.yaxis.labelpad = -9
ax.zaxis.labelpad = -5
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")
ax.set_xlim(-2., 2.4)
ax.set_ylim(-2.4, 2.4)
ax.set_zlim(-0.4, 0.15)
ax.grid()
ax.set_axisbelow(True)
ax.view_init(elev=25, azim=55)
# ax.dist = 12

fig.savefig("fig7_sim2realsopra3d.png", dpi=300, bbox_inches="tight", pad_inches=6*mm)
fig.savefig("fig7_sim2realsopra3d.pdf", bbox_inches="tight", pad_inches=6*mm)
plt.close()


### RMSE error of segment
rmsesSim = np.linalg.norm(qs_sim - qs_gt, axis=1)
rmsesDD = np.linalg.norm(qs_dd - qs_gt, axis=1)
rmsesRes = np.linalg.norm(qs_res - qs_gt, axis=1)

print(f"RMSE Sim: {np.mean(rmsesSim):.4f}mm +- {np.std(rmsesSim):.4f}mm")
print(f"RMSE DD: {np.mean(rmsesDD):.4f}mm +- {np.std(rmsesDD):.4f}mm")
print(f"RMSE Res: {np.mean(rmsesRes):.4f}mm +- {np.std(rmsesRes):.4f}mm")

