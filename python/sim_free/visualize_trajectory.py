
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time


def trajectory_animation (q, save_file):
    """
    Creates a 3D trajectory animation of the cantilever tip.

    Arguments:
        q (ndarray [T, N, 3]): Array of cantilever tip positions.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    qmin, qmax = q.min(), q.max()
    eps = 1.05
    minmaxRange = eps*(qmax - qmin)
    ax.set_xlim([q.mean((0,1))[0]-minmaxRange/2, q.mean((0,1))[0]+minmaxRange/2])
    ax.set_ylim([q.mean((0,1))[1]-minmaxRange/2, q.mean((0,1))[1]+minmaxRange/2])
    ax.set_zlim([q.mean((0,1))[2]-0.6*minmaxRange, q.mean((0,1))[2]+0.4*minmaxRange])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory of Cantilever Tip')
    line, = ax.plot([], [], [], 'k-', lw=0, marker='o', markersize=2)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def animate(i):
        x = q[i, :, 0]
        y = q[i, :, 1]
        z = q[i, :, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=q.shape[0], init_func=init, blit=True)
    ani.save(f'{save_file}', writer='ffmpeg', fps=30)



if __name__ == '__main__':
    loaded = np.load('../../cantilever_data/sim2sim_vibration/data_real/trajectory0.npy', allow_pickle=True)[()]
    q_real, v_real = loaded['q'], loaded['v']

    # loaded = np.load('../../cantilever_data/sim2sim_vibration/cantilever_data_sim2sim/optimized_data_0.npy', allow_pickle=True)[()]
    # q, v, f = loaded['q_trajectory'], loaded['v_trajectory'], loaded['optimized_forces']
    
    # youngs_modulus = 263824
    # poissons_ratio = 0.499
    # angle = 0.799
    # loaded = np.load(f"data/sim2sim_beam_twisting/sim_{youngs_modulus:.0f}_{poissons_ratio:.4f}_trajectory_{angle:.3f}rad.npy", allow_pickle=True)[()]
    # q, v = loaded['q'], loaded['v']

    ### Real data in mm
    q = 1e-3 * np.load('../../cantilever_data/weight_data_ordered/qs_real0_reorder.npy', allow_pickle=True).transpose(2, 0, 1)

    ## Cantilever data
    # data = np.load('../../cantilever_data/cantilever_data_fix_registration/q_trajectoryfull0_reorder.npz', allow_pickle=True)
    # q = []
    # for key in data.keys():
    #     q.append(data[key])
    # q = np.stack(q, axis=0)

    # weights = [0.05, 0.06, 0.07, 0.1, 0.09, 0.08, 0.11, 0.12, 0.15, 0.09, 0.13, 0.14, 0.16, 0.17, 0.2, 0.18, 0.22, 0.21]
    # for i, weight in enumerate(weights):
    #     if i == 1:
    #         continue
    #     dataQ = np.load(f'../../cantilever_data/cantilever_data_fix_registration/q_trajectoryfull{i}_reorder.npz', allow_pickle=True)
    #     dataV = np.load(f'../../cantilever_data/cantilever_data_fix_registration/v_trajectoryfull{i}_reorder.npz', allow_pickle=True)
    #     q, v = [], []
    #     for keyQ, keyV in zip(dataQ.keys(), dataV.keys()):
    #         q.append(dataQ[keyQ])
    #         v.append(dataV[keyV])
    #     q = np.stack(q, axis=0)
    #     v = np.stack(v, axis=0)

    #     data = {"q" : q, "v" : v}
    #     np.save(f"data/sim2real_beam_oscillating/real_trajectory_{weight:.3f}kg.npy", data)

    ## Sopra Data
    for i in range(200):
        loadedData = np.load(f'../../sopra_data/augmented_dataset_smaller_tol/optimized_data_{i}.npy', allow_pickle=True)[()]

        q = loadedData['q_trajectory']
        v = loadedData['v_trajectory']
        f = loadedData['pressure_forces']
        # Filter out weird nodes that are not part of sopra
        q[:, 45*3:54*3] = 0
        q[:, 120*3:126*3] = 0
        v[:, 45*3:54*3] = 0
        v[:, 120*3:126*3] = 0
        f[:, 45*3:54*3] = 0
        f[:, 120*3:126*3] = 0
        # Pad force at the last timestep (since no actuation anymore there, but this point should not be used).
        f = np.concatenate([f, np.ones_like(f[0:1])*np.nan], axis=0)
        data = {"q" : q, "v" : v, "f" : f}
        np.save(f"data/sim2real_arm/real_trajectory_{i}.npy", data)

        print(f"Finished {i}")

    q_real = q_real.reshape(q_real.shape[0], -1, 3)
    q = q.reshape(q.shape[0], -1, 3)

    cantileverTipReal = q_real[:, :, 2].mean(-1)
    cantileverTip = q[:, :, 2].mean(-1)
    plt.plot(q[:, :, 0].mean(-1), q[:, :, 1].mean(-1))
    # plt.plot(cantileverTip, label='Simulation')
    # plt.plot(cantileverTipReal, label='Real')
    plt.legend()
    plt.savefig('outputs/test.png')
    plt.close()

    # Difference
    # plt.plot(cantileverTip - cantileverTipReal[:-1])
    # plt.savefig('outputs/cantileverTipDiff.png')


    # Plot time series of 3d point cloud of q, and store video
    start = time.time()
    trajectory_animation(q, 'outputs/test.mp4')
    print(f"Time taken: {time.time() - start}s")




