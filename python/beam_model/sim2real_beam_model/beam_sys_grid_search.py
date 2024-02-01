import sys
sys.path.append('../')
sys.path.append('../../')

import time
import numpy as np
import torch
from env_cantilever import CantileverEnv3d
import matplotlib.pyplot as plt
from _visualization import error_computation
### Apply System Identification to the Beam Model

seed = 42
np.random.seed(seed)


def system_identification (initial_real_markers, trajectories, num_frames, dt, verbose=False):
    """
    Fit the Young's Modulus of the arm model such that the simulated markers best match the real motion marker locations.

    Arguments:
        real_p [num_trajectories, num_frames, num_chambers]: Real pressure data
        initial_real_markers [num_trajectories, num_frames, num_markers, xyz]: Real marker data

    Returns:
        E_opt: Optimal Young's Modulus
    """
    folder = "sysID"

    # General simulation material parameters
    env_params = {
        "density": 1.07e3,
        "poissons_ratio": 0.499,
        "state_force_parameters": [0, 0, -9.80709],
        "mesh_type": "hex",
        "refinement": 1,
    }
    # Forward simulation stepping parameters
    method = 'pd_eigen'
    # opt = { 'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 16, 'use_bfgs': 1, 'bfgs_history_size': 10 }

    x_lb = 50000
    x_ub = 2000000
    x_init = np.array([np.log(215856)])    # 258321.2 # 263824

    loss_list = []
    E_list = []
    def loss_E(E):
        ### Initialize Environment
        nu = 0.499
        E_list.append(E)
        env_params['youngs_modulus'] = E
        env_params['poissons_ratio'] = nu
        env = CantileverEnv3d(seed, folder, env_params)

        ### Forward Simulation for each trajectory
        total_loss = 0
        total_grad = 0
        env.opt['thread_ct'] = 32
        for traj_i in trajectories:
            initial_real_marker = initial_real_markers[f'arr_{traj_i}'][:,:,-1]*1e-3
            # initial_real_marker = initial_real_markers[f'arr_10'][:,:,830]*1e-3
            R, t = env.fit_realframe(initial_real_marker)
            real_markers_init = initial_real_marker @ R.T + t
            env.interpolate_markers_3d(env._q0.reshape(-1,3), real_markers_init)
            qs_real = np.load(f'weight_data_ordered/qs_real{traj_i}_reorder.npy' )
            real_markers_old = qs_real * 1e-3
            real_markers = np.zeros((real_markers_old.shape[2],real_markers_old.shape[0],real_markers_old.shape[1]),dtype=np.float64)
            for i in range(real_markers.shape[0]):
                real_markers[i] = real_markers_old[:,:,i] @ R.T + t
            env.qs_real_series = real_markers
            interpolated_marker = env.get_markers_3d(torch.from_numpy(env._q0.reshape(-1,3)))
            data_info = np.load(f"cantilever_data_fix_registration/optimized_data_{traj_i}.npy", allow_pickle=True)[()]
            qs = data_info['q_trajectory']
            q0 = qs[0]
            v0 = np.zeros_like(q0)

            # Create a function for compute f_ext on the fly
            f_ext = np.zeros_like(env._q0)
            f_exts = [f_ext for _ in range(num_frames-1)]

            loss, _, info = env.simulate(dt, num_frames-1, method, env.opt, q0=q0, v0=v0, f_ext=f_exts, require_grad=True, vis_folder=None)
            
            total_loss += loss

        total_loss /= len(trajectories)
        print(f"Loss: {total_loss:.4f} \tE: {E:.1f} \t, forward time: {len(trajectories)*info['forward_time']:.2f}s")

        loss_list.append(total_loss)

        return total_loss
    

    t0 = time.time()
    nu_opt = 0.499
    losses = []
    for E in range(x_lb, x_ub, 10000):
        loss = loss_E(E)
        losses.append(loss)
    np.save('grid_search_loss.npy',losses)
    E_opt_idx = np.argmin(losses)
    E_opt = x_lb + E_opt_idx*10000

    ### Plotting Loss History
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(loss_list)
    ax.set_xlim([0, len(loss_list)])
    ax.set_title("Log Loss History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    fig.savefig(f"{folder}/loss_history.png", dpi=300, bbox_inches='tight')
    plt.close()

    ### Plotting Parameter History
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(E_list)
    ax.set_xlim([0, len(E_list)])
    ax.set_title("Parameter History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Young's Modulus")
    ax.grid()
    fig.savefig(f"{folder}/param_history.png", dpi=300, bbox_inches='tight')
    plt.close()


    return E_opt, nu_opt


def run_simulation (trajectories, youngs_modulus, poisson_ratio, dt=0.01):
    """
    Simple wrapper to run the simulation with the specified Young's Modulus.
    """
    ### Define simulation environment
    folder = "sysID"

    env_params = {
        "youngs_modulus": youngs_modulus,
        "density": 1.07e3,
        "poissons_ratio": poisson_ratio,
        "state_force_parameters": [0, 0, -9.80709],
        "mesh_type": "tet",
        "refinement": 1,
    }
    env = CantileverEnv3d(seed, folder, env_params)
    env.method = 'pd_eigen'
    env.opt = { 'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 16, 'use_bfgs': 1, 'bfgs_history_size': 10 }
    print(f"DOFs: {env._dofs}")


    ### Perform Forward Simulation using Real Pressures
    num_trajectories = len(trajectories)#real_p.shape[0]
    qs_sim = [[] for _ in range(num_trajectories)]
    for idx, traj_i in enumerate(trajectories):
        data_info = np.load(f"cantilever_data_fix_registration/optimized_data_{traj_i}.npy", allow_pickle=True)[()]
        qs = data_info['q_trajectory']
        start_time = time.time()
        q = torch.from_numpy(qs[0])
        v = torch.zeros_like(q)
        num_frames = qs.shape[0]

        # Append initial frame
        qs_sim[idx].append(q.reshape(-1,3))

        # Iterate over all frames
        for frame_i in range(1, num_frames):
            f_ext = torch.zeros(env.dofs, dtype=torch.float64)
            q, v = env.forward(q, v, f_ext=f_ext, dt=dt)
            
            # Append simulated markers
            qs_sim[idx].append(q.reshape(-1,3))

            if frame_i % 100 == 0:
                print(f"Trajectory {traj_i} Frame {frame_i}: {time.time() - start_time:.2f}s")
                
        qs_sim[idx] = torch.stack(qs_sim[idx], axis=0)
    qs_sim = torch.stack(qs_sim, axis=0)

    return env, qs_sim


if __name__ == "__main__":
    # Set to sampling rate of real data
    dt = 0.01
    trajectories = [0, 3, 4, 6, 8, 10, 12, 13, 17]
    # trajectories = [3, 4, 6, 8, 10, 12, 13, 17]
    # trajectories = [0]
    test_trajectories = [2, 7, 9, 11, 14, 16]
    num_frames = 140

    ### Loading Real Data


    qs_real_ = np.load("weight_data_ordered/q_data_reorder.npz")
    # print(steady_state)
    opt_youngs_modulus, opt_poisson_ratio = system_identification(qs_real_, trajectories, num_frames, dt, verbose=True)
    # opt_youngs_modulus = 212553.8
    # opt_poisson_ratio = -0.0822
    print(f"Optimized Young's Modulus: {opt_youngs_modulus:.1f}")
    print(f"Optimized Poisson's Ratio: {opt_poisson_ratio:.4f}")

    # opt_youngs_modulus = 394405.1
    # opt_poisson_ratio = 0.499

    ### Run initial and final simulation metrics
    env, qs_sim = run_simulation(test_trajectories, 215856, 0.45)
    _, qs_sim_opt = run_simulation(test_trajectories, opt_youngs_modulus, opt_poisson_ratio)

    ### Find simulated marker locations
    # Either simulation environment above should produce the same initial conditions.
    # Discard the base markers for all subsequent error computation
    # real_markers = real_markers[:num_trajectories, :num_frames, 3:]

    # Generate simulated markers
    sim_markers_init = [[] for _ in range(len(test_trajectories))]
    sim_markers_opt = [[] for _ in range(len(test_trajectories))]
    real_markers_series = []
    for idx, traj_i in enumerate(test_trajectories):
        steady_state = qs_real_[f"arr_{traj_i}"][:, :, -1] * 1e-3
        R, t = env.fit_realframe(steady_state)
        real_marker = steady_state @ R.T + t
        env.interpolate_markers_3d(env._q0.reshape(-1,3), real_marker)
        qs_real = np.load(f'weight_data_ordered/qs_real{traj_i}_reorder.npy' )
        real_markers_old = qs_real * 1e-3
        real_markers = np.zeros((num_frames,real_markers_old.shape[0],real_markers_old.shape[1]),dtype=np.float64)
        for i in range(num_frames):
            real_markers[i] = real_markers_old[:,:,i] @ R.T + t
        real_markers_series.append(real_markers)
        for frame_i in range(num_frames):
            sim_markers_init[idx].append(env.get_markers_3d(qs_sim[idx, frame_i]).numpy())
            sim_markers_opt[idx].append(env.get_markers_3d(qs_sim_opt[idx, frame_i]).numpy())

        sim_markers_init[idx] = np.stack(sim_markers_init[idx], axis=0)
        sim_markers_opt[idx] = np.stack(sim_markers_opt[idx], axis=0)
    real_markers_series = np.stack(real_markers_series, axis=0)
    
    sim_markers_init = np.stack(sim_markers_init, axis=0)
    sim_markers_opt = np.stack(sim_markers_opt, axis=0)


    ### Quantitative Error Analysis
    init_error_mean, init_error_std = error_computation(sim_markers_init, real_markers_series, dt, "plots/plots_init")
    opt_error_mean, opt_error_std = error_computation(sim_markers_opt, real_markers_series, dt, "plots/plots_opt")


    print(f"Initial Young's Modulus: {215856:.1f}")
    print(f"Optimized Young's Modulus: {opt_youngs_modulus:.1f}")
    print('-'*10)
    for traj_i in range(len(test_trajectories)):
        print(f"Trajectory {traj_i}:: \tInitial Marker Error: \t{1000*init_error_mean[traj_i].mean():.4f}mm +- {1000*init_error_std[traj_i].mean():.2f}mm \t---\tOptimized Marker Error: \t{1000*opt_error_mean[traj_i].mean():.4f}mm +- {1000*opt_error_std[traj_i].mean():.2f}mm")