import sys
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')

import time
import numpy as np
import torch
from env_cantilever import CantileverEnv3d
from _visualization import error_computation
### Apply System Identification to the Beam Model

seed = 42
np.random.seed(seed)

def damping_opt(initial_real_markers, trajectories, num_frames, dt):
    youngs_modulus = 808619.4
    poissons_ratio = 0.499
    density = 1.07e3
    state_force = [0, 0, -9.80709]
    hex_params = {
        'density': density,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'hex',
        'refinement': 1,
    }
    weights = [0.05, 0.06, 0.07, 0.1, 0.09, 0.08, 0.11, 0.12, 0.15, 0.09, 0.13, 0.14, 0.16, 0.17, 0.2,0.18,0.22,0.21]
    env = CantileverEnv3d(42, 'beam', hex_params)
    damping_coeff = torch.normal(0, 1e-3, [1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([damping_coeff], lr=1e-2)
    for epoch_i in range(20):
        total_loss = 0
        for traj_i in trajectories:
            initial_real_marker = initial_real_markers[f'arr_{traj_i}'][:,:,-1]*1e-3
            # initial_real_marker = initial_real_markers[f'arr_10'][:,:,830]*1e-3
            R, t = env.fit_realframe(initial_real_marker)
            real_markers_init = initial_real_marker @ R.T + t
            env.interpolate_markers_3d(env._q0.reshape(-1,3), real_markers_init)
            qs_real = np.load(f'weight_data_ordered/qs_real{traj_i}_reorder.npy' )
            target_data = qs_real * 1e-3
            target_data_flatten = np.zeros((target_data.shape[0] * target_data.shape[1], target_data.shape[2]))
            for i in range(num_frames):
                target_data_tmp = target_data[:,:,i]
                target_data_tmp = target_data_tmp @ R.T + t
                target_data_flatten[:, i] = target_data_tmp.flatten()
            target_data_flatten = torch.from_numpy(target_data_flatten)
            data_info = np.load(f"cantilever_data_fix_registration/optimized_data_{traj_i}.npy", allow_pickle=True)[()]
            qs = data_info['q_trajectory']
            q = torch.from_numpy(qs[0])
            v = torch.zeros_like(q)
            
            for frame_i in range(1, num_frames):
                q, v = env.forward(q, v, f_ext=damping_coeff*v, dt=dt)
                qx = q.reshape(-1,3)

                qx_marker = env.get_markers_3d(qx)
                loss = ((-qx_marker.flatten() + target_data_flatten[:, frame_i])**2).sum()
                total_loss += loss
            total_loss /= num_frames
        total_loss /= len(trajectories)
        total_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch_i}, loss {total_loss.item()}, damping_coeff {damping_coeff.item()}")
    return damping_coeff.detach()

def run_simulation (trajectories, damping_coeff, dt=0.01):
    """
    Simple wrapper to run the simulation with the specified Young's Modulus.
    """
    ### Define simulation environment
    folder = "damping_opt"

    env_params = {
        "youngs_modulus": 808619.4,
        "density": 1.07e3,
        "poissons_ratio": 0.499,
        "state_force_parameters": [0, 0, -9.80709],
        "mesh_type": "tet",
        "refinement": 1,
    }
    env = CantileverEnv3d(seed, folder, env_params)
    env.method = 'pd_eigen'
    env.opt = { 'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 16, 'use_bfgs': 1, 'bfgs_history_size': 10 }
    print(f"DOFs: {env._dofs}")


    ### Perform Forward Simulation using Real Pressures
    num_trajectories = len(trajectories)
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
            q, v = env.forward(q, v, f_ext=damping_coeff*v, dt=dt)
            
            # Append simulated markers
            qs_sim[idx].append(q.reshape(-1,3))

            if frame_i % 100 == 0:
                print(f"Trajectory {traj_i} Frame {frame_i}: {time.time() - start_time:.2f}s")
                
        qs_sim[idx] = torch.stack(qs_sim[idx], axis=0)
    qs_sim = torch.stack(qs_sim, axis=0)

    return env, qs_sim

        

if __name__ == '__main__':
    dt = 0.01
    trajectories = [0, 3, 4, 6, 8, 10, 12, 13, 17]
    test_trajectories = [2, 7, 9, 11, 14, 16]
    qs_real_ = np.load("weight_data_ordered/q_data_reorder.npz")
    num_frames = 140
    damping_coeff = damping_opt(qs_real_, trajectories, 150, dt)

    env, qs_sim_opt = run_simulation(test_trajectories, damping_coeff)

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
            sim_markers_opt[idx].append(env.get_markers_3d(qs_sim_opt[idx, frame_i]).numpy())

        sim_markers_opt[idx] = np.stack(sim_markers_opt[idx], axis=0)
    real_markers_series = np.stack(real_markers_series, axis=0)
    
    sim_markers_opt = np.stack(sim_markers_opt, axis=0)


    ### Quantitative Error Analysis
    opt_error_mean, opt_error_std = error_computation(sim_markers_opt, real_markers_series, dt, "plots/plots_opt")

    print("Optimized Damping Coefficient: ", damping_coeff)
    print('-'*10)
    for traj_i in range(len(test_trajectories)):
        print(f"Trajectory {traj_i}::\t---\tOptimized Marker Error: \t{1000*opt_error_mean[traj_i].mean():.4f}mm +- {1000*opt_error_std[traj_i].mean():.2f}mm")