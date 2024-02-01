import sys
sys.path.append('..')
sys.path.append('../..')

import time
import scipy
import numpy as np
import torch

from env_cantilever import CantileverEnv3d
import matplotlib.pyplot as plt
from _visualization import error_computation, plot_histogram
### Apply System Identification to the Beam Model

seed = 42
np.random.seed(seed)


def system_identification (initial_real_markers, trajectories, num_frames, dt, verbose=False, folder="sysID"):
    """
    Fit the Young's Modulus of the arm model such that the simulated markers best match the real motion marker locations.

    Arguments:
        real_p [num_trajectories, num_frames, num_chambers]: Real pressure data
        initial_real_markers [num_trajectories, num_frames, num_markers, xyz]: Real marker data

    Returns:
        E_opt: Optimal Young's Modulus
    """

    # General simulation material parameters
    env_params = {
        "density": 1.07e3,
        # "poissons_ratio": 0.499,
        "state_force_parameters": [0, 0, -9.80709],
        "mesh_type": "hex",
        "refinement": 1,
    }
    # Forward simulation stepping parameters
    method = 'pd_eigen'
    # opt = { 'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 16, 'use_bfgs': 1, 'bfgs_history_size': 10 }

    ### Start Optimization
    x_lb = np.array([np.log(1.5e5), 0.3, -0.2, -0.2, -0.2])
    x_ub = np.array([np.log(1e7), 0.49, 0.2, 0.2, 0.2])
    x_init = np.array([np.log(2e5), 0.4, 0.0, 0.0, 0.1])    # 258321.2 # 263824
    x_bounds = scipy.optimize.Bounds(x_lb, x_ub)

    ### Only optimize for Young's Modulus
    # x_lb = np.array([np.log(50000)])
    # x_ub = np.array([np.log(1000000)])
    # x_init = np.array([np.log(215856)])    # 258321.2 # 263824
    # x_bounds = scipy.optimize.Bounds(x_lb, x_ub)

    loss_list = []
    E_list = []
    nu_list = []
    damping_list = []
    def loss_and_grad (x):
        ### Initialize Environment
        E = np.exp(x[0])
        nu = x[1]
        damping_param = np.array(x[2:]).reshape(1,3)
        nu_list.append(nu)
        #print(E)
        env_params['youngs_modulus'] = E
        env_params['poissons_ratio'] = nu
        env = CantileverEnv3d(seed, folder, env_params)

        ### Forward Simulation for each trajectory
        total_loss = 0
        total_grad = 0
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
            # env.vis_dynamic_sim2real_markers("vis_sysID_all", np.zeros_like(env._q0), torch.zeros_like(interpolated_marker), real_markers[-1])
            data_info = np.load(f"cantilever_data_fix_registration/optimized_data_{traj_i}.npy", allow_pickle=True)[()]
            qs = data_info['q_trajectory']
            q0 = qs[0]
            v0 = np.zeros_like(q0)
            # Discard the base markers for all subsequent error computation
            # real_markers_traj = real_markers[:num_frames]
            # env.qs_real = real_markers_traj

            # Create a function for compute f_ext on the fly
            #f_ext = np.zeros_like(env._q0)
            #f_exts = [f_ext for _ in range(num_frames-1)]
            # Add velocity-dependent force for damping
            def f_ext (i, q, v):
                damping_force = (damping_param * v.reshape(-1,3)).reshape(-1)
                return damping_force

            loss, grad, info = env.simulate(dt, num_frames-1, method, env.opt, q0=q0, v0=v0, f_ext=f_ext, require_grad=True, vis_folder=None)
            
            # Compute gradient
            param_grad = np. array([
                0.1 * info['material_parameter_gradients'][0] * np.exp(x)[0],
                info['material_parameter_gradients'][1],
                0.01 * np.sum(grad[3] * (np.array([[1, 0, 0]]) * np.array(info['v'][:-1]).reshape(-1,3)).reshape(num_frames-1, -1)),
                0.01 * np.sum(grad[3] * (np.array([[0, 1, 0]]) * np.array(info['v'][:-1]).reshape(-1,3)).reshape(num_frames-1, -1)),
                0.01 * np.sum(grad[3] * (np.array([[0, 0, 1]]) * np.array(info['v'][:-1]).reshape(-1,3)).reshape(num_frames-1, -1))
            ])

            total_loss += loss
            total_grad += param_grad

        total_loss /= len(trajectories)
        total_grad /= len(trajectories)

        print('loss: {:8.4e}, |grad|: {:8.3e}, forward time: {:6.2f}s, backward time: {:6.2f}s, E: {:.1f}, nu: {:.4f}, damping: [{:.6f}, {:.6f}, {:.6f}]'.format(total_loss, np.linalg.norm(total_grad), len(trajectories)*info['forward_time'], len(trajectories)*info['backward_time'], E, nu, damping_param[0,0], damping_param[0,1], damping_param[0,2]))
        loss_list.append(total_loss)

        return total_loss, total_grad
    

    t0 = time.time()
    try: 
        # result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
        #     method='L-BFGS-B', jac=True, bounds=x_bounds, options={ 'ftol': 1e-6, 'gtol': 1e-8, 'maxiter': 50 })
        # result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
        #     method='Newton-CG', jac=True, bounds=x_bounds, options={ 'xtol': 1e-6, 'maxiter': 50 })
        # result = scipy.optimize.minimize_scalar(loss_and_grad, bounds=[x_lb, x_ub], tol=1e-6, method='bounded', options={'maxiter': 50 })

        ### Writing out the basinhopping explanations on temperature: It should be the distance between local optima function value differences, but what does this do in practice? Between 1e0 and 1e-6 there is no difference. I interpreted it as if a new global minimum should be considered if the difference to the old global minimum is smaller than T, but this doesn't seem to be the case. Oh but really small T such as 1e-9 does seem to train a bit faster, for some reason.
        ### With stepsize 0.1 we get jumps in YM of 20k, and poisson ratio of 0.1, which is reasonable. 0.5 step size does YM jumps of 100k and reaches bounds for poisson always.
        result = scipy.optimize.basinhopping(loss_and_grad, np.copy(x_init), niter=5, T=1e-9, stepsize=0.05, minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True, 'bounds': x_bounds, 'options': { 'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 5 }}, disp=True)

        E_opt = np.exp(result.x[0])
        nu_opt = result.x[1]
        damping_opt = np.array(result.x[2:]).reshape(1,3)
        # nu_opt = 0.499
        print(f"Optimization time: {time.time()-t0:6.2f}s")

    finally: 
        damping_list = np.concatenate(damping_list, axis=0)

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
        fig.savefig(f"{folder}/paramE_history.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(nu_list)
        ax.set_xlim([0, len(nu_list)])
        ax.set_title("Parameter History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Poisson Ratio")
        ax.grid()
        fig.savefig(f"{folder}/paramnu_history.png", dpi=300, bbox_inches='tight')
        plt.close()

        for i in range(damping_list.shape[1]):
            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(damping_list[:,i])
            ax.set_xlim([0, len(damping_list[:,i])])
            ax.set_title("Parameter History")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"Damping Parameter {i}")
            ax.grid()
            fig.savefig(f"{folder}/paramdamping{i}_history.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Scatter plot of E vs nu with loss as contour plot in background
        fig, ax = plt.subplots(figsize=(4,3))
        ax.tricontourf(np.array(E_list)[:len(loss_list)], np.array(nu_list)[:len(loss_list)], np.array(loss_list), cmap='viridis')
        ax.plot(np.array(E_list), np.array(nu_list), '--', color='black', linewidth=0.5, marker='o', markersize=2)
        # ax.set_xlim([np.exp(x_lb[0]), np.exp(x_ub[0])])
        # ax.set_ylim([x_lb[1], x_ub[1]])
        ax.set_title("Parameter History")
        ax.set_xlabel("Young's Modulus")
        ax.set_ylabel("Poisson Ratio")
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.grid()
        fig.savefig(f"{folder}/param_history_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Store convergence
        np.savetxt(f"{folder}/loss_history.txt", np.array(loss_list))
        np.savetxt(f"{folder}/paramE_history.txt", np.array(E_list))
        np.savetxt(f"{folder}/paramnu_history.txt", np.array(nu_list))
        np.savetxt(f"{folder}/paramdamping_history.txt", damping_list)

    return E_opt, nu_opt, damping_opt


def run_simulation (trajectories, youngs_modulus, poisson_ratio, damping_param, dt=0.01, folder="sysID"):
    """
    Simple wrapper to run the simulation with the specified Young's Modulus, Poisson's Ratio, and damping parameter.
    """
    ### Define simulation environment
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
    env.opt = { 'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-8, 'rel_tol': 1e-8, 'verbose': 0, 'thread_ct': 4, 'use_bfgs': 1, 'bfgs_history_size': 10 }
    print(f"DOFs: {env._dofs}")


    ### Perform Forward Simulation starting from registration position, falling under gravity.
    num_trajectories = len(trajectories)
    qs_sim = [[] for _ in range(num_trajectories)]
    for idx, traj_i in enumerate(trajectories):
        data_info = np.load(f"cantilever_data_fix_registration/optimized_data_{traj_i}.npy", allow_pickle=True)[()]
        qs = data_info['q_trajectory']
        fixed_dt = 0.01 # Data collected at 100Hz
        start_time = time.time()
        q = torch.from_numpy(qs[0])
        v = torch.zeros_like(q)
        num_frames = int(fixed_dt / dt) * qs.shape[0] # In case of higher simulation frequency than data collection frequency

        # Append initial frame
        qs_sim[idx].append(q.reshape(-1,3))

        # Iterate over all frames
        for frame_i in range(1, num_frames):
            #f_ext = torch.zeros(env.dofs, dtype=torch.float64)
            damping_force = (damping_param * v.detach().numpy().reshape(-1,3)).reshape(-1)
            q, v = env.forward(q, v, f_ext=torch.from_numpy(damping_force), dt=dt)
            
            # Append simulated markers
            qs_sim[idx].append(q.reshape(-1,3))

            if frame_i % 100 == 0:
                print(f"Trajectory {traj_i} Frame {frame_i}: {time.time() - start_time:.2f}s")
                
        qs_sim[idx] = torch.stack(qs_sim[idx], axis=0)
    qs_sim = torch.stack(qs_sim, axis=0)

    return env, qs_sim


if __name__ == "__main__":
    print("\033[95m Starting System Identification for Beam Sim2Real\033[0m")
    # Set to sampling rate of real data
    dt = 0.01
    trajectories = [0, 3, 4, 6, 8, 10, 12, 13, 17]
    #trajectories = [0]
    test_trajectories = [2, 7, 9, 11, 14, 16]
    #test_trajectories = trajectories
    num_frames = 140
    folder = "SysID_beam_damping"

    ### Loading Real Data
    qs_real_ = np.load("weight_data_ordered/q_data_reorder.npz")
    # print(steady_state)
    base_youngs_modulus = 200000
    base_poisson_ratio = 0.4
    base_damping_param = np.array([[0.0, 0.0, 0.0]])
    opt_youngs_modulus = 6275138.3
    opt_poisson_ratio = 0.49
    opt_damping_param = np.array([[0.001, 0.001, -0.2]])
    opt_youngs_modulus, opt_poisson_ratio, opt_damping_param = system_identification(qs_real_, trajectories, num_frames, dt, verbose=True, folder=folder)
    print(f"Optimized Young's Modulus: {opt_youngs_modulus:.1f}")
    print(f"Optimized Poisson's Ratio: {opt_poisson_ratio:.4f}")
    print(f"Optimized Damping Parameter: {opt_damping_param[0,0], opt_damping_param[0,1], opt_damping_param[0,2]}")


    ### Run initial and final simulation metrics
    # Ability to run the simulation at a lower dt if desired.
    downsample_factor = 1.0

    ### Run initial and final simulation metrics
    env, qs_sim = run_simulation(test_trajectories, base_youngs_modulus, base_poisson_ratio, base_damping_param, dt*downsample_factor, folder=folder)
    _, qs_sim_opt = run_simulation(test_trajectories, opt_youngs_modulus, opt_poisson_ratio, opt_damping_param, dt*downsample_factor, folder=folder)
    qs_sim = qs_sim[:, ::int(1/downsample_factor)]
    qs_sim_opt = qs_sim_opt[:, ::int(1/downsample_factor)]

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
        real_markers = np.transpose(real_markers_old[:,:,:num_frames], [2, 0, 1]) @ R.T + t
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
    init_error_mean, init_error_std = error_computation(sim_markers_init, real_markers_series, dt,  f"{folder}/plots_init")
    opt_error_mean, opt_error_std = error_computation(sim_markers_opt, real_markers_series, dt,  f"{folder}/plots_opt")

    plot_histogram(init_error_mean, opt_error_mean, n_bins=50, vis_folder=f"{folder}/plots")

    print(f"Initial Young's Modulus: {base_youngs_modulus:.1f}")
    print(f"Optimized Young's Modulus: {opt_youngs_modulus:.1f}")
    print('-'*10)
    for traj_i in range(len(test_trajectories)):
        print(f"Trajectory {traj_i}:: \tInitial Marker Error: \t{1000*init_error_mean[traj_i].mean():.4f}mm +- {1000*init_error_std[traj_i].mean():.2f}mm \t---\tOptimized Marker Error: \t{1000*opt_error_mean[traj_i].mean():.4f}mm +- {1000*opt_error_std[traj_i].mean():.2f}mm")

        
    print('-'*10)
    print("Statistics over all trajectories")
    init_trajectory_errors = init_error_mean.mean(axis=1)
    opt_trajectory_errors = opt_error_mean.mean(axis=1)
    print(f"Initial Marker Error: \t{1000*init_trajectory_errors.mean():.4f}mm +- {1000*init_trajectory_errors.std():.2f}mm \t---\tOptimized Marker Error: \t{1000*opt_trajectory_errors.mean():.4f}mm +- {1000*opt_trajectory_errors.std():.2f}mm")
