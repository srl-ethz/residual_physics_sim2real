import sys
sys.path.append('../')
sys.path.append('../../..')
sys.path.append('../beam_model/sim2sim_beam_model')

import time
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from env_cantilever import CantileverEnv3d


GRAV_ACC = -9.80709


def generate_oscillating_beam (simParams, timesteps=150, dt=1e-2, visualize=False, PREP_DT=1e-2, PREP_MAX=1000):
    """
    PREP_DT will determine how much numerical damping is applied to the system, and hence how fast the dynamic system converges to steady state.
    """
    weights = [0.01*i for i in range(5, 23)] # Weights from 50g to 220g
    # weights = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.6, 0.7, 0.8, 0.9, 1.0]

    save_folder = 'data/sim2sim_beam_oscillating'
    sim = CantileverEnv3d(42, save_folder, simParams)

    plt.figure(figsize=(10, 5))
    plt.grid()

    startTime = time.time()
    for weight in weights:
        # Run simulation until steady state under weight
        q = torch.from_numpy(sim._q0)
        v = torch.zeros_like(q)

        v_hist = []
        for j in range(PREP_MAX):
            weight_force = weight * GRAV_ACC
            res_force = torch.zeros(q.shape, dtype=torch.float64)
            res_force[-49*3+2::3] = weight_force / 16
            q, v = sim.forward(q, v, f_ext=res_force, dt=PREP_DT)
            v_hist.append(v.detach().numpy())

            # Check if we have reached steady state, if no vertex is moving more than 0.1mm/s for more than 5 frames.
            if np.array(v_hist[-5:]).max() < 1e-6:
                break

        assert j < PREP_MAX, "Did not reach steady state"

        q_trajectory = [q.detach().numpy()]
        v_trajectory = [v.detach().numpy()]
        for _ in range(timesteps):
            q, v = sim.forward(q, v, f_ext=torch.zeros_like(q), dt=dt)
            q_trajectory.append(q.detach().numpy())
            v_trajectory.append(v.detach().numpy())
        data = {"q" : np.stack(q_trajectory, axis=0), "v" : np.stack(v_trajectory, axis=0)}
        np.save(f"{save_folder}/sim_{simParams['youngs_modulus']:.0f}_{simParams['poissons_ratio']:.4f}_trajectory_{weight:.3f}kg.npy", data)

        # Visualize trajectory
        if visualize:
            plt.plot(data['q'][:, 2::3].mean(-1), linewidth=1, label=f"Weight {weight:.4f}kg")
            plt.legend()
            plt.savefig(f"{save_folder}/sim_{simParams['youngs_modulus']:.0f}_{simParams['poissons_ratio']:.4f}_trajectory_{weight:.3f}kg.png")


        print(f"\033[94mTime {time.time()-startTime:.2f}s: Finished trajectory --- sim_{simParams['youngs_modulus']:.0f}_{simParams['poissons_ratio']:.4f}_trajectory_{weight:.3f}kg --- \033[0m")


def generate_twisting_beam (simParams, timesteps=100, dt=1e-2, samples=20, visualize=False):
    """
    PREP_DT will determine how much numerical damping is applied to the system, and hence how fast the dynamic system converges to steady state.
    """
    # twistAngles = np.random.uniform(np.pi / 6, np.pi, size=samples)
    twistAngles = np.linspace(np.pi/6, np.pi, samples)

    save_folder = 'data/sim2sim_beam_twisting'

    plt.figure(figsize=(10, 5))
    plt.grid()

    startTime = time.time()
    for angle in twistAngles:
        simParams['twist_angle'] = angle
        sim = CantileverEnv3d(42, save_folder, simParams)
        q = torch.from_numpy(sim._q0)
        v = torch.zeros_like(q)

        q_trajectory = [q.detach().numpy()]
        v_trajectory = [v.detach().numpy()]
        for _ in range(timesteps):
            q, v = sim.forward(q, v, f_ext=torch.zeros_like(q), dt=dt)
            q_trajectory.append(q.detach().numpy())
            v_trajectory.append(v.detach().numpy())
        data = {"q" : np.stack(q_trajectory, axis=0), "v" : np.stack(v_trajectory, axis=0)}
        np.save(f"{save_folder}/sim_{simParams['youngs_modulus']:.0f}_{simParams['poissons_ratio']:.4f}_trajectory_{angle:.3f}rad.npy", data)

        # Visualize trajectory
        if visualize:
            # Track edge at the bottom of the tip
            tipIdx = np.where(q[2::3] - q[2::3].min() < 1e-4)[0]
            plt.plot(data['q'].reshape(data['q'].shape[0], -1, 3)[:, tipIdx, 1].mean(-1), data['q'].reshape(data['q'].shape[0], -1, 3)[:, tipIdx, 2].mean(-1), linewidth=1, label=f"Angle {angle:.4f}rad")
            # plt.plot(data['q'].reshape(data['q'].shape[0], -1, 3)[:, tipIdx, 1].mean(-1), linewidth=1, label=f"Angle {angle:.4f}rad")
            plt.legend(loc='upper right')
            plt.savefig(f"{save_folder}/sim_{simParams['youngs_modulus']:.0f}_{simParams['poissons_ratio']:.4f}_trajectory_{angle:.3f}rad.png")

        print(f"\033[94mTime {time.time()-startTime:.2f}s: Finished trajectory --- sim_{simParams['youngs_modulus']:.0f}_{simParams['poissons_ratio']:.4f}_trajectory_{angle:.3f}rad --- \033[0m")


def generate_realbeam_marker_v (dataFolder, visualize=False):
    """
    Creates central finite difference approximation of velocity from position data for the real oscillating beam data.
    """
    dt = 1e-2
    weights = [0.05, 0.06, 0.07, 0.1, 0.09, 0.08, 0.11, 0.12, 0.15, 0.09, 0.13, 0.14, 0.16, 0.17, 0.2, 0.18, 0.22, 0.21]
    save_folder = 'data/sim2real_beam_oscillating'
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.grid()
    
    startTime = time.time()
    for i, weight in enumerate(weights):
        q = 1e-3*np.load(f'{dataFolder}/qs_real{i}_reorder.npy', allow_pickle=True).transpose(2, 0, 1)
        if np.isnan(q).any():
            print(f"\033[91mData {i} with weight {weight:.3f}kg contains NaN values\033[0m")
            continue
        q = q.reshape(q.shape[0], -1)
        v = np.zeros_like(q)
        # Forward difference for first and last point, then central difference in between.
        v[0] = (q[1] - q[0]) / dt
        v[-1] = (q[-1] - q[-2]) / dt
        v[1:-1] = (q[2:] - q[:-2]) / (2*dt)
        data = {"q" : q, "v" : v}
        np.save(f"{save_folder}/real_marker_trajectory_{weight:.3f}kg.npy", data)

        # Visualize trajectory
        if visualize:
            plt.plot(data['q'][:, 2::3].mean(-1), linewidth=1, label=f"Weight {weight:.4f}kg")
            # plt.plot(data['v'][:, 2::3].mean(-1), linewidth=1, label=f"Vel Weight {weight:.4f}kg")
            plt.legend(loc='upper right')
            plt.savefig(f"{save_folder}/real_marker_trajectory_{weight:.3f}kg.png")

        print(f"\033[94mTime {time.time()-startTime:.2f}s: Finished trajectory --- real_marker_trajectory_{weight:.3f}kg --- \033[0m")



if __name__ == "__main__":
    params_sim1 = {
        'density': 1070.,
        'youngs_modulus': 215856,
        'poissons_ratio': 0.45,
        'state_force_parameters': [0, 0, GRAV_ACC],
        'mesh_type': 'hex',
        'refinement': 1,
    }
    params_sim2 = {
        'density': 1070.,
        'youngs_modulus': 263824,
        'poissons_ratio': 0.499,
        'state_force_parameters': [0, 0, GRAV_ACC],
        'mesh_type': 'hex',
        'refinement': 1,
    }

    # generate_oscillating_beam(params_sim1, visualize=False)
    # generate_oscillating_beam(params_sim2, visualize=True)
    # generate_twisting_beam(params_sim2, visualize=True)
    # generate_realbeam_marker_v(f'../../cantilever_data/weight_data_ordered', visualize=True)





    