import sys
sys.path.append('../')
sys.path.append('../..')

import os
import time
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from env_arm import ArmEnv
from torch import nn

from arm_model._utils import plot_curve, icp


seed = 42
np.random.seed(seed)
np.set_printoptions(precision=3)

def init_realdata (datafile, start_frame=0, num_frames=1000, traj_idx=None):
    """
    Load real data from .npy file. Logged pressure is in mbar.
    """
    data = np.load(f"{datafile}", allow_pickle=True)[()]
    if traj_idx is None:
        markers = data['data'][:, start_frame:start_frame+num_frames]
        pressures = 100*data['p'][:, start_frame:start_frame+num_frames]
    else:
        markers = data['data'][traj_idx, start_frame:start_frame+num_frames]
        pressures = 100*data['p'][traj_idx, start_frame:start_frame+num_frames]

    return pressures, markers


def init_simenv (model, mesh_path, options={}):
    """
    model: str, path to .vtk file
    Initialize DiffPD simulation environment.
    """
    # General simulation material parameters
    youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 263824
    poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
    density = 1.07e3
    state_force = [0, 0, -9.80709]
    params = {
        "density": density,
        "youngs_modulus": youngs_modulus,
        "poissons_ratio": poissons_ratio,
        "state_force_parameters": state_force,
        "mesh_type": "tet",
        "refinement": 1,
        "arm_file": model
    }
    sopra_env = ArmEnv(seed, mesh_path, options=params)

    # Forward simulation stepping parameters
    thread_ct = 8
    method = 'pd_eigen'
    opt = { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-7, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 }

    return sopra_env, method, opt



def plot_trajectory (sim_q, real_q, dim=0, marker_idx=0, dt=1):
    """
    Plot trajectory of real and simulated data.
    """
    plot_curve(inputs={"Simulated": sim_q[:, marker_idx, dim]}, baselines={"Real": real_q[:, marker_idx, dim]}, yaxis="z-displacement (m)", filename=f"displacement_{dim}", folder=f"{sopra_env._folder}", dt=dt)



def align_sim_real (unaligned_q, sim_q):
    """
    Align simulated and real frame by matching the motion marker base.
    """
    real_q, T, distances, marker_idx = icp(unaligned_q, sim_q)

    ### Manually craft the base frame of the robot from the top three markers
    base_q_unaligned = unaligned_q[:,:3]


    ### Define the simulated base frame
    # TODO: Figure out 
    return real_q, marker_idx

if __name__ == '__main__':
    ### Load real data
    real_p, unaligned_q = init_realdata("arm_data/sopra_markers3d_260523.npy", traj_idx=0)


    ### Initialize simulation environment
    sopra_env, method, opt = init_simenv()

    ### Align simulation with real pose
    real_q, marker_idx = align_sim_real(unaligned_q, sopra_env._q0.reshape(-1,3))

    ### Forward simulation, data collected at 100Hz
    dt = 0.01
    num_frames = 1000
    q, v = torch.tensor(sopra_env._q0), torch.tensor(sopra_env._v0)
    log = {
        'q': [q.reshape(-1,3)],
        'v': [v.reshape(-1,3)],
    }
    for i in range(num_frames-1):
        print(f"Pressurization step {i+1}/{num_frames}: {real_p[i]}")
        f_ext = torch.tensor(sopra_env.apply_inner_pressure(real_p[i], q, chambers=[0,1,2,3,4,5])).double()
        q, v = sopra_env.forward(q, v, act=None, f_ext=f_ext, dt=dt)
        log['q'].append(q.reshape(-1,3))
        log['v'].append(v.reshape(-1,3))
    
        sopra_env.vis_dynamic_markers("plots", q.detach().cpu().numpy(), real_q[i], frame=i)

    ### Plot results
    plot_trajectory(np.stack(log['q'], axis=0)[:,marker_idx], real_q[:num_frames], marker_idx=11, dt=dt)

    ### Create video from frame visualization
    framerate = int(1/dt)
    folder = f"{sopra_env._folder}/plots"
    os.system(f"ffmpeg -y -framerate {framerate} -i {folder}/%04d.png -c:v libx264 -crf 0 {sopra_env._folder}/alignment.mp4")


    #import pdb; pdb.set_trace()


