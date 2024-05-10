import sys
sys.path.append('../..')
# sys.path.append('../../..')

import numpy as np
import torch
from env_cantilever import CantileverEnv3d
from init_beam import optimize_init_force
from optimize_trajectory import optimize_trajectoryfull

if __name__ == '__main__':
    weights = [0.05, 0.06, 0.07, 0.1, 0.09, 0.08, 0.11, 0.12, 0.15, 0.09, 0.13, 0.14, 0.16, 0.17, 0.2,0.18,0.22,0.21]
    prepare = 100
    forward = 0
    forward_times = []
    transformed_data = []
    idxs = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    for idx in idxs[:1]:
        youngs_modulus = 215856
        poissons_ratio = 0.45
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
        cantilever = CantileverEnv3d(42, f'test{idx}', hex_params)
        q_init = cantilever._q0
        q0 = torch.from_numpy(cantilever._q0)
        q_ = q0.reshape(-1, 3)
        v0 = torch.zeros(q0.shape, dtype=torch.float64)
        weight = weights[idx] * 9.80709
        res_force = torch.zeros(q0.shape, dtype=torch.float64)
        res_force[-46::3] = - weight / 16
        qs_real_ = np.load("weight_data_ordered/q_data_reorder.npz")
        steady_state = qs_real_[f'arr_0'][:, :, -1] * 1e-3
        # steady_state = np.load(f"weight_data_ordered/qs_real{idxs[13]}_reorder.npy")[:, :, 62]*1e-3
        # qs_real = np.load(f"weight_data_ordered/q_data_reorder.npz")[f'arr_1'] * 1e-3
        # steady_state = qs_real[:, :, 667]

        R, t = cantilever.fit_realframe(steady_state)
        ######################################
        steady_state_transformed = steady_state @ R.T + t

        ########################################
        qs_real = np.load(f'weight_data_ordered/qs_real{idx}_reorder.npy' )
        target = qs_real[:, :, 0] * 1e-3
        target = target @ R.T + t
        cantilever.interpolate_markers_3d(q_.detach().numpy(), steady_state_transformed)
        num_epochs = 300
        num_frames = 150
        optimize_init_force(num_epochs, num_frames, target, cantilever, 0.01, id=idx, res_force=res_force, suffix="straight")
        target_data = qs_real * 1e-3
        target_data_flatten = np.zeros((target_data.shape[0] * target_data.shape[1], target_data.shape[2]))
        for i in range(num_frames):
            target_data_tmp = target_data[:,:,i]
            target_data_tmp = target_data_tmp @ R.T + t
            target_data_flatten[:, i] = target_data_tmp.flatten()
        
        num_frames = 150
        num_epochs = 300
        optimize_trajectoryfull(f'test{idx}', cantilever, num_frames, num_epochs, target_data_flatten, 0.01, idx, suffix="straight")