import sys
sys.path.append('../')
sys.path.append('../../..')
from pathlib import Path
import numpy as np
import torch

from env_cantilever import CantileverEnv3d
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("-t", dest="save_folder", required=False)


if __name__ == '__main__':
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
    weights = [0.05, 0.06, 0.07, 0.1, 0.09, 0.08, 0.11, 0.12, 0.15, 0.09, 0.13, 0.14, 0.16, 0.17, 0.2,0.18,0.22,0.21]
    prepare = 200
    save_folder = 'data_real' if args.parse_args().save_folder is None else args.parse_args().save_folder
    cantilever = CantileverEnv3d(42, save_folder, hex_params)
    q = torch.from_numpy(cantilever._q0)
    v = torch.zeros_like(q)

    for id in range(len(weights)):
        if save_folder == 'data_real':
            for j in range(prepare):
                weight = weights[id] * 9.80709
                res_force = torch.zeros(q.shape, dtype=torch.float64)
                res_force[-49*3+2::3] = - weight / 16
                q, v = cantilever.forward(q, v, f_ext=res_force, dt=0.01)
        elif save_folder == 'data_sim':
            q = torch.from_numpy(np.load(f'data_real/trajectory{id}.npy', allow_pickle=True)[()]['q'][0])
            v = torch.from_numpy(np.load(f'data_real/trajectory{id}.npy', allow_pickle=True)[()]['v'][0])
        else:
            assert False, "save_folder should be either data_real or data_sim"

        q_trajectory = [q.detach().numpy()]
        v_trajectory = [v.detach().numpy()]
        for k in range(150):
            q, v = cantilever.forward(q, v, f_ext=torch.zeros_like(q), dt=0.01)
            q_trajectory.append(q.detach().numpy())
            v_trajectory.append(v.detach().numpy())
        data = {"q" : np.stack(q_trajectory), "v" : np.stack(v_trajectory)}
        np.save(f'{save_folder}/trajectory{id}.npy', data)
        print("Finish trajectory", id)






    