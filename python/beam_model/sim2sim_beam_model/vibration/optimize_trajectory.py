import sys
sys.path.append('../')
sys.path.append('../../..')
from pathlib import Path
import time
import numpy as np
import torch

from env_cantilever import CantileverEnv3d

def optimize_trajectoryfull(cantilever:CantileverEnv3d, num_frames, num_epochs, dt, sample=0, suffix = "fix_registration"):
    force_nodes_num = cantilever._dofs
    params = torch.normal(0, 1e-4, [force_nodes_num, num_frames], dtype=torch.float64, requires_grad=True)
    target_trajectory = torch.from_numpy(np.load(f"data_real/trajectory{sample}.npy", allow_pickle=True)[()]['q'])
    q0 = target_trajectory[0]
    ### Define target location
    q_all = []
    v_all = []
    loss_history_init = []
    loss_history_optimized = []
    q_all.append(q0)
    v_all.append(np.zeros_like(q0))

    q_last_epoch = q0.clone()
    v_last_epoch = torch.zeros(cantilever._dofs, dtype=torch.float64)
    for frame_i in range(1, num_frames):
        with torch.no_grad():
            params[:, frame_i] = params[:, frame_i - 1]
        init_loss = ((target_trajectory[frame_i] - q_last_epoch)**2).sum().item()
        loss_history_init.append(init_loss)
        optimizer = torch.optim.Adam([params],lr=1e-3)
        loss_last = 0
        for epoch in range(num_epochs):
            start_time = time.time()
            optimizer.zero_grad()

            q, v = q_last_epoch.detach().clone(), v_last_epoch.detach().clone()
            f_ext = torch.zeros(cantilever._dofs, dtype=torch.float64)
            f_ext[-force_nodes_num:] += params[:, frame_i]

            q, v = cantilever.forward(q, v, f_ext=f_ext, dt=dt)
            data_loss = ((target_trajectory[frame_i] - q)**2).sum()
            loss = data_loss + 1e-4 * (params[:,frame_i]**2).sum()
            if (torch.abs(loss - loss_last)/loss < 1e-6):
                break
            loss_last = loss
            # Backward gradients so we know which direction to update parameters
            loss.backward()
            # Actually update parameters
            optimizer.step()
        loss_history_optimized.append(loss.item())
        v_last_epoch = v
        q_last_epoch = q
        v_all.append(v.detach().numpy())
        q_all.append(q.detach().numpy())

        with np.printoptions(precision=3):
            print(f"Frame [{frame_i}/{num_frames-1}]/ Epoch {epoch}: {(time.time()-start_time):.2f}s,- Loss {data_loss.item():.4e}, init_loss {init_loss} ")

    ### Plotting Loss History
    print("length of q_all", len(q_all))
    print("length of v_all", len(v_all))
    optimized_data_save = {
        "q_trajectory": np.stack(q_all),
        "v_trajectory": np.stack(v_all),
        "optimized_forces": params.detach().numpy(),
    }
    Path(f"cantilever_data_{suffix}").mkdir(parents=True,exist_ok=True)
    np.save(f"cantilever_data_{suffix}/optimized_data_{sample}.npy", optimized_data_save)
    print("-----------finish visualization---------------")

if __name__ == '__main__':
    weights = [0.05, 0.06, 0.07, 0.1, 0.09, 0.08, 0.11, 0.12, 0.15, 0.09, 0.13, 0.14, 0.16, 0.17, 0.2,0.18,0.22,0.21]
    idxs = [0,3,4,5,6,8,10,12,13,15,17]
    for idx in idxs:
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

        ########################################
        num_frames = 150
        num_epochs = 300
        optimize_trajectoryfull(cantilever, num_frames, num_epochs, 0.01, sample=idx, suffix="sim2sim")