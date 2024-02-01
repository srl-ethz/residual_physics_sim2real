import sys
sys.path.append('../')
sys.path.append('../..')
from pathlib import Path
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser

from py_diff_pd.common.common import ndarray, create_folder, print_info,delete_folder
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import StdRealVector, HexMesh3d, HexDeformable, TetMesh3d, TetDeformable
from py_diff_pd.common.hex_mesh import generate_hex_mesh, voxelize, hex2obj
from py_diff_pd.common.display import render_hex_mesh, export_gif, export_mp4

from env_cantilever import CantileverEnv3d

def optimize_trajectoryfull(beam_folder, cantilever:CantileverEnv3d, num_frames, num_epochs, target_data, dt, sample=0, suffix = "fix_registration"):
    folder = 'trajectoryfull'
    Path(beam_folder+ '/'+ folder).mkdir(parents=True,exist_ok=True)
    # force_nodes_num = 49 * 3
    force_nodes_num = cantilever.dofs
    # force_nodes_num = 16
    params = torch.normal(0, 1e-4, [force_nodes_num, num_frames], dtype=torch.float64, requires_grad=True)
    target_data = torch.from_numpy(target_data)
    q_steady = cantilever._q0
    cantilever._q0 = np.load(f"cantilever_data_{suffix}/q_force_opt{sample}_reorder.npz")['arr_%d' % (len(np.load(f"cantilever_data_{suffix}/q_force_opt{sample}_reorder.npz"))-1)]
    ### Define target location
    q_all = []
    v_all = []
    loss_history_init = []
    loss_history_optimized = []
    q_all.append(cantilever._q0)
    v_all.append(np.zeros_like(cantilever._q0))
    q_init_marked = torch.from_numpy(cantilever._q0).reshape(-1,3)
    qx = q_init_marked
    qx_marker = cantilever.get_markers_3d(qx)

    q_last_epoch = torch.from_numpy(cantilever._q0)
    v_last_epoch = torch.zeros(cantilever.dofs, dtype=torch.float64)
    for frame_i in range(1, num_frames-1):
        with torch.no_grad():
            params[:, frame_i] = params[:, frame_i - 1]
        init_loss = ((qx_marker.flatten() - target_data[:, frame_i])**2).sum().item()
        loss_history_init.append(init_loss)
        optimizer = torch.optim.Adam([params],lr=1e-3)
        loss_last = 0
        for epoch in range(num_epochs):
            start_time = time.time()
            optimizer.zero_grad()
            q, v = q_last_epoch.detach().clone(), v_last_epoch.detach().clone()
            f_ext = torch.zeros(cantilever.dofs, dtype=torch.float64)
            f_ext[-force_nodes_num:] += params[:, frame_i]
            # f_ext[-force_nodes_num*3+2::3] += params[:, i]
            q, v = cantilever.forward(q, v, f_ext=f_ext, dt=dt)
            qx = q.reshape(-1,3)

            qx_marker = cantilever.get_markers_3d(qx)
            loss = ((-qx_marker.flatten() + target_data[:, frame_i])**2).sum()
            loss += 1e-4 * (params[:,frame_i]**2).sum()
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

        if sample == 0:
            cantilever.vis_dynamic_sim2real_markers('trajectoryfull', q.detach().numpy(), qx_marker.detach().numpy(), target_data[:, frame_i].reshape(-1,3).detach().numpy(), frame=frame_i)
        # cantilever.vis_dynamic_sim2real_markers('trajectoryfull', q.detach().numpy(), qx_marker.detach().numpy(), target_data[:, frame_i+1].reshape(-1,3).detach().numpy(), frame=frame_i)
    
        with np.printoptions(precision=3):
            print(f"Frame [{frame_i}/{num_frames-1}]/ Epoch {epoch}: {(time.time()-start_time):.2f}s,- Loss {loss.item():.4e}, init_loss {init_loss} ")#- f_ext: {params[0].detach().cpu().numpy():.6f} - grad: {params.grad[0].detach().cpu().numpy():.2e} ")# - Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

    ### Plotting Loss History
    print("length of q_all", len(q_all))
    print("length of v_all", len(v_all))
    optimized_data_save = {
        "q_trajectory": np.stack(q_all),
        "v_trajectory": np.stack(v_all),
        "optimized_forces": params.detach().numpy(),
    }
    np.save(f"cantilever_data_{suffix}/optimized_data_{sample}.npy", optimized_data_save)
    np.savez(f"cantilever_data_{suffix}/q_trajectoryfull{sample}_reorder.npz", *q_all)
    np.savez(f"cantilever_data_{suffix}/v_trajectoryfull{sample}_reorder.npz", *v_all)