import sys

sys.path.append("../")
sys.path.append("../..")
import time
from pathlib import Path
import numpy as np
import torch

from env_cantilever import CantileverEnv3d


def optimize_init_force(
    num_epochs,
    num_frames,
    target,
    cantilever: CantileverEnv3d,
    dt,
    id=0,
    res_force=None,
    suffix="fix_registration",
):
    force_nodes_num = 16
    params = torch.normal(
        0, 1e-4, [force_nodes_num, num_frames], dtype=torch.float64, requires_grad=True
    )
    with torch.no_grad():
        if res_force is not None:
            for i in range(num_frames):
                params[:, i] = res_force[-46::3]

    optimizer = torch.optim.Adam([params], lr=1e-3)
    loss_history = []
    target = torch.from_numpy(target)

    for epoch in range(num_epochs):
        start_time = time.time()

        def closure():
            optimizer.zero_grad()
            q0 = cantilever._q0
            q, v = torch.from_numpy(q0), torch.zeros(q0.shape).double()
            loss = 0
            for i in range(num_frames):
                f_ext = torch.zeros(cantilever.dofs, dtype=torch.float64)
                f_ext[-force_nodes_num * 3 + 2 :: 3] += params[:, i]
                q, v = cantilever.forward(q, v, f_ext=f_ext, dt=dt)
                qx = q.reshape(-1, 3)
                qx_marker = cantilever.get_markers_3d(qx)
                loss += ((target - qx_marker) ** 2).sum()

            loss /= num_frames

            ### Additionally add condition on f_ext change to not be too sudden
            threshold = 100
            loss += 1e-7 * (params**2).sum()
            # Backward gradients so we know which direction to update parameters
            loss.backward()

            # with torch.no_grad():
            return loss

        # Actually update parameters
        loss = optimizer.step(closure)
        loss_history.append(loss.item())
        rel_loss = abs(loss.item() - loss_history[-2]) / loss.item() if epoch > 10 else 1
        if rel_loss < 1e-3:
                break
        with np.printoptions(precision=3):
            print(
                f"Epoch [{epoch+1}/{num_epochs}]: {(time.time()-start_time):.2f}s - Loss {loss.item():.4e} "
            )  # - f_ext: {params[0].detach().cpu().numpy():.6f} - grad: {params.grad[0].detach().cpu().numpy():.2e} ")# - Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        ### Early stopping
        if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-8:
            break

    q_all = []
    print(f"test{id}")
    path_vis = Path(f"test{id}/init_beam")
    path_vis.mkdir(exist_ok=True, parents=True)
    with torch.no_grad():
        q, v = cantilever._q0, torch.zeros(cantilever.dofs, dtype=torch.float64)
        q_all.append(q)
        q = torch.from_numpy(q)

        for i in range(num_frames):
            start_time = time.time()

            f_ext = torch.zeros(cantilever.dofs, dtype=torch.float64)
            f_ext[-force_nodes_num * 3 + 2 :: 3] += params[:, i]
            end_vis = time.time()
            q, v = cantilever.forward(q, v, f_ext=f_ext, dt=dt)
            q_all.append(q.detach().numpy())

            qx_marker = torch.zeros(target.shape).double()
            qx = q.reshape(-1, 3)
            qx_marker = cantilever.get_markers_3d(qx)

            # Time including visualization
            print(
                f"Frame [{i+1}/{num_frames}]: {1000*(time.time()-end_vis):.2f}ms (+ {1000*(end_vis-start_time):.2f}ms for visualization)"
            )
            cantilever.vis_dynamic_sim2real_markers(
                f"init_beam",
                q.detach().numpy(),
                qx_marker.detach().numpy(),
                target.detach().numpy(),
                frame=i,
            )
        Path(f"cantilever_data_{suffix}").mkdir(exist_ok=True, parents=True)
        np.savez(f"cantilever_data_{suffix}/q_force_opt{id}_reorder.npz", *q_all)
        print("-----------finish visualization---------------")
