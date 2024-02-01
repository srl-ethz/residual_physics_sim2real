import sys

sys.path.append("../")
import time
import os
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from markermatch import init_realdata, init_simenv, align_sim_real
from env_arm import ArmEnv


def optimize_trajectoryfull(
    arm: ArmEnv,
    options: dict,
    base_q: np.ndarray,
    real_p: np.ndarray,
    sim_chambers=[0, 1, 2, 3, 4, 5],
    sample=0,
    data_folder="arm_data",
):
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    # Perform rigd registration
    steady_state = base_q[0]
    sopra_env.set_measured_markers(mikmarkers)
    R, t = sopra_env.fit_realframe(steady_state)
    transformed_real_data = base_q[:, 3:] @ R.T + t
    arm.compute_interpolation_coeff(transformed_real_data[0])
    force_nodes_num = arm._dofs
    num_frames = (
        options["num_frames"] if options["num_frames"] is not None else base_q.shape[0]
    )
    num_epochs = options["num_epochs"] if options["num_epochs"] is not None else 50
    dt = options["dt"] if options["dt"] is not None else 0.01
    learning_rate = (
        options["learning_rate"] if options["learning_rate"] is not None else 1e-3
    )
    # l2_lmbda = options["lmbda"] if "lmbda" in options else 1e-5
    # l1_lambda = options["l1_lambda"] if "l1_lambda" in options else 0
    lmbda = options["lmbda"] if "lmbda" in options else 1e-5
    rel_tol = options["rel_tol"] if "rel_tol" in options else 1e-6

    torch.random.manual_seed(options["random_seed"])
    params = torch.normal(
        0, 1e-4, [force_nodes_num, num_frames], dtype=torch.float64, requires_grad=True
    )
    print(f"params norm: {params[:, 0].norm()}")
    # Initialize containers
    q_all = []
    v_all = []
    pressure_used = []
    loss_history = []
    data_loss_history = []
    pressure_forces = []
    epochs = []
    q_all.append(arm._q0)
    v_all.append(np.zeros_like(arm._q0))
    q = torch.from_numpy(arm._q0)
    v = torch.zeros_like(q)
    q_last_epoch = torch.from_numpy(arm._q0)
    v_last_epoch = torch.zeros_like(q)
    total_time = 0
    if sample == 0:
        q_reshape = q.reshape(-1, 3)
        simulated_markers = arm.return_simulated_markers(q_reshape)
        target_markers = torch.from_numpy(transformed_real_data[0, :, :])
        arm.vis_dynamic_sim2real_markers(
            f"plots_{sample}_{data_folder}",
            q.detach().numpy(),
            simulated_markers.detach().numpy(),
            target_markers.detach().numpy(),
            frame=0,
        )
    for frame_i in range(1, num_frames):
        with torch.no_grad():
            params[:, frame_i] = params[:, frame_i - 1]
        target_markers = torch.from_numpy(transformed_real_data[frame_i, :, :])
        f_ext = arm.apply_inner_pressure(
            real_p[frame_i - 1], q.detach().numpy(), chambers=sim_chambers
        )
        pressure_used.append(real_p[frame_i - 1])
        pressure_forces.append(f_ext)
        f_ext = torch.from_numpy(f_ext)
        optimizer = torch.optim.Adam([params], lr=learning_rate)
        loss_last = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            q, v = q_last_epoch.detach().clone(), v_last_epoch.detach().clone()
            f_epoch = f_ext.clone()
            f_epoch[-force_nodes_num:] += params[:, frame_i]
            q, v = arm.forward(q, v, f_ext=f_epoch, dt=dt)
            q_reshape = q.reshape(-1, 3)
            simulated_markers = arm.return_simulated_markers(q_reshape)
            data_loss = ((target_markers - simulated_markers) ** 2).sum()
            # loss = data_loss + l_lmbda * (params[:, frame_i] ** 2).sum() + l1_lambda * torch.abs(params[:, frame_i]).sum()
            loss = data_loss + lmbda * (params[:, frame_i] ** 2).sum()
            # loss = data_loss + lmbda * (params[:, frame_i] ** 2).sum()
            if (torch.abs(loss - loss_last) / loss) < rel_tol:
                break
            loss_last = loss
            loss.backward()
            optimizer.step()
        print(
            f"sample: {sample}, num_frame: {frame_i}, epoch: {epoch}, loss: {loss.item() :.3E}, data_loss: {data_loss.item() :.3E}, params_norm: {params[:, frame_i].norm() :.3E}, regu_norm: {lmbda *torch.abs(params[:, frame_i]).sum().item() :.3E}"
        )
        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(f"Epoch time: {epoch_time :.3f}")
        if sample == 0:
            arm.vis_dynamic_sim2real_markers(
                f"plots_{sample}_{data_folder}",
                q.detach().numpy(),
                simulated_markers.detach().numpy(),
                target_markers.detach().numpy(),
                frame=frame_i,
            )
        loss_history.append(loss.item())
        data_loss_history.append(data_loss.item())
        epochs.append(epoch)
        q_last_epoch = q
        v_last_epoch = v
        q_all.append(q.detach().numpy())
        v_all.append(v.detach().numpy())

        if (
            (frame_i >= 100)
            and (frame_i % 50 == 1)
            or (frame_i % (num_frames - 1) == 0)
        ):
            optimized_data_save = {
                "q_trajectory": np.stack(q_all),
                "v_trajectory": np.stack(v_all),
                "pressurefull": np.stack(pressure_used),
                "optimized_forces": params.detach().numpy(),
                "pressure_forces": np.stack(pressure_forces),
            }
            np.save(f"{data_folder}/optimized_data_{sample}.npy", optimized_data_save)
            loss_all = {
                "total_loss": np.stack(loss_history),
                "data_loss": np.stack(data_loss_history),
                "epochs": np.stack(epochs),
                "params_norm": params.norm(dim=0).detach().numpy(),
            }
            np.save(f"{data_folder}/loss_{sample}.npy", loss_all)
    print(f"Total time: {total_time :.3f}")


def init_realdata(datafile, start_frame=0, num_frames=1000, traj_idx=None):
    """
    Load real data from .npy file. Logged pressure is in mbar.
    """
    data = np.load(f"{datafile}", allow_pickle=True)[()]
    if traj_idx is None:
        markers = data["data"][:, start_frame : start_frame + num_frames]
        pressures = 100 * data["p"][:, start_frame : start_frame + num_frames]
    else:
        markers = data["data"][traj_idx, start_frame : start_frame + num_frames]
        pressures = 100 * data["p"][traj_idx, start_frame : start_frame + num_frames]

    return pressures, markers


if __name__ == "__main__":
    arm_folder = "optimize_arm"
    model_name = "sopra_494"
    model = f"../sopra_model/{model_name}.vtk"
    # Intializ sopra
    options = {}
    options['poissons_ratio'] = 0.495
    options["youngs_modulus"] = 215856
    sopra_env, method, opt = init_simenv(model, arm_folder, options=options)
    sopra_env.opt['thread_ct'] = 32
    # Load data
    max_pressure = 200
    valves = [1, 3, 4, 5, 6, 7]
    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    real_chambers = [pairs[i][1] for i in range(len(pairs))]
    
    real_p, base_q = init_realdata(
        f"../arm_data_sep_4/captured_data_200traj_1000timesteps_{max_pressure}pressure.npy"
    )
    # Set markers
    mikmarkers = (
        np.array(
            [
                [-62.9, -23, -8.2],  # 0
                [-62.9, 22, -8.2],  # 1
                [31.3, -21.6, -8.2],  # 2
                [18, -17, -128],  # 3
                [21, 9, -128],  # 4
                [-8, 21, -128],  # 5
                [-21, 10, -128],  # 6
                [-20, -20, -128],  # 7
                [18, 10, -146],  # 8
                [-26, 8, -146],  # 9
                [-3, -29, -146],  # 10
                [11, -10, -266],  # 11
                [-6, 14, -266],  # 12
                [-21, -10, -266],  # 13
                [-8, -20, -266],  # 14
            ]
        )
        * 1e-3
        + np.array([0, 0, 2.92047]) * 1e-3
    )
    mikmarkers = mikmarkers[[1, 2, 0, 4, 3, 6, 5, 7, 8, 9, 10, 12, 11, 14, 13]]
    # Set optimization parameters
    options = {
        "num_frames": 1000, # The number of total simulation frames
        "num_data": base_q.shape[0],  # The number of total simulation frames
        "num_epochs": 150,  # The number of epochs to run in optimization for each frame
        "dt": 0.01,  # The time step of the simulation
        "learning_rate": 1e-3,  # The step of the optimizer
        'lmbda': 1e-4, # The regularization parameter
        "random_seed": 42,  # The random seed for the optimization
        "rel_tol": 1e-3,  # The relative tolerance for the optimization loss
    }
    torch.manual_seed(options["random_seed"])
    chambers_mapping = [5, 4, 2, 0, 1, 3]
    p_mapped = real_p[:,:, chambers_mapping]
    save_folder = "augmented_dataset"
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    with open(f"{save_folder}/config.yaml", "w") as f:
        yaml.dump(options, f)
    for sample in range(0,options["num_data"]):
        print(f"Start running optimization for sample {sample}")
        print(options)
        optimize_trajectoryfull(
            sopra_env,
            options,
            base_q[sample],
            p_mapped[sample],
            sample=sample,
            sim_chambers=[0, 1, 2, 3, 4, 5],
            data_folder=save_folder,
        )
