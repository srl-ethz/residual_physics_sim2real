import sys

sys.path.append("../")
import time
import os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

from markermatch import init_realdata, init_simenv
from _utils import ArmDataset
from residual_physics.network import MLPResidual, ResMLPResidual2

args = argparse.ArgumentParser()
args.add_argument("-model", dest="model", required=False)

def test_trajectory(
    sopra_env, save_folder, test_data_idx, transformed_markers, real_p, fitting_options,  start_frame=0, end_frame=999
):
    print("Testing trajectory", test_data_idx)
    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    if training_options['model'] == "skip_connection":
        residual_network = ResMLPResidual2(sopra_env._dofs * 3, sopra_env._dofs, num_mlp_blocks=training_options['num_mlp_blocks'], num_block_layer=training_options['num_block_layer'])
    elif training_options['model'] == "mlp":
        residual_network = MLPResidual(sopra_env._dofs * 3, sopra_env._dofs, hidden_sizes=training_options['hidden_sizes'])
    
    model = torch.load(f"{save_folder}/residual_network.pth")
    model_input = args.parse_args().model
    if model_input == "residual":
        model_input = f"residual_network"
    else:
        model_input = f"best_trained_model"
    print(model_input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f"{save_folder}/{model_input}.pth", map_location=device)

    # create new OrderedDict that does not contain `module.`
    if 'module' in list(model['model'].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model['model'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model['model'] = new_state_dict

    residual_network.load_state_dict(model["model"])
    residual_network.eval()

    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    real_chambers = [pairs[i][1] for i in range(len(pairs))]
    data_folder = "augmented_dataset_smaller_tol"
    ground_truth = np.load(
        f"../preprocess_data/{data_folder}/optimized_data_{test_data_idx}.npy",
        allow_pickle=True,
    )[()]
    f_optimized = torch.from_numpy(ground_truth["optimized_forces"]).t()[1:]
    loss_fn = torch.nn.MSELoss(reduction="mean")

    training_set = ArmDataset(
    training_options["training_set"],
    sopra_env,
    f"../preprocess_data/{data_folder}",
    start_frame=training_options["start_frame"],
    end_frame=training_options["end_frame"],
    )
    q_sim = torch.from_numpy(sopra_env._q0)
    v_sim = torch.zeros_like(q_sim)
    q_init = q_res = torch.from_numpy(sopra_env._q0)
    v_res = torch.zeros_like(q_res)
    frame_i = 0
    qs_sim = []
    vs_sim = []
    qs_sim.append(q_sim.detach().numpy())
    vs_sim.append(v_sim.detach().numpy())
    qs_res = []
    vs_res = []
    qs_res.append(q_res.detach().numpy())
    vs_res.append(v_res.detach().numpy())
    num_frames = transformed_markers.shape[0]
    res_force_errors = []
    predicted_residual_force_norms = []
    ground_truth_residual_force_norms = []
    normalize = True
    for frame_i in range(1, end_frame):
        start_time = time.time()
        pressure = real_p[frame_i - 1]
        f_ext_sim = sopra_env.apply_inner_pressure(
            pressure, q_sim.detach().numpy(), chambers=real_chambers
        )
        f_ext_res = sopra_env.apply_inner_pressure(
            pressure, q_res.detach().numpy(), chambers=real_chambers
        )
        f_ext_sim = torch.from_numpy(f_ext_sim)
        f_ext_res = torch.from_numpy(f_ext_res)
        pressure = torch.from_numpy(pressure)
        if normalize:
            (
                q_res_normalized,
                v_res_normalized,
                f_ext_res_normalized,
            ) = training_set.normalize(q=q_res - q_init, v=v_res, p=f_ext_res)
            res_force_normalized = residual_network(
                torch.cat(
                    (q_res_normalized, v_res_normalized, f_ext_res_normalized),
                    dim=0,
                ).expand(1, -1)
            )[0]
            f_res_normalized_gt = training_set.normalize(f=f_optimized[frame_i - 1, :])[0]
            # means squared error between normalized residual force and ground truth normalized residual force
            res_force_error = loss_fn(res_force_normalized, f_res_normalized_gt)
            res_force = training_set.denormalize(f=res_force_normalized)[0]
        else:
            res_force = residual_network(
                torch.cat((q_res - q_init, v_res, f_ext_res), dim=0)
            )
        res_force_error = torch.norm(res_force - f_optimized[frame_i - 1, :])
        predicted_residual_force_norms.append(torch.norm(res_force).item())
        res_force_errors.append(res_force_error.item())
        ground_truth_residual_force_norms.append(
            torch.norm(f_optimized[frame_i - 1, :]).item()
        )
        try:
            q_sim, v_sim = sopra_env.forward(q_sim, v_sim, f_ext=f_ext_sim, dt=0.01)
            q_res, v_res = sopra_env.forward(
                q_res, v_res, f_ext=f_ext_res + res_force, dt=0.01
            )
        except:
            print("Solver fails at frame", frame_i)
            break
        qs_sim.append(q_sim.detach().numpy())
        qs_res.append(q_res.detach().numpy())
        vs_sim.append(v_sim.detach().numpy())
        vs_res.append(v_res.detach().numpy())
    np.save(f"{save_folder}/qs_sim_{test_data_idx}.npy", qs_sim)
    np.save(f"{save_folder}/vs_sim_{test_data_idx}.npy", vs_sim)
    np.save(f"{save_folder}/qs_res_{test_data_idx}.npy", qs_res)
    np.save(f"{save_folder}/vs_res_{test_data_idx}.npy", vs_res)
    qs_sim = np.array(qs_sim)
    qs_res = np.array(qs_res)

    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    vis_1d_folder = f"2dplots_displacement/{save_folder.replace('training/', '')}_{model_input}"
    os.makedirs(vis_1d_folder, exist_ok=True)
    mm = 1.5 / 25.4
    sim_markers = []
    res_markers = []
    sim_frames = qs_sim.shape[0]
    dt = 0.01
    times = np.linspace(0, sim_frames * dt, sim_frames + 1)[:-1]
    for frame_i in range(sim_frames):
        sim_markers.append(
            sopra_env.return_simulated_markers(
                torch.from_numpy(qs_sim[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
        res_markers.append(
            sopra_env.return_simulated_markers(
                torch.from_numpy(qs_res[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
    sim_markers = np.array(sim_markers)
    res_markers = np.array(res_markers)
    real_frames = res_markers.shape[0]
    sim_markers_error = np.linalg.norm(sim_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    res_markers_error = np.linalg.norm(res_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    predicted_residual_force_norms = np.array(predicted_residual_force_norms)
    res_force_errors = np.array(res_force_errors)
    print(
        f"test id {test_data_idx} sim error {sim_markers_error.mean()*1e3:.3f}mm +-  {sim_markers_error.mean(-1).std()*1e3:.3f} mm"
    )
    print(
        f"res error {test_data_idx} {res_markers_error.mean()*1e3:.3f}mm +-  {res_markers_error.mean(-1).std()*1e3:.3f} mm"
    )
    figsize = (88 * 2 * mm, 60 * mm)
    plot_trajectory(
        vis_1d_folder,
        figsize,
        transformed_markers,
        sim_markers,
        res_markers,
        test_data_idx,
        real_frames,
        dt,
    )
    plot_forces_norm(
        vis_1d_folder,
        test_data_idx,
        figsize,
        predicted_residual_force_norms,
        ground_truth_residual_force_norms,
        dt,
    )

    return sim_markers_error, res_markers_error


def plot_trajectory(
    vis_1d_folder,
    figsize,
    transformed_markers,
    sim_markers,
    res_markers,
    test_data_idx,
    real_frames,
    dt,
):
    mm = 1.5 / 25.4
    fig, axs = plt.subplots(ncols=2, figsize=figsize)
    times = np.linspace(0, real_frames * dt, real_frames + 1)[:-1]
    axs[0].plot(
        times,
        transformed_markers[:real_frames, -4:, 0].mean(1),
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Ground Truth",
        linewidth=2.0,
    )
    axs[0].plot(
        times,
        sim_markers[:, -4:, 0].mean(1),
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Simulation",
        linewidth=2.0,
    )
    axs[0].plot(
        times,
        res_markers[:, -4:, 0].mean(1),
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Residual Physics",
        linewidth=2.0,
    )
    axs[0].legend(
        loc="lower center",
        bbox_to_anchor=[0.5, -0.35],
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    axs[0].set_title(f"x axis")
    axs[1].plot(
        times,
        transformed_markers[:real_frames, -4:, 1].mean(1),
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Real",
        linewidth=2.0,
    )
    axs[1].plot(
        times,
        sim_markers[:, -4:, 1].mean(1),
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Simulation",
        linewidth=2.0,
    )
    axs[1].legend(
        loc="lower center",
        bbox_to_anchor=[0.5, -0.35],
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    axs[1].plot(
        times,
        res_markers[:, -4:, 1].mean(1),
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Residual Physics",
        linewidth=2.0,
    )
    axs[1].set_title(f"y axis")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.set_xlim(0, real_frames * dt)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=[0.5, -0.35],
            ncol=3,
            fancybox=True,
            shadow=True,
        )
    fig.savefig(
        f"{vis_1d_folder}/real_chamber_{test_data_idx}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=1 * mm,
    )
    plt.close()


def plot_forces_norm(
    vis_1d_folder,
    data_idx,
    figsize,
    predicted_residual_force_norms,
    ground_truth_residual_force_norms,
    dt,
):
    mm = 1.5 / 25.4
    fig, ax = plt.subplots(figsize=figsize)
    real_frames = predicted_residual_force_norms.shape[0]
    times = np.linspace(0, real_frames * dt, real_frames + 1)[:-1]
    ax.plot(
        times,
        predicted_residual_force_norms,
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Predicted Residual Force Norm",
        linewidth=2.0,
    )
    ax.plot(
        times,
        ground_truth_residual_force_norms,
        linestyle="-",
        marker="o",
        markersize=0.5,
        label=f"Ground Truth Residual Force Norm",
        linewidth=2.0,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual Force Norm (N)")
    ax.set_xlim(0, real_frames * dt)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    plt.savefig(
        f"{vis_1d_folder}/residual_force_norm_{data_idx}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=1 * mm,
    )
    plt.close()


if __name__ == "__main__":
    max_pressure = 200
    # max_pressure = 350
    real_p, base_q = init_realdata(
        f"../arm_data_sep_4/captured_data_200traj_1000timesteps_{max_pressure}pressure.npy"
    )
    arm_folder = model_name = "sopra_494"
    model = f"../sopra_model/{model_name}.vtk"

    options = {}
    options["poissons_ratio"] = 0.45
    options["youngs_modulus"] = 215856
    # options["poissons_ratio"] =  0.4194
    # options["youngs_modulus"] = 237629.9
    sopra_env, method, opt = init_simenv(model, arm_folder, options)

    sopra_env.set_measured_markers()
    measured_markers = sopra_env.get_measured_markers()
    steady_state = base_q[0, 0]
    R, t = sopra_env.fit_realframe(steady_state)
    real_markers = steady_state @ R.T + t
    error = np.linalg.norm(real_markers[3:] - measured_markers[3:]) / np.linalg.norm(
        real_markers[3:]
    )
    sopra_env.compute_interpolation_coeff(real_markers[3:])
    transformed_markers = base_q[:, :, 3:] @ R.T + t

    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    real_chambers = [pairs[i][1] for i in range(len(pairs))]
    real_p = real_p[:, :, real_chambers]
    lrs = [None]
    steps = [None]
    gammas = [None]
    errors_all = []
    for skip_i in [5]:
        for grad_clip in [None]:
            for lr in lrs:
                for step in steps:
                    for gamma in gammas:
                        save_folder = f"hyperparam_search/learning_rate_1.000E-03_numblocks_5_num_layers_3_hidden_size_512"
                        sim_errors = []
                        res_errors = []
                        for i in range(180,200):
                            sim_error, res_error = test_trajectory(
                                sopra_env, save_folder, i, transformed_markers[i], real_p[i], "force", end_frame=999
                            )
                            sim_errors.append(sim_error)
                            res_errors.append(res_error)
                        sim_errors = np.array(sim_errors)
                        res_errors = np.array(res_errors)
                        sim_error_mean = sim_errors.mean(axis=-1).mean(axis=-1)
                        res_error_mean = res_errors.mean(axis=-1).mean(axis=-1)
                        print(f"sim error {sim_error_mean.mean() *1000 :.3f}mm +-  {sim_error_mean.std() * 1000:.3f} mm")
                        print(f"res error {res_error_mean.mean() * 1000:.3f}mm +-  {res_error_mean.std() * 1000:.3f} mm")
                        np.save(f"{save_folder}/sim_errors_{args.parse_args().model}.npy", sim_errors)
                        np.save(f"{save_folder}/res_errors_{args.parse_args().model}.npy", res_errors)
                        errors_all.append([res_error_mean.mean(), res_error_mean.std()])
                        errors_all = np.array(errors_all)
                        np.save(f"errors_None.npy", np.array(errors_all))
