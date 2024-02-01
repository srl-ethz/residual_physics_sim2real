import sys
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")
import time
import os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from _utils import CantileverDataset
from _visualization import plot_trajectory, plot_forces_norm
from env_cantilever import CantileverEnv3d
from residual_physics.network import ResMLPResidual2, MLPResidual

args = argparse.ArgumentParser()
args.add_argument("-model", dest="model", required=False)

def test_trajectory(
    cantilever:CantileverEnv3d, save_folder, test_data_idx, transformed_markers, start_frame=0, end_frame=150, cantilever_sim=None,
):
    if cantilever_sim is None:
        cantilever_sim = cantilever
    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    
    model_input = args.parse_args().model
    if model_input == "residual":
        model_input = f"residual_network"
    else:
        model_input = f"best_trained_model"
    device = torch.device("cpu")
    print(f"{save_folder}/{model_input}.pth")
    model = torch.load(f"{save_folder}/{model_input}.pth", map_location=device)

    # create new OrderedDict that does not contain `module.`
    if 'module' in list(model['model'].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model['model'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model['model'] = new_state_dict

    dofs = cantilever._dofs
    if training_options["model"] == "skip_connection":
        residual_network = ResMLPResidual2(dofs * 2, dofs, hidden_size=training_options['hidden_size'], num_mlp_blocks=training_options['num_mlp_blocks'], num_block_layer=training_options['num_hidden_layer'])
    elif training_options["model"] == "MLP":
        residual_network = MLPResidual(dofs*2, dofs)
    residual_network.load_state_dict(model["model"])
    print("The model saves at epoch", model["epoch"])
    residual_network.eval()

    ground_truth = np.load(
        f"cantilever_data_fix_registration/optimized_data_{test_data_idx}.npy",
        allow_pickle=True,
    )[()]
    f_optimized = torch.from_numpy(ground_truth["optimized_forces"]).t()[1:]
    loss_fn = torch.nn.MSELoss(reduction="mean")

    training_set = CantileverDataset(
    training_options["training_set"],
    cantilever._q0,
    f"cantilever_data_fix_registration",
    start_frame=training_options["start_frame"],
    end_frame=training_options["end_frame"],
    )

    q0 = torch.from_numpy(ground_truth["q_trajectory"][0])
    v0 = torch.zeros_like(q0)
    q_sim = q0.clone()
    v_sim = v0.clone()
    q_res = q0.clone()
    v_res = v0.clone()
    q_origin = q0.clone()
    v_origin = v0.clone()
    frame_i = 0

    qs_sim = []
    vs_sim = []
    qs_sim.append(q_sim.detach().numpy())
    vs_sim.append(v_sim.detach().numpy())
    qs_res = []
    vs_res = []
    qs_res.append(q_res.detach().numpy())
    vs_res.append(v_res.detach().numpy())
    qs_origin = []
    vs_origin = []
    qs_origin.append(q_origin.detach().numpy())
    vs_origin.append(v_origin.detach().numpy())
    res_force_errors = []
    predicted_residual_force_norms = []
    ground_truth_residual_force_norms = []
    normalize = True
    cantilever.vis_dynamic_sim2real_markers(f"res_{test_data_idx}", q_res.detach().numpy(), cantilever.get_markers_3d(q_res.reshape(-1,3)).detach().numpy(), transformed_markers[0], frame=0)
    time_sim = 0
    time_res = 0
    time_origin = 0
    time_network = 0
    for frame_i in range(1, end_frame):
        if normalize:
            ti = time.time()
            (
                q_res_normalized,
                v_res_normalized,
            ) = training_set.normalize(q=q_res - q_init, v=v_res)
            res_force_normalized = residual_network(
                torch.cat(
                    (q_res_normalized, v_res_normalized),
                    dim=0,
                ).expand(1, -1)
            )[0]
            res_force = training_set.denormalize(f=res_force_normalized)[0]
            ti_end = time.time()
            time_network += ti_end - ti
            time_res += ti_end - ti
        else:
            res_force = residual_network(
                torch.cat((q_res - q_init, v_res), dim=0)
            )
        res_force_error = torch.norm(res_force - f_optimized[frame_i - 1, :])
        predicted_residual_force_norms.append(torch.norm(res_force).item())
        res_force_errors.append(res_force_error.item())
        ground_truth_residual_force_norms.append(
            torch.norm(f_optimized[frame_i - 1, :]).item()
        )
        try:
            t0 = time.time()
            q_res, v_res = cantilever.forward(
                q_res, v_res, f_ext=res_force, dt=0.01
            )
            time_res += time.time() - t0
            t1 = time.time()
            q_sim, v_sim = cantilever_sim.forward(q_sim, v_sim, f_ext=torch.zeros_like(q_sim), dt=0.01)
            time_sim += time.time() - t1
            t1 = time.time()
            q_origin, v_origin = cantilever.forward(q_origin, v_origin, f_ext=torch.zeros_like(q_origin), dt=0.01)
            time_origin += time.time() - t1
        except:
            print("Solver fails at frame", frame_i)
            break
        # cantilever.vis_dynamic_sim2real_markers("res", q_res.detach().numpy(), cantilever.get_markers_3d(q_res.reshape(-1,3)).detach().numpy(), transformed_markers[frame_i], frame=frame_i)
        # cantilever.vis_dynamic_sim2real_markers("sim", q_sim.detach().numpy(), cantilever.get_markers_3d(q_sim.reshape(-1,3)).detach().numpy(), transformed_markers[frame_i], frame=frame_i)
        qs_sim.append(q_sim.detach().numpy())
        qs_res.append(q_res.detach().numpy())
        vs_sim.append(v_sim.detach().numpy())
        vs_res.append(v_res.detach().numpy())
        qs_origin.append(q_origin.detach().numpy())
        vs_origin.append(v_origin.detach().numpy())
    np.save(f"{save_folder}/qs_sim_{test_data_idx}.npy", qs_sim)
    np.save(f"{save_folder}/qs_res_{test_data_idx}.npy", qs_res)
    np.save(f"{save_folder}/vs_sim_{test_data_idx}.npy", vs_sim)
    np.save(f"{save_folder}/vs_res_{test_data_idx}.npy", vs_res)
    np.save(f"{save_folder}/qs_origin_{test_data_idx}.npy", qs_origin)
    np.save(f"{save_folder}/vs_origin_{test_data_idx}.npy", vs_origin)
    qs_sim = np.array(qs_sim)
    qs_res = np.array(qs_res)
    qs_origin = np.array(qs_origin)
    

    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    vis_1d_folder = f"2dplots_displacement/{save_folder.replace('training/', '')}_{model_input}"
    os.makedirs(vis_1d_folder, exist_ok=True)
    mm = 1.5 / 25.4
    sim_markers = []
    res_markers = []
    origin_markers = []
    sim_frames = qs_sim.shape[0]
    dt = 0.01
    times = np.linspace(0, sim_frames * dt, sim_frames + 1)[:-1]
    for frame_i in range(sim_frames):
        sim_markers.append(
            cantilever.get_markers_3d(
                torch.from_numpy(qs_sim[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
        res_markers.append(
            cantilever.get_markers_3d(
                torch.from_numpy(qs_res[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
        origin_markers.append(
            cantilever.get_markers_3d(
                torch.from_numpy(qs_origin[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
    sim_markers = np.array(sim_markers)
    res_markers = np.array(res_markers)
    origin_markers = np.array(origin_markers)
    real_frames = res_markers.shape[0]
    print(sim_markers.shape)
    sim_markers_error = np.linalg.norm(sim_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    res_markers_error = np.linalg.norm(res_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    origin_markers_error = np.linalg.norm(origin_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    predicted_residual_force_norms = np.array(predicted_residual_force_norms)
    res_force_errors = np.array(res_force_errors)
    print(
        f"test id {test_data_idx} sim error {sim_markers_error.mean()*1e3:.3f}mm +-  {sim_markers_error.mean(-1).std()*1e3:.3f} mm"
    )
    print(
        f"res error {test_data_idx} {res_markers_error.mean()*1e3:.3f}mm +-  {res_markers_error.mean(-1).std()*1e3:.3f} mm"
    )
    print(f"origin error {origin_markers_error.mean()*1e3:.3f}mm +-  {origin_markers_error.mean(-1).std()*1e3:.3f} mm")
    print(f"Sim time {time_sim:.3f} s, Res time {time_res:.3f} s, origin time {time_origin:.3f} s, Network time {time_network:.3f} s")
    figsize = (44 * mm, 30 * mm)
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

    return sim_markers_error, res_markers_error, time_sim, time_res, time_network, origin_markers_error, time_origin

if __name__ == "__main__":
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

    cantilever = CantileverEnv3d(42, 'beam', hex_params)
    hex_params['youngs_modulus'] = 810695.1
    hex_params['poissons_ratio'] = 0.499
    cantilever_sim = CantileverEnv3d(42, 'beam', hex_params)
    q_init = torch.from_numpy(cantilever._q0)
    q0 = torch.from_numpy(cantilever._q0)
    q_ = q0.reshape(-1, 3)
    v0 = torch.zeros(q0.shape, dtype=torch.float64)

    qs_real_ = np.load("weight_data_ordered/q_data_reorder.npz")

    steady_state = qs_real_[f'arr_0'][:, :, -1] * 1e-3
    R, t = cantilever.fit_realframe(steady_state)
    steady_state_transformed = steady_state @ R.T + t

    cantilever.interpolate_markers_3d(q_.detach().numpy(), steady_state_transformed)

    # save_folder = f"hyperparams_search/num_layers_3_hidden_size_64_num_blocks_6"
    save_folder = f"training/test_refactor"
    sim_errors = []
    res_errors = []
    origin_errors = []
    time_sim_total = 0
    time_res_total = 0
    time_origin_total = 0
    time_network_total = 0
    for test_i in [2,7,11,14,16]:
        print(f"test id {test_i}")
        real_markers = np.load(f"weight_data_ordered/qs_real{test_i}_reorder.npy") * 1e-3
        transformed_markers = np.zeros((150, 10,3))
        for i in range(150):
            transformed_markers[i] = real_markers[:, :, i] @ R.T + t
        sim_error, res_error, time_sim, time_res, time_network, origin_error, time_origin = test_trajectory(
            cantilever, save_folder, test_i, transformed_markers, end_frame=140, cantilever_sim=cantilever_sim
        )
        time_res_total += time_res
        time_sim_total += time_sim
        time_origin_total += time_origin
        time_network_total += time_network
        sim_errors.append(sim_error)
        res_errors.append(res_error)
        origin_errors.append(origin_error)
    sim_errors = np.array(sim_errors)
    res_errors = np.array(res_errors)
    origin_errors = np.array(origin_errors)
    sim_error_mean = sim_errors.mean(axis=-1).mean(axis=-1)
    res_error_mean = res_errors.mean(axis=-1).mean(axis=-1)
    origin_error_mean = origin_errors.mean(axis=-1).mean(axis=-1)
    print(f"sim error {sim_error_mean.mean() *1000 :.3f}mm +-  {sim_error_mean.std() * 1000:.3f} mm")
    print(f"res error {res_error_mean.mean() * 1000:.3f}mm +-  {res_error_mean.std() * 1000:.3f} mm")
    time_res_mean = time_res_total / 6
    time_sim_mean = time_sim_total / 6
    time_origin_mean = time_origin_total / 6
    time_network_mean = time_network_total / 6
    print(f"Total Sim time {time_sim_mean:.3f} s, Total Res time {time_res_mean:.3f} s, Total Network time {time_network_mean:.3f} s, Total origin time {time_origin_mean:.3f} s")
    print("Frame error: \n")
    print(f"sim error {sim_errors.mean(-1).flatten().mean() *1000 :.3f}mm +-  {sim_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    print(f"res error {res_errors.mean(-1).flatten().mean() * 1000:.3f}mm +-  {res_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    print(f"origin error {origin_error_mean.mean() * 1000:.3f}mm +-  {origin_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    np.save(f"{save_folder}/sim_errors_{args.parse_args().model}.npy", sim_errors)
    np.save(f"{save_folder}/res_errors_{args.parse_args().model}.npy", res_errors)
