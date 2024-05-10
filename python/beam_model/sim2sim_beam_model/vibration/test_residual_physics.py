import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../..")
import time
import os
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
    cantilever:CantileverEnv3d, save_folder, test_data_idx,  start_frame=0, end_frame=150, cantilever_sim=None,
):
    if cantilever_sim is None:
        cantilever_sim = cantilever
    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    dofs = cantilever._dofs
    if training_options["model"] == "skip_connection":
        residual_network = ResMLPResidual2(dofs * 2, dofs, hidden_size=training_options['hidden_size'], num_mlp_blocks=training_options['num_mlp_blocks'], num_block_layer=training_options['num_hidden_layer'])
    elif training_options["model"] == "MLP":
        residual_network = MLPResidual(dofs*2, dofs)
    
    model_input = args.parse_args().model
    if model_input == "residual":
        model_input = f"residual_network"
    else:
        model_input = f"best_trained_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    residual_network.load_state_dict(model["model"])
    print("The model saves at epoch", model["epoch"])
    residual_network.eval()

    ground_truth = np.load(
        f"cantilever_data_sim2sim/optimized_data_{test_data_idx}.npy",
        allow_pickle=True,
    )[()]
    f_optimized = torch.from_numpy(ground_truth["optimized_forces"]).t()[1:]
    loss_fn = torch.nn.MSELoss(reduction="mean")

    training_set = CantileverDataset(
    training_options["training_set"],
    cantilever.q0,
    f"cantilever_data_sim2sim",
    start_frame=training_options["start_frame"],
    end_frame=training_options["end_frame"],
    )

    q0 = torch.from_numpy(ground_truth["q_trajectory"][0])
    q_init = torch.from_numpy(cantilever._q0)
    v0 = torch.zeros_like(q0)
    q_sim = q0.clone()
    v_sim = v0.clone()
    q_res = q0.clone()
    v_res = v0.clone()
    frame_i = 0

    qs_sim = []
    vs_sim = []
    qs_sim.append(q_sim.detach().numpy())
    vs_sim.append(v_sim.detach().numpy())
    qs_res = []
    vs_res = []
    qs_res.append(q_res.detach().numpy())
    vs_res.append(v_res.detach().numpy())
    res_force_errors = []
    predicted_residual_force_norms = []
    ground_truth_residual_force_norms = []
    normalize = True
    sim_t = 0
    res_t = 0
    NN_t = 0
    for frame_i in range(1, end_frame):
        if normalize:
            t0 = time.time()
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
            NN_t += time.time() - t0
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
            res_t += time.time() - t0
            t1 = time.time()
            q_sim, v_sim = cantilever_sim.forward(q_sim, v_sim, f_ext=torch.zeros_like(q_sim), dt=0.01)
            sim_t += time.time() - t1
        except:
            print("Solver fails at frame", frame_i)
            break
        qs_sim.append(q_sim.detach().numpy())
        qs_res.append(q_res.detach().numpy())
        vs_sim.append(v_sim.detach().numpy())
        vs_res.append(v_res.detach().numpy())
    np.save(f"{save_folder}/qs_sim_{test_data_idx}.npy", qs_sim)
    np.save(f"{save_folder}/qs_res_{test_data_idx}.npy", qs_res)
    np.save(f"{save_folder}/vs_sim_{test_data_idx}.npy", vs_sim)
    np.save(f"{save_folder}/vs_res_{test_data_idx}.npy", vs_res)
    qs_sim = np.array(qs_sim)
    qs_res = np.array(qs_res)
    qs_ground_truth = np.array(ground_truth["q_trajectory"])

    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    vis_1d_folder = f"sim2sim/{save_folder.replace('training/', '')}_{model_input}"
    os.makedirs(vis_1d_folder, exist_ok=True)
    mm = 1.5 / 25.4
    sim_frames = qs_sim.shape[0]
    real_frames = sim_frames
    dt = 0.01
    times = np.linspace(0, sim_frames * dt, sim_frames + 1)[:-1]
    qs_sim = qs_sim.reshape(qs_sim.shape[0], -1,3)
    qs_res = qs_res.reshape(qs_res.shape[0], -1,3)
    qs_ground_truth = qs_ground_truth.reshape(qs_ground_truth.shape[0], -1,3)
    sim_error = np.linalg.norm(qs_sim[:real_frames] - qs_ground_truth[:real_frames], axis=-1)
    res_error = np.linalg.norm(qs_res[:real_frames] - qs_ground_truth[:real_frames], axis=-1)
    predicted_residual_force_norms = np.array(predicted_residual_force_norms)
    res_force_errors = np.array(res_force_errors)
    print(
        f"test id {test_data_idx} sim error {sim_error.mean()*1e3:.3f}mm +-  {sim_error.mean(-1).std()*1e3:.3f} mm"
    )
    print(
        f"res error {test_data_idx} {res_error.mean()*1e3:.3f}mm +-  {res_error.mean(-1).std()*1e3:.3f} mm"
    )
    print(f"NN time {NN_t:.3f}s, res time {res_t:.3f}s, sim time {sim_t:.3f}s")
    figsize = (88 * mm, 60 * mm)
    plot_trajectory(
        vis_1d_folder,
        figsize,
        qs_ground_truth,
        qs_sim,
        qs_res,
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

    return sim_error, res_error


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
    cantilever = CantileverEnv3d(42, 'beam', hex_params)
    # save_folder = f"training/sim2simResMLP5"
    save_folder = "training/test_refactor"
    sim_errors = []
    res_errors = []
    for test_i in [2,7,11,14,16]:
        print(f"test id {test_i}")
        sim_error, res_error = test_trajectory(
            cantilever, save_folder, test_i, end_frame=150, cantilever_sim=cantilever
        )
        sim_errors.append(sim_error)
        res_errors.append(res_error)
    sim_errors = np.array(sim_errors)
    res_errors = np.array(res_errors)
    print(sim_errors.shape)
    sim_error_mean = sim_errors.mean(axis=-1).mean(axis=-1)
    res_error_mean = res_errors.mean(axis=-1).mean(axis=-1)
    print(f"sim error {sim_error_mean.mean() *1000 :.3f}mm +-  {sim_error_mean.std() * 1000:.3f} mm")
    print(f"res error {res_error_mean.mean() * 1000:.3f}mm +-  {res_error_mean.std() * 1000:.3f} mm")
    print("Frame error: \n")
    print(f"sim error {sim_errors.mean(-1).flatten().mean() *1000 :.3f}mm +-  {sim_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    print(f"res error {res_errors.mean(-1).flatten().mean() * 1000:.3f}mm +-  {res_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    np.save(f"{save_folder}/sim_errors_{args.parse_args().model}.npy", sim_errors)
    np.save(f"{save_folder}/res_errors_{args.parse_args().model}.npy", res_errors)
