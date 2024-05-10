import sys

sys.path.append("../")
import time
import yaml
import numpy as np
import torch
import argparse
from _utils import CantileverDataset
from env_cantilever import CantileverEnv3d
# from model import CantileverSimpleMLP, ResMLP_Residual
from residual_physics.network import ResMLPResidual2

args = argparse.ArgumentParser()
args.add_argument("-model", dest="model", required=False)

def validate_trajectory(
    cantilever:CantileverEnv3d, save_folder, test_data_idx, transformed_markers, start_frame=0, end_frame=150,  residual_network=None,
):
    config = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    if residual_network is None:
        # if config['model'] == "skip_connection":
        #     model_class = ResMLP_Residual
        # elif config['model'] == "MLP":
        #     model_class = CantileverSimpleMLP
        
        model_input = f"residual_network"
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
        residual_network = ResMLPResidual2(dofs * 2, dofs, hidden_size=config['hidden_size'], num_mlp_blocks=config['num_mlp_blocks'], num_block_layer=config['num_hidden_layer'])
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
    config["training_set"],
    cantilever._q0,
    f"cantilever_data_fix_registration",
    start_frame=config["start_frame"],
    end_frame=config["end_frame"],
    )

    q0 = torch.from_numpy(ground_truth["q_trajectory"][0])
    q_init = torch.from_numpy(cantilever._q0)
    v0 = torch.zeros_like(q0)
    q_res = q0.clone()
    v_res = v0.clone()
    frame_i = 0

    qs_res = []
    vs_res = []
    qs_res.append(q_res.detach().numpy())
    vs_res.append(v_res.detach().numpy())
    res_force_errors = []
    predicted_residual_force_norms = []
    ground_truth_residual_force_norms = []
    time_res = 0
    for frame_i in range(1, end_frame):
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
        except:
            print("Solver fails at frame", frame_i)
            break
        qs_res.append(q_res.detach().numpy())
        vs_res.append(v_res.detach().numpy())
    res_markers = []
    qs_res = np.array(qs_res)
    sim_frames = qs_res.shape[0]
    for frame_i in range(sim_frames):
        res_markers.append(
            cantilever.get_markers_3d(
                torch.from_numpy(qs_res[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
    res_markers = np.array(res_markers)
    real_frames = res_markers.shape[0]
    res_markers_error = np.linalg.norm(res_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    return res_markers_error

def main(save_folder, validation_set=[5,15], residual_network=None):
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
    q_init = torch.from_numpy(cantilever._q0)
    q0 = torch.from_numpy(cantilever._q0)
    q_ = q0.reshape(-1, 3)
    v0 = torch.zeros(q0.shape, dtype=torch.float64)

    qs_real_ = np.load("weight_data_ordered/q_data_reorder.npz")

    steady_state = qs_real_[f'arr_0'][:, :, -1] * 1e-3
    R, t = cantilever.fit_realframe(steady_state)
    steady_state_transformed = steady_state @ R.T + t

    cantilever.interpolate_markers_3d(q_.detach().numpy(), steady_state_transformed)
    res_errors = []


    for test_i in validation_set:
        print(f"test id {test_i}")
        real_markers = np.load(f"weight_data_ordered/qs_real{test_i}_reorder.npy") * 1e-3
        transformed_markers = np.zeros((150, 10,3))
        for i in range(150):
            transformed_markers[i] = real_markers[:, :, i] @ R.T + t
        res_error = validate_trajectory(
        cantilever, save_folder, test_i, transformed_markers, start_frame=0, end_frame=150,  residual_network=residual_network
        )
        res_errors.append(res_error)
    res_errors = np.array(res_errors)
    res_error_mean = res_errors.mean(axis=-1).mean(axis=-1)
    print(f"res error {res_error_mean.mean() * 1000:.3f}mm +-  {res_error_mean.std() * 1000:.3f} mm")
    metrics = {
        'validate_trajectory_error_mean': res_error_mean.mean(),
        'validate_trajectory_error_std': res_error_mean.std(),
               }

    return metrics

if __name__ == '__main__':
    main("training/test_val_5_num_blocks_5_num_layers_3_2")

