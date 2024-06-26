import sys

sys.path.append("../")
import yaml
import numpy as np
import torch
import argparse
from markermatch import init_realdata, init_simenv
from _utils import ArmDataset
from env_arm import ArmEnv
from sopra_residual_physics import MLPResidual, ResMLPResidual2

args = argparse.ArgumentParser()
args.add_argument("-model", dest="model", required=False)

def validate_trajectory(
    sopra_env:ArmEnv, save_folder:str, test_data_idx:list, transformed_markers:np.ndarray, real_p:np.ndarray, fitting_options:dict, lookback=1, start_frame=0, end_frame=999, residual_network:torch.nn.Module=None,
):
    print("Testing trajectory", test_data_idx)
    assert fitting_options in ["pressure", "force"]
    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    if residual_network is None:
        if training_options['model'] == "skip_connection":
            model_class = ResMLPResidual2
        elif training_options['model'] == "mlp":
            model_class = MLPResidual
        
        if fitting_options == "pressure":
            residual_network = model_class(sopra_env._dofs * 2 + 6, sopra_env._dofs)
        elif fitting_options == "force":
            residual_network = model_class(sopra_env._dofs * 3, sopra_env._dofs, num_mlp_blocks=training_options['num_mlp_blocks'],num_block_layer=training_options['num_block_layer'], hidden_size=training_options['hidden_size'])
        # model = torch.load(f"{save_folder}/residual_network.pth")
        model_input = f"residual_network"
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
        print("The model saves at epoch", model["epoch"])
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
    q_init = q_res = torch.from_numpy(sopra_env._q0)
    v_res = torch.zeros_like(q_res)
    frame_i = 0
    qs_res = []
    vs_res = []
    qs_res.append(q_res.detach().numpy())
    vs_res.append(v_res.detach().numpy())
    res_force_errors = []
    predicted_residual_force_norms = []
    ground_truth_residual_force_norms = []
    normalize = True
    for frame_i in range(1, end_frame):
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
            q_res, v_res = sopra_env.forward(
                q_res, v_res, f_ext=f_ext_res + res_force, dt=0.01
            )
            
        except:
            print("Solver fails at frame", frame_i)
            break
        qs_res.append(q_res.detach().numpy())
        vs_res.append(v_res.detach().numpy())
    # np.save(f"{save_folder}/qs_res_{test_data_idx}.npy", qs_res)
    # np.save(f"{save_folder}/vs_res_{test_data_idx}.npy", vs_res)
    qs_res = np.array(qs_res)

    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    mm = 1.5 / 25.4
    sim_markers = []
    res_markers = []
    sim_frames = qs_res.shape[0]
    dt = 0.01
    times = np.linspace(0, sim_frames * dt, sim_frames + 1)[:-1]
    for frame_i in range(sim_frames):
        res_markers.append(
            sopra_env.return_simulated_markers(
                torch.from_numpy(qs_res[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
    res_markers = np.array(res_markers)
    real_frames = res_markers.shape[0]
    res_markers_error = np.linalg.norm(res_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    predicted_residual_force_norms = np.array(predicted_residual_force_norms)
    res_force_errors = np.array(res_force_errors)
    print(
        f"res error {test_data_idx} {res_markers_error.mean()*1e3:.3f}mm +-  {res_markers_error.mean(-1).std()*1e3:.3f} mm"
    )
    return res_markers_error

def main(save_folder, validation_set=list(range(160,180)), residual_network=None):
    max_pressure = 200
    # max_pressure = 350
    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    real_p, base_q = init_realdata(
        f"../arm_data_sep_4/captured_data_200traj_1000timesteps_{max_pressure}pressure.npy"
    )
    arm_folder = model_name = "sopra_494"
    model = f"../sopra_model/{model_name}.vtk"

    options = {}
    options["poissons_ratio"] = 0.45
    options["youngs_modulus"] = 215856
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
    res_errors = []
    for i in validation_set:
        res_error = validate_trajectory(
            sopra_env, save_folder, i, transformed_markers[i], real_p[i], "force", end_frame=999, residual_network=residual_network
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
    # main("training/num_blocks_5_num_layers_3")
    # main("hyperparams_search/learning_rate_1.047E-03_numblocks_4_num_layers_4_hidden_size_32")
    main("training/test_val_5_num_blocks_5_num_layers_3_2")

