import torch
import yaml
from sopra_residual_physics import SoPrAResidualPhysics
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-config", dest="config", required=False)

if __name__ == "__main__":
    poissons_ratio = 0.45
    youngs_modulus = 215856
    density = 1.07e3
    state_force = [0, 0, -9.80709]
    model_name = "sopra_494"
    model = f"../sopra_model/{model_name}.vtk"
    params = {
        "density": density,
        "youngs_modulus": youngs_modulus,
        "poissons_ratio": poissons_ratio,
        "state_force_parameters": state_force,
        "mesh_type": "tet",
        "refinement": 1,
        "arm_file": model,
    }
    if ap.parse_args().config is not None:
        with open(ap.parse_args().config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
        config["seed"] = 42
        config["epochs"] = 1000
        config["batch_size"] = 256
        config["learning_rate"] = 1e-3
        config["optimizer"] = "adam"
        config["start_frame"] = 0
        config["end_frame"] = 999
        config["training_set"] = list(range(1))#list(range(160))
        config["validate_set"] = list(range(160,161))#,180))
        config["cuda"] = 1
        config["normalize"] = True 
        config["Inialization"] = 1e-3
        config["scale"] = 1#e6
        config["data_type"] = "optimized"
        config["weight_decay"] = 1e-5
        # config["validate_physics"] = True
        # config["validate_epochs"] = 20
        # config["transfer_learning_model"] = f'training/fit_500_batch_sopra_4942/residual_network.pth'
        config["fit"] = "forces"
        # config["fit"] = "SITL"  
        # config["model"] = "skip_connection"
        config["model"] = "MLP"
        config["tolerance"] = 121
        grad_clips = [None]
        learning_rates      = [1e-3]
        scheduler_steps     = [None]
        scheduler_gammas    = [None]
        weight_decays = [1e-5]
        for num_blocks in [5]:
            for num_layers in [3]:

                hidden_sizes = [512] * num_layers
                hidden_sizes.insert(0, 4446)
                hidden_sizes.append(1482)

                config["num_mlp_blocks"] = num_blocks
                config["hidden_size"] = 512
                config["num_block_layer"] = num_layers

                # save_folder = f"training/test_refactor"
                save_folder = f"training/test_refactor_SITL"
                config["data_folder"] = save_folder.replace("training/", "")
                sopra_residual = SoPrAResidualPhysics(config, save_folder, params)
                torch.manual_seed(config["seed"])
                with open(f'{save_folder}/config.yaml', 'w') as f:
                    yaml.dump(params, f)
                    yaml.dump(config, f)
                print("Training Options: ", config)
                print(save_folder)
                sopra_residual.train("../preprocess_data/augmented_dataset_smaller_tol", config)
