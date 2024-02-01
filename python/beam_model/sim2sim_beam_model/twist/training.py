import sys
import argparse

sys.path.append("..")
sys.path.append("../..")
import torch
import yaml
import numpy as np

from beam_residual_physics import CantileverResidualPhysics

ap = argparse.ArgumentParser()
ap.add_argument("-config", dest="config", required=False)

if __name__ == "__main__":
    poissons_ratio = 0.45
    youngs_modulus = 215856
    density = 1.07e3
    state_force = [0, 0, -9.80709]
    params = {
        "density": density,
        "youngs_modulus": youngs_modulus,
        "poissons_ratio": poissons_ratio,
        "state_force_parameters": state_force,
        "mesh_type": "tet",
        "refinement": 1,
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
        config["end_frame"] = 100
        config["training_set"] = list(range(10))
        config["validate_set"] = [10,11]

        config["cuda"] = 2
        config["normalize"] = True 
        config["Inialization"] = 1e-3
        config["scale"] = 1#e6
        config["data_type"] = "optimized"
        config["weight_decay"] = 1e-5
        config["fit"] = "forces"
        # config["fit"] = "SITL"
        config["model"] = "skip_connection"
        # config["model"] = "MLP"
        config["num_mlp_blocks"] = 5
        config["hidden_size"] = 512
        config["num_hidden_layer"] = 3
        # save_folder = f"training/sim2simResMLP5"
        save_folder = f"training/test_refactor"
        config["data_folder"] = save_folder.replace("training/", "")
        cantilever_residual = CantileverResidualPhysics(config, save_folder, params)
        torch.manual_seed(config["seed"])
        with open(f'{save_folder}/config.yaml', 'w') as f:
            yaml.dump(params, f)
            yaml.dump(config, f)
        print("Training Options: ", config)
        print(save_folder)
        cantilever_residual.train("cantilever_data_sim2sim", config)
