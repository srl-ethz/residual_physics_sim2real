from beam_residual_physics import CantileverResidualPhysics
import torch
import wandb
import yaml
from validate_residual_physics import main as validate_main

import traceback, sys

config = {}
config["seed"] = 42
config["epochs"] = 1000
config["batch_size"] = 256
config["optimizer"] = "adam"
config["start_frame"] = 0
config["end_frame"] = 140
config["training_set"] = [0, 3, 4, 6, 8, 10, 12, 13, 17]
config["validate_set"] = [5,15]
config["cuda"] = 2#"cuda"
config["normalize"] = True 
config["Inialization"] = 1e-3
config["scale"] = 1#e6
config["data_type"] = "optimized"
config["weight_decay"] = 1e-5
# config["validate_physics"] = True
# config["validate_epochs"] = 20
config["fit"] = "forces"
config["model"] = "skip_connection"
config["tolerance"] = 121
grad_clips = [None]
learning_rates      = [1e-3]
scheduler_steps     = [None]
scheduler_gammas    = [None]
weight_decays = [1e-5]

config["learning_rate"] = 1e-3
# config["num_mlp_blocks"] = 5
# config["num_block_layer"] = 3
# config["hidden_size"] = 512

poissons_ratio = 0.45
youngs_modulus = 215856
density = 1.07e3
state_force = [0, 0, -9.80709]
save_folder = f"training/hyperparams_sweeping"
config["data_folder"] = save_folder.replace("training/", "")
params = {
    "density": density,
    "youngs_modulus": youngs_modulus,
    "poissons_ratio": poissons_ratio,
    "state_force_parameters": state_force,
    "mesh_type": "tet",
    "refinement": 1,
}

def run_sweep():
    try:
        wandb.init(config=config)
        
        save_folder = f"hyperparams_search/num_layers_{wandb.config['num_hidden_layer']}_hidden_size_{wandb.config['hidden_size']}_num_blocks_{wandb.config['num_mlp_blocks']}"
        config["data_folder"] = save_folder.replace("hyperparams_search/", "")
        cantilever_residual = CantileverResidualPhysics(config, save_folder, params)
        torch.manual_seed(config["seed"])
        config["num_hidden_layer"] = wandb.config["num_hidden_layer"]
        config["hidden_size"] = wandb.config["hidden_size"]
        config["num_mlp_blocks"] = wandb.config["num_mlp_blocks"]
        with open(f"{save_folder}/config.yaml", "w") as f:
            yaml.dump(config, f)
        print("Training Options: ", wandb.config)
        print(save_folder)
        cantilever_residual.train("cantilever_data_fix_registration", config)
        validate_metrics = validate_main(save_folder, validation_set=[5,15])
        wandb.log(validate_metrics)

        
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)

if __name__ == "__main__":
    ## default config
    with open("sweep_forward.yaml") as stream:
       try:
           sweep_config = yaml.safe_load(stream)
           print(sweep_config)
       except yaml.YAMLError as exc:
           print(exc)

    sweep_id = wandb.sweep(sweep_config, entity='junpeng_eth')
    wandb.agent(sweep_id, function=run_sweep)