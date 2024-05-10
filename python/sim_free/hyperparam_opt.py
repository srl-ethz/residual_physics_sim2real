### Run hyperparameter search for specific network architectures.

import torch
import wandb
import yaml
from argparse import ArgumentParser

import traceback, sys

from python.sim_free.training_simfree import main


################################################################
# configs
################################################################
default_configs = {
    'device':               torch.device("cuda" if torch.cuda.is_available() else "cpu"),   # Define device for training & inference - GPU/CPU
    'logEpoch':             10,                      # Define epoch interval for logging
    'load':                 False,                   # Define whether to load a model or not
    
    'data':                 "beam_oscillating",      # Define data to use for training - 'beam_oscillating', 'beam_twisting', 'beam_real_markers', 'beam_real', 'arm_real'
    'model':                "resmlp",                # Define model to use - 'mlp' or 'resmlp'
    'hidden_dim':           64,                     # Define hidden dimension of model
    'num_layers':           5,                       # Define number of layers in model
    'num_blocks':           5,                       # Define number of blocks in model
    'loss_fn':              'L2',                    # Define loss function to use - 'L1', 'L2'

    'learning_rate':        10e-3,                   # Define learning rate
    'scheduler_step':       10,                      # Define step size for learning rate scheduler
    'scheduler_gamma':      0.95,                    # Define gamma for learning rate scheduler
    'batch_size':           32,                      # Define batch size for training
    'epochs':               1000,                     # Define number of epochs for training

    'youngsModulus':        263824,                  # Define Young's Modulus for beam
    'poissonsRatio':        0.499,                   # Define Poisson's Ratio for beam
}

def run_sweep():
    try:
        wandb.init(config=default_configs)

        ### Run training
        rmses = main(wandb.config)
    
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


if __name__ == "__main__":
    ### Choose model type from command line
    args = ArgumentParser()
    args.add_argument("-m", dest="model", default="resmlp", help="Model type to use for training. Choose from ['mlp']", choices=['mlp', 'resmlp'])
    args.add_argument("-d", dest="data", default="arm_real", help="Data to use for training.", choices=['beam_oscillating', 'beam_twisting', 'beam_real_markers', 'beam_real', 'arm_real'])
    args.add_argument("-s", dest="sweep_id", default=None, help="Sweep ID for WandB.")
    args.add_argument("-e", dest="epochs", default=1000, type=int, help="Number of epochs to train.")
    args.add_argument
    args = args.parse_args()

    # Set default configs.
    default_configs['model'] = args.model
    default_configs['data'] = args.data
    default_configs['epochs'] = args.epochs

    with open("sweep_config.yaml") as stream:
        try:
            sweep_config = yaml.safe_load(stream)
            print(sweep_config)
        except yaml.YAMLError as exc:
            print(exc)

    sweep_config['project'] = f"residual_physics_{default_configs['data']}"

    # Continue running existing sweeps
    if args.sweep_id is not None:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep_config, entity='srl_ethz')
    # Multiple machines can run the same sweep
    print(f"### Sweep ID: {sweep_id}")
    ### The configs from wandb are updated with sweep config!
    wandb.agent(sweep_id, function=run_sweep)
