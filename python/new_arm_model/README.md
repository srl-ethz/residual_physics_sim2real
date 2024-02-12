
### Code structure for SoPrA model

    .
    ├── arm_data_sep_4 # Data files need to be put here to run scripts.
    ├── colearning_resphy # Residual physics framework
    ├── preprocess_data # Build augmented dataset
    ├── sopra_model # SoPrA mesh files
    ├── test_experiments # Calibrate the chambers of sopra
    ├── env_arm.py # SoPrA DiffPD class
    ├── _utils.py
    ├── markermatch.py
    └── README.md

You can download raw SoPrA data from [Google Drive](
https://drive.google.com/drive/folders/1_r7lP5Zgi_Jhu-A_-lMfV0zqxWmuu4Fq?usp=sharing).

We temporarily save our optimized data on [Google Drive](https://drive.google.com/drive/folders/1_r7lP5Zgi_Jhu-A_-lMfV0zqxWmuu4Fq?usp=sharing) too to help verify our experiments easily but we may not maintain it.

To generate augmented dataset, you can run `preprocess_data/build_augmented_data.py` directly.

To train residual physics network, You can run `colearning_resphy/training.py` with custum training options defined in the script. The best model is saved as `residual_network.pth`. `colearning_resphy/hyperparam_opt.py` sweeps over the hidden sizes, layers and blocks of the network. With the trained network, you can run `colearning_resphy/test_residual_physics.py` with `python test_residual_physics.py -model residual`.

