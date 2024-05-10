### Code Structure Sim-to-real beam model

    .
    ├── _utils.py
    ├── _visualization.py
    ├── beam_sys_all.py # Run system identification (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics framework.
    ├── beam_sys_grid_search.py # Grid search verification for system identification results.
    ├── env_base.py # Base class of DiffPD model, modifying for damping optimization. 
    ├── env_cantilever.py # Beam DiffPD class.
    ├── init_beam.py # Optimizing virtual forces to match the initial state.
    ├── model.py # Neural netowrk.
    ├── optimize_trajectory.py # Optimize the full state model step by step
    ├── build_augmented_data.py # Generate augmented data
    ├── beam_residual_physics.py # Residual physics training framework
    ├── test_residual_physics.py # Test residual physics framework
    ├── training.py # Training script
    └── README.md


You can download Sim2real beam real data from [Google Drive](https://drive.google.com/drive/folders/1LnwYr0sBDyMs6rvVQbn1PKMmMCQAy-ky?usp=sharing)
, where `weight_data_ordered` stores the collected raw data and `cantilver_data_fix_registration` stores the optimized full-state trajectories by `optimize_trajectory.py`.

To run the complete residual physics framework, we first need to build augmented dataset by run `build_augmented_data.py`. Then we can run `training.py` to train the residual physics network. We save the best model performed on validation set as `residual_network.pth`. With the trained network, we can run `python test_residual_physics.py -model residual`.



