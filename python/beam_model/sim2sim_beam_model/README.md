    
### Code structure for Sim-to-sim beam model
    .
    ├── sim2real_prep # Inverstigate how many markers are needed to reconstruct the full state information
    ├── vibration # Sim2sim experiments for oscillating beams
    ├── twist # Sim2sim experiments for sim2sim twisting beams

You can download sim2sim data from [Google drive](https://drive.google.com/drive/folders/1LnwYr0sBDyMs6rvVQbn1PKMmMCQAy-ky?usp=sharing) and put the subfolder data into the corresponding experimental folder to verify our result easily.

`sim2real_prep` includes marker ablation experiments to investigate how many markers can reconstruct the full state information.

For oscillating and twisting sim2sim experiments in `vibration` and `twist`:

    .
    ├── data_generation.py # We need to generate target data by changing the options in the file.
    ├── optimize_trajectory.py # Optimize residual forces to match the real trajectory.
    ├── training.py # Training residual physics network
    ├── test_residual_physics.py # Run tests for simulation and residual physics framework.