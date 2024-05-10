# Residual Physics

Note: this is a modified version of https://github.com/mit-gfx/diff_pd, with the purpose of learning residual physics of soft deformable objects.


## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)

## Installation
```
git clone --recursive git@github.com:srl-ethz/residual_physics_sim2real.git
cd residual_physics_sim2real
conda env create -f environment.yml
conda activate residual_physics
./install.sh
```

## Experiments
Navigate to python folder to one of the experiments below, detailed instructions can be found in the subfolder README.md.

### beam_model

sim2sim_beam_model: Sim-to-sim osicillating and twisting beam experiments; Marker ablation experiments.

sim2real_beam_model: Sim-to-real experiments for the cantilever beam.

### arm_model

Sim-to-real experiments for SoPrA.

### residual physics

Residual physics base class.

### paper_figures

Experimental result data and plots in the paper.
