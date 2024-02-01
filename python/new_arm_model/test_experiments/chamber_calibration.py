


import sys
import os
import time
sys.path.append('../')
sys.path.append('../..')

import torch
import numpy as np
import matplotlib.pyplot as plt

from markermatch import init_realdata, init_simenv
from _visualization import plot_displacement, plot_displacement_xy, plot_sim2real_errors


if __name__ == '__main__':
    arm_folder = "arm_measure_test_sep_4"
    # model = '../sopra_model/Segment.vtk'
    # model = '../sopra_model/sopra_test3.vtk'
    model = '../sopra_model/sopra_494.vtk'
    sopra_env, _, _ = init_simenv(model, arm_folder)
    print(f"DOFs: {sopra_env._dofs}")

    # Set to sampling rate of real data
    dt = 0.01
    ### Marker shape is [trajectories, timesteps, markers, xyz]
    real_p, initial_real_markers = init_realdata("../arm_data_sep_4/captured_data_singleChamberCalibration.npy")
    # Map chambers correctly from real chamber index to simulated chamber index
    chamber_mapping = [5, 4, 2, 0, 1, 3]
    real_p = real_p[:, chamber_mapping]
    # DEBUG: Use only first 20 frames so I can debug fast
    #initial_real_markers = initial_real_markers[:, :30, :, :]


    ### Rigid Registration to get sim2real marker transformation
    steady_state = initial_real_markers[0, 0]
    R, t = sopra_env.fit_realframe(steady_state)
    real_markers = initial_real_markers @ R.T + t
    # Discard the base markers for all subsequent error computation
    real_markers = real_markers[:, :, 3:]


    ### Steady State Error
    # By default the base markers are ignored in the comparison
    registration_distance_error = np.linalg.norm(real_markers[0,0] - sopra_env.measured_markers[3:], axis=-1)
    print(f"Steady State Marker Distance: \t\t\t{1000*registration_distance_error.mean():.4f}mm +- {1000*registration_distance_error.std():.4f}mm")
    
    ### Nodes might not be located on the surface mesh nodes, hence we interpolate to find closer matches.
    sopra_env.compute_interpolation_coeff(real_markers[0,0])
    sim_markers = sopra_env.return_simulated_markers(torch.from_numpy(sopra_env._q0.reshape(-1, 3))).numpy()
    interpolation_distance_error = np.linalg.norm(real_markers[0,0] - sim_markers, axis=-1)
    print(f"Interpolated Steady State Marker Distance: \t{1000*interpolation_distance_error.mean():.4e}mm +- {1000*interpolation_distance_error.std():.4e}mm")



    ### Perform Forward Simulation using Real Pressures
    num_chambers = 6
    real_markers = real_markers[:num_chambers]
    num_frames = real_markers.shape[1]
    if os.path.isfile(f"sim_markers_{sopra_env._dofs}.npy"):
        print("Found existing simulation data, loading...")
        sim_markers = np.load(f"sim_markers_{sopra_env._dofs}.npy")
    else:
        sim_markers = [[] for _ in range(num_chambers)]
        for chamber_i in range(num_chambers):
            start_time = time.time()
            q = torch.from_numpy(sopra_env._q0)
            v = torch.zeros_like(q)

            # Append initial frame
            sim_markers[chamber_i].append(sopra_env.return_simulated_markers(q.reshape(-1,3)).numpy())
            sopra_env.vis_dynamic_sim2real_markers(f"chamber_{chamber_i}_dofs_{sopra_env._dofs}", q.detach().numpy(), sim_markers[chamber_i][0], real_markers[chamber_i][0], frame=0)

            # Iterate over all frames
            for frame_i in range(1, num_frames):
                start_frame = time.time()
                f_ext = torch.from_numpy(sopra_env.apply_inner_pressure(real_p[chamber_i], q.detach().numpy(), chambers=[0,1,2,3,4,5]))
                q, v = sopra_env.forward(q, v, f_ext=f_ext, dt=dt)
                
                # Append simulated markers
                sim_markers[chamber_i].append(sopra_env.return_simulated_markers(q.reshape(-1,3)).numpy())

                sopra_env.vis_dynamic_sim2real_markers(f"chamber_{chamber_i}_dofs_{sopra_env._dofs}", q.detach().numpy(), sim_markers[chamber_i][frame_i], real_markers[chamber_i][frame_i], frame=frame_i)
        
                #print(f"Frame [{frame_i+1}/{num_frames}]: {(time.time()-start_frame):.2f}s")

                if frame_i % 10 == 0:
                    print(f"Chamber {chamber_i} Frame {frame_i}: {time.time() - start_time:.2f}s")
                
        sim_markers = np.array(sim_markers)
        np.save(f"sim_markers_{sopra_env._dofs}.npy", sim_markers)



    ### Quantitative Error Analysis
    # Error shapes [chambers, timesteps]
    marker_errors_mean = np.linalg.norm(real_markers - sim_markers, axis=-1).mean(axis=-1)
    marker_errors_std = np.linalg.norm(real_markers - sim_markers, axis=-1).std(axis=-1)

    # Error per chamber
    for chamber_i in range(num_chambers):
        print(f"Chamber {chamber_i} Time-Averaged Marker Error: \t{1000*marker_errors_mean[chamber_i].mean():.4f}mm +- {1000*marker_errors_std[chamber_i].mean():.4f}mm")


    ### Visualization
    vis_folder = f"plots/dofs_{sopra_env._dofs}"
    os.makedirs(vis_folder, exist_ok=True)

    # Plot mean tip marker sim2real trajectories 
    plot_displacement(sim_markers, real_markers, dt=dt, vis_folder=vis_folder)

    # Plot mean tip marker sim2real trajectories in projected XY plane
    plot_displacement_xy(sim_markers, real_markers, dt=dt, vis_folder=vis_folder)

    # Plot errors
    plot_sim2real_errors(marker_errors_mean, marker_errors_std, dt=dt, vis_folder=vis_folder)


    # Store Displacement Videos
    # Slowed down factor 0.1x
    slow_down_factor = 0.1
    for chamber_i in range(num_chambers):
        os.system(f"ffmpeg -y -framerate {int(slow_down_factor/dt)} -i {sopra_env._folder}/chamber_{chamber_i}_dofs_{sopra_env._dofs}/%04d.png -c:v libx264 -crf 0 {sopra_env._folder}/chamber_{chamber_i}_alignment.mp4")

