import sys
sys.path.append('../../')
import time
import os
import cv2
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from mikmarkermatch import init_realdata, init_simenv

def visualize_trajetory(num_trajectory=1):
    max_pressure = 200
    real_p, base_q = init_realdata(f"../../arm_data_sep_4/captured_data_200traj_1000timesteps_{max_pressure}pressure.npy")
    arm_folder = model_name = 'sopra_494'
    model = f'../../sopra_model/{model_name}.vtk'
    options = {}
    options['youngs_modulus'] = 215856
    sopra_env, method, opt = init_simenv(model, arm_folder, options)
    sopra_env.set_measured_markers()
    for data_i in range(num_trajectory):
        save_folder = f"visulization/optimized_data_{data_i}"
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        steady_state = base_q[data_i,0]
        R, t = sopra_env.fit_realframe(steady_state)
        real_markers = steady_state @ R.T + t
        transformed_markers = base_q[data_i, :, 3:] @ R.T + t
        sopra_env.compute_interpolation_coeff(real_markers[3:])
        data_info = np.load(f"../augmented_dataset_fix_ordering/optimized_data_{data_i}.npy",allow_pickle=True)[()]
        qs = data_info['q_trajectory']
        for frame_i in range(qs.shape[0]):
            q = torch.from_numpy(qs[frame_i])
            simulated_markers = sopra_env.return_simulated_markers(q.reshape(-1, 3))
            sopra_env.vis_dynamic_sim2real_markers(save_folder, q.detach().numpy(),simulated_markers,transformed_markers[frame_i] ,frame=frame_i)
        image_folder = save_folder
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()

        # Set the video file name and frame rate (fps)
        video_name = f'{image_folder}.mp4'
        fps = 50  # Adjust this as needed

        # Get the first image to obtain its width and height
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        # Create the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        # Loop through the images and add them to the video
        for image in images:
            img = cv2.imread(os.path.join(image_folder, image))
            video.write(img)

        # Release the VideoWriter and close any open windows
        video.release()
        shutil.rmtree(save_folder)

def optimized_data_check(data_nums):
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    mm = 2.0/25.4
    for data_i in range(data_nums):
        data_info = np.load(f"../augmented_dataset_fix_ordering/optimized_data_{data_i}.npy",allow_pickle=True)[()]
        loss_info = np.load(f"../augmented_dataset_fix_ordering/loss_{data_i}.npy",allow_pickle=True)[()]
        optimized_force_norm= torch.from_numpy(data_info['optimized_forces']).norm(dim=0)
        figsize = (88*mm, 60*mm)
        plt.figure(figsize=figsize)
        plt.plot(optimized_force_norm)
        optimized_force_norm = torch.from_numpy(data_info['optimized_forces']).norm(dim=0)
        plt.plot(optimized_force_norm)
        plt.title(f"Optimized Force Norm Sample {data_i}")
        plt.xlabel("Frame (steps)")
        plt.ylabel("Force Norm / N")
        plt.grid()
        plt.savefig(f"optimized_data_plots/optimized_force_norm_{data_i}.png", )
        plt.close()

        loss = loss_info['data_loss']
        plt.figure(figsize=figsize)
        plt.plot(np.sqrt(loss/12))
        plt.title(f"Data Loss Sample {data_i}")
        plt.xlabel("Frame (steps)")
        plt.ylabel("Rooted Mean Square Error")
        plt.grid()
        plt.savefig(f"optimized_data_plots/data_loss_{data_i}.png")
        plt.close()