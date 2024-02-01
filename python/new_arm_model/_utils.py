### Utility functions, such as visualization.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors



def plot_curve (inputs, baselines=None, yaxis="Value ()", filename="plot", folder="plots"):
    """
    Creates a 2D plot of data with time-axis being a linspace. Inputs given in form of a dictionary. Baselines are plotted with dashed lines. By default stored in the plots. folder.
    
    Arguments:
        inputs (dict {"name": np.ndarray([T])}): Dictionary with entries where the keys are the labels of the plotted lines, and the values are 1D time-series data that should be plotted.
        yaxis (str): Label for the y axis.
    """
    Path(folder).mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.size': 7})     # Font size should be max 7pt and min 5pt
    plt.rcParams.update({'pdf.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
    plt.rcParams.update({'ps.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
    mm = 1/25.4
    figsize = (88*mm, 60*mm)
    fig, ax = plt.subplots(figsize=figsize)
    
    if baselines is not None:
        for k in baselines:
            ax.plot(np.linspace(0, baselines[k].shape[0]-1, baselines[k].shape[0]), baselines[k], '--', label=k)
    for k in inputs:
        ax.plot(np.linspace(0, inputs[k].shape[0]-1, inputs[k].shape[0]), inputs[k], label=k)
        
    ax.set_title(filename)
    ax.set_xlabel("Time (frame)")
    ax.set_ylabel(yaxis)
    ax.set_xlim(-2, inputs[k].shape[0]-1)  # Little buffer on the minimum to see starting state coinciding.
    #ax.set_ylim(minValy, maxValy)
    #ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    ax.set_axisbelow(True)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3, fancybox=True, shadow=False)
    Path(f"{folder}").mkdir(exist_ok=True, parents=True)
    fig.savefig(f"{folder}/{filename}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(f"{folder}/{filename}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()

def icp (real_q, B):
    """
    Returns result from rigid registration on markers. A is moved to match B. Give a whole time sequence of markers for A of real data.
    """
    init_pose=None
    max_iterations=200
    tolerance=1e-5

    A = real_q[0]

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    aligned_q =  np.ones((real_q.shape[0], m+1, A.shape[0]))
    aligned_q[:,:m,:] = np.copy(np.transpose(real_q, (0,2,1)))
        
    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    def nearest_neighbor(src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()


    def best_fit_transform(A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
        A: Nxm numpy array of corresponding points
        B: Nxm numpy array of corresponding points
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        '''

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t


    for _ in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        print(f"Mean Error: {mean_error:.4e}")
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    aligned_q = np.einsum('ij, bjk -> bik', T, aligned_q)

    return np.transpose(aligned_q, (0,2,1))[:,:,:3], T, distances, indices



