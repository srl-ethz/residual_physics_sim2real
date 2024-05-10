###
# Utility functions for training models in a purely data-driven manner.
###

import time
import copy
import wandb
from argparse import ArgumentParser

import torch
import numpy as np

import matplotlib.pyplot as plt

from _architecture import get_model

import sys
sys.path.append('../arm_model')

from env_arm import ArmEnv

WANDB = True


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
    'num_layers':           6,                       # Define number of layers in model
    'num_blocks':           6,                       # Define number of blocks in model
    'loss_fn':              'L2',                    # Define loss function to use - 'L1', 'L2'

    'learning_rate':        0.00974460123480158,                   # Define learning rate
    'scheduler_step':       10,                      # Define step size for learning rate scheduler
    'scheduler_gamma':      0.8070177410830386,                    # Define gamma for learning rate scheduler
    'batch_size':           8,                      # Define batch size for training
    'epochs':               100,                     # Define number of epochs for training

    'youngsModulus':        263824,                  # Define Young's Modulus for beam
    'poissonsRatio':        0.499,                   # Define Poisson's Ratio for beam
}


def train_model(model, trainLoader, testLoader, epochs, optimizer, scheduler, loss_fn=torch.nn.MSELoss(), device=torch.device('cpu'), logEpoch=10, earlyStop=100):
    """
    Trains the model.
    
    Arguments:
        model: Model to train.
        trainLoader: Training data loader.
        testLoader: Testing data loader.
        epochs (int): Number of epochs to train.
        optimizer: Optimizer to use.
        scheduler: Learning rate scheduler.
        loss_fn: Loss function to use.
        device: Device to use.

    Returns:
        bestModel: Model with best test loss.
        lossHist: Dictionary containing train and test loss history.
    """
    lossHist = {'train': [], 'test': []}
    bestModel = copy.deepcopy(model)
    startTime = time.time()
    for i in range(epochs):
        ### Training
        trainLoss = 0
        model.train()
        for data in trainLoader:
            if len(data) == 2:
                X, Y = data 
            else:
                X, Y, _ = data
            output = model(X.to(device))
            assert(output.shape == Y.shape)

            loss = loss_fn(output, Y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item()

        scheduler.step()
        trainLoss /= len(trainLoader)

        ### Validation
        testLoss = 0
        model.eval()
        for data in testLoader:
            if len(data) == 2:
                X, Y = data 
            else:
                X, Y, _ = data
            output = model(X.to(device))

            loss = loss_fn(output, Y.to(device))
            testLoss += loss.item()
        
        testLoss /= len(testLoader)

        lossHist['train'].append(trainLoss)
        lossHist['test'].append(testLoss)

        if WANDB:
            wandb.log({"Epoch": i, "Train Loss": trainLoss, "Validation Loss": testLoss})

        ### Logging
        if i % logEpoch == 0:
            print(f"Epoch [{i:03d}/{epochs}] with LR {scheduler.get_last_lr()[0]:.2e} at {time.time()-startTime:.4f}s: Train Loss: {trainLoss:.2e} -\tTest Loss: {testLoss:.2e}")
            startTime = time.time()

            ### Plot loss history
            plt.figure(figsize=(4,3))
            xEpoch = np.arange(0, i+1, 1)
            plt.plot(xEpoch, [np.log(l) for l in lossHist['train']], label='Train Loss')
            plt.plot(xEpoch, [np.log(l) for l in lossHist['test']], label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Log Loss')
            plt.legend()
            plt.grid()
            plt.savefig("outputs/loss.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()


        ### Store model with best test loss
        if testLoss == min(lossHist['test']):
            bestModel = copy.deepcopy(model)
            # Save lowest test loss model
            torch.save(model, f"outputs/model_val.pt")
        # Save model anyways
        torch.save(model, f"outputs/model.pt")

        ### Early stopping
        if i - np.argmin(lossHist['test']) > earlyStop:
            print(f"Early stopping at epoch {i}")
            break

    return bestModel, lossHist


def load_data (parameters, filePrefix, numFrames=None):
    """
    Parameters should be a list of strings describing the parameter that is varied in the data.

    Note: If the data is actuated, the force is stored in the last 3 dimensions of the state, and will not be included in dataY for prediction ground truth. The data should have been stored in a specific order of keys, [q,v,(f)].

    Note: Right now limited to 3D data.
    """
    dataX = []
    dataY = []
    dataU = []
    for param in parameters:
        loadedData = np.load(f"{filePrefix}_trajectory_{param}.npy", allow_pickle=True)[()]

        # Either q,v or q,v,f when actuated. Assume nT has to be the same for all data. f_opt is the augmented dataset of optimized forces, and should always be present.
        state = []
        control = []
        for key in loadedData:
            nT = loadedData[key].shape[0]
            numFrames = nT if numFrames is None else numFrames
            if key == 'f_opt':
                continue
            elif key == 'p':
                # Control input pressure is not defined per vertex.
                control.append(torch.tensor(loadedData[key][:numFrames], dtype=torch.float32))
            else:
                state.append(torch.tensor(loadedData[key].reshape(nT, -1, 3)[:numFrames], dtype=torch.float32))
        state = torch.cat(state, axis=-1)
        control = torch.cat(control, axis=-1) if len(control) > 0 else torch.tensor([])
        dataX.append(state[:-1])
        dataY.append(state[1:, :, :6])
        dataU.append(control[:-1])
    dataX = torch.cat(dataX, axis=0)
    dataY = torch.cat(dataY, axis=0)
    dataU = torch.cat(dataU, axis=0) if len(dataU) > 0 else torch.zeros_like(dataX)

    return dataX, dataY, dataU


def normalize_data (trainX, trainY, valX, valY, testX, testY, method='gaussian'):
    """
    Normalizes data and returns a denormalizer function. This function takes inputs of shape [T, N, D].
    """
    if method == 'gaussian':
        meanX = []
        stdX = []
        meanY = trainY.mean((0,1))
        stdY = trainY.std((0,1))
        # Element-wise mean and std
        for i in range(trainX.shape[-1]//3):
            meanX.append(trainX[:,:,3*i:3*(i+1)].mean((0,1)))
            stdX.append(trainX[:,:,3*i:3*(i+1)].std((0,1)))
        meanX = torch.cat(meanX)
        stdX = torch.cat(stdX)

        # Target data only contains position and velocity.
        trainX, valX, testX = [((X - meanX) / stdX) for X in [trainX, valX, testX]]
        trainY, valY, testY = [((Y - meanY) / stdY) for Y in [trainY, valY, testY]]

        class Normalizer:
            def __init__(self, meanX, stdX, meanY, stdY):
                self.meanX = meanX
                self.stdX = stdX
                self.meanY = meanY
                self.stdY = stdY

            def normalizeX (self, x):
                return (x - self.meanX) / self.stdX

            def normalizeY (self, y):
                return (y - self.meanY) / self.stdY
            
            def denormalizeX (self, x):
                return x * self.stdX + self.meanX
            
            def denormalizeY (self, y):
                return y * self.stdY + self.meanY
            
        normalization = Normalizer(meanX, stdX, meanY, stdY)

    else:
        raise ValueError("Normalization method not supported.")

    return normalization, trainX, trainY, valX, valY, testX, testY
        

def get_realbeam_markers (q):
    """
    Interpolate the markers on the full beam hexahedral mesh. Interpolation coefficients are stored separately and loaded.

    Arguments:
        q (torch.Tensor [N_markers, 3]): Tensor of marker positions.
    """
    data = np.load("data/sim2real_beam_oscillating/real_beam_interpolationCoeff.npy", allow_pickle=True)[()]
    neighborIdx = data['neighbor_idx']
    barycentricCoeff3d = data['barycentric_coeff_3d']
    normalScale = data['normal_scale']

    qMarker = torch.zeros((len(neighborIdx), 3), dtype=q.dtype)
    for i in range(len(neighborIdx)):
        v1 = q[neighborIdx[i][0]]
        v2 = q[neighborIdx[i][1]]
        v3 = q[neighborIdx[i][2]]
        v4 = q[neighborIdx[i][3]]
        x = barycentricCoeff3d[i]
        a = v1
        b = v2 - v1
        c = v4 - v1
        d = v1 - v2 - v4 + v3
        barycentric_coord = a + b * x[0] + c * x[1] + d * x[0] * x[1]
        normal_vector = torch.cross(b, c)
        normal_vector = normal_vector / torch.norm(normal_vector)
        barycentric_coord = barycentric_coord + normalScale[i] * normal_vector
        qMarker[i, :] = barycentric_coord

    return qMarker

def get_realarm_markers (q):
    """
    Interpolate the markers on the full sopra tetrahedral mesh. Interpolation coefficients are stored separately and loaded.

    Arguments:
        q (torch.Tensor [N_markers, 3]): Tensor of marker positions.
    """
    data = np.load("data/sim2real_arm/real_arm_interpolationCoeff.npy", allow_pickle=True)[()]
    neighborIdx = data['neighbor_idx']
    barycentricCoeff3d = data['barycentric_coeff_3d']
    normalScale = data['normal_scale']

    qMarker = torch.zeros((len(neighborIdx), 3), dtype=q.dtype)
    for i in range(len(neighborIdx)):
        v1 = q[neighborIdx[i][0]]
        v2 = q[neighborIdx[i][1]]
        v3 = q[neighborIdx[i][2]]
        x = barycentricCoeff3d[i]
        b = v2 - v1
        c = v3 - v1
        barycentric_coord = x[0] * v1 + x[1] * v2 + x[2] * v3
        normal_vector = torch.cross(b, c)
        normal_vector = normal_vector / torch.norm(normal_vector)
        barycentric_coord = barycentric_coord + normalScale[i] * normal_vector
        qMarker[i, :] = barycentric_coord

    return qMarker


def rollout_trajectory (model, testloader, device, args, normalization, sim=None, video=False):
    model.eval()
    rmses = []
    testFigs = []
    with torch.no_grad():
        ### Start from first frame, and rollout the trajectory. 
        # Assume the testloader gives the full trajectory in one batch!
        for i, (X, Y, U) in enumerate(testloader):
            startTime = time.time()
            nT = X.shape[0]
            numNodes = Y[0:1].view(1, -1, 6).shape[1]
            outQV = X[0:1].view(1, numNodes, -1)[..., :6]
            q = normalization.denormalizeY(outQV)[..., :3]

            trajectoryq = []
            for t in range(nT):
                state = outQV
                if len(U) != 0:
                    # In case actuation is present, compute current pressure forces of current state.
                    fPressure = torch.from_numpy(sim.apply_inner_pressure(U[t].numpy(), q.numpy(), chambers=[0, 1, 2, 3, 4, 5]).reshape(1, numNodes, 3)).float()
                    normfPressure = normalization.normalizeX(torch.cat([outQV, fPressure], dim=-1))[..., 6:]
                    state = torch.cat([
                        outQV,
                        normfPressure
                    ], dim=-1).view(1, -1)

                outQV = model(state.to(device)).view(1, numNodes, 6).detach().cpu()

                q = normalization.denormalizeY(outQV)[..., :3]
                trajectoryq.append(q)
            trajectoryq = torch.cat(trajectoryq, axis=0)
            print(f"Time taken for rollout: {1e3*(time.time() - startTime):.4f}ms")

            # Denormalize
            Yq = normalization.denormalizeY(Y.reshape(nT, -1, 6))[:, :, :3]

            ### Only compute error using markers
            if args['data'] == "beam_real":
                trajMarkers = []
                YMarkers = []
                for frame_i in range(trajectoryq.shape[0]):
                    trajMarkers.append(get_realbeam_markers(trajectoryq[frame_i]))
                    YMarkers.append(get_realbeam_markers(Yq[frame_i]))
                trajectoryq = torch.stack(trajMarkers, axis=0)
                Yq = torch.stack(YMarkers, axis=0)

            elif args['data'] == "arm_real":
                trajMarkers = []
                YMarkers = []
                for frame_i in range(trajectoryq.shape[0]):
                    trajMarkers.append(get_realarm_markers(trajectoryq[frame_i]))
                    YMarkers.append(get_realarm_markers(Yq[frame_i]))
                trajectoryq = torch.stack(trajMarkers, axis=0)
                Yq = torch.stack(YMarkers, axis=0)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.grid()

            if args['data'] == "beam_twisting":
                tipIdx = np.where(Yq[-1,:,2] - Yq[-1,:,2].min() < 1e-4)[0]
                ax.plot(trajectoryq[:,tipIdx,1].mean(-1), trajectoryq[:,tipIdx,2].mean(-1), linewidth=1, label="Prediction")
                ax.plot(Yq[:,tipIdx,1].mean(-1), Yq[:,tipIdx,2].mean(-1), '--', linewidth=1, label="Ground Truth")

            elif args['data'] == "arm_real":
                # tipIdx = np.where(abs(X[-1,:,2] - X[-1,:,2].min()) < 1e-4)[0]
                # plt.plot(trajectory[:,tipIdx,0].mean(-1), trajectory[:,tipIdx,1].mean(-1), linewidth=1, label="Prediction")
                # plt.plot(X[:,tipIdx,0].mean(-1), X[:,tipIdx,1].mean(-1), linewidth=1, label="Ground Truth")
                ax.plot(trajectoryq[:,:,1].mean(-1), linewidth=1, label="Prediction")
                ax.plot(Yq[:,:,1].mean(-1), '--', linewidth=1, label="Ground Truth")

            else:
                ax.plot(trajectoryq[:,:,2].mean(-1), linewidth=1, label="Prediction")
                ax.plot(Yq[:,:,2].mean(-1), '--', linewidth=1, label="Ground Truth")

            ax.legend()
            fig.savefig(f"outputs/test_traj{i}.png")

            # Save figure in WandB
            if WANDB:
                testFigs.append(wandb.Image(fig, caption=f"Test Trajectory {i}"))
            
            plt.close(fig)

            # Print error
            rmse = np.linalg.norm(trajectoryq - Yq, axis=-1)
            print(f"Test Trajectory {i}: RMSE: {rmse.mean():.4e}")
            rmses.append(rmse)


            # save marker trajectory as npy
            np.save(f"outputs/dd_markers_{i}.npy", trajectoryq)
    
    if WANDB:
        wandb.log({"Test Trajectory": testFigs})
                
    rmses = np.stack(rmses, axis=0)

    return rmses


def main (configs):
    ### Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    ### Load Data
    if configs['data'] == "beam_oscillating":
        trainParam = [f'{w:.3f}kg' for w in [0.05, 0.09, 0.10, 0.11, 0.13, 0.15, 0.16, 0.17, 0.21]]
        valParam = [f'{w:.3f}kg' for w in [0.08, 0.18]]
        testParam = [f'{w:.3f}kg' for w in [0.07, 0.12, 0.14, 0.20, 0.22]]

        filePrefix = f"data/sim2sim_beam_oscillating/sim_{configs['youngsModulus']:.0f}_{configs['poissonsRatio']:.4f}"

    elif configs['data'] == "beam_twisting":
        trainParam = [f'{i}' for i in np.arange(0, 10)]
        valParam = [f'{i}' for i in np.arange(10, 12)]
        testParam = [f'{i}' for i in np.arange(12, 20)]

        filePrefix = f"data/sim2sim_beam_twisting/sim_{configs['youngsModulus']:.0f}_{configs['poissonsRatio']:.4f}"

    elif configs['data'] == "beam_real_markers":
        trainParam = [f'{w:.3f}kg' for w in [0.05, 0.09, 0.10, 0.11, 0.13, 0.15, 0.16, 0.17, 0.21]]
        valParam = [f'{w:.3f}kg' for w in [0.08, 0.18]]
        testParam = [f'{w:.3f}kg' for w in [0.07, 0.12, 0.14, 0.20, 0.22]]

        filePrefix = f"data/sim2real_beam_oscillating/real_marker"

    elif configs['data'] == "beam_real":
        trainParam = [f'{w:.3f}kg' for w in [0.05, 0.09, 0.10, 0.11, 0.13, 0.15, 0.16, 0.17, 0.21]]
        valParam = [f'{w:.3f}kg' for w in [0.08, 0.18]]
        testParam = [f'{w:.3f}kg' for w in [0.07, 0.12, 0.14, 0.20, 0.22]]

        filePrefix = f"data/sim2real_beam_oscillating/real"
    
    elif configs['data'] == "arm_real":
        trainParam = [f'{i}' for i in np.arange(0, 35)]
        valParam = [f'{i}' for i in np.arange(35, 40)]
        testParam = [f'{i}' for i in np.arange(40, 50)]

        filePrefix = f"data/sim2real_arm/real"

    nT = np.load(f"{filePrefix}_trajectory_{trainParam[0]}.npy", allow_pickle=True)[()]['q'].shape[0]
    startTime = time.time()
    trainX, trainY, trainU = load_data(trainParam, filePrefix, numFrames=nT)
    valX, valY, valU = load_data(valParam, filePrefix, numFrames=nT)
    testX, testY, testU = load_data(testParam, filePrefix, numFrames=nT)
    print(f"\033[95mTime taken to load data: {time.time() - startTime:.4f}s \033[0m")

    ### Normalize Data using min-max of training data, for position and velocity separately.
    normalization, trainX, trainY, valX, valY, testX, testY = normalize_data(trainX, trainY, valX, valY, testX, testY, method='gaussian')

    # Flatten the data so all vertices are passed together as feature dimension
    trainX = trainX.reshape(trainX.shape[0], -1)
    trainY = trainY.reshape(trainY.shape[0], -1)
    valX = valX.reshape(valX.shape[0], -1)
    valY = valY.reshape(valY.shape[0], -1)
    testX = testX.reshape(testX.shape[0], -1)
    testY = testY.reshape(testY.shape[0], -1)


    ### Create Dataloaders
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(trainX, trainY, trainU), batch_size=configs['batch_size'], shuffle=True)
    valloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(valX, valY, valU), batch_size=nT-1, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(testX, testY, testU), batch_size=nT-1, shuffle=False)


    ### Initialize Model
    device = configs['device']
    modelParams = {
        'input_dim': trainX.shape[-1],
        'hidden_dim' : configs['hidden_dim'],
        'num_layers' : configs['num_layers'],
        'num_blocks' : configs['num_blocks'], # Only for ResMLP
        'output_dim' : trainY.shape[-1],
    }
    model = get_model(configs['model'], modelParams)
    numParams = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print(f"\033[95mNumber of trainable parameters in model: {numParams}\033[0m")
    model.to(device)

    if WANDB:
        ### Initialize Weights and Biases
        run = wandb.init(
            project=f"residual_physics_{configs['data']}",
            entity="srl_ethz",
            config=configs
        )
        wandb.log({"Number of Trainable Parameters": numParams})

    if configs['load']:
        ### Load model
        model = torch.load(f"outputs/model.pt")
        model.eval()
        print(f"Loading model.pt")
    else:
        ### Run Training
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs['scheduler_step'], gamma=configs['scheduler_gamma'])
        if configs['loss_fn'] == 'L1':
            loss_fn = torch.nn.L1Loss()
        elif configs['loss_fn'] == 'L2':
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Loss function not supported.")

        model, losshist = train_model(model, trainloader, valloader, configs['epochs'], optimizer, scheduler, loss_fn, device, logEpoch=configs['logEpoch'])

    ### Rollout Test
    sim = None
    if 'arm' in configs['data']:
        tet_params = {
            'density': 1.07e3,
            'youngs_modulus': configs['youngsModulus'],
            'poissons_ratio': configs['poissonsRatio'],
            'state_force_parameters': [0, 0, -9.80709],
            'mesh_type': 'tet',
            'arm_file': 'sopra_model/sopra.vtk'
        }
        sim = ArmEnv(42, 'outputs/sim', tet_params)

    rmsesVal = rollout_trajectory(model, valloader, device, configs, normalization, sim=sim, video=False)
    ### No rollout for sopra for now (too computationally expensive)
    rmsesTest = np.zeros_like(rmsesVal)
    if configs['load'] or configs['data'] != "arm_real":
        rmsesTest = rollout_trajectory(model, testloader, device, configs, normalization, sim=sim, video=False)
    
    print("Validation RMSEs:")
    print(f"Overall RMSE: {1e3*np.mean(rmsesVal):.4f}mm +- {1e3*np.std(rmsesVal.mean((1,2))):.3f}mm")
    print(f"Frame RMSE: {rmsesVal.mean(-1).mean()*1e3 :.4f}mm +-  {rmsesVal.mean(-1).std()*1e3:.3f}mm")
    print("Test RMSEs:")
    print(f"Overall RMSE: {1e3*np.mean(rmsesTest):.4f}mm +- {1e3*np.std(rmsesTest.mean((1,2))):.3f}mm")
    print(f"Frame RMSE: {rmsesTest.mean(-1).mean()*1e3 :.4f}mm +-  {rmsesTest.mean(-1).std()*1e3:.3f}mm")

    if WANDB:
        wandb.log({"Val RMSE Mean (mm)": rmsesVal.mean(-1).mean()*1e3, "Val RMSE Std (mm)": rmsesVal.mean(-1).std()*1e3})
        wandb.log({"Test RMSE Mean (mm)": rmsesTest.mean(-1).mean()*1e3, "Test RMSE Std (mm)": rmsesTest.mean(-1).std()*1e3})
        wandb.finish()

    return rmsesVal, rmsesTest
    


if __name__ == "__main__":
    ### Choose model type from command line
    args = ArgumentParser()
    args.add_argument("-m", dest="model", default="resmlp", help="Model type to use for training. Choose from ['mlp']", choices=['mlp', 'resmlp'])
    args.add_argument("-e", dest="epochs", default=100, type=int, help="Number of epochs to train.")
    args.add_argument("-d", dest="data", default="arm_real", help="Data to use for training.", choices=['beam_oscillating', 'beam_twisting', 'beam_real_markers', 'beam_real', 'arm_real'])
    args.add_argument("-l", dest="load", action='store_true', help="Flag to load a model for testing, or train a new model.")
    args.add_argument
    args = args.parse_args()

    # Set default configs.
    default_configs['model'] = args.model
    default_configs['epochs'] = args.epochs
    default_configs['data'] = args.data
    default_configs['load'] = args.load

    if args.load:
        WANDB = False

    rmses = main(default_configs)


