import sys
sys.path.append("../..")
import torch
import numpy as np
import time
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self,  validation_loss):
        val_min_idx = np.argmin(validation_loss)
        val_idx = len(validation_loss)
        if val_idx - val_min_idx > self.tolerance:
            self.early_stop = True

class ResidualPhysicsBase:
    def __init__(self, config):
        cuda = config["cuda"] if "cuda" in config else 0
        if config["fit"] == "SITL":
            device = "cpu"
        else:
            if type(cuda) is int:
                device = torch.device(
                    f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
                )
            else:
                device = torch.device(
                    f"cuda" if torch.cuda.is_available() else "cpu"
                )
        self.device = device
        # Initialize parameters
        self.epochs = config["epochs"] if "epochs" in config else 100
        self.batch_size = config["batch_size"] if "batch_size" in config else 32
        self.learning_rate = config["learning_rate"] if "learning_rate" in config else 5e-6
        self.optimizer = config["optimizer"] if "optimizer" in config else "adam"
        self.start_frame = config["start_frame"] if "start_frame" in config else 1
        self.end_frame = config["end_frame"] if "end_frame" in config else 1000
        self.normalize = config["normalize"] if "normalize" in config else False
        self.scaling = config["scale"] if "scale" in config else 1
        self.validation = False if "validate_set" not in config else True
        self.validate_physics = (
            config["validate_physics"] if "validate_physics" in config else False
        )
        self.grad_clip = config["grad_clip"] if "grad_clip" in config else None
        self.weight_decay = config["weight_decay"] if "weight_decay" in config else 0.0
        self.validate_epoch = config["validate_epochs"] if "validate_epochs" in config else 50
        self.gamma = config["scheduler_gamma"] if "scheduler_gamma" in config else None
        self.scheduler_step = config["scheduler_step"] if "scheduler_step" in config else None
        self.transfer_learning_model = (
            config["transfer_learning_model"]
            if "transfer_learning_model" in config
            else None
        )
        self.hidden_sizes = config["hidden_sizes"] if "hidden_sizes" in config else None
        self.num_mlp_blocks = config["num_mlp_blocks"] if "num_mlp_blocks" in config else 2
        act = config["activation"] if "activation" in config else "elu"
        if act == "relu":
            fn_act = torch.nn.ReLU()
        elif act == "elu":
            fn_act = torch.nn.ELU()
        elif act == "selu":
            fn_act = torch.nn.SELU()
        elif act == "gelu":
            fn_act = torch.nn.GELU()
        self.fn_act = fn_act

        self.early_stopping_tolerance = config["tolerance"] if "tolerance" in config else None 
        if self.early_stopping_tolerance is not None:
            self.early_stopping = EarlyStopping(
                tolerance=self.early_stopping_tolerance
            )
        else :
            self.early_stopping = EarlyStopping(tolerance=self.epochs)

    def load_model(self):
        model = torch.load(self.transfer_learning_model, map_location=self.device)
        if "module" in list(model["model"].keys())[0]:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in model["model"].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model["model"] = new_state_dict
        print("Loading pretrained model ...")
        self.residual_network.load_state_dict(model["model"])
    
    def initialize_optimizer(self, optimizer="adam", weight_decay=0.0):
        # Initialize optimizer
        assert optimizer in ["adam"]
        self.optimizer = torch.optim.Adam(
            self.residual_network.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.83
        )
        if self.transfer_learning_model is not None:
            self.optimizer.load_state_dict(
                torch.load(self.transfer_learning_model)["optimizer"]
            )
    
    def save_training_history(self, val_loss, epoch, start_time):
        min_val_loss = (
            np.min(self.validation_loss_history)
            if len(self.validation_loss_history) > 0
            else 1e10
        )
        if (
            len(self.validation_loss_history) == 0
            or val_loss < min_val_loss
        ):
            self.epoch = epoch
            self.training_time = time.time() - start_time
            self.save_model()
        self.validation_loss_history.append(val_loss)
        np.save(
            f"{self.diffpd_model._folder}/validation_loss_history.npy",
            np.array(self.validation_loss_history),
        )
        min_train_loss = (
            np.min(self.total_loss_history)
            if len(self.total_loss_history) > 0
            else 1e10
        )
        if self.total_loss_history[-1] < min_train_loss:
            self.epoch = epoch
            self.training_time = time.time() - start_time
            self.save_model(model_name="best_trained_model")
        if self.early_stopping_tolerance is not None:
            self.early_stopping(self.validation_loss_history)


    def save_model(self, model_name="residual_network"):
        """
        To call save model, you need to save the network, optimizer, scheduler, loss, epoch, and training time into self.residual_network, self.optimizer, self.scheduler, self.total_loss_history, self.epoch, and self.training_time, respectively.
        """
        torch.save(
            {
                "model": self.residual_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "loss": self.total_loss_history,
                "validated_loss": self.validation_loss_history,
                "epoch": self.epoch,
                "training_time": self.training_time,
            },
            f"{self.diffpd_model._folder}/{model_name}.pth",
        )
    