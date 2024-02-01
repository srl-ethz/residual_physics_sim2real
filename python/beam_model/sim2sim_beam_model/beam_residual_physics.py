import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import time
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from _utils import CantileverDataset
from env_cantilever import CantileverEnv3d
from residual_physics.residual_physics import ResidualPhysicsBase
from residual_physics.network import ResMLPResidual2, MLPResidual

class CantileverResidualPhysics(ResidualPhysicsBase):
    def __init__(self, config, folder, options):
        super().__init__(config)
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.diffpd_model = CantileverEnv3d(config['seed'], folder, options)

    def train(self, data_path, config):
        diffpd_model = self.diffpd_model
        if config["model"] == "skip_connection":
            self.residual_network = ResMLPResidual2(diffpd_model._dofs * 2, diffpd_model._dofs, hidden_size=config['hidden_size'], num_mlp_blocks=config['num_mlp_blocks'], num_block_layer=config['num_hidden_layer'])
        elif config["model"] == "MLP":
            self.residual_network = MLPResidual(diffpd_model._dofs*2, diffpd_model._dofs)
        # Initialize dataset
        training_set_index = config["training_set"]
        assert config["fit"] in ["SITL", "forces"]
        training_set = CantileverDataset(
            training_set_index,
            diffpd_model.q0,
            data_path,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
        )
        training_loader = DataLoader(
            training_set, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        
        if self.validation:
            validation_set_index = config["validate_set"]
            self.validation_set_index = validation_set_index
            validation_set = CantileverDataset(
                validation_set_index,
                diffpd_model.q0,
                data_path,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
            )
            batch_size = 1 if self.validate_physics else config["batch_size"]
            validation_loader = DataLoader(
                validation_set, batch_size=batch_size, shuffle=False
            )
        else:
            validation_set = None
            validation_loader = None

        # Load model
        self.residual_network.to(self.device)
        self.residual_network.train()
        if self.transfer_learning_model is not None:
            self.load_model()
        assert self.optimizer in ["adam"]
        self.initialize_optimizer(weight_decay=config["weight_decay"])
        # Fit model
        assert config["fit"] in ["SITL", "forces"]
        self.fit_residual_physics(
                training_set,
                training_loader,
                validation_set,
                validation_loader,
            )
    
    def fit_residual_physics(
        self,
        training_set,
        training_loader,
        validation_set,
        validation_loader,
    ):
        device = self.device
        with tqdm(total=self.epochs) as qbar:
            start_time = time.time()
            self.total_loss_history = []
            self.batch_loss_history = []
            self.f_ext_loss = []
            self.validation_loss_history = []
            for epoch in range(self.epochs):
                train_loss = 0
                batch_iter = 0
                self.residual_network.train()
                for (
                    q_start_batch,
                    q_target_batch,
                    v_start_batch,
                    v_target_batch,
                    f_optimized,
                ) in training_loader:
                    batch_size = q_start_batch.shape[0]
                    if self.normalize:
                        (
                            q_start_batch,
                            v_start_batch,
                            f_optimized,
                        ) = training_set.normalize(
                            q_start_batch,
                            v_start_batch,
                            f_optimized,
                        )
                    q_start_batch = q_start_batch.to(device)
                    v_start_batch = v_start_batch.to(device)
                    f_optimized = f_optimized.to(device)
                    batch_iter += 1
                    self.optimizer.zero_grad()
                    residual_forces = self.residual_network(
                        torch.cat(
                            (q_start_batch, v_start_batch), dim=1
                        )
                    )
                    loss = self.loss_fn(residual_forces, f_optimized) * self.scaling
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.residual_network.parameters(), self.grad_clip
                        )
                    self.batch_loss_history.append(loss.item())
                    self.optimizer.step()
                    train_loss += loss.item() * batch_size
                grad_norm = 0
                for p in self.residual_network.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm() ** 2
                grad_norm = grad_norm**0.5
                train_loss /= len(training_set)
                self.total_loss_history.append(train_loss)
                np.save(
                    f"{self.diffpd_model._folder}/total_loss_history.npy",
                    np.array(self.total_loss_history),
                )
                ####Validation#####
                val_loss = 0
                if self.validation:
                    self.residual_network.eval()
                    val_iter = 0
                    for (
                        q_start_batch,
                        q_target_batch,
                        v_start_batch,
                        v_target_batch,
                        f_optimized,
                    ) in validation_loader:
                        val_iter += 1
                        batch_size = q_start_batch.shape[0]
                        if self.normalize:
                            (
                                q_start_batch,
                                v_start_batch,
                                f_optimized,
                            ) = training_set.normalize(
                                q_start_batch,
                                v_start_batch,
                                f_optimized,
                            )
                        q_start_batch = q_start_batch.to(device)
                        v_start_batch = v_start_batch.to(device)
                        f_optimized = f_optimized.to(device)
                        residual_forces = self.residual_network(
                            torch.cat(
                                (q_start_batch, v_start_batch),
                                dim=1,
                            )
                        )
                        f_ext_loss = self.loss_fn(residual_forces, f_optimized) * self.scaling
                        val_loss += f_ext_loss.item() * batch_size
                    val_loss /= len(validation_set)
                else:
                    val_loss = self.validation_loss_history[-1] if len(self.validation_loss_history) != 0 else 1e10
                self.save_training_history(val_loss, epoch, start_time)
                qbar.set_description(
                    f"Epoch {epoch+1}, Loss: {train_loss:.3E}, Validation Loss: {val_loss:.3E}"
                )
                qbar.update(1)
