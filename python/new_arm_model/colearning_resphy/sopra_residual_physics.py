import sys

sys.path.append("../..")
sys.path.append("..")
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from new_arm_model.env_arm import ArmEnv
from new_arm_model.colearning_resphy._utils import (
    ArmDataset,
)
from residual_physics.network import MLPResidual, ResMLPResidual2
from model import SupervisedLearningForward, PhysicsForward, LearningFoward
from validate_residual_physics import main as val_main
from residual_physics.residual_physics import ResidualPhysicsBase

class SoPrAResidualPhysics(ResidualPhysicsBase):
    def __init__(self, config, save_folder, params):
        super().__init__(config)
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.diffpd_model = ArmEnv(config['seed'], save_folder, params)

    def train(self, data_path, config):
        diffpd_model = self.diffpd_model
        assert config["model"] in ["skip_connection", "MLP"]
        assert config["fit"] in ["SITL", "forces"]

        if config["model"] == "skip_connection":
            self.residual_network = ResMLPResidual2(diffpd_model._dofs * 3, diffpd_model._dofs, num_mlp_blocks=config['num_mlp_blocks'],num_block_layer=config['num_block_layer'], hidden_size=config['hidden_size'], act_fn=self.fn_act)
        elif config["model"] == "MLP":
            self.residual_network = MLPResidual(diffpd_model._dofs * 3, diffpd_model._dofs, hidden_sizes=self.hidden_sizes, act_fn=self.fn_act)

        # Initialize dataset
        training_set_index = config["training_set"]
        training_set = ArmDataset(
            training_set_index,
            diffpd_model,
            data_path,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
        )
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        if self.validation:
            validation_set_index = config["validate_set"]
            self.validation_set_index = validation_set_index
            validation_set = ArmDataset(
                validation_set_index,
                diffpd_model,
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
        print(f"Number of parameters: {self.residual_network.count_parameters()}")

        config["hidden_sizes"] = self.residual_network.hidden_sizes
        # Initialize optimizer
        self.initialize_optimizer(weight_decay=config["weight_decay"])

        # Fit model
        if config['fit'] == "SITL":
            forward_model : LearningFoward = PhysicsForward(diffpd_model, training_set, self.residual_network, self.loss_fn)
        elif config['fit'] == "forces":
            forward_model : LearningFoward = SupervisedLearningForward(self, training_set, self.residual_network, self.loss_fn) 
        
        self.fit_residual_physics(
                forward_model,
                training_set,
                training_loader,
                validation_set,
                validation_loader,
            )

    def fit_residual_physics(
        self,
        forward_model,
        training_set,
        training_loader,
        validation_set,
        validation_loader,
    ):
        # Initialize auxiliary variables
        if self.device == 'cuda':
            self.residual_network = torch.nn.DataParallel(self.residual_network)
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
                for (
                    q_start_batch,
                    q_target_batch,
                    v_start_batch,
                    v_target_batch,
                    pressure_forces_batch,
                    f_optimized,
                ) in training_loader:
                    batch_size = q_start_batch.shape[0]
                    if self.normalize:
                        (
                            q_start_batch,
                            v_start_batch,
                            pressure_forces_batch,
                            f_optimized,
                        ) = training_set.normalize(
                            q_start_batch,
                            v_start_batch,
                            pressure_forces_batch,
                            f_optimized,
                        )
                    q_start_batch = q_start_batch.to(device)
                    v_start_batch = v_start_batch.to(device)
                    pressure_forces_batch = pressure_forces_batch.to(device)
                    f_optimized = f_optimized.to(device)
                    batch_iter += 1
                    self.optimizer.zero_grad()
                    loss = forward_model.forward(
                        q_start_batch,
                        q_target_batch,
                        v_start_batch,
                        v_target_batch,
                        pressure_forces_batch,
                        f_optimized,
                    )

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
                        grad_norm += p.grad.norm()**2
                grad_norm = grad_norm**0.5
                train_loss /= len(training_set)
                if self.scheduler_step is not None:
                    print("Scheduler step: ", self.scheduler_step)
                    self.scheduler.step()
                self.total_loss_history.append(train_loss)
                np.save(
                    f"{self.diffpd_model._folder}/total_loss_history.npy",
                    np.array(self.total_loss_history),
                )
                ####Validation#####
                if self.validation and (not self.validate_physics):
                    val_loss = self.validate_residual_forces(epoch, training_set, validation_set, validation_loader)
                elif self.validation and self.validate_physics and epoch % self.validate_epoch == 0:
                    self.residual_network.eval()
                    self.residual_network.to("cpu")
                    val_metrics = val_main(self.diffpd_model._folder, self.validation_set_index, self.residual_network)
                    val_loss = val_metrics["validate_trajectory_error_mean"]
                    self.residual_network.train()
                    self.residual_network.to(self.device)
                else:
                    val_loss = self.validation_loss_history[-1] if len(self.validation_loss_history) != 0 else 1e10
                self.save_training_history(val_loss, epoch, start_time)
                qbar.set_description(
                    f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.3E}, Validation Loss: {val_loss:.3E}"
                )
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
                qbar.update(1)

    def validate_residual_forces(self, epoch, training_set, validation_set, validation_loader):
        val_loss = 0
        device = self.device
        if self.validation and (not self.validate_physics):
            self.residual_network.eval()
            val_iter = 0
            for (
                q_start_batch,
                q_target_batch,
                v_start_batch,
                v_target_batch,
                pressure_forces_batch,
                f_optimized,
            ) in validation_loader:
                val_iter += 1
                batch_size = q_start_batch.shape[0]
                if self.normalize:
                    (
                        q_start_batch,
                        v_start_batch,
                        pressure_forces_batch,
                        f_optimized,
                    ) = training_set.normalize(
                        q_start_batch,
                        v_start_batch,
                        pressure_forces_batch,
                        f_optimized,
                    )
                q_start_batch = q_start_batch.to(device)
                v_start_batch = v_start_batch.to(device)
                pressure_forces_batch = pressure_forces_batch.to(device)
                f_optimized = f_optimized.to(device)
                residual_forces = self.residual_network(
                    torch.cat(
                        (q_start_batch, v_start_batch, pressure_forces_batch),
                        dim=1,
                    )
                )
                f_ext_loss = (
                    self.loss_fn(residual_forces, f_optimized) * self.scaling
                )
                val_loss += f_ext_loss.item() * batch_size
            val_loss /= len(validation_set)
            
        return val_loss
        