import torch
import numpy as np
from torch.utils.data.dataset import Dataset

class LearningFoward:
    def __init__(self, diffpd_model, training_set:Dataset, residual_network:torch.nn.Module, loss_fn):
        self.diffpd_model = diffpd_model
        self.training_set = training_set
        self.residual_network = residual_network
        self.loss_fn = loss_fn
    
    def forward(self, q_start, q_target, v_start, v_target, pressure_forces, f_optimized):
        raise NotImplementedError

class SupervisedLearningForward(LearningFoward):
    def __init__(self, diffpd_model, training_set:Dataset, residual_network:torch.nn.Module, loss_fn) -> None:
        super().__init__(diffpd_model, training_set, residual_network, loss_fn)
        self.batch_loss_history = []

    def forward(self, q_start, q_target, v_start, v_target, f_optimized):
        residual_forces = self.residual_network(
            torch.cat(
                (q_start, v_start), dim=1
            )
        )
        f_ext_loss = self.loss_fn(residual_forces, f_optimized)
        return f_ext_loss 
    
class PhysicsForward(LearningFoward):
    def __init__(self, diffpd_model, training_set:Dataset, residual_network:torch.nn.Module, loss_fn, normalize=True) -> None:
        super().__init__(diffpd_model, training_set, residual_network, loss_fn)
        self.total_loss_history = []
        self.batch_loss_history = []
        self.data_loss_history = []
        self.velocity_loss_history = []
        self.validation_loss_history = []
        self.f_ext_loss = []
        self.normalize = normalize
        self.diffpd_model = diffpd_model
    
    def forward(self, q_start, q_target, v_start, v_target, _):
        training_set = self.training_set
        q_init = training_set.get_q_init()
        cur_batch_size = q_start.shape[0]
        if self.normalize:
            (
                q_input,
                v_input,
                # f_normalized,
            ) = training_set.normalize(
                q=q_start,
                v=v_start,
                # f=f_optimized,
            )
        batch_loss = 0
        data_loss = 0
        velocity_loss = 0
        f_ext_loss = 0
        residual_forces_normalized = self.residual_network(
            torch.cat((q_input, v_input), dim=1)
        )
        residual_forces = training_set.denormalize(
            f=residual_forces_normalized
        )[0]
        for batch_i in range(cur_batch_size):
            q_start_i = q_start[batch_i, :] + q_init
            q_target_i = q_target[batch_i, :] + q_init
            v_start_i = v_start[batch_i, :]
            residual_forces_batch_i = residual_forces[batch_i, :]
            q, v = self.diffpd_model.forward(
                q_start_i,
                v_start_i,
                f_ext=residual_forces_batch_i,
                dt=0.01,
            )
            q_normalized = training_set.normalize(q=q - q_init)[0]
            q_target_normalized = training_set.normalize(q=q_target_i - q_init)[0]
            data_loss += self.loss_fn(q_normalized, q_target_normalized)
            # f_ext_loss += self.loss_fn(
            #     residual_forces_normalized[batch_i], f_normalized[batch_i]
            # )
            velocity_loss += self.loss_fn(v, v_target[batch_i, :])
            batch_loss = data_loss #+ f_ext_loss
            data_loss /= cur_batch_size
            velocity_loss /= cur_batch_size
            f_ext_loss /= cur_batch_size
            batch_loss /= cur_batch_size
        self.batch_loss_history.append(batch_loss.item())
        self.data_loss_history.append(data_loss.item())
        self.velocity_loss_history.append(velocity_loss.item())
        # self.f_ext_loss.append(f_ext_loss.item())
        np.save(
            f"{self.diffpd_model._folder}/batch_loss_history.npy",
            np.array(self.batch_loss_history),
        )
        np.save(
            f"{self.diffpd_model._folder}/data_loss_history.npy",
            np.array(self.data_loss_history),
        )
        np.save(
            f"{self.diffpd_model._folder}/velocity_loss_history.npy",
            np.array(self.velocity_loss_history),
        )
        # np.save(f"{self._folder}/f_ext_loss.npy", np.array(self.f_ext_loss))

        return batch_loss
