import torch
import numpy as np
from arm_model.env_arm import ArmEnv

class ArmDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list,
        arm: ArmEnv,
        data_path: str,
        start_frame=1,
        end_frame=1000,
        normalization=False,
        data_type="optimized",
    ):
        """
        samples: list of integers dataset
        """
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.q_init = torch.from_numpy(arm._q0)
        dofs = arm._dofs
        qs = []
        vs = []
        ps = []
        fs = []
        chamber_pressures = []
        assert data_type in ["optimized", "sim2sim"]
        if data_type == "optimized":
            for i in samples:
                data_info = np.load(
                    data_path + f"/optimized_data_{i}.npy", allow_pickle=True
                )[()]
                q = data_info["q_trajectory"]
                v = data_info["v_trajectory"]
                pressure_forces = data_info["pressure_forces"]
                optimized_forces = data_info["optimized_forces"]
                pressure = data_info["pressurefull"]
                q_i = torch.from_numpy(q) - self.q_init
                v_i = torch.from_numpy(v)
                p_i = torch.from_numpy(pressure_forces)
                f_i = torch.from_numpy(optimized_forces[:, 1:]).t()
                pressure_i = torch.from_numpy(pressure)
                qs.append(q_i[start_frame:end_frame + 1, :])
                vs.append(v_i[start_frame:end_frame + 1, :])
                ps.append(p_i[start_frame:end_frame, :])
                fs.append(f_i[start_frame:end_frame, :])
                chamber_pressures.append(pressure_i[start_frame:end_frame, :])
            chamber_pressures = torch.stack(chamber_pressures, axis=0)
            qs = torch.stack(qs, axis=0)
            vs = torch.stack(vs, axis=0)
            ps = torch.stack(ps, axis=0)
            fs = torch.stack(fs, axis=0)
        elif data_type == "sim2sim":
            data_info = np.load(data_path, allow_pickle=True)[()]
            qs = torch.from_numpy(data_info["q_trajectory"]) - self.q_init
            vs = torch.from_numpy(data_info["v_trajectory"])
            ps = torch.from_numpy(data_info["pressure_forces"])
            fs = torch.from_numpy(data_info["optimized_forces"]) if "optimized_forces" in data_info else torch.zeros_like(ps)
            qs = qs[samples, start_frame:end_frame + 1, :]
            vs = vs[samples, start_frame:end_frame + 1, :]
            ps = ps[samples, start_frame:end_frame, :]
            fs = fs[samples, start_frame:end_frame, :]

        self.q_mean = qs.mean(axis=[0, 1])
        self.q_std = qs.std(axis=[0, 1])
        self.q_std[self.q_std == 0] = 1
        self.v_mean = vs.mean(axis=[0, 1])
        self.v_std = vs.std(axis=[0, 1])
        self.v_std[self.v_std == 0] = 1
        self.p_mean = ps.mean(axis=[0, 1])
        self.p_std = ps.std(axis=[0, 1])
        self.p_std[self.p_std == 0] = 1
        self.f_mean = fs.mean(axis=[0, 1])
        self.f_std = fs.std(axis=[0, 1])
        self.ps = ps
        self.fs = fs
        self.chamber_pressures = chamber_pressures
        self.q_start = qs[:, :-1, :]
        self.q_target = qs[:, 1:, :]
        self.v_start = vs[:, :-1, :]
        self.v_target = vs[:, 1:, :]
        self.num_data, self.num_timesteps, _ = self.q_start.shape
        if data_type == "optimized":
            self.pressures_mean = chamber_pressures.mean(axis=[0, 1])
            self.pressures_std = chamber_pressures.std(axis=[0, 1])
            self.pressures_std[self.pressures_std == 0] = 1

    def get_init_q(self):
        return self.q_init

    def __len__(self):
        return self.num_data * self.num_timesteps

    def __getitem__(self, idx):
        data_idx = idx // self.num_timesteps
        timestep_idx = idx % self.num_timesteps
        q_start = self.q_start[data_idx, timestep_idx, :]
        v_start = self.v_start[data_idx, timestep_idx, :]
        q_target = self.q_target[data_idx, timestep_idx, :]
        v_target = self.v_target[data_idx, timestep_idx, :]
        p = self.ps[data_idx, timestep_idx, :]
        f = self.fs[data_idx, timestep_idx, :]
        return q_start, q_target, v_start, v_target, p, f

    def normalize(self, q=None, v=None, p=None, f=None, normalization_params=None):
        """
        Returns always the ordering of [q, v, p, f]. If one is not given, that slot will be skipped.
        """
        if normalization_params is None:
            res = []
            if q is not None:
                q = (q - self.q_mean) / self.q_std
                res.append(q)
            if v is not None:
                v = (v - self.v_mean) / self.v_std
                res.append(v)
            if p is not None:
                p = (p - self.p_mean) / self.p_std
                res.append(p)
            if f is not None:
                f = (f - self.f_mean) / self.f_std
                res.append(f)

            return res
        else:
            (
                q_mean,
                q_std,
                v_mean,
                v_std,
                p_mean,
                p_std,
                f_mean,
                f_std,
            ) = normalization_params
            res = []
            if q is not None:
                q = (q - q_mean) / q_std
                res.append(q)
            if v is not None:
                v = (v - v_mean) / v_std
                res.append(v)
            if p is not None:
                p = (p - p_mean) / p_std
                res.append(p)
            if f is not None:
                f = (f - f_mean) / f_std
                res.append(f)

            return res

    def denormalize(self, q=None, v=None, p=None, f=None, normalization_params=None):
        if normalization_params is None:
            res = []
            if q is not None:
                q = q * self.q_std + self.q_mean
                res.append(q)
            if v is not None:
                v = v * self.v_std + self.v_mean
                res.append(v)
            if p is not None:
                p = p * self.p_std + self.p_mean
                res.append(p)
            if f is not None:
                f = f * self.f_std + self.f_mean
                res.append(f)

            return res
        else:
            (
                q_mean,
                q_std,
                v_mean,
                v_std,
                p_mean,
                p_std,
                f_mean,
                f_std,
            ) = normalization_params
            res = []
            if q is not None:
                q = q * q_std + q_mean
                res.append(q)
            if v is not None:
                v = v * v_std + v_mean
                res.append(v)
            if p is not None:
                p = p * p_std + p_mean
                res.append(p)
            if f is not None:
                f = f * f_std + f_mean
                res.append(f)

            return res

    def return_normalization(self):
        return (
            self.q_mean,
            self.q_std,
            self.v_mean,
            self.v_std,
            self.p_mean,
            self.p_std,
            self.f_mean,
            self.f_std,
        )
