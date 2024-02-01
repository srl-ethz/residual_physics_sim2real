import torch
import numpy as np

class CantileverDataset(torch.utils.data.Dataset):
    def __init__(self, data_set, q_init, path, start_frame=0, end_frame=149):
        # q_init is a numpy array
        self.q_tensor = []
        self.v_tensor = []
        self.starting_frame = start_frame
        self.frames = end_frame
        self.q_init = q_init
        self.data_set = data_set
        self.train_set = data_set

        qs = torch.empty(
            (len(data_set), self.frames - self.starting_frame - 1, len(q_init))
        ).double()
        vs= torch.empty(
            (len(data_set), self.frames - self.starting_frame - 1, len(q_init))
        ).double()
        fs = torch.empty(
            (len(data_set), self.frames - self.starting_frame - 1, len(q_init))
        ).double()
        for i, data_idx in enumerate(data_set):
            data_i = np.load(path + f"/optimized_data_{data_idx}.npy", allow_pickle=True)[()]
            q = torch.from_numpy(data_i["q_trajectory"])[start_frame:end_frame-1]
            v = torch.from_numpy(data_i["v_trajectory"])[start_frame:end_frame-1]
            f = torch.from_numpy(data_i["optimized_forces"][:, 1:])[:, start_frame:end_frame-1]
            q = q - self.q_init
            qs[i] = q
            vs[i] = v
            fs[i] = f.t()
        self.q_mean = torch.mean(qs, dim=(0, 1))
        self.v_mean = torch.mean(vs, dim=(0, 1))
        self.f_mean = torch.mean(fs, dim=(0, 1))
        self.q_std = torch.std(qs, dim=(0, 1))
        self.v_std = torch.std(vs, dim=(0, 1))
        self.f_std = torch.std(fs, dim=(0, 1))
        self.q_std[self.q_std == 0] = 1
        self.v_std[self.v_std == 0] = 1
        self.f_std[self.f_std == 0] = 1
        self.fs = fs
        self.q_start = qs[:, :-1]
        self.q_target = qs[:, 1:]
        self.v_start = vs[:, :-1]
        self.v_target = vs[:, 1:]
        self.num_data, self.num_timesteps, _ = self.q_start.shape
        

    def get_init_q(self):
        return self.q_init

    def __len__(self):
        return self.num_data * (self.num_timesteps)

    def __getitem__(self, idx):
        data_idx = idx // self.num_timesteps
        timestep_idx = idx % self.num_timesteps
        q_start = self.q_start[data_idx, timestep_idx, :]
        v_start = self.v_start[data_idx, timestep_idx, :]
        q_target = self.q_target[data_idx, timestep_idx, :]
        v_target = self.v_target[data_idx, timestep_idx, :]
        f = self.fs[data_idx, timestep_idx, :]
        return q_start, q_target, v_start, v_target, f

    def normalize(self, q=None, v=None, f=None, normalization_params=None):
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
            if f is not None:
                f = (f - f_mean) / f_std
                res.append(f)

            return res

    def denormalize(self, q=None, v=None, f=None, normalization_params=None):
        if normalization_params is None:
            res = []
            if q is not None:
                q = q * self.q_std + self.q_mean
                res.append(q)
            if v is not None:
                v = v * self.v_std + self.v_mean
                res.append(v)
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
            self.f_mean,
            self.f_std,
        )