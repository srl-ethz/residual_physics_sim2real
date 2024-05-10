import sys

sys.path.append("../")
sys.path.append("../../..")

from pathlib import Path
import numpy as np

# from twist_beam_env import CantileverEnv3d
from env_cantilever import CantileverEnv3d



def data_generate(sample_num, frame_num, hex_params, vis=True):
    seed = 42
    folder = "twist"
    refinement = 1
    twist_angle_range = np.random.uniform(np.pi / 6, np.pi, size=sample_num)
    for idx, twist_angle in enumerate(twist_angle_range):
        qs = []
        vs = []
        hex_params["twist_angle"] = twist_angle
        env = CantileverEnv3d(
            seed,
            folder,
            hex_params,
        )
        deformable = env.deformable()

        thread_cts = [8]

        methods = ("pd_eigen",)
        opts = (
            {
                "max_pd_iter": 3000,
                "max_ls_iter": 10,
                "abs_tol": 1e-7,
                "rel_tol": 1e-5,
                "verbose": 0,
                "thread_ct": 16,
                "use_bfgs": 1,
                "bfgs_history_size": 10,
            },
        )

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = env.q0
        v0 = env.v0
        a0 = np.random.uniform(size=act_dofs)
        a0 = np.zeros(a0.shape)
        f0 = np.zeros((dofs))
        # Visualization.
        dt = 1e-2
        f0_series = [f0 for _ in range(int(frame_num))]

        for method, opt in zip(methods, opts):
            vis_folder = method if vis else None
            _, _, qv = env.simulate(
                dt,
                int(frame_num),
                "pd_eigen" if method == "pd_no_bfgs" else method,
                opt,
                q0,
                v0,
                None,
                f0_series,
                require_grad=False,
                vis_folder=vis_folder,
            )

        qs.append(qv["q"])
        vs.append(qv["v"])
        data = {"q": np.stack(qs), "v": np.stack(vs)}
        data_folder = 'data_real'
        Path(f"{data_folder}").mkdir(parents=True, exist_ok=True)
        np.save(f"{data_folder}/trajectory{idx}.npy", data)


if __name__ == "__main__":
    youngs_modulus = 263824
    sample_num = 20 # 200
    frame_num = 100
    hex_params = {
                "refinement": 1,
                "youngs_modulus": youngs_modulus,
                "twist_angle": 0.0,
                'density': 1.07e3,
                'poissons_ratio': 0.499,
                'state_force_parameters': [0, 0, -9.80709],
            }
    data_generate(sample_num, frame_num, hex_params, vis=False)
