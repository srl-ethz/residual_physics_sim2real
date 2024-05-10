from pathlib import Path
import numpy as np
import torch
import os
import sys

from env_base import EnvBase
from py_diff_pd.common.sim import Sim
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable

class CantileverEnv3d (EnvBase):
    # Refinement is an integer controlling the resolution of the mesh.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        refinement = options["refinement"] if "refinement" in options else 2
        youngs_modulus = (
            options["youngs_modulus"] if "youngs_modulus" in options else 1e6
        )
        poissons_ratio = (
            options["poissons_ratio"] if "poissons_ratio" in options else 0.45
        )
        state_force_parameters = (
            options["state_force_parameters"]
            if "state_force_parameters" in options
            else ndarray([0.0, 0.0, -9.81])
        )
        density = options["density"] if "density" in options else 5e3
        twist_angle = options["twist_angle"] if "twist_angle" in options else 0

        ### Material Parameters
        la = (
            youngs_modulus
            * poissons_ratio
            / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        )
        mu = youngs_modulus / (2 * (1 + poissons_ratio))

        ### Mesh Parameters
        # Cantilever is 0.1m long, 0.03m wide, and 0.03m tall
        dx = 0.01 / refinement
        cell_nums = (round(0.1 / dx), round(0.03 / dx), round(0.03 / dx))
        assert cell_nums[0] * dx == 0.1 and cell_nums[1] * dx == 0.03, "Refinement does not properly divide the cantilever dimensions!" 
        origin = ndarray([0.0, 0.0, 0.0])

        bin_file_name = folder + "mesh.bin"
        bin_file_name = Path(bin_file_name)
        voxels = np.ones(cell_nums)
        generate_hex_mesh(voxels, dx, origin, bin_file_name)

        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))
        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, "none", youngs_modulus, poissons_ratio)
        os.remove(bin_file_name)

        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])

        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        self._obj_center = (max_corner - min_corner) / 2


        ### Boundary Conditions
        self.force_nodes = []
        for i in range(vert_num):
            vx, vy, vz = verts[i]
            if abs(vx - min_corner[0]) < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)

            # Forces are applied on the front-top edge of the cantilever
            elif abs(vx - max_corner[0]) < 1e-3 and abs(vz - max_corner[2]) < 1e-3:
                self.force_nodes.append(i)

        # State-based forces.
        deformable.AddStateForce("gravity", state_force_parameters)
        # Elasticity.
        deformable.AddPdEnergy("corotated", [2 * mu, ], [])
        deformable.AddPdEnergy("volume", [la,], [])

        ### Twist and Rotate
        q0 = ndarray(mesh.py_vertices())
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
        max_theta = twist_angle
        for i in range(1, node_nums[0]):
            theta = max_theta * i / (node_nums[0] - 1)
            c, s = np.cos(theta), np.sin(theta)
            R = ndarray([[1, 0, 0], [0, c, -s], [0, s, c]])
            center = (
                ndarray([i * dx, cell_nums[1] / 2 * dx, cell_nums[2] / 2 * dx]) + origin
            )
            for j in range(node_nums[1]):
                for k in range(node_nums[2]):
                    idx = i * node_nums[1] * node_nums[2] + j * node_nums[2] + k
                    v = ndarray(mesh.py_vertex(idx))
                    q0[3 * idx : 3 * idx + 3] = R @ (v - center) + center


        ### Simulation Parameters
        self.method = 'pd_eigen'
        self.opt = {'max_pd_iter': 10000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-7, 'verbose': 0, 'thread_ct': 16, 'use_bfgs': 1, 'bfgs_history_size': 10}
        create_folder(f"{folder}/{self.method}", exist_ok=False)

        dofs = deformable.dofs()
        self._dofs = dofs
        act_dofs = deformable.act_dofs()
        self.act_dofs = act_dofs

        self._deformable = deformable
        self.sim = Sim(deformable)

        self.q0 = torch.tensor(verts.flatten())
        self.v0 = torch.zeros_like(self.q0)
        self.f_ext = torch.zeros_like(self.q0)

        # Reload _q0, _v0 for env base to run simulation for twisting beam
        self._q0 = self.q0.clone().detach().numpy()
        self._v0 = self.v0.clone().detach().numpy()
        

    def forward (self, q, v, act=None, f_ext=None, dt=0.01):
        if f_ext is None:
            f_ext = self.f_ext
        if act is None:
            act = torch.zeros(self.act_dofs)

        q, v = self.sim(self._dofs, self.act_dofs, self.method, q, v, act, f_ext, dt, self.opt)

        return q, v


    def display_mesh (self, q, file_name, extra_points=None):
        """
        Allow to get images of the simulation
        """
        options = {
            "file_name": file_name,
            "light_map": "uffizi-large.exr",
            "sample": 8,
            "max_depth": 2,
            "camera_pos": (0.5, -1, 0.5),  # Position of camera
            "camera_lookat": (0, 0, 0.2),  # Position that camera looks at
        }
        renderer = PbrtRenderer(options)
        transforms = [("s", 2.4), ("t", [0.0, -0.2, 0.25])]

        tmp_bin_file_name = '.tmp.bin'
        self._deformable.PySaveToMeshFile(ndarray(q), tmp_bin_file_name)

        mesh = HexMesh3d()
        mesh.Initialize(tmp_bin_file_name)
        os.remove(tmp_bin_file_name)

        if extra_points is not None:
            for q_v in extra_points:
                renderer.add_shape_mesh(
                    {"name": "sphere", "center": ndarray((q_v)), "radius": 0.0025},
                    color="ff3025", #"2aaa8a",  # green
                    transforms=transforms,
                )

        renderer.add_hex_mesh(
            mesh, transforms=transforms, render_voxel_edge=True, color="0096c7"
        )
        renderer.add_tri_mesh(
            Path(root_path) / "asset/mesh/curved_ground.obj",
            texture_img="chkbd_24_0.7",
            transforms=[("s", 2)],
        )

        renderer.render()
