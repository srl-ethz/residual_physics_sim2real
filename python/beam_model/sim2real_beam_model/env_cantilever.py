from importlib.resources import path
import time
from pathlib import Path

import numpy as np
import torch
import scipy
import os

from env_base import EnvBase
from py_diff_pd.common.sim import Sim
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable

from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable


class CantileverEnv3d(EnvBase):
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
        twist_angle = options["twist_angle"] if "twist_angle" in options else 0
        state_force_parameters = (
            options["state_force_parameters"]
            if "state_force_parameters" in options
            else ndarray([0.0, 0.0, -9.81])
        )
        density = options["density"] if "density" in options else 5e3
        self.refinement = refinement

        # # Mesh parameters.
        la = (
            youngs_modulus
            * poissons_ratio
            / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        )
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        cell_nums = (10 * refinement, 3 * refinement, 3 * refinement)
        origin = ndarray([0.0, 0.0, 0.0])
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
        dx = 0.01 / refinement
        bin_file_name = folder + "/mesh.bin"
        voxels = np.ones(cell_nums)
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))
        # mesh.Scale(scale_factor=0.02) ##mm ->m
        deformable = HexDeformable()
        deformable.Initialize(
            str(bin_file_name), density, "none", youngs_modulus, poissons_ratio
        )

        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])

        self._mesh_type = "hex"
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        self._obj_center = (max_corner - min_corner) / 2
        self.qs_real = options["qs_real"] if "qs_real" in options else None
        self.qs_real_series = (
            options["qs_real_series"] if "qs_real_series" in options else None
        )
        self.dynamic_marker = (
            options["dynamic_marker"] if "dynamic_marker" in options else None
        )

        min_x = min_corner[0]
        max_x = max_corner[0]
        min_y = min_corner[1]
        max_y = max_corner[1]
        min_z = min_corner[2]
        max_z = max_corner[2]
        self.__min_x_nodes = []
        self.__max_x_nodes = []
        self.__min_y_nodes = []
        self.__max_y_nodes = []
        self.__min_z_nodes = []
        self.__max_z_nodes = []

        self.target_points = np.array(
            [
                [0.0247, 0.0, 0.015],
                [0.0447, 0.0, 0.015],
                [0.0648, 0.0, 0.015],
                [0.0447, 0.015, 0.03],
                [0.0648, 0.006, 0.03],
                [0.0648, 0.024, 0.03],
                [0.0247, 0.03, 0.006],
                [0.0247, 0.03, 0.024],
                [0.0447, 0.03, 0.006],
                [0.0648, 0.03, 0.015],
            ]
        )

        self.neighbor_idx = [
            [33, 34, 49, 50],
            [65, 66, 81, 82],
            [97, 98, 113, 114],
            [71, 75, 87, 91],
            [99, 103, 115, 119],
            [107, 111, 123, 127],
            [44, 45, 60, 61],
            [46, 47, 62, 63],
            [76, 77, 92, 93],
            [109, 110, 125, 126],
        ]
        self.barycentric_coeff = []
        self.barycentric_coeff_3d = []
        self.normal_scale = []
        self.target_idx = []
        for i in range(vert_num):
            vx, vy, vz = verts[i]
            # if abs(vz - min_z) < 1e-5:
            if abs(vx - min_x) < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)

        for point in self.target_points:
            norm = np.linalg.norm(verts - point, axis=1)
            self.target_idx.append(int(np.argmin(norm)))
        # State-based forces.
        deformable.AddStateForce("gravity", state_force_parameters)
        # Elasticity.
        deformable.AddPdEnergy(
            "corotated",
            [
                2 * mu,
            ],
            [],
        )
        deformable.AddPdEnergy(
            "volume",
            [
                la,
            ],
            [],
        )

        #base mesh
        cell_nums_base = (2,6,6)
        cell_nums_base = (cell_nums_base[0]*refinement, cell_nums_base[1]*refinement, cell_nums_base[2]*refinement)
        node_nums_base = (cell_nums_base[0] + 1, cell_nums_base[1] + 1, cell_nums_base[2] + 1)
        dx_base = 0.01 / refinement
        bin_file_name_base = "mesh_base.bin"
        bin_file_name_base = Path(bin_file_name_base)
        base_origin = ndarray([-0.1, 0.0, 0.0])
        voxels_base = np.ones(cell_nums_base)
        generate_hex_mesh(voxels_base, dx_base, origin, bin_file_name_base)
        mesh_base = HexMesh3d()
        mesh_base.Initialize(str(bin_file_name_base))
        deformable_base = HexDeformable()
        deformable_base.Initialize(
            str(bin_file_name_base), density, "none", youngs_modulus, poissons_ratio
        )

        # Initial state set by rotating the cuboid kinematically.
        dofs = deformable.dofs()
        self.dofs = dofs
        self._dofs = dofs
        act_dofs = deformable.act_dofs()
        self.act_dofs = act_dofs
        vertex_num = mesh.NumOfVertices()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.random.normal(scale=0.1, size=dofs) * density * (dx**3)
        # Data members.
        self.method = "pd_eigen"
        self.opt = {
            "max_pd_iter": 3000,
            "max_ls_iter": 10,
            "abs_tol": 1e-7,
            "rel_tol": 1e-5,
            "verbose": 0,
            "thread_ct": 16,
            "use_bfgs": 1,
            "bfgs_history_size": 10,
        }
        self._deformable = deformable
        self.sim = Sim(deformable)
        self._q0 = verts.flatten()
        self._constant_q0 = verts.flatten().copy()
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = True
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.__node_nums = node_nums

        self.__spp = 16
        self._camera_pos = (0.7, -0.8, 0.7)
        self._camera_lookat = (0, 0.15, -0.05)
        self._color = (0.3, 0.7, 0.5)
        self._scale = 1.6
        self._resolution = (1800, 1800)
        self._under_gravity_state = self.get_steady_state_under_gravity()
    
    def get_steady_state_under_gravity(self):
        q = torch.from_numpy(self._q0)
        v = torch.from_numpy(self._v0)
        f_ext = torch.from_numpy(self._f_ext)
        q_last = torch.zeros_like(q).clone()
        while torch.norm(q-q_last, p=2) > 1e-8:
            q_last = q
            q, v = self.forward(q, v, f_ext=f_ext, dt=0.01)
        return q.detach().numpy()
    
    def get_under_weight_state(self, weight):
        weight_force =  weight * 9.80709
        q = torch.from_numpy(self._q0)
        q_last = torch.zeros_like(q).clone()
        v = torch.zeros_like(q)
        f_ext_steady = torch.zeros_like(q)
        f_ext_steady[-46::3] = - weight_force / 16
        while torch.norm(q-q_last, p=2) > 1e-8:
            q_last = q
            q, v = self.forward(q, v, f_ext=f_ext_steady, dt=0.01)
        return q.detach().numpy()

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def forward(self, q, v, act=None, f_ext=None, dt=0.01):
        if f_ext is None:
            f_ext = self.f_ext
        if act is None:
            act = torch.zeros(self.act_dofs)

        q, v = self.sim(
            self.dofs, self.act_dofs, self.method, q, v, act, f_ext, dt, self.opt
        )

        return q, v

    def is_dirichlet_dof(self, dof):
        i = dof // (self.__node_nums[1] * self.__node_nums[2])
        return i == 0

    def _stepwise_loss_and_grad(self, q, v, i):
        mesh_file = self._folder / "groundtruth" / "{:04d}.bin".format(i)
        if not mesh_file.exists():
            return 0, np.zeros(q.size), np.zeros(q.size)

        mesh = HexMesh3d()
        mesh.Initialize(str(mesh_file))
        q_ref = ndarray(mesh.py_vertices())
        grad = q - q_ref
        loss = 0.5 * grad.dot(grad)
        return loss, grad, np.zeros(q.size)

    def fit_realframe(self, qs_init, MAX_ITER=200):
        # qs_init is nx3 numpy array
        """
        Optimize for the frame that would best make the real data fit the initial frame of the simulated beam.
        """
        totalR = np.eye(3)
        totalt = ndarray([0, 0, 0])
        # print("MAX_ITER", MAX_ITER)

        for i in range(MAX_ITER):
            new_qs = qs_init @ totalR.T + totalt
            R, mse_error = scipy.spatial.transform.Rotation.align_vectors(
                self.target_points, new_qs
            )
            R = R.as_matrix()
            totalR = R @ totalR
            rotated = new_qs @ R.T

            res = scipy.optimize.minimize(
                lambda x: np.mean(
                    np.sum((self.target_points - (rotated + x)) ** 2, axis=-1)
                ),
                ndarray([0, 0, 0]),
                method="BFGS",
                options={"gtol": 1e-8},
            )
            totalt = R @ totalt + res.x
            if res.fun < 1e-9:
                break

        return totalR, totalt


    def interpolate_markers_3d(
        self, steady_state: np.ndarray, real_markers: np.ndarray
    ):
        """
        Compute interpolation coefficients for 3d markers.
        barycentric_coeff stores the 2d interpolation coefficients for each marker, and normal_scale stores the distance from the marker to the surface.
        However, there are two sets of coefficients, barycentric_coeff and barycentric_coeff_3d. In default, we use barycentric_coeff as it looks better in visualization.

        barycentric_coeff_3d computes the 2d interpolation coeffs by projecting the steady markers onto the surface, and then compute the 2d interpolation coeffs.
        """
        target_pts_2d = []
        if len(self.barycentric_coeff_3d) == 0:
            for i in range(len(self.neighbor_idx)):
                face_idx = self.neighbor_idx[i]
                vert1, vert2, vert3 = (
                    steady_state[face_idx[0]],
                    steady_state[face_idx[1]],
                    steady_state[face_idx[2]],
                )
                vec1 = vert2 - vert1
                vec2 = vert3 - vert1
                face_normal = np.cross(vec1, vec2) / np.linalg.norm(
                    np.cross(vec1, vec2)
                )
                vec = real_markers[i] - vert1
                dist = np.dot(vec, face_normal)
                projected_pt = vec - dist * face_normal
                projected_pt = projected_pt + vert1
                target_pts_2d.append(projected_pt)
                self.normal_scale.append(dist)
            k = 0
            for point in target_pts_2d:
                verts = self._q0.reshape(-1, 3)
                neighboring_pt_index = self.neighbor_idx[k]
                # compute the barycentric coordinate
                # reference :https://numfactory.upc.edu/web/FiniteElements/Pract/P4-QuadInterpolation/html/QuadInterpolation.html:
                v1 = verts[neighboring_pt_index[0]]
                v2 = verts[neighboring_pt_index[1]]
                v3 = verts[neighboring_pt_index[2]]
                v4 = verts[neighboring_pt_index[3]]
                target_point = point
                a = v1 - target_point
                b = v2 - v1
                c = v4 - v1
                d = v1 - v2 - v4 + v3
                x = 0.5
                y = 0.5
                init = np.array([x, y])
                res = scipy.optimize.minimize(
                    lambda x: sum(((a + b * x[0] + c * x[1] + d * x[0] * x[1])) ** 2),
                    init,
                    method="BFGS",
                    options={"gtol": 1e-10, "disp": False, "maxiter": 1000},
                )
                x = res.x
                self.barycentric_coeff_3d.append(x)
                k += 1

    def get_markers_3d(self, qx):
        """Interpolate the simulataed markers with normal vector to the face corrected."""
        qx_marker = torch.zeros((len(self.neighbor_idx), 3), dtype=qx.dtype)

        for i in range(len(self.neighbor_idx)):
            v1 = qx[self.neighbor_idx[i][0]]
            v2 = qx[self.neighbor_idx[i][1]]
            v3 = qx[self.neighbor_idx[i][2]]
            v4 = qx[self.neighbor_idx[i][3]]
            x = self.barycentric_coeff_3d[i]
            a = v1
            b = v2 - v1
            c = v4 - v1
            d = v1 - v2 - v4 + v3
            barycentric_coord = a + b * x[0] + c * x[1] + d * x[0] * x[1]
            normal_vector = torch.cross(b, c)
            normal_vector = normal_vector / torch.norm(normal_vector)
            barycentric_coord = barycentric_coord + self.normal_scale[i] * normal_vector
            qx_marker[i, :] = barycentric_coord
        return qx_marker
    
    def get_closest_mesh_nodes_index(self, real_markers):
        """Get the closest mesh nodes index of the real_markers."""
        A = self._q0.reshape(-1, 3)
        B = real_markers
        for i in range(len(B)):
            dist_matrix = np.sqrt(np.sum((A[:, np.newaxis] - B) ** 2, axis=2))
            closest_indices = np.argmin(dist_matrix, axis=0)
        self.closest_nodes_index = closest_indices

        return closest_indices
    
    def get_closest_mesh_nodes(self, q: torch.Tensor):
        """Get the closest mesh nodes of the q."""
        q_reshape = q.reshape(-1, 3)
        closest_nodes = []
        for i in range(len(self.closest_nodes_index)):
            closest_nodes.append(q_reshape[self.closest_nodes_index[i]])
        closest_nodes = torch.stack(closest_nodes)
        return closest_nodes

    def _display_mesh(self, mesh_file, file_name):
        """
        Allow to get images of the simulation
        """
        options = {
            "file_name": file_name,
            "light_map": "uffizi-large.exr",
            "sample": self.__spp,
            "max_depth": 2,
            "camera_pos": (0.5, -1.0, 0.5),  # Position of camera
            "camera_lookat": (0, 0, 0.2),  # Position that camera looks at
            "resolution": (1800, 1800),
        }
        renderer = PbrtRenderer(options)
        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        scale = 0.8
        transforms = [("s", 2.2), ("t", [0.0, -0.2, 0.25])]  # scale),
        if self.qs_real is not None:
            for q_v in self.qs_real:
                renderer.add_shape_mesh(
                    {"name": "sphere", "center": ndarray((q_v)), "radius": 0.0025},
                    color="2aaa8a",  # green
                    transforms=transforms,
                )
        if self.dynamic_marker is not None:
            for q_v in self.dynamic_marker:
                renderer.add_shape_mesh(
                    {"name": "sphere", "center": ndarray((q_v)), "radius": 0.0025},
                    color="ff3025",  #red
                    transforms=transforms,
                )

        renderer.add_hex_mesh(
            mesh, transforms=transforms, render_voxel_edge=True, color="0096c7"
        )
        
        transforms_cylinder = [("s", 2.4), ("t", [-0.048, -0.232, 0.213]), ]#("r", [np.pi/2, 0, 1, 0])]  # scale),
        mesh_base = HexMesh3d()
        mesh_base.Initialize(str("mesh_base.bin"))
        renderer.add_hex_mesh(
            mesh_base, transforms=transforms_cylinder, render_voxel_edge=False, color="2ca02c"
        )

        renderer.add_tri_mesh(
            Path(root_path) / "asset/mesh/curved_ground.obj",
            texture_img="chkbd_24_0.7",
            transforms=[("s", 2)],
        )

        renderer.render()

    def visualization(self, vis_folder, q, render_frame_skip=1, errors=None):
        # print(self.qs_real)
        if errors is not None:
            normalized_errors = (errors - errors.min()) / (
                errors.max() - errors.min()
            )  # Normalize over all frames of the simulation so the colorbar doesn't shift. OR normalize over every timestep so we see at every frame where the error occurs in the mesh
            normalized_errors[errors.max() - errors.min() == 0] = 1
        if vis_folder is not None:
            path_vis = self._folder / Path(vis_folder)
            path_vis.mkdir(exist_ok=True)
            for i, qi in enumerate(q):
                if i % render_frame_skip != 0:
                    continue
                mesh_file = str(path_vis) + "{:04d}.bin".format(i)
                self._deformable.PySaveToMeshFile(qi, mesh_file)
                # print(qi)
                self._display_mesh(mesh_file, path_vis / "{:04d}.png".format(i))
                os.remove(str(path_vis) + "{:04d}.bin".format(i))

    def vis_dynamic_markers(
        self, vis_folder, q, dynamic_target, frame=0, render_frame_skip=1, errors=None
    ):
        mesh_file = str(self._folder / vis_folder) + "{:04d}.bin".format(frame)
        self.qs_real = dynamic_target
        self._deformable.PySaveToMeshFile(q, mesh_file)
        self._display_mesh(
            mesh_file, self._folder / vis_folder / "{:04d}.png".format(frame)
        )

    def vis_dynamic_sim2real_markers(
        self,
        vis_folder,
        q,
        sim_target,
        real_target,
        frame=0,
        render_frame_skip=1,
        errors=None,
    ):
        path_vis = self._folder / Path(vis_folder)
        path_vis.mkdir(exist_ok=True)
        mesh_file = str(self._folder / vis_folder) + "{:04d}.bin".format(frame)
        self.dynamic_marker = sim_target
        self.qs_real = real_target
        self._deformable.PySaveToMeshFile(q, mesh_file)
        self._display_mesh(
            mesh_file, self._folder / vis_folder / "{:04d}.png".format(frame)
        )
        os.remove(str(path_vis) + "{:04d}.bin".format(frame))

    def update_qs_real_series(self, qs_real_series):
        self.qs_real_series = qs_real_series

    def _stepwise_loss_and_grad(self, q, v, i):
        # This method is for system identification of arm
        if self.qs_real_series is None:
            return 0.0, np.zeros_like(q), np.zeros_like(q)
        # Match z coordinate of the target motion with reality of specific target point
        q = torch.tensor(q, requires_grad=True)
        # sim2real
        q_markers = self.get_markers_3d(q.reshape(-1, 3))
        diff = -q_markers[:, -1] + torch.from_numpy(
            self.qs_real_series[i, :, -1]
        )
        loss = 0.5 * torch.dot(diff.flatten(), diff.flatten())
        loss.backward()
        grad = q.grad.detach().numpy()
        loss = loss.detach().numpy()

        return loss, grad, np.zeros_like(q.detach().numpy())
