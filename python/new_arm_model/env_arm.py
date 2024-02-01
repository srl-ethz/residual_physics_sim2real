import sys

# this is a terrible hack
sys.path.append("../")
sys.path.append("../../..")

from pathlib import Path
import time
import os
import numpy as np
import scipy.optimize
import trimesh
import meshio
import torch
from tqdm import tqdm
from collections import OrderedDict
from trimesh import Trimesh

from py_diff_pd.common.common import ndarray, print_info
from py_diff_pd.common.project_path import root_path

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.sim import Sim

from py_diff_pd.core.py_diff_pd_core import HexMesh3d, TetMesh3d, TetDeformable
from py_diff_pd.common.display import export_gif, export_mp4
from py_diff_pd.common.tet_mesh import generate_tet_mesh
from py_diff_pd.common.tet_mesh import (
    get_contact_vertex as get_tet_contact_vertex,
    get_boundary_face,
)


class ArmEnv(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)
        Path(folder).mkdir(parents=True, exist_ok=True)

        np.random.seed(seed)
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
        material = options["material"] if "material" in options else "none"
        mesh_type = options["mesh_type"] if "mesh_type" in options else "tet"
        assert mesh_type in ["tet", "hex"], "Invalid mesh type!"
        arm_file = (
            options["arm_file"] if "arm_file" in options else "sopra_model/Segment.vtk"
        )

        la = (
            youngs_modulus
            * poissons_ratio
            / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        )
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1.07e3

        ### Create Mesh
        arm_mesh = meshio.read(arm_file)
        tmp_bin_file_name = ".tmp.bin"
        if "Segment" in arm_file:
            generate_tet_mesh(
                arm_mesh.points * 1e-3, arm_mesh.cells[-1].data, tmp_bin_file_name
            )
        else:
            generate_tet_mesh(
                arm_mesh.points, arm_mesh.cells[-1].data, tmp_bin_file_name
            )
        fn = tmp_bin_file_name
        mesh = TetMesh3d()
        mesh.Initialize(tmp_bin_file_name)

        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])
        element_num = mesh.NumOfElements()
        self._elements = np.array(
            [np.array(mesh.py_element(i), dtype=np.int64) for i in range(element_num)],
            dtype=np.int64,
        )
        self._vertices = verts
        self._boundary = self._get_boundary_ordered()
        # Rotate along x by 90 degrees.
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        # as this shows, corner of voxel mesh is aligned to the origin...
        self._obj_center = (max_corner + min_corner) / 2
        deformable = TetDeformable()
        deformable.Initialize(
            tmp_bin_file_name, density, "none", youngs_modulus, poissons_ratio
        )
        os.remove(fn) if os.path.exists(fn) else None
        ### Boundary conditions: Glue vertices spatially
        for i in range(vert_num):
            vx, vy, vz = verts[i]

            if abs(vz - max_corner[2]) < 1e-4:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)

        # Add inner chamber face
        components = self.split_boundary()
        longest_list = max(components, key=len)
        self.outer_faces = self._boundary[longest_list]
        components.remove(longest_list)
        inner_faces = []
        for component in components:
            boundary_component = self._boundary[component]
            inner_faces.append(boundary_component)
        self._inner_faces = inner_faces

        # State-based forces.
        deformable.AddStateForce("gravity", state_force_parameters)

        if material == "none":
            # For corotated material
            deformable.AddPdEnergy("corotated", [2 * mu], [])
            deformable.AddPdEnergy("volume", [la], [])

        # visualize points
        self.dynamic_marker = None
        self.qs_real = None

        # simulation setting
        self.method = "pd_eigen"
        self.opt = {
            "max_pd_iter": 3000,
            "max_ls_iter": 10,
            "abs_tol": 1e-6,
            "rel_tol": 1e-7,
            "verbose": 0,
            "thread_ct": 16,
            "use_bfgs": 1,
            "bfgs_history_size": 10,
        }
        act_dofs = deformable.act_dofs()
        self.act_dofs = act_dofs
        self.sim = Sim(deformable)

        # Initialize data members.
        dofs = deformable.dofs()
        q0 = np.copy(verts)
        q0 = q0.ravel()
        v0 = ndarray(np.zeros(dofs)).ravel()
        f_ext = ndarray(np.zeros(dofs)).ravel()
        self._deformable = deformable
        self._dofs = dofs
        self._q0 = q0  # N x 1 shape ndarray
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self._mesh_type = mesh_type

        # Visualization parameters
        self._camera_pos = (0.05, -0.05, 0.05)
        self._camera_lookat = (0, 0, 0.02)
        self._scale = 1.6
        self._spp = options["spp"] if "spp" in options else 8
        self._resolution = (1800, 1800)#(600, 600)
        self.closest_tri = None
        self.interpolation_coeffs = None
        self._stepwise_loss = True

        # print("Loading ", arm_file)
        # print(
        #     "Initialize sopra with youngs_modulus: ",
        #     youngs_modulus,
        #     " and poissons_ratio: ",
        #     poissons_ratio,
        #     "Degress of freedom: ",
        #     self._dofs,
        # )

    def set_measured_markers(self, measured_markers: np.ndarray = None):
        """
        measured_markers: np.array of shape (#markers, 3)
        measured_markers save the measured points in simulation coordinate system.
        """
        if measured_markers is not None:
            self.measured_markers = measured_markers
        else:
            self.measured_markers = (
                np.array(
                    [
                        [-62.9, 22, -8.2],  # 1
                        [31.3, -21.6, -8.2],  # 2
                        [-62.9, -23, -8.2],  # 0
                        [21, 9, -128],  # 4
                        [18, -17, -128],  # 3
                        [-21, 10, -128],  # 6
                        [-8, 21, -128],  # 5
                        [-20, -20, -128],  # g
                        [18, 10, -146],  # 8
                        [-26, 8, -146],  # 9
                        [-3, -29, -146],  # 10
                        [-6, 14, -266],  # 12
                        [11, -10, -266],  # 11
                        [-8, -20, -266],  # 14
                        [-21, -10, -266],  # 13
                    ]
                )
                * 1e-3
                + np.array([0, 0, 2.92047]) * 1e-3
            )
    
    def get_measured_markers(self):
        return self.measured_markers

    def fix_tet_faces(self, verts):
        verts = ndarray(verts)
        v0, v1, v2, v3 = verts
        f = []
        if np.cross(v1 - v0, v2 - v1).dot(v3 - v0) < 0:
            f = [(0, 1, 2), (2, 1, 3), (1, 0, 3), (0, 2, 3)]
        else:
            f = [(1, 0, 2), (1, 2, 3), (0, 1, 3), (2, 0, 3)]

        return ndarray(f).astype(np.int)

    def _get_boundary_ordered(self) -> ndarray:
        """
        The boundary mesh whose normal points outward will be returned in the order of right hand rule. This is necessary to apply pressure to chambers
        """
        boundary = OrderedDict()
        for e in self._elements:
            element_vert = []
            for vi in e:
                element_vert.append(self._vertices[vi])
            element_vert = ndarray(element_vert)
            face_indices = self.fix_tet_faces(element_vert)
            for indices in face_indices:
                face = e[indices]
                sorted_face = tuple(sorted(face))
                if sorted_face in boundary:
                    del boundary[sorted_face]
                else:
                    boundary[sorted_face] = face
        faces = np.vstack(boundary.values())
        return faces

    def split_boundary(self):
        all_meshes = Trimesh(
            self._vertices, self._boundary, process=False, validate=False
        )
        components = trimesh.graph.connected_component_labels(
            edges=all_meshes.face_adjacency, node_count=len(all_meshes.faces)
        )
        components_indices = []
        for label in np.unique(components):
            components_indices.append(np.argwhere(components == label).ravel())
        return components_indices

    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)

    def forward(self, q, v, act=None, f_ext=None, dt=0.01):
        if f_ext is None:
            f_ext = self.f_ext
        if act is None:
            act = torch.zeros(self.act_dofs)

        q, v = self.sim(
            self._dofs, self.act_dofs, self.method, q, v, act, f_ext, dt, self.opt
        )

        return q, v

    def normalized_forward(self, dataset, q, v, p, f, dt=0.01):
        """
        Have normalized inputs according to data in dataset. This wrapper will call the forward function with denormalization for the inputs, and normalization for the outputs. NOTE: pressure and external forces (residual forces) are split as separate inputs in this case!!! Not the convention for the rest of DiffPD.
        """
        q = dataset.denormalize(q=q.view(-1, 3))[0].view(-1)
        v = dataset.denormalize(v=v.view(-1, 3))[0].view(-1)
        p = dataset.denormalize(p=p.view(-1, 3))[0].view(-1)
        f = dataset.denormalize(f=f.view(-1, 3))[0].view(-1)

        q, v = self.forward(q, v, f_ext=p + f, dt=dt)

        q = dataset.normalize(q=q.view(-1, 3))[0].view(-1)
        v = dataset.normalize(v=v.view(-1, 3))[0].view(-1)

        return q, v

    def _display_mesh(self, mesh_file, file_name):
        """
        the implementation in env_base.py only allows for hex meshes
        """
        options = {
            "file_name": file_name,
            "light_map": "uffizi-large.exr",
            "sample": self._spp,
            "max_depth": 2,
            "camera_pos": (0.5, -1.0, 0.5),  # Position of camera
            "camera_lookat": (0, 0, 0.2),  # Position that camera looks at
            "resolution": self._resolution,
        }
        renderer = PbrtRenderer(options)
        transforms = [("s", self._scale), ("t", [0.0, 0.0, 0.50])]  # scale),

        if self._mesh_type == "tet":
            mesh = TetMesh3d()
            mesh.Initialize(mesh_file)
            renderer.add_tri_mesh(
                mesh, render_tet_edge=True, color="0096c7", transforms=transforms
            )
        elif self._mesh_type == "hex":
            mesh = HexMesh3d()
            mesh.Initialize(mesh_file)
            renderer.add_hex_mesh(
                mesh,
                render_voxel_edge=True,
                color=self._color,
                transforms=[("s", self._scale), ("t", [0.0, -0.0, -0.3])],
            )
        if self.dynamic_marker is not None:
            for q_v in self.dynamic_marker:
                renderer.add_shape_mesh(
                    {"name": "sphere", "center": ndarray((q_v)), "radius": 0.0035},
                    color="ff3025",  # red
                    transforms=transforms,
                )
        if self.qs_real is not None:
            for q_v in self.qs_real:
                renderer.add_shape_mesh(
                    {"name": "sphere", "center": ndarray((q_v)), "radius": 0.0035},
                    color="2aaa8a",  # green
                    transforms=transforms,
                )

        renderer.add_tri_mesh(
            Path(root_path) / "asset/mesh/curved_ground.obj",
            texture_img="chkbd_24_0.7",
            transforms=[("s", 3)],
        )

        renderer.render()

    def apply_inner_pressure(self, p, q=None, chambers=[0, 1, 2, 3, 4, 5]):
        """
        Applies some pressure on all nodes on the inner surface of specific chambers.
        Arguments:
            p (list of float) : pressure difference uniformly in the cube, difference with pressure outside the cube
            q (ndarray with shape [3*N]) : (optional) node positions at this timestep
            chambers (list) : (optional) chambers where pressure should be applied input as a list of integers.

        Returns:
            f (ndarray with shape [3*N]) : external forces on all nodes for this one timestep.
        """
        f_ext = np.zeros_like(self._f_ext)
        f_ext_count = np.zeros_like(
            f_ext, dtype=int
        )  # We apply forces multiple times on same vertex, we take the average in the end

        verts = q.reshape(-1, 3) if q is not None else self._q0.reshape(-1, 3)

        # chamber_faces = np.concatenate([self._inner_faces[i] for i in chambers])
        for chamber_idx in range(len(chambers)):
            chamber_faces = self._inner_faces[chambers[chamber_idx]]
            chamber_p = p[chambers[chamber_idx]]
            for face in chamber_faces:
                # Find surface normal (same for tet and hex)
                v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                cross_prod = np.cross((v1 - v0), (v2 - v0))
                if self._mesh_type == "tet":
                    # Triangle area
                    area_factor = 0.5

                elif self._mesh_type == "hex":
                    # Square area
                    area_factor = 1

                f_pressure = -chamber_p * area_factor * cross_prod

                for vertex_idx in face:
                    # Apply forces in x, y and z directions (3 dimensional)
                    for d in range(3):
                        # Increase occurence count of vertex
                        f_ext_count[3 * vertex_idx + d] += 1
                        # Set pressure force
                        # The computation refers to SOFA SurfacePressureConstraint.
                        f_ext[3 * vertex_idx + d] += f_pressure[d] / 3

        return f_ext

    def fit_realframe(self, qs_init, MAX_ITER=300):
        """
        qs_init: N x 3 numpy array, initial position of markers in the real world.
        Optimize for the frame that would best make the real data fit the initial frame of the simulated beam.
        """
        totalR = np.eye(3)
        totalt = ndarray([0, 0, 0])

        for i in range(MAX_ITER):
            new_qs = qs_init @ totalR.T + totalt
            R, mse_error = scipy.spatial.transform.Rotation.align_vectors(
                self.measured_markers, new_qs
            )

            R = R.as_matrix()
            totalR = R @ totalR
            rotated = new_qs @ R.T

            res = scipy.optimize.minimize(
                lambda x: np.mean(
                    np.sum((self.measured_markers - (rotated + x)) ** 2, axis=-1)
                ),
                ndarray([0, 0, 0]),
                method="BFGS",
                options={"gtol": 1e-8},
            )
            totalt = R @ totalt + res.x

            if res.fun < 1e-9:
                break

        return totalR, totalt

    # Compute simulated markers
    def compute_interpolation_coeff(self, real_markers=None):
        """
        real_markers: Nx3 numpy array, transformed coordiantes of markers at initial state in simulation coordinate system.
        Compute the interpolation coefficients for each measured marker to the closest triangle in the mesh.
        """
        # Find the closest triangle on the outer surface for each target point to make interpolation
        self.closest_tri = []
        self.interpolation_coeffs = []
        outer_faces_nodes = np.unique(self.outer_faces)
        outer_faces_vertices = self._vertices[outer_faces_nodes]
        for i, marker in enumerate(real_markers):
            closest_node = np.argmin(
                np.linalg.norm(outer_faces_vertices - marker, axis=1)
            )
            closest_node_idx = outer_faces_nodes[closest_node]
            near_mesh = self.outer_faces[
                np.where((self.outer_faces == closest_node_idx).any(axis=1))
            ]
            closest_mesh_vertices = self._vertices[near_mesh]
            closest_mesh = near_mesh[
                np.argmin(np.linalg.norm(closest_mesh_vertices - marker, axis=(1, 2)))
            ]
            self.closest_tri.append(closest_mesh)

        # The code refers to https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
        for i, face_idx in enumerate(self.closest_tri):
            face_idx = self.closest_tri[i]
            vert1, vert2, vert3 = (
                self._vertices[face_idx[0]],
                self._vertices[face_idx[1]],
                self._vertices[face_idx[2]],
            )
            u = vert2 - vert1
            v = vert3 - vert1
            n = np.cross(u, v)
            w = real_markers[i] - vert1
            gamma = np.dot(np.cross(u, w), n) / np.linalg.norm(n) ** 2
            beta = np.dot(np.cross(w, v), n) / np.linalg.norm(n) ** 2
            alpha = 1 - gamma - beta
            project_point = alpha * vert1 + beta * vert2 + gamma * vert3
            marker_diff = real_markers[i] - project_point
            normal_coeff = np.dot(marker_diff, n) / np.linalg.norm(n)
            self.interpolation_coeffs.append([alpha, beta, gamma, normal_coeff])

    def return_simulated_markers(self, arm: torch.Tensor) -> torch.Tensor:
        simulated_markers = []
        for idx, face_idx in enumerate(self.closest_tri):
            vert1, vert2, vert3 = arm[face_idx[0]], arm[face_idx[1]], arm[face_idx[2]]
            u = vert2 - vert1
            v = vert3 - vert1
            n = torch.cross(u, v) / torch.norm(torch.cross(u, v))
            simulated_marker = (
                self.interpolation_coeffs[idx][0] * vert1
                + self.interpolation_coeffs[idx][1] * vert2
                + self.interpolation_coeffs[idx][2] * vert3
            )
            normal_correct = self.interpolation_coeffs[idx][3] * n
            simulated_markers.append(simulated_marker + normal_correct)
        simulated_markers = torch.stack(simulated_markers, dim=0)
        return simulated_markers

    # Visualization
    def vis_dynamic_sim2real_markers(
        self,
        vis_folder,
        q: np.ndarray,
        sim_target: np.ndarray = None,
        real_target: np.ndarray = None,
        frame=0,
        render_frame_skip=1,
        errors=None,
    ):
        path_vis = self._folder / Path(vis_folder)
        path_vis.mkdir(exist_ok=True, parents=True)
        mesh_file = str(self._folder / vis_folder) + "{:04d}.bin".format(frame)
        self.dynamic_marker = sim_target
        self.qs_real = real_target
        self._deformable.PySaveToMeshFile(q, mesh_file)
        self._display_mesh(
            mesh_file, self._folder / vis_folder / "{:04d}.png".format(frame)
        )
        os.remove(str(path_vis) + "{:04d}.bin".format(frame))

    def visualization(self, vis_folder: str, q: list, render_frame_skip=1):
        if vis_folder is not None:
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder, exist_ok=True)
            with tqdm(total=len(q)) as qbar:
                for i, qi in enumerate(q):
                    if i % render_frame_skip != 0:
                        continue
                    mesh_file = str("{:04d}.bin".format(i))
                    self.deformable().PySaveToMeshFile(qi, mesh_file)
                    self._display_mesh(mesh_file, vis_folder + "/{:04d}.png".format(i))
                    os.remove(mesh_file)
                    qbar.update(1)
            export_mp4(vis_folder, "arm.mp4", 10)

    # For system identification
    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        # Using Corotated
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        # Material Jacobian returns d(la, mu)/d(E, nu) for lame, shear modulus and youngs, poisson ratio.
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]

        return jac_total

    def _stepwise_loss_and_grad(self, q, v, i):
        # This method is for system identification of arm
        if self.qs_real is None:
            raise ValueError("Real markers are not set yet!")
            return 0.0, np.zeros_like(q), np.zeros_like(q)

        # Match z coordinate of the target motion with reality of specific target point
        q = torch.tensor(q, requires_grad=True)
        # Discard 3 base markers for loss computation
        q_markers = self.return_simulated_markers(q.reshape(-1, 3))[3:]
        diff = q_markers - torch.from_numpy(self.qs_real[i])

        loss = 0.5 * torch.dot(diff.flatten(), diff.flatten())
        loss.backward()
        grad = q.grad.detach().numpy()
        loss = loss.detach().numpy()

        return loss, grad, np.zeros_like(q.detach().numpy())
