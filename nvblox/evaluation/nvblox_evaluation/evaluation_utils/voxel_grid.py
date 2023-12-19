#
# Copyright 2022 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pathlib import Path
from typing import Any, Tuple

import numpy as np

import open3d as o3d
from plyfile import PlyData, PlyElement
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip


class VoxelGrid:
    unobserved_sentinal = -1000.0

    def __init__(self, voxels: np.ndarray, min_indices: np.ndarray, voxel_size: float):
        """An object representing an VoxelGrid.

        Args:
            voxels (_type_): A 3D numpy array containing VoxelGrid values
            min_indices (_type_): An 3x1 array representing the low-side corner of the grid in voxel indices.
            voxel_size (_type_): side length of a single voxel.
        """
        assert len(min_indices) == 3
        self.voxels = voxels
        self.min_indices = min_indices
        self.voxel_size = voxel_size

    def shape(self) -> Tuple:
        """Get the grid size.

        Returns:
            _type_: a 3x1 tuple representing the size of the grid
        """
        return self.voxels.shape

    def voxel_centers_along_axis(self, axis_idx: int) -> np.ndarray:
        """Generates a ndarray of the voxel centers along a certain dimension

        Args:
            axis_idx (int): The index of the axis (0, 1, 2)

        Returns:
            np.ndarray: returns a vector of the voxel centers.
        """
        return (
            np.arange(
                self.min_indices[axis_idx],
                self.min_indices[axis_idx] + self.shape()[axis_idx],
            )
            + 0.5
        ) * self.voxel_size

    def get_valid_voxel_centers(self) -> np.ndarray:
        """Get the centers of all valid voxels as an Nx3 numpy array

        Returns:
            np.ndarray: Nx3 numpy array containing the center locations of valid voxels.
        """
        x_range = self.voxel_centers_along_axis(0)
        y_range = self.voxel_centers_along_axis(1)
        z_range = self.voxel_centers_along_axis(2)
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        return np.vstack(
            (
                X[self.voxels != VoxelGrid.unobserved_sentinal],
                Y[self.voxels != VoxelGrid.unobserved_sentinal],
                Z[self.voxels != VoxelGrid.unobserved_sentinal],
            )
        ).transpose()

    def get_valid_voxel_values(self) -> np.ndarray:
        """Get the centers of all valid voxels as an Nx3 numpy array

        Returns:
            np.ndarray: Nx1 numpy array containing the center locations of valid voxels.
        """
        return self.voxels[self.voxels != VoxelGrid.unobserved_sentinal]

    @staticmethod
    def createFromPly(ply_path: Path) -> "VoxelGrid":
        """Creates an VoxelGrid object from a nvblox ESDF pointcloud ply.

        Args:
            ply_path (Path): Path to the nvblox file

        Returns:
            VoxelGrid: The object representing the VoxelGrid.
        """
        # Get the xyz position of voxels
        sdf_pointcloud_xyz = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
        # Get the ESDF values
        sdf_pointcloud_values = np.array(
            PlyData.read(str(ply_path)).elements[0]["intensity"]
        )
        return VoxelGrid.createFromSparseVoxels(
            sdf_pointcloud_xyz, sdf_pointcloud_values
        )

    def writeToPly(self, ply_path: Path) -> None:
        """Writes the ESDF as a pointcloud ply to file.

        Args:
            ply_path (Path): Path to the ply file to write
        """
        xyz = self.get_valid_voxel_centers()
        distances = self.get_valid_voxel_values()
        xyzi = np.hstack([xyz, distances.reshape((-1, 1))])
        xyzi_structured = np.array(
            [tuple(row) for row in xyzi],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")],
        )
        point_elements = PlyElement.describe(xyzi_structured, "vertex")
        PlyData([point_elements], text=True).write(str(ply_path))

    @staticmethod
    def createFromSparseVoxels(
        voxels_xyz: np.ndarray, voxel_values: np.ndarray
    ) -> "VoxelGrid":
        """Creates an VoxelGrid object from a list of valid voxels values and their locations.

        Args:
            voxels_xyz (np.ndarray): Nx3 array representing the position of the voxel centers
            voxel_values (np.ndarray): NX1 array representing the value of the voxels.

        Returns:
            VoxelGrid: The object representing the VoxelGrid.
        """
        # Detect the voxel size
        element_wise_diffs = np.diff(voxels_xyz, axis=0).flatten()
        voxel_size = np.min(element_wise_diffs[element_wise_diffs > 0])
        # Convert these two pointcloud parts to our VoxelGrid object
        voxel_indices = (np.around((voxels_xyz / voxel_size) - 0.5)).astype(
            dtype=np.intc
        )
        min_indices = np.min(voxel_indices, axis=0)
        max_indices = np.max(voxel_indices, axis=0)
        voxel_indices_zero_based = voxel_indices - min_indices
        sdf = VoxelGrid.unobserved_sentinal * np.ones(max_indices - min_indices + 1)
        sdf[
            voxel_indices_zero_based[:, 0],
            voxel_indices_zero_based[:, 1],
            voxel_indices_zero_based[:, 2],
        ] = voxel_values
        return VoxelGrid(sdf, min_indices, voxel_size)

    def get_slice_mesh_at_ratio(
        self,
        slice_level_ratio: float,
        axis: str = "x",
        cube_size: float = 0.75,
        bounds: Tuple = (None, None),
        highlight_positive: bool = False,
    ) -> o3d.geometry.TriangleMesh:
        """Gets a mesh representing a slice at ratio (0.0-1.0) along dimension axis.

        Args:
            slice_level_ratio (float): Where to slice.
            axis (str, optional): The axis to slice along. Defaults to 'x'.
            cube_size (float, optional): Size of the mesh cube that will be used to
                represent voxels. Given as a fraction of voxel size (i.e between 0.0 and 1.0).
            bounds (Tuple, optional): The min and max values to clip the SDF at. Defaults to (None, None).
            highlight_positive (bool, optional): Whether to highlight positive values in the colormap. Defaults to False.

        Returns:
            o3d.geometry.TriangleMesh: Mesh representing the slice.
        """
        assert slice_level_ratio >= 0.0 and slice_level_ratio <= 1.0
        assert axis == "x" or axis == "y" or axis == "z"
        if axis == "x":
            axis_idx = 0
        elif axis == "y":
            axis_idx = 1
        else:
            axis_idx = 2
        slice_level_idx = int(self.shape()[axis_idx] * slice_level_ratio)
        return self.get_slice_mesh_at_index(slice_level_idx, axis, cube_size, bounds, highlight_positive)

    def get_slice_mesh_at_index(
        self,
        slice_level_idx: int,
        axis: str = "x",
        cube_size: float = 0.75,
        bounds: Tuple = (None, None),
        highlight_positive: bool = False,
    ) -> o3d.geometry.TriangleMesh:
        """Gets a mesh representing a slice at slice_level_idx along dimension axis.

        Args:
            slice_level_idx (int): The index to slice at.
            axis (str, optional): The axis to slice along. Defaults to 'x'.
            cube_size (float, optional): Size of the mesh cube that will be used to
                represent voxels. Given as a fraction of voxel size (i.e between 0.0 and 1.0).
            bounds (Tuple, optional): The min and max values to clip the SDF at. Defaults to (None, None).
            highlight_positive (bool, optional): Whether to highlight positive values in the colormap. Defaults to False.

        Returns:
            o3d.geometry.TriangleMesh: Mesh representing the slice.
        """
        assert axis == "x" or axis == "y" or axis == "z"
        assert cube_size > 0.0 and cube_size <= 1.0

        # The VoxelGrid values to clip at
        percentile_lim_upper = 90
        percentile_lim_lower = 10
        if bounds[1] is None:
            sdf_clip_max = np.percentile(
                self.get_valid_voxel_values(), percentile_lim_upper
            )
        else:
            sdf_clip_max = bounds[1]
        if bounds[0] is None:
            sdf_clip_min = np.percentile(
                self.get_valid_voxel_values(), percentile_lim_lower
            )
        else:
            sdf_clip_min = bounds[0]

        # Size of the cubes
        voxel_cube_size = self.voxel_size * cube_size

        # Slice
        if axis == "x":
            slice_level_m = self.voxel_centers_along_axis(0)[slice_level_idx]
            slice = self.voxels[slice_level_idx, :, :]
            dim_1_vec = self.voxel_centers_along_axis(1)
            dim_2_vec = self.voxel_centers_along_axis(2)

            def to_3d(y, z):
                return np.array([slice_level_m, y, z])

        elif axis == "y":
            slice_level_m = self.voxel_centers_along_axis(1)[slice_level_idx]
            slice = self.voxels[:, slice_level_idx, :]
            dim_1_vec = self.voxel_centers_along_axis(0)
            dim_2_vec = self.voxel_centers_along_axis(2)

            def to_3d(x, z):
                return np.array([x, slice_level_m, z])

        else:
            slice_level_m = self.voxel_centers_along_axis(2)[slice_level_idx]
            slice = self.voxels[:, :, slice_level_idx]
            dim_1_vec = self.voxel_centers_along_axis(0)
            dim_2_vec = self.voxel_centers_along_axis(1)

            def to_3d(x, y):
                return np.array([x, y, slice_level_m])

        # Normalizing/Clipping the distances
        slice_normalized = (slice - sdf_clip_min) / (sdf_clip_max - sdf_clip_min)
        slice_normalized = slice_normalized.clip(min=0.0, max=1.0)

        # Create the slice mesh
        slice_mesh = o3d.geometry.TriangleMesh()
        for idx_1, pos_1 in np.ndenumerate(dim_1_vec):
            for idx_2, pos_2 in np.ndenumerate(dim_2_vec):
                if slice[idx_1, idx_2] == self.unobserved_sentinal:
                    continue
                box = o3d.geometry.TriangleMesh.create_box(
                    width=voxel_cube_size, height=voxel_cube_size, depth=voxel_cube_size
                )
                color = plt.cm.viridis(slice_normalized[idx_1, idx_2])
                if highlight_positive:
                    # Un-normalized value
                    if slice[idx_1, idx_2] > 1.0*self.voxel_size:  # TODO(rgg): remove one-voxel margin?
                        red = np.array([[1.0, 0.0, 0.0]])
                        color = red
                        print(f"Highlighting positive value: {slice[idx_1, idx_2]} with color {color}")
                box.compute_vertex_normals()
                box.paint_uniform_color(color[0, 0:3])
                box.translate(to_3d(pos_1, pos_2))
                slice_mesh += box

        return slice_mesh

    def num_valid_voxels(self) -> int:
        """Returns the number of observed voxels

        Returns:
            int: number of observed voxels
        """
        return np.sum(self.voxels != self.unobserved_sentinal)

    def get_voxel_by_global_idx(self, x_idx: int, y_idx: int, z_idx: int) -> Any:
        """Gets the voxel value at a global index.

        Args:
            x_idx (int): The x index of the voxel
            y_idx (int): The y index of the voxel
            z_idx (int): The z index of the voxel

        Returns:
            Any: The value of the voxel
        """
        return self.voxels[
            x_idx - self.min_indices[0],
            y_idx - self.min_indices[1],
            z_idx - self.min_indices[2],
        ]

    def set_voxel_by_global_idx(
        self, x_idx: int, y_idx: int, z_idx: int, value: Any
    ) -> None:
        """Sets the voxel value at a global index.

        Args:
            x_idx (int): The x index of the voxel
            y_idx (int): The y index of the voxel
            z_idx (int): The z index of the voxel
            value (Any): The value to set the voxel to
        """
        self.voxels[
            x_idx - self.min_indices[0],
            y_idx - self.min_indices[1],
            z_idx - self.min_indices[2],
        ] = value

    def __sub__(self, other: "VoxelGrid") -> "VoxelGrid":
        """
        Subtracts the values of two VoxelGrids. The resulting VoxelGrid will
        have the same voxel size and min_indices as self.
        When a value does not exist in the other VoxelGrid, the value in the
        resulting VoxelGrid will be the same as self.
        """
        self_min_indices = self.min_indices
        # Create a new VoxelGrid with the same size as self
        new_voxel_grid = VoxelGrid(
            np.zeros(self.shape()), self_min_indices, self.voxel_size
        )
        # Copy self values into new_voxel_grid
        new_voxel_grid.voxels = np.copy(self.voxels)
        # Find the range in global indices valid in *both* VoxelGrids
        other_min_indices = np.array(other.min_indices)
        other_max_indices = np.array(other_min_indices + other.shape() - 1)
        self_min_indices = np.array(self_min_indices)
        self_max_indices = np.array(self_min_indices + self.shape() - 1)
        min_indices = np.maximum(self_min_indices, other_min_indices)
        max_indices = np.minimum(self_max_indices, other_max_indices)
        # Subtract other values from new_voxel_grid
        # Iterate over valid indices in both VoxelGrids and subtract
        for x_idx in range(min_indices[0], max_indices[0] + 1):
            for y_idx in range(min_indices[1], max_indices[1] + 1):
                for z_idx in range(min_indices[2], max_indices[2] + 1):
                    diff = self.get_voxel_by_global_idx(
                        x_idx, y_idx, z_idx
                    ) - other.get_voxel_by_global_idx(x_idx, y_idx, z_idx)
                    # Check if voxel in either is unobserved, and if so, set to unobserved
                    # in output
                    if (
                        other.get_voxel_by_global_idx(x_idx, y_idx, z_idx)
                        == VoxelGrid.unobserved_sentinal
                        or self.get_voxel_by_global_idx(
                            x_idx, y_idx, z_idx
                        )  # Technically redundant, but more readable
                        == VoxelGrid.unobserved_sentinal
                    ):
                        diff = VoxelGrid.unobserved_sentinal
                    new_voxel_grid.set_voxel_by_global_idx(x_idx, y_idx, z_idx, diff)
        return new_voxel_grid

    def __repr__(self) -> str:
        return (
            "VoxelGrid of voxels with shape: "
            + str(self.voxels.shape)
            + " and "
            + str(self.num_valid_voxels())
            + " valid voxels."
        )

    def get_z_slice_animation_clip(
        self,
        mesh: o3d.geometry.TriangleMesh = None,
        viewpoint: o3d.camera.PinholeCameraParameters = None,
    ) -> ImageSequenceClip:
        """Creates a image sequence containing horizontal slices moving through the z dimension of the VoxelGrid

        Args:
            mesh (o3d.geometry.TriangleMesh, optional): Additional mesh to add to the animation. Defaults to None.
            viewpoint (o3d.camera.PinholeCameraParameters, optional): Viewpoint to record the slice from. Defaults to None.

        Returns:
            ImageSequenceClip: sequence of images of the slicing results
        """
        images = []
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        if viewpoint is not None:
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(viewpoint)
        if mesh is not None:
            vis.add_geometry(mesh)
        slice_mesh = o3d.geometry.TriangleMesh()
        vis.add_geometry(slice_mesh)
        images = []
        first = True
        for z_idx in range(self.shape()[2]):
            vis.remove_geometry(slice_mesh, reset_bounding_box=False)
            slice_mesh = self.get_slice_mesh_at_index(z_idx, axis="z", cube_size=1.0)
            if first and mesh is None:
                vis.add_geometry(slice_mesh, reset_bounding_box=True)
                first = False
            else:
                vis.add_geometry(slice_mesh, reset_bounding_box=False)
            if viewpoint is not None:
                ctr.convert_from_pinhole_camera_parameters(viewpoint)
            vis.poll_events()
            vis.update_renderer()
            image_float = np.asarray(vis.capture_screen_float_buffer())
            image_uint8 = (image_float * 255).astype(np.uint8)
            images.append(image_uint8)
        vis.destroy_window()
        return ImageSequenceClip(images, fps=10)
