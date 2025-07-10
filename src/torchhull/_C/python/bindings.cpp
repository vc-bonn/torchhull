#include <torch/python.h>

#include <torchhull/gaussian_blur.h>
#include <torchhull/io.h>
#include <torchhull/marching_cubes.h>
#include <torchhull/visual_hull.h>

using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("visual_hull",
          &torchhull::visual_hull,
          "masks"_a,
          "transforms"_a,
          "level"_a,
          "cube_corner_bfl"_a,
          "cube_length"_a,
          "masks_partial"_a,
          "unique_verts"_a = true,
          R"(
        Compute the visual hull of the given masks in terms of a mesh.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        masks
            Single-channel mask images with binary values {0, 1}. B x H x W x 1.
        transforms
            The combined transformations from world coordinates to OpenGL clip space (right before perspective division). B x 4 x 4.
        level
            The hierarchy level to compute the visual hull at.
        cube_corner_bfl
            The 3D world position of the bottom-front-left corner indexed as (0, 0, 0) in the grid.
        cube_length
            The length of the cube in world space.
        masks_partial
            Whether some masks may only contain the object partially.
        unique_verts
            Whether a compact mesh without duplicate vertices (\|F\| approx. 2 * \|V\|) if true, or a triangle soup
            (\|F\| = (1/3) * \|V\|) if false should be returned.

        Returns
        -------
        verts: The vertex tensor of the extracted visual hull. \|V\| x 3. float32.
        faces: The face tensor of the extracted visual hull. \|F\| x 3. int64.
    )");

    m.def("visual_hull_with_candidate_voxels",
          &torchhull::visual_hull_with_candidate_voxels,
          "masks"_a,
          "transforms"_a,
          "level"_a,
          "cube_corner_bfl"_a,
          "cube_length"_a,
          "masks_partial"_a,
          "unique_verts"_a = true,
          R"(
        Compute the visual hull of the given masks in terms of a mesh.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        masks
            Single-channel mask images with binary values {0, 1}. B x H x W x 1.
        transforms
            The combined transformations from world coordinates to OpenGL clip space (right before perspective division). B x 4 x 4.
        level
            The hierarchy level to compute the visual hull at.
        cube_corner_bfl
            The 3D world position of the bottom-front-left corner indexed as (0, 0, 0) in the grid.
        cube_length
            The length of the cube in world space.
        masks_partial
            Whether some masks may only contain the object partially.
        unique_verts
            Whether a compact mesh without duplicate vertices (\|F\| approx. 2 * \|V\|) if true, or a triangle soup
            (\|F\| = (1/3) * \|V\|) if false should be returned.

        Returns
        -------
        verts: The vertex tensor of the extracted visual hull. \|V\| x 3. float32.
        faces: The face tensor of the extracted visual hull. \|F\| x 3. int64.
        candidate_voxels: List of ravelled candidate voxels at each hierarchy level. [\|I_0\|, ... \|I_level\|]. int64.
    )");

    m.def("candidate_voxels_to_wireframes",
          &torchhull::candidate_voxels_to_wireframes,
          "candidate_voxels"_a,
          "cube_corner_bfl"_a,
          "cube_length"_a,
          R"(
        Converts the ravelled candidate voxels into wireframes.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        candidate_voxels
            List of ravelled candidate voxels at each hierarchy level. [\|I_0\|, ... \|I_level\|]. int64.
        cube_corner_bfl
            The 3D world position of the bottom-front-left corner indexed as (0, 0, 0) in the grid.
        cube_length
            The length of the cube in world space.

        Returns
        -------
        List of vertex and edge tensors. [(\|V_0\| x 3. float32, \|E_0\| x 3. int64), ... [(\|V_level\| x 3. float32, \|E_level\| x 3. int64)].
    )");

    m.def("sparse_visual_hull_field",
          &torchhull::sparse_visual_hull_field,
          "masks"_a,
          "transforms"_a,
          "level"_a,
          "cube_corner_bfl"_a,
          "cube_length"_a,
          "masks_partial"_a,
          R"(
        Compute a sparse scalar field of the sum of projected foreground pixels per detected candidate voxel. In this
        field, the visual hull is located at isolevel \|M\| - 0.5.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        masks
            Single-channel mask images with binary values {0, 1}. B x H x W x 1.
        transforms
            The combined transformations from world coordinates to OpenGL clip space (right before perspective division). B x 4 x 4.
        level
            The hierarchy level to compute the counts at.
        cube_corner_bfl
            The 3D world position of the bottom-front-left corner indexed as (0, 0, 0) in the grid.
        cube_lenth
            The length of the cube in world space.
        masks_partial
            Whether some masks may only contain the object partially.

        Returns
        -------
        A sparse scalar field containing the visual hull. 1 x resolution^3 with resolution = 2^level + 1
    )");

    m.def("marching_cubes",
          &torchhull::marching_cubes,
          "volume"_a,
          "isolevel"_a,
          "return_local_coords"_a = true,
          "unique_verts"_a = true,
          R"(
        Runs Marching Cubes on the given volume.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        volume
            The volume tensor. 1 x D x H x W.
        isolevel
            The value to determine the inside and outside of the scene parts within the volume.
        return_local_coords
            Whether local coordinates [-1, 1] x [-1, 1] x [-1, 1] if true, or global coordinates
            [0, W-1] x [0, H-1] x [0, D-1] if false should be used.
        unique_verts
            Whether a compact mesh without duplicate vertices (\|F\| approx. 2 * \|V\|) if true, or a triangle soup
            (\|F\| = (1/3) * \|V\|) if false should be returned.

        Returns
        -------
        verts: The vertex tensor of the extracted scene. \|V\| x 3. float32.
        faces: The face tensor of the extracted scene. \|F\| x 3. int64.
    )");

    m.def("gaussian_blur",
          &torchhull::gaussian_blur,
          "images"_a,
          "kernel_size"_a,
          "sigma"_a,
          "sparse"_a = true,
          "dtype"_a = py::none(),
          R"(
        Blurs the given images with a Gaussian kernel.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        images
            The image tensor. B x H x W x C.
        kernel_size
            The size of the kernel. Must be an odd number.
        sigma
            The standard deviation of the kernel.
        sparse
            Whether to use the sparse implementation optimized for masks or the standard dense version.
        dtype
            Optional dtype of the returned tensor. If set to `None`, the dtype of the input images is selected.
            Must be a floating-point type.

        Returns
        -------
        Blurred images. B x H x W x C.
    )");

    m.def("store_curve_network",
          &torchhull::store_curve_network,
          "filename"_a,
          "curve_network"_a,
          "verbose"_a = false,
          R"(
        Stores the given curve network into a file.

        Parameters
        ----------
        filename
            The path to the file.
        curve_network
            A curve network/wireframe with vertex and edge tensors. \|V\| x 3. float32, \|E\| x 3. int64.
        verbose
            Whether to print intermediate progress.
    )");
}
