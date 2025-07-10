from __future__ import annotations

import functools
import pathlib
import sys

import pytest
import torch

import torchhull

DATA_DIR = pathlib.Path(__file__).parents[1] / "data"
sys.path.append(str(DATA_DIR))
from generate_dataset import generate_dataset  # noqa: E402

generate_dataset = functools.cache(generate_dataset)


DEVICE = torch.device("cuda")


def is_closed_manifold(faces: torch.Tensor) -> bool:
    assert faces.shape[1] == 3  # noqa: PLR2004
    edges = torch.cat([faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]], dim=0)

    edges_ordered, _ = torch.sort(edges, dim=1)
    _, edges_counts = torch.unique(edges_ordered, dim=0, return_counts=True)

    return torch.equal(edges_counts, torch.full_like(edges_counts, 2))


def list_levels() -> list[int]:
    return list(range(1, 8 + 1))


@pytest.mark.parametrize("level", list_levels())
@pytest.mark.parametrize("masks_partial", [True, False])
@pytest.mark.parametrize("unique_verts", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_transforms", [torch.float32, torch.float64])
def test_visual_hull_empty_masks(
    level: int,
    masks_partial: bool,
    unique_verts: bool,
    dtype_masks: torch.dtype,
    dtype_transforms: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    projection_matrices, view_matrices, masks_reference = generate_dataset(mesh_file=data_dir / file, device=DEVICE)
    transforms = projection_matrices @ view_matrices

    masks_reference = masks_reference.to(dtype=dtype_masks, device=DEVICE)
    masks = torch.zeros_like(masks_reference)

    transforms = transforms.to(dtype=dtype_transforms, device=DEVICE)

    scale = 1.1

    vertices, faces = torchhull.visual_hull(
        masks=masks,
        transforms=transforms,
        level=level,
        cube_corner_bfl=(-scale, -scale, -scale),
        cube_length=2.0 * scale,
        masks_partial=masks_partial,
        unique_verts=unique_verts,
    )

    assert vertices.size(0) == 0
    assert faces.size(0) == 0


@pytest.mark.parametrize("level", list_levels())
@pytest.mark.parametrize("masks_partial", [True, False])
@pytest.mark.parametrize("unique_verts", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_transforms", [torch.float32, torch.float64])
def test_visual_hull_full_masks(
    level: int,
    masks_partial: bool,
    unique_verts: bool,
    dtype_masks: torch.dtype,
    dtype_transforms: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    projection_matrices, view_matrices, masks_reference = generate_dataset(mesh_file=data_dir / file, device=DEVICE)
    transforms = projection_matrices @ view_matrices

    masks_reference = masks_reference.to(dtype=dtype_masks, device=DEVICE)
    masks = torch.ones_like(masks_reference)

    transforms = transforms.to(dtype=dtype_transforms, device=DEVICE)

    scale = 1.1

    vertices, faces = torchhull.visual_hull(
        masks=masks,
        transforms=transforms,
        level=level,
        cube_corner_bfl=(-scale, -scale, -scale),
        cube_length=2.0 * scale,
        masks_partial=masks_partial,
        unique_verts=unique_verts,
    )

    assert vertices.size(0) == 0
    assert faces.size(0) == 0


@pytest.mark.parametrize("level", list_levels())
@pytest.mark.parametrize("masks_partial", [True, False])
@pytest.mark.parametrize("unique_verts", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_transforms", [torch.float32, torch.float64])
def test_visual_hull(
    level: int,
    masks_partial: bool,
    unique_verts: bool,
    dtype_masks: torch.dtype,
    dtype_transforms: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    projection_matrices, view_matrices, masks = generate_dataset(mesh_file=data_dir / file, device=DEVICE)
    transforms = projection_matrices @ view_matrices

    masks = masks.to(dtype=dtype_masks, device=DEVICE)
    transforms = transforms.to(dtype=dtype_transforms, device=DEVICE)

    scale = 1.1

    vertices, faces = torchhull.visual_hull(
        masks=masks,
        transforms=transforms,
        level=level,
        cube_corner_bfl=(-scale, -scale, -scale),
        cube_length=2.0 * scale,
        masks_partial=masks_partial,
        unique_verts=unique_verts,
    )

    assert torch.all(vertices >= torch.tensor([[-scale, -scale, -scale]], dtype=vertices.dtype, device=vertices.device))
    assert torch.all(vertices <= torch.tensor([[scale, scale, scale]], dtype=vertices.dtype, device=vertices.device))
    if unique_verts:
        assert is_closed_manifold(faces)
    else:
        assert vertices.size(0) == 3 * faces.size(0)


@pytest.mark.parametrize("level", list_levels())
@pytest.mark.parametrize("masks_partial", [True, False])
@pytest.mark.parametrize("unique_verts", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_transforms", [torch.float32, torch.float64])
def test_visual_hull_with_blur(
    level: int,
    masks_partial: bool,
    unique_verts: bool,
    dtype_masks: torch.dtype,
    dtype_transforms: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    projection_matrices, view_matrices, masks = generate_dataset(mesh_file=data_dir / file, device=DEVICE)
    transforms = projection_matrices @ view_matrices

    masks = masks.to(dtype=dtype_masks, device=DEVICE)
    transforms = transforms.to(dtype=dtype_transforms, device=DEVICE)

    kernel_size = 9
    sigma = 2
    masks = torchhull.gaussian_blur(
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=True,
        dtype=dtype_transforms,
    )

    scale = 1.1

    vertices, faces = torchhull.visual_hull(
        masks=masks,
        transforms=transforms,
        level=level,
        cube_corner_bfl=(-scale, -scale, -scale),
        cube_length=2.0 * scale,
        masks_partial=masks_partial,
        unique_verts=unique_verts,
    )

    assert torch.all(vertices >= torch.tensor([[-scale, -scale, -scale]], dtype=vertices.dtype, device=vertices.device))
    assert torch.all(vertices <= torch.tensor([[scale, scale, scale]], dtype=vertices.dtype, device=vertices.device))
    if unique_verts:
        assert is_closed_manifold(faces)
    else:
        assert vertices.size(0) == 3 * faces.size(0)


@pytest.mark.parametrize("level", list_levels())
@pytest.mark.parametrize("masks_partial", [True, False])
@pytest.mark.parametrize("unique_verts", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_transforms", [torch.float32, torch.float64])
def test_visual_hull_with_candidate_voxels(
    level: int,
    masks_partial: bool,
    unique_verts: bool,
    dtype_masks: torch.dtype,
    dtype_transforms: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    projection_matrices, view_matrices, masks = generate_dataset(mesh_file=data_dir / file, device=DEVICE)
    transforms = projection_matrices @ view_matrices

    masks = masks.to(dtype=dtype_masks, device=DEVICE)
    transforms = transforms.to(dtype=dtype_transforms, device=DEVICE)

    scale = 1.1

    vertices, faces, candidates = torchhull.visual_hull_with_candidate_voxels(
        masks=masks,
        transforms=transforms,
        level=level,
        cube_corner_bfl=(-scale, -scale, -scale),
        cube_length=2.0 * scale,
        masks_partial=masks_partial,
        unique_verts=unique_verts,
    )

    wireframes = torchhull.candidate_voxels_to_wireframes(
        candidates,
        cube_corner_bfl=(-scale, -scale, -scale),
        cube_length=2.0 * scale,
    )

    assert torch.all(vertices >= torch.tensor([[-scale, -scale, -scale]], dtype=vertices.dtype, device=vertices.device))
    assert torch.all(vertices <= torch.tensor([[scale, scale, scale]], dtype=vertices.dtype, device=vertices.device))
    if unique_verts:
        assert is_closed_manifold(faces)
    else:
        assert vertices.size(0) == 3 * faces.size(0)

    for i, (w_vertices, _) in enumerate(wireframes):
        assert torch.all(
            w_vertices >= torch.tensor([-scale, -scale, -scale], dtype=w_vertices.dtype, device=w_vertices.device)
        )
        assert torch.all(
            w_vertices <= torch.tensor([scale, scale, scale], dtype=w_vertices.dtype, device=w_vertices.device)
        )

        if i > 0:
            w_previous_vertices, _ = wireframes[i - 1]
            bb_min, _ = torch.min(w_previous_vertices, dim=0)
            bb_max, _ = torch.max(w_previous_vertices, dim=0)
            assert torch.all(w_vertices >= bb_min)
            assert torch.all(w_vertices <= bb_max)
