from __future__ import annotations

import enum

import pytest
import torch

import torchhull

DEVICE = torch.device("cuda")


def sdf_sphere(center: torch.Tensor, radius: float, samples: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(samples - torch.unsqueeze(center, 0), dim=1) - radius


def is_closed_manifold(faces: torch.Tensor) -> bool:
    assert faces.shape[1] == 3  # noqa: PLR2004

    edges = torch.cat([faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]], dim=0)

    edges_ordered, _ = torch.sort(edges, dim=1)
    _, edges_counts = torch.unique(edges_ordered, dim=0, return_counts=True)

    return torch.equal(edges_counts, torch.full_like(edges_counts, 2))


def list_sizes() -> list[int]:
    return [100, 200, 300, 400]


class TensorType(enum.Enum):
    DENSE = enum.auto()
    SPARSE = enum.auto()

    def __str__(self) -> str:
        return f"{self.name}"


@pytest.mark.parametrize("size", list_sizes())
@pytest.mark.parametrize("tensor_type", [TensorType.DENSE, TensorType.SPARSE])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_marching_cubes_empty_input(
    size: int,
    tensor_type: TensorType,
    dtype: torch.dtype,
) -> None:
    if tensor_type == TensorType.DENSE:
        sdf = torch.full([1, *(3 * [size])], -1.0, dtype=dtype, device=DEVICE)
    elif tensor_type == TensorType.SPARSE:
        sdf = torch.sparse_coo_tensor(
            indices=torch.empty([4, 0], dtype=torch.int64, device=DEVICE),
            values=torch.empty([0], dtype=dtype, device=DEVICE),
            size=[1, *(3 * [size])],
        )

    verts, faces = torchhull.marching_cubes(sdf, isolevel=0, return_local_coords=False)

    assert verts.size(0) == 0
    assert faces.size(0) == 0


@pytest.mark.parametrize("size", list_sizes())
@pytest.mark.parametrize("tensor_type", [TensorType.DENSE, TensorType.SPARSE])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_marching_cubes(
    size: int,
    tensor_type: TensorType,
    dtype: torch.dtype,
) -> None:
    grid_1d = torch.arange(size, device=DEVICE)

    grid_verts = torch.stack(
        torch.meshgrid(3 * [grid_1d], indexing="ij"),
        dim=0,
    )  # 3 x size x size x size
    grid_verts = grid_verts.reshape([3, -1]).T.to(dtype=dtype)  # size^3 x 3

    center = torch.full((3,), size / 2, dtype=dtype, device=DEVICE)
    radius = 0.8731945 * (size / 2)  # Use very uneven fraction to avoid zero values at the vertices
    sdf = sdf_sphere(center, radius, grid_verts)

    sdf = sdf.reshape([1, *(3 * [size])])

    if tensor_type == TensorType.SPARSE:
        truncation_distance = 2  # max(3, 0.01 * max(sdf.shape))
        sdf[sdf.abs() > truncation_distance] = 0

        sdf = sdf.to_sparse()

    verts, faces = torchhull.marching_cubes(sdf, isolevel=0, return_local_coords=False)

    sdf_verts = sdf_sphere(center, radius, verts)
    assert torch.allclose(sdf_verts, torch.zeros_like(sdf_verts), atol=1e-2)
    assert is_closed_manifold(faces)
