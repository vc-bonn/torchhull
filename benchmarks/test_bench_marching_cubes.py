from __future__ import annotations

import enum

import pytest
import torch

import torchhull

try:
    import pytorch3d.ops.marching_cubes

    pytorch3d_available = True
except ImportError:
    pytorch3d_available = False


DEVICE = torch.device("cuda")


class TensorType(enum.Enum):
    DENSE = enum.auto()
    SPARSE = enum.auto()

    def __str__(self) -> str:
        return f"{self.name}"


class ImplementationType(enum.Enum):
    TORCHHULL = enum.auto()
    PYTORCH3D = enum.auto()

    def __str__(self) -> str:
        return f"{self.name}"


def sdf_sphere(center: torch.Tensor, radius: float, samples: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(samples - torch.unsqueeze(center, 0), dim=1) - radius


def run_marching_cubes(sdf: torch.Tensor, implementation_type: ImplementationType) -> tuple[torch.Tensor, torch.Tensor]:
    if implementation_type == ImplementationType.TORCHHULL:
        v, f = torchhull.marching_cubes(sdf, isolevel=0, return_local_coords=False)
        verts, faces = [], []
        verts.append(v)
        faces.append(f)
    elif implementation_type == ImplementationType.PYTORCH3D and pytorch3d_available:
        verts, faces = pytorch3d.ops.marching_cubes.marching_cubes(sdf, isolevel=0, return_local_coords=False)
    else:
        verts, faces = [], []
        verts.append(torch.empty([0, 3], dtype=torch.float32, device=DEVICE))
        faces.append(torch.empty([0, 3], dtype=torch.int64, device=DEVICE))
    return verts[0], faces[0]


def list_sizes_mc() -> list[int]:
    return [100, 200, 300, 400, 500, 600]


@pytest.mark.parametrize("size", list_sizes_mc())
@pytest.mark.parametrize(
    ("implementation_type", "tensor_type"),
    [
        pytest.param(
            ImplementationType.PYTORCH3D,
            TensorType.DENSE,
            marks=pytest.mark.skipif(not pytorch3d_available, reason="PyTorch3D not available"),
        ),
        (ImplementationType.TORCHHULL, TensorType.DENSE),
        (ImplementationType.TORCHHULL, TensorType.SPARSE),
    ],
)
def test_marching_cubes(
    benchmark,  # noqa: ANN001
    size: int,
    implementation_type: ImplementationType,
    tensor_type: TensorType,
) -> None:
    torch.cuda.empty_cache()

    grid_1d = torch.arange(size, device=DEVICE)

    grid_verts = torch.stack(
        torch.meshgrid(3 * [grid_1d], indexing="ij"),
        dim=0,
    )  # 3 x size x size x size
    grid_verts = grid_verts.reshape([3, -1]).T  # size^3 x 3

    center = torch.full((3,), size / 2, dtype=torch.float32, device=DEVICE)
    radius = 0.8731945 * (size / 2)  # Use very uneven fraction to avoid zero values at the vertices
    sdf = sdf_sphere(center, radius, grid_verts)

    sdf = sdf.reshape([1, *(3 * [size])])

    if tensor_type == TensorType.SPARSE:
        truncation_distance = 2  # max(3, 0.01 * max(sdf.shape))
        sdf[sdf.abs() > truncation_distance] = 0

        sdf = sdf.to_sparse()

    # Warmup
    run_marching_cubes(sdf=sdf, implementation_type=implementation_type)

    benchmark(run_marching_cubes, sdf=sdf, implementation_type=implementation_type)
