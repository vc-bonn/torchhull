from __future__ import annotations

import enum
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


class ImplementationType(enum.Enum):
    DENSE = enum.auto()
    SPARSE = enum.auto()

    def __str__(self) -> str:
        return f"{self.name}"


@pytest.mark.parametrize("number_cameras", [10, 30, 50])
@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9, 11])
@pytest.mark.parametrize("implementation_type", [ImplementationType.DENSE, ImplementationType.SPARSE])
def test_gaussian_blur(
    benchmark,  # noqa: ANN001
    number_cameras: int,
    kernel_size: int,
    implementation_type: ImplementationType,
) -> None:
    torch.cuda.empty_cache()

    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    _, _, masks, _, _, _ = generate_dataset(
        mesh_file=data_dir / file,
        number_cameras=number_cameras,
        device=DEVICE,
    )

    masks = masks.to(dtype=torch.float32, device=DEVICE)
    sigma = 1.0
    sparse = implementation_type == ImplementationType.SPARSE

    # Warmup
    torchhull.gaussian_blur(
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=sparse,
    )

    benchmark(
        torchhull.gaussian_blur,
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=sparse,
    )
