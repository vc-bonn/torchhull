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


@pytest.mark.parametrize("level", [7, 8, 9, 10, 11])
@pytest.mark.parametrize("number_cameras", [10, 20, 30, 40, 50])
def test_visual_hull(benchmark, level: int, number_cameras: int) -> None:  # noqa: ANN001
    torch.cuda.empty_cache()

    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    projection_matrices, view_matrices, masks, _, _, _ = generate_dataset(
        mesh_file=data_dir / file,
        number_cameras=number_cameras,
        device=DEVICE,
    )
    transforms = projection_matrices @ view_matrices

    masks = masks.to(dtype=torch.float32, device=DEVICE)
    transforms = transforms.to(dtype=torch.float32, device=DEVICE)

    scale = 1.1

    # Warmup
    torchhull.visual_hull(
        masks=masks,
        transforms=transforms,
        level=level,
        cube_corner_bfl=(-scale, -scale, -scale),
        cube_length=2.0 * scale,
        masks_partial=False,
        transforms_convention="opengl",
        unique_verts=True,
    )

    benchmark(
        torchhull.visual_hull,
        masks=masks,
        transforms=transforms,
        level=level,
        cube_corner_bfl=(-scale, -scale, -scale),
        cube_length=2.0 * scale,
        masks_partial=False,
        transforms_convention="opengl",
        unique_verts=True,
    )
