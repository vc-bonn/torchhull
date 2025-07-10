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


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_blurred", [torch.float16, torch.float32, torch.float64])
def test_gaussian_blur_empty_masks(
    sparse: bool,
    dtype_masks: torch.dtype,
    dtype_blurred: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    _, _, masks_reference = generate_dataset(mesh_file=data_dir / file, device=DEVICE)

    masks_reference = masks_reference.to(dtype=dtype_masks, device=DEVICE)
    masks = torch.zeros_like(masks_reference)

    kernel_size = 5
    sigma = 1

    blurred_masks = torchhull.gaussian_blur(
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=sparse,
        dtype=dtype_blurred,
    )

    assert blurred_masks.dtype == dtype_blurred
    assert torch.allclose(blurred_masks, masks.to(dtype_blurred))


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_blurred", [torch.float16, torch.float32, torch.float64])
def test_gaussian_blur_full_masks(
    sparse: bool,
    dtype_masks: torch.dtype,
    dtype_blurred: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    _, _, masks_reference = generate_dataset(mesh_file=data_dir / file, device=DEVICE)

    masks_reference = masks_reference.to(dtype=dtype_masks, device=DEVICE)
    masks = torch.ones_like(masks_reference)

    kernel_size = 5
    sigma = 1

    blurred_masks = torchhull.gaussian_blur(
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=sparse,
        dtype=dtype_blurred,
    )

    assert blurred_masks.dtype == dtype_blurred
    assert torch.allclose(blurred_masks, masks.to(dtype_blurred))


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_blurred", [torch.float16, torch.float32, torch.float64])
def test_gaussian_blur_kernel_size_1(
    sparse: bool,
    dtype_masks: torch.dtype,
    dtype_blurred: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    _, _, masks = generate_dataset(mesh_file=data_dir / file, device=DEVICE)

    masks = masks.to(dtype=dtype_masks, device=DEVICE)

    kernel_size = 1
    sigma = 1

    blurred_masks = torchhull.gaussian_blur(
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=sparse,
        dtype=dtype_blurred,
    )

    assert blurred_masks.dtype == dtype_blurred
    assert torch.allclose(blurred_masks, masks.to(dtype_blurred))


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9, 11])
@pytest.mark.parametrize("sigma", [1.0, 100.0])
@pytest.mark.parametrize("masks_partial", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("dtype_masks", [torch.float16, torch.float32, torch.float64])
def test_gaussian_blur_default_type(
    kernel_size: int,
    sigma: float,
    masks_partial: bool,
    sparse: bool,
    dtype_masks: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    _, _, masks = generate_dataset(mesh_file=data_dir / file, device=DEVICE)

    masks = masks.to(dtype=dtype_masks, device=DEVICE)

    crop_x = 700
    crop_y = 400
    masks = masks[:, crop_y:-crop_y, crop_x:-crop_x, :].clone() if masks_partial else masks

    blurred_masks = torchhull.gaussian_blur(
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=sparse,
    )

    assert blurred_masks.dtype == masks.dtype
    assert torch.all(blurred_masks >= 0.0 - 1e-5)
    assert torch.all(blurred_masks <= 1.0 + 1e-5)


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9, 11])
@pytest.mark.parametrize("sigma", [1.0, 100.0])
@pytest.mark.parametrize("masks_partial", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_blurred", [torch.float16, torch.float32, torch.float64])
def test_gaussian_blur(
    kernel_size: int,
    sigma: float,
    masks_partial: bool,
    sparse: bool,
    dtype_masks: torch.dtype,
    dtype_blurred: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    _, _, masks = generate_dataset(mesh_file=data_dir / file, device=DEVICE)

    masks = masks.to(dtype=dtype_masks, device=DEVICE)

    crop_x = 700
    crop_y = 400
    masks = masks[:, crop_y:-crop_y, crop_x:-crop_x, :].clone() if masks_partial else masks

    blurred_masks = torchhull.gaussian_blur(
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=sparse,
        dtype=dtype_blurred,
    )

    assert blurred_masks.dtype == dtype_blurred
    assert torch.all(blurred_masks >= 0.0 - 1e-5)
    assert torch.all(blurred_masks <= 1.0 + 1e-5)


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9, 11])
@pytest.mark.parametrize("sigma", [1.0, 100.0])
@pytest.mark.parametrize("masks_partial", [True, False])
@pytest.mark.parametrize(
    "dtype_masks", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dtype_blurred", [torch.float16, torch.float32, torch.float64])
def test_gaussian_blur_consistent(
    kernel_size: int,
    sigma: float,
    masks_partial: bool,
    dtype_masks: torch.dtype,
    dtype_blurred: torch.dtype,
) -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    file = "Armadillo.ply"

    _, _, masks = generate_dataset(mesh_file=data_dir / file, device=DEVICE)

    masks = masks.to(dtype=dtype_masks, device=DEVICE)

    crop_x = 700
    crop_y = 400
    masks = masks[:, crop_y:-crop_y, crop_x:-crop_x, :].clone() if masks_partial else masks

    blurred_masks_dense = torchhull.gaussian_blur(
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=False,
        dtype=dtype_blurred,
    )

    blurred_masks_sparse = torchhull.gaussian_blur(
        images=masks,
        kernel_size=kernel_size,
        sigma=sigma,
        sparse=True,
        dtype=dtype_blurred,
    )

    if dtype_blurred == torch.float16:
        epsilon = 3e-3
    if dtype_blurred == torch.float32:
        epsilon = 1e-7
    if dtype_blurred == torch.float64:
        epsilon = 1e-7

    assert torch.allclose(blurred_masks_dense, blurred_masks_sparse, atol=epsilon)
