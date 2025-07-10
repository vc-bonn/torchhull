"""
torchhull is an extremely fast Torch C++/CUDA implementation for computing visual hulls from mask images and comes
with Python bindings through `charonload <https://github.com/vc-bonn/charonload>`_:

.. code-block:: python

    import torchhull

    verts, faces = torchhull.visual_hull(masks,  # [B, H, W, 1]
                                         transforms,  # [B, 4, 4]
                                         level,
                                         cube_corner_bfl,
                                         cube_length,
                                         masks_partial=False,
                                         unique_verts=True,
                                        )
"""

import sys

required_version = (3, 9)

if sys.version_info[:2] < required_version:  # pragma: no cover
    msg = "%s requires Python %d.%d+" % (__package__, *required_version)  # noqa: UP031
    raise RuntimeError(msg)

del required_version
del sys


import email.utils
import importlib.metadata
import pathlib

import charonload

PROJECT_ROOT_DIRECTORY = pathlib.Path(__file__).parents[2]

VSCODE_STUBS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "typings"


charonload.module_config["_c_torchhull"] = charonload.Config(
    pathlib.Path(__file__).parent / "_C",
    stubs_directory=VSCODE_STUBS_DIRECTORY,
    verbose=False,
)


from _c_torchhull import (
    candidate_voxels_to_wireframes,
    gaussian_blur,
    marching_cubes,
    sparse_visual_hull_field,
    store_curve_network,
    visual_hull,
    visual_hull_with_candidate_voxels,
)

__author__ = ", ".join(
    [
        email.utils.parseaddr(author.strip())[0]
        for author in importlib.metadata.metadata(__package__)["Author-email"].split(",")
    ]
)
__version__ = importlib.metadata.version(__package__)
__copyright__ = f"2024, {__author__}"

__all__ = [
    "candidate_voxels_to_wireframes",
    "gaussian_blur",
    "marching_cubes",
    "sparse_visual_hull_field",
    "store_curve_network",
    "visual_hull",
    "visual_hull_with_candidate_voxels",
]
