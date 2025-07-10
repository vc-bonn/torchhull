<h1 align="center">torchhull: A fast Visual Hull implementation</h1>

<!-- start readme -->

<p align="center">
<a href="https://pypi.python.org/pypi/torchhull">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/torchhull">
</a>
<a href="https://pypi.python.org/pypi/torchhull">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/torchhull">
</a>
<a href="https://github.com/vc-bonn/torchhull/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/badge/License-MIT-green.svg"/>
</a>
<a href="https://github.com/vc-bonn/torchhull/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/badge/License-BSD--3--Clause-green.svg"/>
</a>
<a href="https://github.com/vc-bonn/torchhull/actions/workflows/build.yml">
    <img alt="Build" src="https://github.com/vc-bonn/torchhull/actions/workflows/build.yml/badge.svg">
</a>
<a href="https://github.com/vc-bonn/torchhull/actions/workflows/lint.yml">
    <img alt="Lint" src="https://github.com/vc-bonn/torchhull/actions/workflows/lint.yml/badge.svg">
</a>
<a href="https://vc-bonn.github.io/torchhull">
    <img alt="Documentation" src="https://img.shields.io/badge/docs-Latest-green.svg"/>
</a>
</p>


torchhull is an extremely fast Torch C++/CUDA implementation for computing visual hulls from mask images and comes with Python bindings through [charonload](https://github.com/vc-bonn/charonload):

- ⚡ Up to real-time capable speed depending on chosen resolution
- 🗜️ Memory-efficient computation by constructing sparse voxel octrees
- 🌊 Watertight mesh generation via Marching Cubes
- 🎈 Smooth surfaces with sparse Gaussian blur preprocessing tailored for mask images
- 🛠️ Support for partially visible objects, i.e. clipped mask images, and fully observed objects


In particular, torchhull is a GPU implementation of the following paper:

```bib
@article{scharr2017fast,
  title={{Fast High Resolution Volume Carving for 3D Plant Shoot Reconstruction}},
  author={Scharr, Hanno and Briese, Christoph and Embgenbroich, Patrick and Fischbach, Andreas and Fiorani, Fabio and M{\"u}ller-Linow, Mark},
  journal={Frontiers in Plant Science},
  volume={8},
  pages={303692},
  year={2017},
  publisher={Frontiers}
}
```


## Installation

torchhull requires the following prerequites (for JIT compilation):

- Python >= 3.9
- CUDA >= 12.1
- C++17 compiler

The package itself can be installed from PyPI:

```sh
pip install torchhull
```


## Quick Start

torchhull gets as input mask images with camera information:

- `masks`: Single-channel images `M` with binary values {0, 1}.
- `transforms`: Fused extrinsic and intrinsic matrix `K * T`, i.e. transformation from world coordinates to OpenGL clip space (right before perspective division).

The visual hull is then evaluated inside a cube with bottom-front-left corner `cube_corner_bfl` and extent `cube_length` at extracted at octree level `level`. The remaining flags control how the output mesh `(verts, faces)` should look like.

```python
import torchhull

# Optional
masks = torchhull.gaussian_blur(masks, # [B, H, W, 1]
                                kernel_size,
                                sigma,
                                sparse=True,
                               )

verts, faces = torchhull.visual_hull(masks,  # [B, H, W, 1]
                                     transforms,  # [B, 4, 4]
                                     level,
                                     cube_corner_bfl,
                                     cube_length,
                                     masks_partial=False,
                                     unique_verts=True,
                                    )
```


## License

This software is provided under MIT license, with parts under BSD 3-Clause license. See [`LICENSE`](https://github.com/vc-bonn/torchhull/blob/main/LICENSE) for more information.


## Contact

Patrick Stotko - <a href="mailto:stotko@cs.uni-bonn.de">stotko@cs.uni-bonn.de</a><br/>

<!-- end readme -->
