# Third-party software

This repository contains **only source code and scripts**. No third-party binaries are redistributed:
`setup.bat` downloads every dependency from its original source onto your own computer and builds it there.

## Code this repository is derived from

| Component | Source (pinned) | License | How it is used |
|---|---|---|---|
| SIBR core | https://gitlab.inria.fr/sibr/sibr_core, branch `gaussian_code_release_union`, commit `4c4a953` | Apache-2.0 (`LICENSE_SIBR_APACHE2.md`) | cloned by setup, patched with `sibr_patch/` |
| SIBR `gaussianviewer` project (Inria GRAPHDECO) | same repository, `src/projects/gaussianviewer` | Gaussian-Splatting License, non-commercial (`LICENSE_GAUSSIAN_SPLATTING.md`) | modified and extended by `sibr_patch/` |
| diff-gaussian-rasterization (Inria GRAPHDECO) | https://github.com/graphdeco-inria/diff-gaussian-rasterization, commit `255c532` | Gaussian-Splatting License, non-commercial | fetched by SIBR CMake, unchanged |
| 4DGaussians (hustvl) | https://github.com/hustvl/4DGaussians, commit `843d5ac` | Apache-2.0 | cloned by setup; `scene/deformation.py`, `scene/hexplane.py`, `scene/grid.py`, `utils/graphics_utils.py` are imported by `tools/export_g4d.py` |

## Libraries downloaded while building (by SIBR's CMake, from Inria's dependency server or their own repositories)

| Library | Version | License |
|---|---|---|
| Boost | 1.71 | BSL-1.0 |
| OpenCV | 4.8 | Apache-2.0 |
| Assimp | 4.1.0 | BSD-3-Clause |
| Eigen | 3 | MPL-2.0 |
| Embree | 3.6.1 | Apache-2.0 |
| FFmpeg | 4.0.2 (Windows build with `--enable-gpl`, x264) | GPL-3.0-or-later |
| GLEW | 2.0.0 | BSD-3-Clause / MIT |
| GLFW | 3.2.1 | zlib |
| Dear ImGui | 1.60 WIP (SIBR fork) | MIT |
| nativefiledialog | (SIBR bundle) | zlib |
| nanoflann, picojson, rapidxml, xatlas, mrf | SIBR forks | BSD / MIT / Boost (see each repository) |

Because the FFmpeg build is GPL and the Gaussian-Splatting License is non-commercial, a ready-to-run binary
package that combines them is **not** published here. Building on your own machine with `setup.bat` avoids
redistributing that combination.

## Python / CUDA environment created by setup.bat (in `env\`)

| Package | Version | License |
|---|---|---|
| Python (conda-forge) | 3.10 | PSF |
| NVIDIA CUDA Toolkit (conda, `nvidia/label/cuda-12.4.1`) | 12.4.1 | NVIDIA CUDA Toolkit EULA |
| PyTorch (download.pytorch.org, cu124) | 2.5.1 | BSD-3-Clause |
| NumPy | < 2.3 | BSD-3-Clause |
| CMake, Ninja (PyPI) | latest | BSD-3-Clause / Apache-2.0 |

## Build tools you install yourself

* Microsoft Visual Studio 2022 (Community or Build Tools) with the C++ workload - Microsoft license
* Git for Windows - GPL-2.0
* Miniconda - BSD-3-Clause (the default Anaconda channel is not used: setup only uses conda-forge and NVIDIA channels)

## Datasets

No dataset or trained model is included. If you train on Neu3D (DyNeRF, CC BY-NC 4.0) or D-NeRF scenes,
follow their own licenses when sharing the exported model folders.
