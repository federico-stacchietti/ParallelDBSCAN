# Parallel DBSCAN

This folder contains the corrected version of the recovered CPU/CUDA DBSCAN
prototype. Distance and neighborhood computation run on the GPU using CUDA and
cuBLAS; cluster expansion runs deterministically on the host.

Requirements: CMake 3.18 or newer, a CUDA-capable NVIDIA GPU, and the CUDA
Toolkit.

```sh
cmake -S . -B build
cmake --build build
./build/parallel_dbscan datasets.txt 7 3
```

To run only the GPU path for one dataset:

```sh
./build/parallel_dbscan datasets.txt 7 3 --gpu-only 0
```

The manifest format is: dataset count, dataset directory, then name, point
count, and dimension on separate lines for each `.txt` dataset.
