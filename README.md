# Parallel DBSCAN

This folder contains a CPU/CUDA DBSCAN implementation. Distance and neighborhood computation run on the GPU using CUDA and
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
