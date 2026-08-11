#include "functions.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

static void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed");
}

__global__ static void square_root_matrix(double* matrix, int matrix_size) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= matrix_size)
        return;
    matrix[id] = sqrt(fmax(matrix[id], 0.0));
}

__global__ static void search_neighbors(const double* distance_matrix,
                                        bool* neighbor_matrix, double epsilon,
                                        int matrix_size) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= matrix_size)
        return;
    neighbor_matrix[id] = distance_matrix[id] <= epsilon;
}

__global__ static void count_neighbors(int* points,
                                       const bool* neighbor_matrix,
                                       int n_points, int matrix_size) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= matrix_size)
        return;
    if (neighbor_matrix[id])
        atomicAdd(&points[id / n_points], 1);
}

__global__ static void mark_core_points(int* points, int min_points,
                                        int n_points) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_points)
        return;
    points[id] = points[id] >= min_points ? id : -1;
}

static int* cluster_on_host(const int* points, const bool* neighbor_matrix,
                            int n_points) {
    int* labels = static_cast<int*>(std::calloc(n_points, sizeof(int)));
    if (labels == nullptr)
        return nullptr;

    int cluster_id = 0;
    for (int point = 0; point < n_points; ++point) {
        if (labels[point] != 0)
            continue;
        if (points[point] == -1) {
            labels[point] = -1;
            continue;
        }

        ++cluster_id;
        labels[point] = cluster_id;
        std::vector<int> neighbors;
        std::vector<char> queued(n_points, 0);
        for (int i = 0; i < n_points; ++i) {
            if (neighbor_matrix[point * n_points + i]) {
                neighbors.push_back(i);
                queued[i] = 1;
            }
        }

        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            const int current = neighbors[i];
            if (labels[current] == -1)
                labels[current] = cluster_id;
            if (labels[current] != 0)
                continue;

            labels[current] = cluster_id;
            if (points[current] == -1)
                continue;
            for (int candidate = 0; candidate < n_points; ++candidate) {
                if (neighbor_matrix[current * n_points + candidate] &&
                    !queued[candidate]) {
                    queued[candidate] = 1;
                    neighbors.push_back(candidate);
                }
            }
        }
    }
    return labels;
}

int* parallel_DBSCAN(long* dataset, int n_points, int dimensions,
                     int dataset_size, int distance_matrix_size,
                     double epsilon, int min_points, cublasHandle_t handle) {
    if (dataset == nullptr || n_points <= 0 || dimensions <= 0 ||
        dataset_size != n_points * dimensions ||
        distance_matrix_size != n_points * n_points)
        throw std::invalid_argument("Invalid DBSCAN dimensions");

    std::vector<double> norms(n_points, 0.0);
    std::vector<double> ones(n_points, 1.0);
    std::vector<double> matrix(dataset_size);
    std::vector<double> transpose(dataset_size);
    for (int point = 0; point < n_points; ++point) {
        for (int dimension = 0; dimension < dimensions; ++dimension) {
            const double value =
                static_cast<double>(dataset[point * dimensions + dimension]);
            norms[point] += value * value;
            matrix[point + dimension * n_points] = value;
            transpose[dimension + point * dimensions] = value;
        }
    }

    double* d_norms = nullptr;
    double* d_ones = nullptr;
    double* d_matrix = nullptr;
    double* d_transpose = nullptr;
    double* d_distances = nullptr;
    int* points = nullptr;
    bool* neighbor_matrix = nullptr;

    check_cuda(cudaMalloc(&d_norms, n_points * sizeof(double)), "cudaMalloc norms");
    check_cuda(cudaMalloc(&d_ones, n_points * sizeof(double)), "cudaMalloc ones");
    check_cuda(cudaMalloc(&d_matrix, dataset_size * sizeof(double)), "cudaMalloc matrix");
    check_cuda(cudaMalloc(&d_transpose, dataset_size * sizeof(double)), "cudaMalloc transpose");
    check_cuda(cudaMalloc(&d_distances, distance_matrix_size * sizeof(double)), "cudaMalloc distances");
    check_cuda(cudaMallocManaged(&points, n_points * sizeof(int)), "cudaMallocManaged points");
    check_cuda(cudaMallocManaged(&neighbor_matrix,
                                 distance_matrix_size * sizeof(bool)),
               "cudaMallocManaged neighbors");

    check_cuda(cudaMemcpy(d_norms, norms.data(), n_points * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy norms");
    check_cuda(cudaMemcpy(d_ones, ones.data(), n_points * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy ones");
    check_cuda(cudaMemcpy(d_matrix, matrix.data(), dataset_size * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy matrix");
    check_cuda(cudaMemcpy(d_transpose, transpose.data(),
                          dataset_size * sizeof(double), cudaMemcpyHostToDevice),
               "copy transpose");
    check_cuda(cudaMemset(points, 0, n_points * sizeof(int)), "clear points");

    const double zero = 0.0;
    const double one = 1.0;
    const double minus_two = -2.0;
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_points,
                             n_points, 1, &one, d_norms, n_points, d_ones,
                             n_points, &zero, d_distances, n_points),
                 "distance norms (rows)");
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_points,
                             n_points, 1, &one, d_ones, n_points, d_norms,
                             n_points, &one, d_distances, n_points),
                 "distance norms (columns)");
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_points,
                             n_points, dimensions, &minus_two, d_matrix,
                             n_points, d_transpose, dimensions, &one,
                             d_distances, n_points),
                 "distance dot products");

    constexpr int threads = 256;
    const int matrix_blocks = (distance_matrix_size + threads - 1) / threads;
    const int point_blocks = (n_points + threads - 1) / threads;
    square_root_matrix<<<matrix_blocks, threads>>>(d_distances,
                                                   distance_matrix_size);
    search_neighbors<<<matrix_blocks, threads>>>(
        d_distances, neighbor_matrix, epsilon, distance_matrix_size);
    count_neighbors<<<matrix_blocks, threads>>>(points, neighbor_matrix,
                                                n_points,
                                                distance_matrix_size);
    mark_core_points<<<point_blocks, threads>>>(points, min_points, n_points);
    check_cuda(cudaGetLastError(), "launch DBSCAN kernels");
    check_cuda(cudaDeviceSynchronize(), "run DBSCAN kernels");

    int* labels = cluster_on_host(points, neighbor_matrix, n_points);

    cudaFree(d_norms);
    cudaFree(d_ones);
    cudaFree(d_matrix);
    cudaFree(d_transpose);
    cudaFree(d_distances);
    cudaFree(points);
    cudaFree(neighbor_matrix);
    return labels;
}
