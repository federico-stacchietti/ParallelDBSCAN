#pragma once

#include <cublas_v2.h>

int* DBSCAN(long* dataset, int n_points, int dimension, double epsilon,
            int min_points);

int* parallel_DBSCAN(long* dataset, int n_points, int dimensions,
                     int dataset_size, int distance_matrix_size,
                     double epsilon, int min_points, cublasHandle_t handle);
