#include "functions.h"
#include "utils.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>

using Clock = std::chrono::steady_clock;

static double elapsed_microseconds(Clock::time_point start,
                                   Clock::time_point stop) {
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

static int print_result(const char* device, int* labels, int n_points,
                        double time) {
    relabel(labels, n_points);
    const int result = reduce(labels, n_points);
    std::cout.precision(8);
    std::cout << device << " RESULT: " << result
              << " --- TIME microsec: " << time
              << "  TIME sec: " << time / 1000000.0 << '\n';
    return result;
}

static void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed");
}

static void run(const std::string& path, double epsilon, int min_points,
                bool parallel_only = false, int dataset_index = 0) {
    Dataset* datasets =
        datasets_constructor(path, parallel_only ? 1 : 0, dataset_index);

    int n_datasets = 1;
    if (!parallel_only) {
        std::ifstream manifest(path);
        if (!(manifest >> n_datasets)) {
            delete[] datasets;
            throw std::runtime_error("Invalid dataset manifest: " + path);
        }
    }

    cublasHandle_t handle = nullptr;
    const cublasStatus_t create_status = cublasCreate(&handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) {
        for (int i = 0; i < n_datasets; ++i)
            delete[] datasets[i].data;
        delete[] datasets;
        check_cublas(create_status, "cublasCreate");
    }

    try {
        for (int i = 0; i < n_datasets; ++i) {
            Dataset& dataset = datasets[i];
            const int dataset_size = dataset.n_points * dataset.dimension;
            const int matrix_size = dataset.n_points * dataset.n_points;

            std::cout << "DATASET is \"" << dataset.name << "\"\n";

            double cpu_time = 0.0;
            if (!parallel_only) {
                const auto start = Clock::now();
                int* labels = DBSCAN(dataset.data, dataset.n_points,
                                     dataset.dimension, epsilon, min_points);
                const auto stop = Clock::now();
                if (labels == nullptr)
                    throw std::bad_alloc();
                cpu_time = elapsed_microseconds(start, stop);
                print_result("CPU", labels, dataset.n_points, cpu_time);
                std::free(labels);
            }

            const auto start = Clock::now();
            int* labels = parallel_DBSCAN(
                dataset.data, dataset.n_points, dataset.dimension, dataset_size,
                matrix_size, epsilon, min_points, handle);
            const auto stop = Clock::now();
            if (labels == nullptr)
                throw std::bad_alloc();
            const double gpu_time = elapsed_microseconds(start, stop);
            print_result("GPU", labels, dataset.n_points, gpu_time);
            if (parallel_only)
                write(labels, dataset.n_points);
            std::free(labels);

            if (!parallel_only)
                std::cout << "SPEEDUP: " << cpu_time / gpu_time << "\n\n";

            delete[] dataset.data;
            dataset.data = nullptr;
        }
    } catch (...) {
        for (int i = 0; i < n_datasets; ++i)
            delete[] datasets[i].data;
        delete[] datasets;
        cublasDestroy(handle);
        throw;
    }

    delete[] datasets;
    check_cublas(cublasDestroy(handle), "cublasDestroy");
}

static void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " <dataset-manifest> <epsilon> <min-points>"
                 " [--gpu-only [dataset-index]]\n";
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        usage(argv[0]);
        return 1;
    }

    try {
        const double epsilon = std::stod(argv[2]);
        const int min_points = std::stoi(argv[3]);
        if (epsilon < 0.0 || min_points <= 0)
            throw std::invalid_argument("epsilon and min-points must be positive");

        bool gpu_only = false;
        int dataset_index = 0;
        if (argc >= 5) {
            if (std::strcmp(argv[4], "--gpu-only") != 0) {
                usage(argv[0]);
                return 1;
            }
            gpu_only = true;
        }
        if (argc == 6)
            dataset_index = std::stoi(argv[5]);

        run(argv[1], epsilon, min_points, gpu_only, dataset_index);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
