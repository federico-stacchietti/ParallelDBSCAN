#include "utils.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;

long* dataset_reader(const std::string& path, int dataset_size) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Cannot open dataset: " + path);

    long* dataset = new long[dataset_size];
    for (int i = 0; i < dataset_size; ++i) {
        double value;
        if (!(input >> value)) {
            delete[] dataset;
            throw std::runtime_error("Dataset has fewer values than expected: " +
                                     path);
        }
        if (std::trunc(value) != value ||
            value < static_cast<double>(std::numeric_limits<long>::min()) ||
            value > static_cast<double>(std::numeric_limits<long>::max())) {
            delete[] dataset;
            throw std::runtime_error("Dataset contains a non-integral value: " +
                                     path);
        }
        dataset[i] = static_cast<long>(value);
    }

    double extra;
    if (input >> extra) {
        delete[] dataset;
        throw std::runtime_error("Dataset has more values than expected: " + path);
    }
    return dataset;
}

Dataset* datasets_constructor(const std::string& info_path, int one_only,
                              int dataset_index) {
    std::ifstream input(info_path);
    if (!input)
        throw std::runtime_error("Cannot open dataset manifest: " + info_path);

    int n_datasets;
    std::string base_path;
    if (!(input >> n_datasets) || n_datasets <= 0 || !(input >> base_path))
        throw std::runtime_error("Invalid dataset manifest: " + info_path);
    if (one_only && (dataset_index < 0 || dataset_index >= n_datasets))
        throw std::runtime_error("Dataset index is outside the manifest");

    fs::path dataset_directory(base_path);
    if (dataset_directory.is_relative())
        dataset_directory = fs::path(info_path).parent_path() / dataset_directory;

    Dataset* datasets = new Dataset[one_only ? 1 : n_datasets];
    int output_index = 0;
    try {
        for (int i = 0; i < n_datasets; ++i) {
            std::string name;
            int n_points;
            int dimension;
            if (!(input >> name >> n_points >> dimension) || n_points <= 0 ||
                dimension <= 0)
                throw std::runtime_error("Invalid dataset entry in: " +
                                         info_path);
            if (one_only && i != dataset_index)
                continue;

            Dataset dataset{name, nullptr, n_points, dimension};
            const fs::path path = dataset_directory / (name + ".txt");
            dataset.data = dataset_reader(path.string(), n_points * dimension);
            datasets[output_index++] = dataset;
        }
    } catch (...) {
        for (int i = 0; i < output_index; ++i)
            delete[] datasets[i].data;
        delete[] datasets;
        throw;
    }
    return datasets;
}

void write(const int* labels, int n) {
    std::ofstream output("CUDA_DBSCAN_RESULT.txt");
    if (!output)
        throw std::runtime_error("Cannot open CUDA_DBSCAN_RESULT.txt for writing");
    for (int i = 0; i < n; ++i)
        output << labels[i] << '\n';
}

void relabel(int* labels, int n) {
    for (int i = 0; i < n; ++i)
        labels[i] = labels[i] > -1 ? labels[i] - 1 : labels[i];
}

int reduce(const int* labels, int n) {
    int result = 0;
    for (int i = 0; i < n; ++i)
        result += labels[i];
    return result;
}
