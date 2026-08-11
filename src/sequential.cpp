#include "functions.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

using std::vector;

static vector<int> region_query(const long* dataset, int n_points,
                                int dimension, int point, double epsilon) {
    vector<int> neighbors;
    for (int i = 0; i < n_points; ++i) {
        double squared_distance = 0.0;
        for (int j = 0; j < dimension; ++j) {
            const double difference = static_cast<double>(
                dataset[point * dimension + j] - dataset[i * dimension + j]);
            squared_distance += difference * difference;
        }
        if (std::sqrt(squared_distance) <= epsilon)
            neighbors.push_back(i);
    }
    return neighbors;
}

static void grow_cluster(const long* dataset, int* labels,
                         vector<int> neighbors, int n_points, int dimension,
                         int point, int cluster_id, double epsilon,
                         int min_points) {
    labels[point] = cluster_id;
    std::vector<char> queued(n_points, 0);
    for (int neighbor : neighbors)
        queued[neighbor] = 1;

    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const int current = neighbors[i];
        if (labels[current] == -1)
            labels[current] = cluster_id;

        if (labels[current] != 0)
            continue;

        labels[current] = cluster_id;
        const vector<int> current_neighbors = region_query(
            dataset, n_points, dimension, current, epsilon);
        if (static_cast<int>(current_neighbors.size()) >= min_points) {
            for (int neighbor : current_neighbors) {
                if (!queued[neighbor]) {
                    queued[neighbor] = 1;
                    neighbors.push_back(neighbor);
                }
            }
        }
    }
}

int* DBSCAN(long* dataset, int n_points, int dimension, double epsilon,
            int min_points) {
    int* labels = static_cast<int*>(std::calloc(n_points, sizeof(int)));
    if (labels == nullptr)
        return nullptr;

    int cluster_id = 0;
    for (int i = 0; i < n_points; ++i) {
        if (labels[i] != 0)
            continue;

        vector<int> neighbors =
            region_query(dataset, n_points, dimension, i, epsilon);
        if (static_cast<int>(neighbors.size()) < min_points) {
            labels[i] = -1;
        } else {
            ++cluster_id;
            grow_cluster(dataset, labels, std::move(neighbors), n_points,
                         dimension, i, cluster_id, epsilon, min_points);
        }
    }
    return labels;
}
