#include <stdio.h>
#include <stdlib.h>
#include "host_DBSCAN.h"
#define n_points 30
#define dimension 10

std::vector<std::vector<int>> region_matrix(float *dataset, float epsilon) {
	float distance = 0;
	std::vector<std::vector<int>> region_matrix;
	for (int i = 0; i < n_points; i++) {
		std::vector<int> neighbors;
		for (int j = 0; j < n_points; j++) {
			for (int k = 0; k < dimension; k++)
				distance += pow(
						*(dataset + i * (dimension) + k)
								- *(dataset + j * (dimension) + k), 2);
			if (sqrt(distance) <= epsilon)
				neighbors.push_back(j);
			distance = 0;
		}
		region_matrix.push_back(neighbors);
	}
	return region_matrix;
}

void grow_cluster(std::vector<std::vector<int>> region_matrix, int *labels,
		int cluster_tag, int min_points, int start) {
	int i = 0;
	int index;
	std::vector<int> neighbors = region_matrix.at(start);
	while (i < neighbors.size()) {
		index = neighbors.at(i);
		if (*(labels + index) == -1)
			*(labels + index) = cluster_tag;
		if (*(labels + index) == 0) {
			*(labels + index) = cluster_tag;
			if (region_matrix.at(i).size() >= min_points)
				for (int elem : region_matrix.at(index))
					if (std::find(neighbors.begin(), neighbors.end(), elem)
							== neighbors.end())
						neighbors.push_back(elem);
		}
		i++;
	}
}

int * DBSCAN(std::vector<std::vector<int>> region_matrix, int min_points) {
	int cluster_tag = 0;
	int *labels = (int *) malloc(n_points * sizeof(int));
	for (int i = 0; i < n_points; i++)
		*(labels + i) = 0;
	for (int i = 0; i < n_points; i++) {
		if (*(labels + i) == 0) {
			if (region_matrix.at(i).size() < min_points)
				*(labels + i) = -1;
			else {
				cluster_tag += 1;
				*(labels + i) = cluster_tag;
				grow_cluster(region_matrix, labels, cluster_tag, min_points, i);
			}
		}
	}
	return labels;
}
