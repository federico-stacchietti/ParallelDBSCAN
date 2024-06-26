#include <iostream>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <vector>

std::vector<std::vector<int>> region_matrix(float *dataset, float epsilon);
void grow_cluster(std::vector<std::vector<int>> region_matrix, int *labels, int cluster_tag,
        int min_points, int start);
int * DBSCAN(std::vector<std::vector<int>> region_matrix, int min_points);
