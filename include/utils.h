#pragma once

#include <string>

struct Dataset {
    std::string name;
    long* data;
    int n_points;
    int dimension;
};

long* dataset_reader(const std::string& path, int dataset_size);
Dataset* datasets_constructor(const std::string& info_path, int one_only = 0,
                              int dataset_index = 0);
void write(const int* labels, int n);
void relabel(int* labels, int n);
int reduce(const int* labels, int n);
