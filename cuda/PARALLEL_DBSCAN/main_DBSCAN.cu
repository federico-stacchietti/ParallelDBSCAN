#include <stdio.h>
#include <stdlib.h>
#include "host_DBSCAN.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#define dimension 10
#define n_points 30

int main() {
	float *dataset;
	dataset = (float *) malloc(n_points * dimension * sizeof(float));

	int n = 0;
	FILE *fptr;
	fptr = fopen("/home/federico/cuda-workspace/PARALLEL_DBSCAN/10D_random_dataset.txt", "r");
	while (fscanf(fptr, "%f", &dataset[n++]) != EOF);
	fclose(fptr);
	int *ptr;
	ptr = DBSCAN(region_matrix(dataset, 7), 3);
	for (int i = 0; i < n_points; i++) {
		printf("%d\n", *(ptr + i) - 1);
	}


	return 0;
}
