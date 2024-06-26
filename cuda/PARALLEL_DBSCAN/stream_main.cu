#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <unistd.h>
#define dimension 15
#define n_points 5000
#define s n_points * n_points

__global__ void distance_matrix_kernel(float *dataset, float *distance_matrix) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < s) {
		int offset = thread_id / n_points;
		int mod = thread_id % n_points;
		float distance = 0;
		for (int j = 0; j < dimension; j++)
			distance += pow(
					dataset[j + offset * dimension]
							- dataset[j + mod * dimension], 2);
		distance_matrix[thread_id] = sqrt(distance);
	}
}

float * CPU_distance_matrix(float *dataset) {
	float *distance_matrix;
	float distance = 0;
	distance_matrix = (float *) malloc(n_points * n_points * sizeof(float));
	for (int i = 0; i < n_points; i++) {
		for (int j = 0; j < n_points; j++) {
			for (int k = 0; k < dimension; k++)
				distance += pow(
						*(dataset + i * dimension + k)
								- *(dataset + j * dimension + k), 2);
			*(distance_matrix + i * n_points + j) = sqrt(distance);
			distance = 0;
		}
	}
	return distance_matrix;
}

int main() {
	float *h_dataset, *h_distance_matrix, *h_shared;
	h_dataset = (float *) malloc(n_points * dimension * sizeof(float));
	h_distance_matrix = (float *) malloc(n_points * n_points * sizeof(float));
	h_shared = (float *) malloc(n_points * n_points * sizeof(float));
	int n = 0;
	double sum_CPU = 0;
	double sum_GPU = 0;
	double sum_shared = 0;
	float *dstmtr;
	FILE *fptr;
	fptr = fopen("/home/federico/cuda-workspace/new_stream/15d_dataset.txt",
			"r");
	while (fscanf(fptr, "%f", &h_dataset[n++]) != EOF)
		;
	fclose(fptr);
	float *d_dataset, *d_distance_matrix, *shared_distance_matrix;
	int dataset_size = n_points * dimension * sizeof(float);
	int distance_matrix_dimension = n_points * n_points * sizeof(float);

	cudaMalloc(&d_dataset, dataset_size);
	cudaMalloc(&d_distance_matrix, distance_matrix_dimension);
	cudaMalloc(&shared_distance_matrix, distance_matrix_dimension);

	cudaMemcpy(d_dataset, h_dataset, dataset_size, cudaMemcpyHostToDevice);
	cudaMemset(d_distance_matrix, 0, distance_matrix_dimension);
	//cudaMemset(shared_distance_matrix, 0, distance_matrix_dimension);

	int gridSize = 24415;
	int blockSize = 1024;
	int nStreams = 257;
	int k = gridSize / nStreams;
	int bytesPerStream = k * blockSize * sizeof(float);
	cudaStream_t stream[nStreams];
	for (int i = 0; i < nStreams; ++i) {
		cudaStreamCreate(&stream[i]);
	}
	int offset = 0;

	for (int i = 0; i < nStreams; ++i) {
		offset = i * bytesPerStream;
		distance_matrix_kernel<<<95, blockSize, 0, stream[i]>>>(d_dataset,
				d_distance_matrix);
		cudaMemcpyAsync(&h_distance_matrix[offset], &d_distance_matrix[offset],
				bytesPerStream, cudaMemcpyDeviceToHost, stream[i]);
	}

	for (int i = 0; i < nStreams; ++i) {
		cudaStreamDestroy(stream[i]);
	}

	dstmtr = CPU_distance_matrix(h_dataset);
	/*distance_matrix_kernel<<<24415, 1024>>>(d_dataset, d_distance_matrix);
	cudaMemcpy(h_distance_matrix, d_distance_matrix, distance_matrix_dimension,
			cudaMemcpyDeviceToHost);*/
	for (int i = 0; i < n_points * n_points; i++) {
		sum_CPU += *(dstmtr + i);
		sum_GPU += h_distance_matrix[i];
		sum_shared += h_shared[i];
	}
	printf("%f  ---   %f --- %f\n", sum_CPU, sum_GPU, sum_shared);

	free(h_dataset);
	free(h_distance_matrix);
	free(h_shared);
	free(dstmtr);
	cudaFree(d_dataset);
	cudaFree(d_distance_matrix);
	cudaFree(shared_distance_matrix);

	return 0;
}
