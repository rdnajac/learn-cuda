#include<stdio.h>
#include<stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c)
{
	for(int idx=0;idx<N;idx++)
		c[idx] = a[idx] + b[idx];
}

__global__ void device_add(int *a, int *b, int *c)
{
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

//basically just fills the array with index.
void fill_array(int *data)
{
	for(int idx=0;idx<N;idx++)
		data[idx] = idx;
}

void print_output(int *a, int *b, int*c)
{
	for(int idx=0;idx<N;idx++)
		printf("\n %d + %d  = %d",  a[idx] , b[idx], c[idx]);
	printf("\n");
}

int main(void) {

	int *h_a, *h_b, *h_c;	// host copies of a, b, c
	int *d_a, *d_b, *d_c; 	// device copies of a, b, c

	// Determine the size of the arrays
	int size = N * sizeof(int);

	// Alloc space for host copies of a, b, c and setup input values
	h_a = (int *)malloc(size);
	h_b = (int *)malloc(size);
	h_c = (int *)malloc(size);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Fill input array
	fill_array(h_a);
	fill_array(h_b);

	// Copy inputs to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	device_add<<<N,1>>>(d_a, d_b, d_c);

	// host_add(a,b,c);
	// Instead of calling host_add, we call device_add
	// and copy the result back to host
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	// print the output
	print_output(h_a, h_b, h_c);

	free(h_a);
	free(h_b);
	free(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
