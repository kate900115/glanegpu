#include "vadd.h"
	
extern "C" __global__ void vadd(float *A, float* B, float* C, int* d_lock, int* flag){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int count = 0;
	while(count<1000000){
		count++;
		while (*d_lock!=0){
			atomicCAS(d_lock, 0,0);
		}

	/*	if ((i<m)&&(j<n)) {
			C[i*n+j] = A[i*n+j]+B[i*n+j];
		}
*/
		__syncthreads();

		if ((i==0)&&(j==0)){
			atomicCAS(d_lock, 0, 1);
			//printf("GPU: lock is set to be 1\n");
		}

		__syncthreads();

		if ((i==0)&&(j==0)){
			*flag = 0;
			//printf("GPU: flag is set to be 0\n");
		}
	}
}

