#include "vadd.h"
	
extern "C" __global__ void vadd(float *A, float* B, float* C, int* d_lock, int* flag){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	//GPU checks its memory location to see if fpga finishes it's execution
	if ((i==0)&&(j==0)){
		while (*d_lock!=10003){
			atomicCAS(d_lock, 0,0);
		}
	}
	__syncthreads();
}

