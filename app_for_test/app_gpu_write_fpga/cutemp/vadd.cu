#include "vadd.h"
	
extern "C" __global__ void vadd(float *A, float* B, float* C, volatile int* d_lock, volatile int* flag){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int count = 0;
	while(count<20000){
		count++;
		*flag = count;
	}
}

