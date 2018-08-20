#include "vadd.h"

__device__ int monitor;
__device__ int monitor2;
	
extern "C" __global__ void vadd(float *A, float* B, float* C, int* d_lock, volatile int* flag){
	int j = threadIdx.x;
	int i = threadIdx.y;

	int jj = blockIdx.x * blockDim.x + threadIdx.x;
	int ii = blockIdx.y * blockDim.y + threadIdx.y; 

	int blockNum = blockDim.x * blockDim.y * blockDim.z;	

	int count = 0;

	while(count<1000000){
		count++;
		
		if ((i==0)&&(j==0)){
			while (*d_lock!=0){
				atomicCAS(d_lock, 0,0);
			}
			atomicAdd(&monitor, 1);
			while(atomicCAS(&monitor, blockNum, blockNum)!=blockNum);
			atomicCAS(d_lock,0 ,1);
		}
		__syncthreads();
		
		// the original GPU kernel code here


		// waiting for all threads finishes the code.
		if ((i==0)&&(j==0)){
			atomicAdd(&monitor2, 1);
			while(atomicCAS(&monitor2, blockNum, blockNum)!=blockNum);
		}
		__syncthreads();


		// clean up monitor and monitor2 for next round of execution
		// notify FPGA to release the lock
		if ((ii==0)&&(jj==0)){
			monitor = 0;
			monitor2 = 0;
			*flag = 0;
		}
	}
}

