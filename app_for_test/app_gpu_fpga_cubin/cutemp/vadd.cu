#include "vadd.h"
#include <cooperative_groups.h>

using namespace cooperative_groups;
	
extern "C" __global__ void vadd(float *A, float* B, float* C, int* d_lock, volatile int* flag){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y; 

	int blockNum = blockDim.x * blockDim.y * blockDim.z;	

	int count = 0;

	while(count<10){
		count++;
		
		if ((i==0)&&(j==0)){
			while (*d_lock!=0){
				atomicCAS(d_lock, 0,0);
			}
		}

		//grid_group g = this_grid();	
//		g.sync();
		//printf("count = %d\n", count);	
		// the original GPU kernel code here
		if ((i==0)&&(j==0)){
			atomicCAS(d_lock, 0, 1);
		}		

		// waiting for all threads finishes the code.
//		g.sync();

		// clean up monitor and monitor2 for next round of execution
		// notify FPGA to release the lock
		if ((i==0)&&(j==0)){
			*flag = 0;
		}
	}
}

