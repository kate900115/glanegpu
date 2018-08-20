#include "vadd.h"
//#include <cooperative_groups.h>

//using namespace cooperative_groups;

struct packet{
	unsigned long CR_address;
	unsigned long inputBufAddress;
	unsigned long remoteInputAddress;
};
	
extern "C" __global__ void vadd(float *A, float* B, float* C, int* d_lock, volatile int* flag){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y; 

	int blockNum = blockDim.x * blockDim.y * blockDim.z;	

	int count = 0;

	struct packet* temp;
	struct packet temp2;
	temp = (struct packet*)flag;

	temp->CR_address = 0x111111;
	temp->inputBufAddress = 0x9999999;
	temp->remoteInputAddress = 0x88888;	
	//*temp = temp2;
	*flag = 1;
	__syncthreads();

	while(count<10000000){
		count++;
		
		if ((i==0)&&(j==0)){
			while (*d_lock!=0){
				atomicCAS(d_lock, 0,0);
			}
		}

//		cooperative_groups::grid_group g = cooperative_groups::this_grid();	
//		g.sync();
		//printf("count = %d\n", count);	
		// the original GPU kernel code here
		__syncthreads();
		if ((i==0)&&(j==0)){
			atomicCAS(d_lock, 0, 1);
		}		

		// waiting for all threads finishes the code.
//		g.sync();

		__syncthreads();
		// clean up monitor and monitor2 for next round of execution
		// notify FPGA to release the lock
		if ((i==0)&&(j==0)){
			*flag = 0;
		}
	}
}

