#include "../GlaneGPUstack.h"

__device__ void* inBuf;
__device__ void* outBuf;

__device__ int cursor;

__device__ void CUDAkernelInitialization(void* dptr){

	// initialize AQ and cursor
	struct AQentry* AQ = (struct AQentry*) dptr;

	for (int i=0; i<16; i++){
		AQ[i].isInUse = 0;
		AQ[i].MemFreelistIdx = i;
	}
	cursor = 0;
	printf("initialization finished!\n");
	// initialize inBuf & outBuf
	inBuf = (void*)dptr + AQsize * sizeof (struct AQentry);
	outBuf = inBuf + 2 * MemBufferSize * m * n * sizeof (float);	
}

__device__ void AQmoveCursor(){
	if (cursor !=15){
		cursor++;
	}
	else{
		cursor = 0;
	}
	printf("cursor = %d\n", cursor);	
}

__device__ void pushRequest(){

}

	
extern "C" __global__ void vadd(float *A, float* B, float* C, int* d_lock, int* flag){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int count = 0;
	
	if ((i==0)&&(j==0)){
		CUDAkernelInitialization((void*)d_lock+100*sizeof(int));
	}
	__syncthreads();
	float* c = (float*)outBuf;
	float* a = (float*)inBuf;	
	//__syncthreads();

	while(count<100){
		count++;
		if ((i==0)&&(j==0)){
			while (*d_lock!=0){
				atomicCAS(d_lock, 0,0);
			}
		}

		// CUDA kernel execution
		if ((i<m)&&(j<n)) {
			c[i*n+j] = a[i*n+j] + i + j;
		}

		__syncthreads();

		if ((i==0)&&(j==0)){
			atomicCAS(d_lock, 0, 1);
			//printf("GPU: lock is set to be 1\n");
		}

		__syncthreads();

		if ((i==0)&&(j==0)){
			*flag = 0;
			//printf("GPU: flag is set to be 0\n");
			AQmoveCursor();
		}
	}
}

