#include "../GlaneGPUstack.h"

// virtual address on GPU
__device__ void* inBuf;
__device__ void* outBuf;
__device__ void* AQueue;
__device__ void* requestBuf;

// physical address on GPU
__device__ unsigned long p_inBuf;
__device__ unsigned long p_outBuf;
__device__ unsigned long p_AQueue;
__device__ unsigned long p_reqBuf;
__device__ int kernelID;

// cursor of AQueue
__device__ int cursor;


__device__ void CUDAkernelInitialization(void* dptr){

	// initialize AQ and cursor
	struct AQentry* AQ = (struct AQentry*) dptr;
	AQueue = (void*) AQ;
	for (int i=0; i<16; i++){
		AQ[i].isInUse = 0;
		AQ[i].MemFreelistIdx = i;
	}
	cursor = 0;
	printf("initialization finished!\n");

	// initialize request buffer
	requestBuf = dptr + AQsize * sizeof (struct AQentry);
	struct reqBuf* requestBuffer = (struct reqBuf*) requestBuf;
	requestBuffer->isInUse = false;		

	// initialize inBuf & outBuf
	inBuf = requestBuf + sizeof (struct reqBuf);
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
	struct AQentry* AQ = (struct AQentry*) AQueue;

	// to check wait until the next AQ entry is available
	while (AQ[cursor].isInUse);
}

__device__ void pushRequest(){
	struct reqBuf* requestBuffer = (struct reqBuf*) requestBuf;
	while (requestBuffer->isInUse);
}

	
extern "C" __global__ void vadd(int* virtualAddr, int* FPGAreqBuf, struct physAddr* addrPacket){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int count = 0;
	
	if ((i==0)&&(j==0)){
		CUDAkernelInitialization((void*)virtualAddr+100*sizeof(int));
		printf("GPU side address = %p\n",addrPacket->dptrPhyAddrOnGPU);
		printf("kernel ID = %d\n", addrPacket->kernelID);
	}
	__syncthreads();
	float* c = (float*)outBuf;
	float* a = (float*)inBuf;	
	//__syncthreads();

	while(count<100){
		count++;
		if ((i==0)&&(j==0)){
			while (*virtualAddr!=0){
				atomicCAS(virtualAddr, 0,0);
			}
		}

		// CUDA kernel execution
		if ((i<m)&&(j<n)) {
			c[i*n+j] = a[i*n+j] + i + j;
		}

		__syncthreads();

		if ((i==0)&&(j==0)){
			atomicCAS(virtualAddr, 0, 1);
			//printf("GPU: lock is set to be 1\n");
		}

		__syncthreads();

		if ((i==0)&&(j==0)){
			*FPGAreqBuf = 0;
			//printf("GPU: flag is set to be 0\n");
			AQmoveCursor();
		}
	}
}


