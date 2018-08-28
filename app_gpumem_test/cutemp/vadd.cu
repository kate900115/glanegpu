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


__device__ void sendDoorBell(void* FPGAreqBuf, unsigned long p_reqBuf){
	unsigned long* FPGAreq = (unsigned long*) FPGAreqBuf;
	*FPGAreq = p_reqBuf;
}


__device__ void CUDAkernelInitialization(void* dptr, struct physAddr* physicalAddr){

	// initialize AQ and cursor
	struct AQentry* AQ = (struct AQentry*) dptr;
	AQueue = (void*) AQ;
	for (int i=0; i<16; i++){
		AQ[i].isInUse = 1;
		AQ[i].MemFreelistIdx = i;
	}
	cursor = 0;
	p_AQueue = physicalAddr->dptrPhyAddrOnGPU;

	// this part is for test
	for (int i = 0; i<5; i++){
		AQ[i].isInUse = 1;
	}



	// initialize request buffer
	requestBuf = dptr + AQsize * sizeof (struct AQentry);
	struct reqBuf* requestBuffer = (struct reqBuf*) requestBuf;
	requestBuffer->isInUse = 0;		
	p_reqBuf = p_AQueue + AQsize * sizeof (struct AQentry);

	// initialize inBuf & outBuf
	inBuf = requestBuf + sizeof (struct reqBuf);
	outBuf = inBuf + 2 * MemBufferSize * m * n * sizeof (float);	
	p_inBuf = p_reqBuf + sizeof(struct reqBuf);
	p_outBuf = p_inBuf + 2 * MemBufferSize * m * n * sizeof(float);

	// initialize kernel ID
	kernelID = physicalAddr->kernelID;
	printf("dptr = %p, inBuf addr = %p, outBuf addr = %p\n",dptr, inBuf, outBuf);





	printf("initialization finished!\n");
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
	while (!AQ[cursor].isInUse);
}


__device__ void pushRequest(void* FPGAreqBuf){
	struct reqBuf* requestBuffer = (struct reqBuf*) requestBuf;
	
	// need change** should be atomic operation
	while (atomicCAS(&requestBuffer->isInUse, 0, 1));

	// fill in the request buffer
	struct AQentry* AQ = (struct AQentry*) AQueue;
	requestBuffer->kernelID = kernelID;
	requestBuffer->AQaddr = p_AQueue;
	requestBuffer->inBufAddr = p_inBuf;
	requestBuffer->outBufAddr = p_outBuf;
	requestBuffer->idx = AQ[cursor].MemFreelistIdx;
	
	// send doorbell to FPGA
	sendDoorBell(FPGAreqBuf, p_reqBuf);

	// ************for test*************
	requestBuffer->isInUse = 0;
	
}

	
extern "C" __global__ void vadd(int* virtualAddr, int* FPGAreqBuf, struct physAddr* addrPacket){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int count = 0;
	
	struct physAddr* paddrPacket = addrPacket;
	paddrPacket->dptrPhyAddrOnGPU = addrPacket->dptrPhyAddrOnGPU + 100*sizeof(int);
	if ((i==0)&&(j==0)){
		CUDAkernelInitialization((void*)virtualAddr + 100*sizeof(int), paddrPacket);
		printf("GPU side address = %p\n",addrPacket->dptrPhyAddrOnGPU);
		printf("kernel ID = %d\n", addrPacket->kernelID);
	}
	__syncthreads();

	struct AQentry* AQ = (struct AQentry*) AQueue;

	float* c = (float*)(outBuf + AQ[cursor].MemFreelistIdx * m * n * sizeof(float) * MemBufferSize);
	float* a = (float*)(inBuf + AQ[cursor].MemFreelistIdx * m * n * sizeof(float) * MemBufferSize);	
	__syncthreads();

	while(count<10){
		count++;
		if ((i==0)&&(j==0)) printf("count = %d\n", count);
		//printf("count = %d\n", count);
		if ((i==0)&&(j==0)){
			printf("virtualAddr = %d\n", *virtualAddr);
			//while (*virtualAddr!=0){
			//	atomicCAS(virtualAddr, 0,0);
			//}
			while (atomicCAS(virtualAddr, 1, 0));
			printf("just for test\n");
		}

		// CUDA kernel execution
		if ((i<m)&&(j<n)) {
			c[i*n+j] = a[i*n+j] + i + j;
			//printf("c = %p, c[%d][%d] = %f\n", c, i, j, c[i*n+j]);
		}

		__syncthreads();

		if ((i==0)&&(j==0)){
			atomicCAS(virtualAddr, 0, 1);
			printf("GPU: lock is set to be 1\n");
		}

		__syncthreads();

		if ((i==0)&&(j==0)){
	//		*FPGAreqBuf = 0;
	//		printf("GPU: flag is set to be 0\n");
			pushRequest((void*)FPGAreqBuf);
			AQmoveCursor();
		}

		__syncthreads();
		if ((i==0)&&(j==0)) printf("!!!!!!!!!!!!!!!!\n");
		c = (float*)(outBuf + AQ[cursor].MemFreelistIdx * m * n * sizeof(float) );
		a = (float*)(inBuf + AQ[cursor].MemFreelistIdx * m * n * sizeof(float) );
		//printf("c = %p, a = %p\n", (void*)c, (void*)a);	

	}
}


