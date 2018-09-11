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
//__device__ volatile int cursor;
__device__ int cursor;

__device__ void sendDoorBell(void* FPGAreqBuf, int kernel_ID){
	unsigned long* FPGAreq = (unsigned long*) FPGAreqBuf;
	*FPGAreq = kernel_ID;
}


__device__ void CUDAkernelInitialization(void* dptr, struct physAddr* physicalAddr){

	// initialize AQ and cursor
	struct AQentry* AQ = (struct AQentry*) dptr;
	AQueue = (void*) AQ;
	cursor = 0;
	p_AQueue = physicalAddr->dptrPhyAddrOnGPU;

	// initialize request buffer
	requestBuf = dptr + AQsize * sizeof (struct AQentry);
	struct reqBuf* requestBuffer = (struct reqBuf*) requestBuf;
	requestBuffer->isInUse = 0;		
	p_reqBuf = p_AQueue + AQsize * sizeof (struct AQentry);

	// initialize inBuf & outBuf
	inBuf = requestBuf + sizeof (struct reqBuf);
	outBuf = inBuf + MemBufferSize * m * n * sizeof (float);	
	p_inBuf = p_reqBuf + sizeof(struct reqBuf);
	p_outBuf = p_inBuf + MemBufferSize * m * n * sizeof(float);

	// initialize kernel ID
	kernelID = physicalAddr->kernelID;
	//printf("GPU: dptr = %p, inBuf addr = %p, outBuf addr = %p\n",dptr, inBuf, outBuf);

	//printf("GPU: initialization finished!\n");
}

__device__ void AQmoveCursor(int* CPU_AQcursor){
	struct AQentry* AQ = (struct AQentry*) AQueue;

	// to check wait until the next AQ entry is available
	//printf("before atomic check cursor = %d\n", cursor);
	while (!atomicCAS(&(AQ[cursor].isInUse),1,1));
	//printf("after atomic cursor = %d, AQ[cursor].isInUse = %d, AQ[cursor+1].isInUse = %d\n", cursor, AQ[cursor].isInUse, AQ[cursor+1].isInUse);
	*CPU_AQcursor = cursor;
	//while (!AQ[cursor].isInUse);
}


__device__ void pushRequest(int* FPGAreqBuf, int* CPU_AQcursor){
	struct reqBuf* requestBuffer = (struct reqBuf*) requestBuf;

	//printf("GPU: requestBuf->isInUse = %d\n", requestBuffer->isInUse);

	// waiting until request buffer is available
	// and then break the while look and set the 
	// isInUse bit to be 1 (in use) again
	while (atomicCAS(&requestBuffer->isInUse, 0, 1));

	// fill in the request buffer
	struct AQentry* AQ = (struct AQentry*) AQueue;
	requestBuffer->kernelID = kernelID;
	requestBuffer->AQaddr = p_AQueue;
	requestBuffer->inBufAddr = p_inBuf;
	requestBuffer->outBufAddr = p_outBuf;
	requestBuffer->idx = AQ[cursor].MemFreelistIdx;
	
	// send doorbell to FPGA
	// the passed doorbell is the CUDA kernel ID
	if (cursor !=AQsize-1) cursor++;
	else cursor = 0;
	//printf("Cursor = %d\n", cursor);	
	//*CPU_AQcursor = cursor;

	sendDoorBell(FPGAreqBuf, kernelID);

	// the is in use bit will be clean up by FPGA.
}

	
extern "C" __global__ void vadd(int* virtualAddr, int* FPGAreqBuf, struct physAddr* addrPacket, int* CPU_AQcursor){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int ii = threadIdx.x;
 	int jj = threadIdx.y;
	int count = 0;
	
	struct physAddr* paddrPacket = addrPacket;
	paddrPacket->dptrPhyAddrOnGPU = addrPacket->dptrPhyAddrOnGPU;
	if ((ii==0)&&(jj==0)){
		CUDAkernelInitialization((void*)virtualAddr, paddrPacket);
		*CPU_AQcursor = 0;
		//printf("GPU: GPU side address = %p\n",addrPacket->dptrPhyAddrOnGPU);
		//printf("GPU: kernel ID = %d\n", addrPacket->kernelID);
	}
	__syncthreads();

	struct AQentry* AQ = (struct AQentry*) AQueue;

	float* c = (float*)(outBuf + AQ[cursor].MemFreelistIdx * m * n * sizeof(float));
	float* a = (float*)(inBuf + AQ[cursor].MemFreelistIdx * m * n * sizeof(float));	
	//__syncthreads();

	while(count<iterationNum){
		count++;
		//if ((i==0)&&(j==0)) printf("GPU: count = %d\n", count);
		// CUDA kernel execution
		if ((i<m)&&(j<n)) {
			c[i*n+j] = a[i*n+j]/7;
		}

		__syncthreads();
	
		// push request to FPGA 
		// and then move the AQ cursor
		if ((i==0)&&(j==0)){
			pushRequest(FPGAreqBuf, CPU_AQcursor);
			AQmoveCursor(CPU_AQcursor);
			//printf("%d\n", count);
		}

		__syncthreads();// this sync needs to across blocks
		c = (float*)(outBuf + AQ[cursor].MemFreelistIdx * m * n * sizeof(float) );
		a = (float*)(inBuf + AQ[cursor].MemFreelistIdx * m * n * sizeof(float) );
	}
}

