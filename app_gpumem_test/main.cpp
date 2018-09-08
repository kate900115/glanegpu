#include "cuda.h"
//#include "cuda_runtime_api.h"
#include "gpumemioctl.h"

#include <dirent.h>
#include <signal.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <sys/uio.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "GlaneGPUstack.h"
// time measurement
#include <chrono>
#include <ctime>
#include <iostream>

//-----------------------------------------------------------------------------

void checkError(CUresult status);
bool wasError(CUresult status);

//-----------------------------------------------------------------------------
pthread_mutex_t AQlock;
pthread_mutex_t RQlock;
pthread_mutex_t printLock;
//-----------------------------------------------------------------------------
struct RQentry{
	int MemFreelistIdx;
	int KernelID;
};

struct ParamForRQhead{
	//FPGA side
	struct RQentry* RQ;
	int* RQhead;  
	int* RQtail;
	int* RQcursor;

	//GPU side
	struct AQentry* AQ;
	float* GPUrecvBuf;
	int* AQhead;
	int* AQtail;

	//for control
	bool* killThread;
};

struct ParamForRQcursor{
	// FPGA side
	struct RQentry* RQ; 
	int* RQhead;
	int* RQtail;
	int* RQcursor;

	//GPU side
	struct AQentry* AQ;
	float* GPUsendBuf;
	int* AQhead;
	int* AQtail;	

	//for control
	bool* killThread;
};	

//---------------------------------------------------------------------------------
void* f_movingRQcursor(void* ptr){
	struct ParamForRQcursor* param = (struct ParamForRQcursor*) ptr;
	struct RQentry* RQ = param->RQ;
	int* RQhead = param->RQhead;
	int* RQtail = param->RQtail;
	int* RQcursor = param->RQcursor;

	struct AQentry* AQ = param->AQ;
	int* AQhead = param->AQhead;
	int* AQtail = param->AQtail;
	float* GPUsendBufBase = param->GPUsendBuf;
	float* GPUsendBuf = GPUsendBufBase;

	// output destination
	float a[m*n];
	bool cursorValid = true;

	bool* killThread = param->killThread;
	bool workFinish = false;

	while(!(workFinish)){
		// copy data out of GPU send buffer
		pthread_mutex_lock(&printLock);
		printf("RQ CURSOR: copy data out of send buffer\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail);	
		pthread_mutex_unlock(&printLock);
		if (cursorValid){
			for (int i=0; i<m*n; i++){
				a[i] = GPUsendBuf[i];
			}
			cursorValid = false;
		}


		pthread_mutex_lock(&printLock);
		printf("RQ CURSOR: before moving AQ head\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail);	
		pthread_mutex_unlock(&printLock);


		// move AQ head
		bool breakLoop = false;
		bool breakRQLoop = false;
		while(!breakLoop){
			pthread_mutex_lock(&AQlock);
			// 00000000111100000
			//         H  T
			pthread_mutex_lock(&printLock);
			printf("RQ CURSOR: moving AQ head\n");
			printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail);	
			pthread_mutex_unlock(&printLock);

			if (*AQhead<*AQtail){
				breakLoop = true;
				AQ[*AQhead].isInUse = 0;
				AQ[*AQhead].MemFreelistIdx = 0;
				(*AQhead)++;
			}
			// 11110000000011111
			//    T        H
			else if (*AQhead>*AQtail){
				if (*AQhead==(AQsize-1)){
					breakLoop = true;
					AQ[*AQhead].isInUse = 0;
					AQ[*AQhead].MemFreelistIdx = 0;
					*AQhead = 0;
				}
				else {

					breakLoop = true;
					AQ[*AQhead].isInUse = 0;
					AQ[*AQhead].MemFreelistIdx = 0;
					(*AQhead)++;
				}
			}
			else if (*AQhead == *AQtail){
				breakLoop = true;
				breakRQLoop = true;
			}
			pthread_mutex_unlock(&AQlock);
		}	

		// move RQ cursor
		
		pthread_mutex_lock(&printLock);
		printf("RQ CURSOR: after moving AQ head\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail);	
		pthread_mutex_unlock(&printLock);



		pthread_mutex_lock(&printLock);
		printf("RQ CURSOR: before moving RQ cursor\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail);	
		pthread_mutex_unlock(&printLock);

		breakLoop = false;
		while(!breakLoop){
		//	if (breakRQLoop) break;
			pthread_mutex_lock(&RQlock);
			if (*RQhead<=*RQtail){
				// 0000111111110000
				//     H   C  T
				if (*RQcursor<*RQtail) {
					(*RQcursor)++;
					cursorValid = true;
					breakLoop = true;
				}	
			}
			else{
				// 1110000000111111
				//   T       H  C
				if ((*RQcursor<RQsize-1)&&(*RQcursor<*RQtail)){
					(*RQcursor)++;
					cursorValid = true;
					breakLoop = true;
				}
				else if (*RQcursor==RQsize-1){
					*RQcursor = 0;
					cursorValid = true;
					breakLoop = true;
				}	
			}
			pthread_mutex_unlock(&RQlock);	
		}
		pthread_mutex_lock(&printLock);
		printf("RQ CURSOR: before moving RQ cursor\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail);	
		pthread_mutex_unlock(&printLock);

		if (*killThread){
			if((*RQtail == *RQhead)&&(*RQtail == *RQcursor)) workFinish=true;
		}

	}
}


// this function must be invoked after the initialization
void* f_movingRQhead(void* ptr){
	struct ParamForRQhead* param = (struct ParamForRQhead*) ptr;
	struct RQentry* RQ = param->RQ;
	int* RQhead = param->RQhead;
	int* RQtail = param->RQtail;
	int* RQcursor = param->RQcursor;

	struct AQentry* AQ = param->AQ;
	int* AQhead = param->AQhead;
	int* AQtail = param->AQtail;
	float* GPUrecvBufBase = param->GPUrecvBuf;
	float* GPUrecvBuf = GPUrecvBufBase;
 
	struct RQentry tempRQentry;
	bool headValid = true;
	//
	bool workFinish =  false;
	bool* killThread = param->killThread;
	while (!(workFinish)){
		// if head pointer is valid,
		// copy data into GPU receive buffer
		pthread_mutex_lock(&printLock);
		printf("RQ HEAD: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail);	
		pthread_mutex_unlock(&printLock);

		if (headValid){
			pthread_mutex_lock(&printLock);
			printf("RQ HEAD: copy data into GPU receive buffer\n");	
			pthread_mutex_unlock(&printLock);

			for (int i=0; i<m*n; i++){
				GPUrecvBuf[i] = i/17;
			}
			// set head to be invalid
			headValid = false;
		}

		// move AQ pointer
		bool breakLoop = false;
		while (!breakLoop){
			pthread_mutex_lock(&AQlock);
			//000000000111111
			//         H    T
			pthread_mutex_lock(&printLock);
			printf("RQ HEAD: before moving AQ tail\n");
			printf("RQ HEAD: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail);	
			pthread_mutex_unlock(&printLock);

			if (*AQtail>=*AQhead){
				if (*AQtail==(AQsize-1)){
					if (*AQhead!=0){
						*AQtail = 0;
						AQ[*AQtail].isInUse = 1;
						AQ[*AQtail].MemFreelistIdx = RQ[*RQhead].MemFreelistIdx;	
						breakLoop = true;
					}
				}
				else{
					(*AQtail)++;
					AQ[*AQtail].isInUse = 1;
					AQ[*AQtail].MemFreelistIdx = RQ[*RQhead].MemFreelistIdx;
					breakLoop = true;
				}
			}
			//11111000111111
			//    T   H
			else{
				pthread_mutex_lock(&printLock);
				printf("RQ HEAD: AQ tail = %d\n", *AQtail);
				pthread_mutex_unlock(&printLock);

				if (*AQhead!=((*AQtail)+1)){
					(*AQtail)++;
					AQ[*AQtail].isInUse = 1;
					AQ[*AQtail].MemFreelistIdx = RQ[*RQhead].MemFreelistIdx;
					breakLoop = true;	
				}
			}

			pthread_mutex_lock(&printLock);
			printf("RQ HEAD: after moving AQ tail\n");
			printf("RQ HEAD: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail);	
			pthread_mutex_unlock(&printLock);



			pthread_mutex_unlock(&AQlock);
		}

		// move RQ pointer
		
		pthread_mutex_lock(&printLock);
		printf("RQ HEAD: before moving RQ head\n");
		printf("RQ HEAD: RQhead = %d\n", *RQhead);
		pthread_mutex_unlock(&printLock);


		breakLoop = false;
		while(!(breakLoop)){
			pthread_mutex_lock(&RQlock);
			if (*RQhead<=*RQtail){
				//00001111110000
				//    H  C T
				if ((*RQcursor>*RQhead)&&(*RQcursor<=*RQtail)){
					(*RQhead)++;
					headValid = true;
					breakLoop = true;
				}
			}
			else {
				//11100000011111
				//  T      H  C
				if ((*RQcursor>*RQhead)&&(*RQcursor<=RQsize-1)){
					if (*RQcursor!=RQsize-1){
						(*RQhead)++;
						headValid = true;
						breakLoop = true;
					}
					else{
						*RQhead=0;
						headValid = true;
						breakLoop = true;
					}
	
				}
				//11100000011111
				//C T      H    
				else if ((*RQcursor<*RQtail)&&(*RQcursor<*RQhead)){
					(*RQhead)++;
					headValid = true;
					breakLoop = true;
				}
		
			}
			pthread_mutex_unlock(&RQlock);
		}
		pthread_mutex_lock(&printLock);
		printf("RQ HEAD: after moving RQ head\n");
		printf("RQ HEAD: RQhead = %d\n", *RQhead);
		pthread_mutex_unlock(&printLock);

	
		// changing the receive buffer address
		GPUrecvBuf = GPUrecvBufBase + m * n * RQ[*RQhead].MemFreelistIdx; 

		if (*killThread){
			printf("999999999999999999999999999999999999\n");
			if((*RQtail == *RQhead)&&(*RQtail == *RQcursor)) workFinish=true;
		}

	}		
}

// zyuxuan: struct for pthread parameter
// zyuxuan: function for the thread

int main(int argc, char *argv[])
{
	gpudma_lock_t lock;
	gpudma_unlock_t unlock;
	gpudma_state_t *state = 0;
	int statesize = 0;
	int res = -1;
	unsigned count=0x0A000000;

	int fd = open("/dev/"GPUMEM_DRIVER_NAME, O_RDWR, 0);
	if (fd < 0) {
		printf("Error open file %s\n", "/dev/"GPUMEM_DRIVER_NAME);
		return -1;
	}

	int fd_v2p2v = open("/dev/v2p2v", O_RDWR, 0);
	if (fd_v2p2v < 0){
		printf("Error open file %s\n","/dev/v2p2v");
		return -1;
	}

	checkError(cuInit(0));

	int total = 0;
	checkError(cuDeviceGetCount(&total));
	fprintf(stderr, "Total devices: %d\n", total);

	CUdevice device;
	checkError(cuDeviceGet(&device, 0));

	char name[256];
	checkError(cuDeviceGetName(name, 256, device));
	fprintf(stderr, "Select device: %s\n", name);

	// get compute capabilities and the devicename
	int major = 0, minor = 0;
	checkError( cuDeviceComputeCapability(&major, &minor, device));
	fprintf(stderr, "Compute capability: %d.%d\n", major, minor);

	size_t global_mem = 0;
	checkError( cuDeviceTotalMem(&global_mem, device));
	fprintf(stderr, "Global memory: %llu MB\n", (unsigned long long)(global_mem >> 20));
	if(global_mem > (unsigned long long)4*1024*1024*1024L)
		fprintf(stderr, "64-bit Memory Address support\n");

	CUcontext  context;
	checkError(cuCtxCreate(&context, CU_CTX_MAP_HOST, device));




	// zyuxuan: allocate memory space on host and device
	CUdeviceptr d_a, d_b, d_c;
	float h_a[m*n], h_b[m*n], h_c[m*n];

	for (int i=0; i<m*n; i++){
		h_a[i] = i;
		h_b[i] = i*i;
		h_c[i] = 0;
	}

	// zyxuan: setup device memory
	checkError(cuMemAlloc(&d_a,sizeof(int)*m*n));
	checkError(cuMemAlloc(&d_b,sizeof(int)*m*n));
	checkError(cuMemAlloc(&d_c,sizeof(int)*m*n));

	// zyuxuan: offload data from CPU to GPU
	checkError(cuMemcpyHtoD(d_a, h_a, sizeof(int)*m*n));
	checkError(cuMemcpyHtoD(d_b, h_b, sizeof(int)*m*n));

	
	

	// zyuxuan: to insert our cuda function
	// zyuxuan: we need to load module 
	char* module_file = (char*) "cutemp/vadd.ptx";
	char* kernel_name = (char*) "vadd";
	CUmodule module;
	
	checkError(cuModuleLoad(&module, module_file));

	CUfunction function;
	checkError(cuModuleGetFunction(&function, module, kernel_name));

	// zyuxuan: to allow GPU to access to CPU memory space
	void* p;
	p = (int*)malloc(20*sizeof(int));
	checkError(cuMemHostRegister(p, 20*sizeof(int), CU_MEMHOSTREGISTER_DEVICEMAP));
	CUdeviceptr CPUflag;
	checkError(cuMemHostGetDevicePointer(&CPUflag, p,0));

	int* p_flag = (int*) p;


	size_t size = 0x100000;
	CUdeviceptr dptr = 0;
	unsigned int flag = 1;
	unsigned char *h_odata = NULL;
	h_odata = (unsigned char *)malloc(size);

	CUresult status = cuMemAlloc(&dptr, size);

	/* this code does not work */
	//CUresult status = cuModuleGetGlobal(&dptr, &size, module, "d_lock");
	//size = 0x100000;

	if(wasError(status)) {
        	goto do_free_context;	
	}

	fprintf(stderr, "Allocate memory address: 0x%llx\n",  (unsigned long long)dptr);

	status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);
	if(wasError(status)) {
		goto do_free_memory;
	}

	fprintf(stderr, "Press enter to lock\n");
	//getchar();

	// TODO: add kernel driver interaction...
	lock.addr = dptr;
	lock.size = size;
	res = ioctl(fd, IOCTL_GPUMEM_LOCK, &lock);
	if(res < 0) {
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		goto do_free_attr;
	}

	fprintf(stderr, "Press enter to get state. We lock %ld pages\n", lock.page_count);
	//getchar();

	statesize = (lock.page_count*sizeof(uint64_t) + sizeof(struct gpudma_state_t));
	state = (struct gpudma_state_t*)malloc(statesize);
	if(!state) {
		goto do_free_attr;
	}
	memset(state, 0, statesize);
	state->handle = lock.handle;
	state->page_count = lock.page_count;
	res = ioctl(fd, IOCTL_GPUMEM_STATE, state);
	if(res < 0) {
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_UNLOCK\n");
		goto do_unlock;
	}

	fprintf(stderr, "Page count 0x%lx\n", state->page_count);
	fprintf(stderr, "Page size 0x%lx\n", state->page_size);

	for(unsigned i=0; i<state->page_count; i++) {
		fprintf(stderr, "%02d: 0x%lx\n", i, state->pages[i]);
		//void* va = mmap(0, state->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)state->pages[i]);
		void* va = mmap(0, state->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)state->pages[0]);
		if (va == MAP_FAILED ) {
			fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
			va = 0;
		} 
		else {
       		     	//memset(va, 0x55, state->page_size);
       		 	//unsigned *ptr=(unsigned*)va;
		 	//for( unsigned jj=0; jj<(state->page_size/4); jj++ ){
        		//	*ptr++=count++;
        		//}
	
			fprintf(stderr, "%s(): Physical Address 0x%lx -> Virtual Address %p\n", __FUNCTION__, state->pages[i], va);

			CUdeviceptr d_physAddr;
			struct physAddr h_physAddr;
			h_physAddr.dptrPhyAddrOnGPU = state->pages[i];
			h_physAddr.kernelID = 1234; // note: kernelID cannot be 0! otherwise the program will be blocked.
			checkError(cuMemAlloc(&d_physAddr,sizeof(struct physAddr)));
			checkError(cuMemcpyHtoD(d_physAddr, &h_physAddr, sizeof(struct physAddr)));
			
			printf("CPU side address = %p\n", state->pages[i]);
				
			// the virtual pointer that points to GPU global 
			// memory for the corresponding element
			struct AQentry* AQ = (struct AQentry*)(va);
			void* reqBufAddr = va + AQsize * sizeof(struct AQentry);
			void* inBuf = reqBufAddr + sizeof(struct reqBuf);
			void* outBuf = inBuf + MemBufferSize * m * n *sizeof(float);   
			struct reqBuf* requestBuffer = (struct reqBuf*) reqBufAddr;
			
				
			// set AQ head & AQ tail
			int AQhead = 0;
			int AQtail = MemBufferSize - 1;

			// initialize AQ
			for (int j=0; j<AQsize; j++){
				AQ[j].isInUse = 0;
				AQ[j].MemFreelistIdx = 0;
			}
					
			// copy data into input buffer
			for (int j=0; j<MemBufferSize; j++){
				// fill in the receive buffer
				float* a = (float*)(inBuf + m*n*j*sizeof(float));
				for (int k=0; k<m*n; k++) a[k] = j*1000+k;	
				
				// fill in the AQ entry
				AQ[j].isInUse = 1;
				AQ[j].MemFreelistIdx = j;			
			}

			// note: this p_flag is on FPGA(now: CPU side)
			// initialize doorbell register
			unsigned long* doorbell = (unsigned long*)p_flag;
			*doorbell = 0;

			// create Request Queue (FPGA side) 
			// and then initialize it
			struct RQentry RQ[RQsize];
			for (int j=0; j<RQsize; j++){
				RQ[j].MemFreelistIdx = 0;
				RQ[j].KernelID = 0;
			}
			int RQhead = 0;
			//int RQtail = MemBufferSize-1;
			int RQtail =0;
			int RQcursor = 0;

			// initialize RQ
			//for (int j=0; j<MemBufferSize; j++){
			//	RQ[j].MemFreelistIdx = j;
			//	RQ[j].KernelID = 1234;
			//}	


			// to create multiple threads
			pthread_t movingRQcursor;
			pthread_t movingRQhead;	

			// the parameters that need to pass to movingRQhead
			// and the parameters that need to pass to movingRQtail
			bool killThread = false;

			struct ParamForRQhead Phead;
			Phead.RQ = RQ;
			Phead.RQhead = &RQhead;
			Phead.RQtail = &RQtail;
			Phead.RQcursor = &RQcursor;
			Phead.GPUrecvBuf = (float*)inBuf;
			Phead.AQ = AQ;
			Phead.AQhead = &AQhead;
			Phead.AQtail = &AQtail;
			Phead.killThread = &killThread;

			struct ParamForRQcursor Pcursor;
			Pcursor.RQ = RQ;
			Pcursor.RQhead = &RQhead;
			Pcursor.RQtail = &RQtail;
			Pcursor.RQcursor = &RQcursor;
			Pcursor.GPUsendBuf = (float*)outBuf;
			Pcursor.AQ = AQ;
			Pcursor.AQhead = &AQhead;
			Pcursor.AQtail = &AQtail;
			Pcursor.killThread = &killThread;

			pthread_create(&movingRQcursor, NULL, f_movingRQcursor, (void*)&Pcursor);	
			pthread_create(&movingRQhead, NULL, f_movingRQhead, (void*)&Phead);

			

			// launch kernel
			void* args[3] = {&dptr, &CPUflag, &d_physAddr};
			checkError(cuLaunchKernel(function, m, n, 1, 16, 16, 1, 0, 0, args,0));
			printf("CPU: kernel launched!\n");
			
			int countNum = 0;

			auto start = std::chrono::high_resolution_clock::now();

			while(countNum<iterationNum){
				// waiting when there is no doorbell in

				pthread_mutex_lock(&printLock);
				printf("RQ TAIL: before doorbell\n");
				pthread_mutex_unlock(&printLock);

				while (!(*doorbell));
			
				pthread_mutex_lock(&printLock);
				printf("RQ TAIL: after doorbell\n");
				printf("RQ TAIL: before moving RQ tail\n");
				pthread_mutex_unlock(&printLock);


				// get information from GPU request buffer
				// according the address send by doorbell
				unsigned long outBufAddr = requestBuffer->outBufAddr;
				int idx = requestBuffer->idx;

				// push new request into RQ
				bool breakLoop = false;
				while (!breakLoop){

					pthread_mutex_lock(&RQlock);
					
					pthread_mutex_lock(&printLock);
					printf("RQ TAIL: move RQ tail\n");
					printf("RQ TAIL: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", RQhead, RQtail, RQcursor, AQhead, AQtail);					 
					pthread_mutex_unlock(&printLock);
					
					// 00000011111000000
					//       H   T
					if (RQtail>=RQhead){
						if (RQtail!=(RQsize-1)){
							RQtail++;
							RQ[RQtail].MemFreelistIdx = idx;
							RQ[RQtail].KernelID = 1234;
							breakLoop = true;	
							
						}
						else if (RQhead!=0){
							RQtail = 0;
							RQ[RQtail].MemFreelistIdx = idx;
							RQ[RQtail].KernelID = 1234;
							breakLoop = true;

						}
					}
					// 1110000000000001111
					//   T            H
					else{
						if ((RQtail+1)!=RQhead){
							RQtail++;
							RQ[RQtail].MemFreelistIdx = idx;
							RQ[RQtail].KernelID = 1234;
							breakLoop = true;	

						} 
					}
					pthread_mutex_unlock(&RQlock);
				}
				*doorbell = 0;
				requestBuffer->isInUse = 0;

				pthread_mutex_lock(&printLock);
				printf("RQ TAIL: after moving RQ tail\n");
				printf("RQ TAIL: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d\n", RQhead, RQtail, RQcursor, AQhead, AQtail);	
				pthread_mutex_unlock(&printLock);

				// clean up the request buffer entry on GPU
				// and the doorbell register on FPGA
					
				countNum++;
			}

			killThread = true;

			pthread_join(movingRQcursor, NULL);
			pthread_join(movingRQhead,NULL);

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> diff = end - start;
			std::cout<<"it took me "<<diff.count()<<" seconds."<<std::endl;
			cuCtxSynchronize();
			munmap(va, state->page_size);
		}
	}

	{
        	//const void* d_idata = (const void*)dptr;
	    	//cudaMemcpy(h_odata, d_idata, size, cudaMemcpyDeviceToHost);
    		//cudaDeviceSynchronize();

    		cuMemcpyDtoH( h_odata, dptr, size );
    		cuCtxSynchronize();

		void* head = h_odata + AQsize * sizeof(struct AQentry) + sizeof(struct reqBuf) + 2 * MemBufferSize*m*n*sizeof(float);
		float* floatHead = (float*)head;
		//float* h_odata_head = (float*)h_odata;
		
		//for (int i=0; i<3000; i++){
		//	printf("i=%d, %f\n", i, h_odata_head[i]);
		//}	
	
		for (int i= 0; i<1500; i++){
			//printf("i=%d, %f\n", i, floatHead[i]);
		}

    		unsigned *ptr = (unsigned*)h_odata;
    		unsigned val;
    		unsigned expect_data=0x0A000000;
    		unsigned cnt=size/4;
    		unsigned error_cnt=0;
	    	for( unsigned ii=0; ii<cnt; ii++ )
    		{
    			val=*ptr++;
	    		if(val!=expect_data){
    				error_cnt++;
    				if( error_cnt<32 )
    			 	fprintf(stderr, "%4d 0x%.8X - Error  expect: 0x%.8X\n", ii, val, expect_data );
	    		} 
			else if(ii<16){
				fprintf(stderr, "%4d 0x%.8X \n", ii, val );
			}
			expect_data++;

    		}
	    	if( 0==error_cnt ){
			fprintf(stderr, "\nTest successful\n" );
		} 
		else{
			fprintf(stderr, "\nTest with error\n" );
		}
	}


	fprintf(stderr, "Press enter to unlock\n");
	//getchar();

do_unlock:
	unlock.handle = lock.handle;
	res = ioctl(fd, IOCTL_GPUMEM_UNLOCK, &unlock);
	if(res < 0) {
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_UNLOCK\n");
		goto do_free_state;
	}
do_free_state:
	free(state);
do_free_attr:
	flag = 0;
	cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);

do_free_memory:
	cuMemFree(dptr);

do_free_context:
	cuCtxDestroy(context);

	close(fd);

	return 0;
}

// -------------------------------------------------------------------

void checkError(CUresult status)
{
    if(status != CUDA_SUCCESS) {
        const char *perrstr = 0;
        CUresult ok = cuGetErrorString(status,&perrstr);
        if(ok == CUDA_SUCCESS) {
            if(perrstr) {
                fprintf(stderr, "info: %s\n", perrstr);
            } else {
                fprintf(stderr, "info: unknown error\n");
            }
        }
        exit(0);
    }
}

//-----------------------------------------------------------------------------

bool wasError(CUresult status)
{
    if(status != CUDA_SUCCESS) {
        const char *perrstr = 0;
        CUresult ok = cuGetErrorString(status,&perrstr);
        if(ok == CUDA_SUCCESS) {
            if(perrstr) {
                fprintf(stderr, "info: %s\n", perrstr);
            } else {
                fprintf(stderr, "info: unknown error\n");
            }
        }
        return true;
    }
    return false;
}

//-----------------------------------------------------------------------------
