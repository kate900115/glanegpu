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
	int* AQcursor;

	//for control
	bool* killThread;
	int* startSignal;
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
	int* AQcursor;	

	//for control
	bool* killThread;
	int* startSignal;
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
	int* AQcursor = param->AQcursor;
	float* GPUsendBufBase = param->GPUsendBuf;
	float* GPUsendBuf = GPUsendBufBase;
	int* startSignal = param->startSignal;

	// output destination
	float a[m*n];
	bool cursorValid = true;

	bool* killThread = param->killThread;
	bool workFinish = false;
	
	//waiting until GPU kernel is launched and intialized
	while (*startSignal!=1);

	while(!(workFinish)){

		if (*killThread){
			//printf("@@@@@@@@@@@@@ RQ cursor: kill thread = 1\n");
			if(*RQtail == *RQcursor) {
				workFinish=true;
				//printf("@@@@@@@@@@@@ the thread is being killed\n");
				#ifdef DEBUG
				pthread_mutex_lock(&printLock);
				printf("RQ CURSOR: the thread is being killed\n");
				printf("RQtail = %d, RQcursor = %d, RQhead = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", *RQtail, *RQcursor, *RQhead, *AQhead, *AQtail, *AQcursor);
				pthread_mutex_unlock(&printLock);
				#endif
				break;
			}
		}

		// copy data out of GPU send buffer
		#ifdef DEBUG
		pthread_mutex_lock(&printLock);
		printf("RQ CURSOR: copy data out of send buffer\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail, *AQcursor);	
		pthread_mutex_unlock(&printLock);
		#endif

		#ifdef DATA_TRANSFER
		if (cursorValid){
			for (int i=0; i<m*n; i++){
				a[i] = GPUsendBuf[i];
			}
			cursorValid = false;
		}
		#endif
	
		#ifdef DEBUG
		pthread_mutex_lock(&printLock);
		printf("RQ CURSOR: before moving AQ head\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail, *AQcursor);	
		pthread_mutex_unlock(&printLock);
		#endif

		// move AQ head
		bool breakLoop = false;
		bool breakRQLoop = false;
		while(!breakLoop){
			pthread_mutex_lock(&AQlock);

			#ifdef DEBUG
			pthread_mutex_lock(&printLock);
			printf("RQ CURSOR: moving AQ head\n");
			printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d, AQ[cursor+1].isInUse = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail, *AQcursor, AQ[(*AQcursor)+1].isInUse);	
			printf("AQ freelist:\n");
			for (int u=0; u<AQsize; u++){
				printf("%d",AQ[u].isInUse);
			}
			printf("\n");
			pthread_mutex_unlock(&printLock);
			#endif

			// 000000001111111100000
			//         H  C   T
			if ((*AQhead<*AQtail)&&(*AQhead<*AQcursor)){
				breakLoop = true;
				AQ[*AQhead].isInUse = 0;
				AQ[*AQhead].MemFreelistIdx = 0;
				(*AQhead)++;
			}
			// 11110000000011111
			//    T        H C
			else if ((*AQhead>*AQtail)&&(*AQhead<*AQcursor)){	
				breakLoop = true;
				AQ[*AQhead].isInUse = 0;
				AQ[*AQhead].MemFreelistIdx = 0;
				(*AQhead)++;	
			}
			else if ((*AQhead>*AQtail)&&(*AQcursor<=*AQtail)){
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

			/////////////////////////////////////
			else if (*AQhead == *AQcursor){	   //
				breakLoop = true;	   //
				breakRQLoop = true;        //
			}				   //
			/////////////////////////////////////
			pthread_mutex_unlock(&AQlock);
		}	

		// move RQ cursor
		#ifdef DEBUG	
		pthread_mutex_lock(&printLock);
		printf("RQ CURSOR: after moving AQ head\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail, *AQcursor);	
		printf("RQ CURSOR: before moving RQ cursor\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail, *AQcursor);	
		pthread_mutex_unlock(&printLock);
		#endif
		breakLoop = false;
		while(!breakLoop){
		//	if (breakRQLoop) break;
			pthread_mutex_lock(&RQlock);
			if (*RQhead<=*RQtail){
				// 0000111111110000
				//     H   C  T
				if (*RQcursor<*RQtail) {
					(*RQcursor)++;
					GPUsendBuf = GPUsendBufBase+m*n*RQ[*RQcursor].MemFreelistIdx;
					cursorValid = true;
					breakLoop = true;
				}	
			}
			else{
				// 1110000000111111
				//   T       H  C
				if ((*RQcursor<RQsize-1)&&(*RQcursor<*RQtail)){
					(*RQcursor)++;
					GPUsendBuf = GPUsendBufBase+m*n*RQ[*RQcursor].MemFreelistIdx;
					cursorValid = true;
					breakLoop = true;
				}
				else if (*RQcursor==RQsize-1){
					*RQcursor = 0;
					GPUsendBuf = GPUsendBufBase+m*n*RQ[*RQcursor].MemFreelistIdx;
					cursorValid = true;
					breakLoop = true;
				}	
			}
			pthread_mutex_unlock(&RQlock);	
		}
		#ifdef DEBUG
		pthread_mutex_lock(&printLock);
		printf("RQ CURSOR: after moving RQ cursor\n");
		printf("RQ CURSOR: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail, *AQcursor);	
		pthread_mutex_unlock(&printLock);
		#endif

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
	int* AQcursor = param->AQcursor;
	float* GPUrecvBufBase = param->GPUrecvBuf;
	float* GPUrecvBuf = GPUrecvBufBase;

	int tmpMemFreelistIdx; 
	//
	bool workFinish =  false;
	bool* killThread = param->killThread;
	
	int* startSignal = param->startSignal;

	while (*startSignal!=1);

	while (!(workFinish)){

		if (*killThread){
			if((*RQtail == *RQhead)&&(*RQtail == *RQcursor)) {
				workFinish=true;
				#ifdef DEBUG
				pthread_mutex_lock(&printLock);
				printf("RQ HEAD: the thread is being killed\n");
				pthread_mutex_unlock(&printLock);
				#endif
				break;
			}
		}


		// if head pointer is valid,
		// copy data into GPU receive buffer
		#ifdef DEBUG	
		pthread_mutex_lock(&printLock);
		printf("RQ HEAD: before moving RQ head\n");
		printf("RQ HEAD: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail, *AQcursor);	
		pthread_mutex_unlock(&printLock);
		#endif

		bool breakLoop = false;
		while(!(breakLoop)){
			pthread_mutex_lock(&RQlock);
			if (*RQhead<=*RQtail){
				//00001111110000
				//    H  C T
				if ((*RQcursor>*RQhead)&&(*RQcursor<=*RQtail)){
					// changing the receive buffer address
					GPUrecvBuf = GPUrecvBufBase + m * n * RQ[*RQhead].MemFreelistIdx; 
					tmpMemFreelistIdx = RQ[*RQhead].MemFreelistIdx;
					(*RQhead)++;
					breakLoop = true;
				}
			}
			else {
				//11100000011111
				//  T      H  C
				if ((*RQcursor>*RQhead)&&(*RQcursor<=RQsize-1)){
					// changing the receive buffer address
					GPUrecvBuf = GPUrecvBufBase + m * n * RQ[*RQhead].MemFreelistIdx; 
					tmpMemFreelistIdx = RQ[*RQhead].MemFreelistIdx;
					(*RQhead)++;
					breakLoop = true;
				}
				//11100000011111
				//C T      H    
				else if (*RQcursor<=*RQtail){
					if (*RQhead!=RQsize-1){
						// changing the receive buffer address
						GPUrecvBuf = GPUrecvBufBase + m * n * RQ[*RQhead].MemFreelistIdx; 
						tmpMemFreelistIdx = RQ[*RQhead].MemFreelistIdx;
						(*RQhead)++;
						breakLoop = true;
					}
					else{
						// changing the receive buffer address
						GPUrecvBuf = GPUrecvBufBase + m * n * RQ[*RQhead].MemFreelistIdx; 
						tmpMemFreelistIdx = RQ[*RQhead].MemFreelistIdx;
						*RQhead = 0;
						breakLoop = true;

					}
				}
		
			}
			pthread_mutex_unlock(&RQlock);
		}


		#ifdef DEBUG
		pthread_mutex_lock(&printLock);
		printf("RQ HEAD: after moving RQ head\n");
		printf("RQ HEAD: RQhead = %d\n", *RQhead);	
		printf("RQ HEAD: copy data into GPU receive buffer\n");	
		pthread_mutex_unlock(&printLock);
		#endif

		#ifdef DATA_TRANSFER
		for (int i=0; i<m*n; i++){
			GPUrecvBuf[i] = i/17;
		}
		#endif

		// move AQ pointer
		breakLoop = false;
		while (!breakLoop){
			pthread_mutex_lock(&AQlock);
			//000000000111111
			//         H    T
			
			#ifdef DEBUG
			pthread_mutex_lock(&printLock);
			printf("RQ HEAD: before moving AQ tail\n");
			printf("RQ HEAD: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail, *AQcursor);	
			pthread_mutex_unlock(&printLock);
			#endif

			if (*AQtail>=*AQhead){
				if (*AQtail==(AQsize-1)){
					if (*AQhead!=0){
						*AQtail = 0;
						AQ[*AQtail].isInUse = 1;
						AQ[*AQtail].MemFreelistIdx = tmpMemFreelistIdx;	
						breakLoop = true;
					}
				}
				else{
					(*AQtail)++;
					AQ[*AQtail].isInUse = 1;
					AQ[*AQtail].MemFreelistIdx = tmpMemFreelistIdx;
					breakLoop = true;
				}
			}
			//11111000111111
			//    T   H
			else{

				if (*AQhead!=((*AQtail)+1)){
					(*AQtail)++;
					AQ[*AQtail].isInUse = 1;
					AQ[*AQtail].MemFreelistIdx = tmpMemFreelistIdx;
					breakLoop = true;	
				}
			}
			
			#ifdef DEBUG
			pthread_mutex_lock(&printLock);
			printf("RQ HEAD: after moving AQ tail\n");
			printf("RQ HEAD: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", *RQhead, *RQtail, *RQcursor, *AQhead, *AQtail, *AQcursor);	
			pthread_mutex_unlock(&printLock);
			#endif
	

			pthread_mutex_unlock(&AQlock);
		}


	

	}		
	#ifdef DEBUG
	pthread_mutex_lock(&printLock);
	printf("RQ HEAD: I'm not kidding you.\n");
	pthread_mutex_unlock(&printLock);
	#endif
}

// zyuxuan: struct for pthread parameter
// zyuxuan: function for the thread

int main(int argc, char *argv[])
{
	gpudma_lock_t lock;
	gpudma_unlock_t unlock;
	gpudma_state_t *state = 0;

	gpudma_lock_t lock1;
	gpudma_unlock_t unlock1;
	gpudma_state_t *state1 = 0;

	gpudma_lock_t lock2;
	gpudma_unlock_t unlock2;
	gpudma_state_t *state2 = 0;



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
	p = (int*)malloc(sizeof(int));
	checkError(cuMemHostRegister(p, sizeof(int), CU_MEMHOSTREGISTER_DEVICEMAP));
	CUdeviceptr CPUflag;
	checkError(cuMemHostGetDevicePointer(&CPUflag, p,0));

	int* p_flag = (int*) p;

	// we also need a AQ cursor on CPU side
	// each time AQ cursor updates, GPU also 
	// notifies to CPU
	void* CPUsideAQcursor;
	CPUsideAQcursor= (int*)malloc(sizeof(int));
	checkError(cuMemHostRegister(CPUsideAQcursor,sizeof(int), CU_MEMHOSTREGISTER_DEVICEMAP));
	CUdeviceptr CPU_AQcursor;
	checkError(cuMemHostGetDevicePointer(&CPU_AQcursor, CPUsideAQcursor,0));

	int* CPUside_AQcursor = (int*)CPUsideAQcursor;

	void* start_signal;
	start_signal = (int*)malloc(sizeof(int));
	checkError(cuMemHostRegister(start_signal,sizeof(int), CU_MEMHOSTREGISTER_DEVICEMAP));
	CUdeviceptr startSignal;
	checkError(cuMemHostGetDevicePointer(&startSignal, start_signal, 0));

	int* StartSignal = (int*)start_signal;
	*StartSignal = 0;
	


	size_t size = 0x100000;
	CUdeviceptr dptr = 0;
	CUdeviceptr dptr1 = 0;
	CUdeviceptr dptr2 = 0;

	unsigned int flag = 1;
	unsigned char *h_odata = NULL;
	h_odata = (unsigned char *)malloc(size);

	CUresult status = cuMemAlloc(&dptr, size);


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



	/////////////////////////////////////////////////

	status = cuMemAlloc(&dptr1, size);

	if(wasError(status)) {
        	goto do_free_context;	
	}

	fprintf(stderr, "Allocate memory address: 0x%llx\n",  (unsigned long long)dptr1);

	status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr1);
	if(wasError(status)) {
		goto do_free_memory;
	}

	fprintf(stderr, "Press enter to lock\n");
	//getchar();

	// TODO: add kernel driver interaction...
	lock1.addr = dptr1;
	lock1.size = size;
	res = ioctl(fd, IOCTL_GPUMEM_LOCK, &lock1);
	if(res < 0) {
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		goto do_free_attr;
	}

	fprintf(stderr, "Press enter to get state. We lock %ld pages\n", lock1.page_count);
	//getchar();

	statesize = (lock1.page_count*sizeof(uint64_t) + sizeof(struct gpudma_state_t));
	state1 = (struct gpudma_state_t*)malloc(statesize);
	if(!state1) {
		goto do_free_attr;
	}
	memset(state1, 0, statesize);
	state1->handle = lock1.handle;
	state1->page_count = lock1.page_count;
	res = ioctl(fd, IOCTL_GPUMEM_STATE, state1);
	if(res < 0) {
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_UNLOCK\n");
		goto do_unlock;
	}

	fprintf(stderr, "Page count 0x%lx\n", state1->page_count);
	fprintf(stderr, "Page size 0x%lx\n", state1->page_size);






	//////////////////////////////////////////
	status = cuMemAlloc(&dptr2, size);

	if(wasError(status)) {
        	goto do_free_context;	
	}

	fprintf(stderr, "Allocate memory address: 0x%llx\n",  (unsigned long long)dptr2);

	status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr2);
	if(wasError(status)) {
		goto do_free_memory;
	}

	fprintf(stderr, "Press enter to lock\n");
	//getchar();

	// TODO: add kernel driver interaction...
	lock1.addr = dptr2;
	lock1.size = size;
	res = ioctl(fd, IOCTL_GPUMEM_LOCK, &lock1);
	if(res < 0) {
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		goto do_free_attr;
	}

	fprintf(stderr, "Press enter to get state. We lock %ld pages\n", lock1.page_count);
	//getchar();

	statesize = (lock1.page_count*sizeof(uint64_t) + sizeof(struct gpudma_state_t));
	state2 = (struct gpudma_state_t*)malloc(statesize);
	if(!state2) {
		goto do_free_attr;
	}
	memset(state2, 0, statesize);
	state2->handle = lock1.handle;
	state2->page_count = lock1.page_count;
	res = ioctl(fd, IOCTL_GPUMEM_STATE, state2);
	if(res < 0) {
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_UNLOCK\n");
		goto do_unlock;
	}

	fprintf(stderr, "Page count 0x%lx\n", state2->page_count);
	fprintf(stderr, "Page size 0x%lx\n", state2->page_size);










//	for(unsigned i=0; i<state->page_count; i++) {
	for(unsigned i=0; i<1; i++) {
	
		fprintf(stderr, "%02d: 0x%lx\n", i, state->pages[i]);
		//void* va0 = mmap(0, state->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)state->pages[i]);
		void* va0 = mmap(0, state->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)state->pages[0]);
		void* va1 = mmap(0, state1->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)state1->pages[0]);
		void* va2 = mmap(0, state2->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)state2->pages[0]);
	

		if (va0 == MAP_FAILED ) {
			fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
			va0 = 0;
		}
		else if (va1 == MAP_FAILED){
			fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
			va1 = 0;
		} 
		else {
       		     	//memset(va, 0x55, state->page_size);
       		 	//unsigned *ptr=(unsigned*)va;
		 	//for( unsigned jj=0; jj<(state->page_size/4); jj++ ){
        		//	*ptr++=count++;
        		//}
	
			fprintf(stderr, "%s(): Physical Address 0x%lx -> Virtual Address %p\n", __FUNCTION__, state->pages[i], va0);

			CUdeviceptr d_physAddr;
			struct physAddr h_physAddr;
			h_physAddr.dptrPhyAddrOnGPU = state->pages[i];
			h_physAddr.kernelID = 1234; // note: kernelID cannot be 0! otherwise the program will be blocked.
			checkError(cuMemAlloc(&d_physAddr,sizeof(struct physAddr)));
			checkError(cuMemcpyHtoD(d_physAddr, &h_physAddr, sizeof(struct physAddr)));
			
			printf("CPU side address = %p\n", state->pages[i]);
				
			// the virtual pointer that points to GPU global 
			// memory for the corresponding element
			struct AQentry* AQ = (struct AQentry*)(va0);
			void* reqBufAddr = va0 + AQsize * sizeof(struct AQentry);
			void* inBuf = va1; 
			void* outBuf = va2;   
			struct reqBuf* requestBuffer = (struct reqBuf*) reqBufAddr;
			
				
			// set AQ head & AQ tail
			int AQhead = 0;
			int AQtail = MemBufferSize - 1;
			//CPUside_AQcursor = 0;
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
			int* doorbell = (int*)p_flag;
			*doorbell = 0;

			// create Request Queue (FPGA side) 
			// and then initialize it
			struct RQentry RQ[RQsize];
			for (int j=0; j<RQsize; j++){
				RQ[j].MemFreelistIdx = 0;
				RQ[j].KernelID = 0;
			}
			int RQhead = 0;
			int RQtail =0;
			int RQcursor = 0;

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
			Phead.AQcursor = CPUside_AQcursor;
			Phead.killThread = &killThread;
			Phead.startSignal = StartSignal; 
	
			struct ParamForRQcursor Pcursor;
			Pcursor.RQ = RQ;
			Pcursor.RQhead = &RQhead;
			Pcursor.RQtail = &RQtail;
			Pcursor.RQcursor = &RQcursor;
			Pcursor.GPUsendBuf = (float*)outBuf;
			Pcursor.AQ = AQ;
			Pcursor.AQhead = &AQhead;
			Pcursor.AQtail = &AQtail;
			Pcursor.AQcursor = CPUside_AQcursor;
			Pcursor.killThread = &killThread;
			Pcursor.startSignal = StartSignal;

			pthread_create(&movingRQcursor, NULL, f_movingRQcursor, (void*)&Pcursor);	
			pthread_create(&movingRQhead, NULL, f_movingRQhead, (void*)&Phead);

			

			// launch kernel
			void* args[7] = {&dptr, &dptr1, &dptr2, &CPUflag, &d_physAddr, &CPU_AQcursor, &startSignal};
//			checkError(cuLaunchKernel(function, m, n, 1, 16, 16, 1, 0, 0, args,0));
			checkError(cuLaunchKernel(function, m/16, n/16, 1, 16, 16, 1, 0, 0, args,0));

			printf("CPU: kernel launched!\n");
			
			int countNum = 0;

			auto start = std::chrono::high_resolution_clock::now();

			while(countNum<iterationNum){
				// waiting when there is no doorbell in

//				printf("@@@ count = %d\n", countNum);
				#ifdef DEBUG
				pthread_mutex_lock(&printLock);
				printf("RQ TAIL: before doorbell\n");
				printf("RQ TAIL: this is the %d times of iteration\n", countNum);
				printf("RQ TAIL: GPU request buf isInUse: %d\n", requestBuffer->isInUse);
				pthread_mutex_unlock(&printLock);
				#endif

				while (!(*doorbell));
			
				#ifdef DEBUG
				pthread_mutex_lock(&printLock);
				printf("RQ TAIL: after doorbell\n");
				printf("RQ TAIL: before moving RQ tail\n");
				pthread_mutex_unlock(&printLock);
				#endif

				// get information from GPU request buffer
				// according the address send by doorbell
				unsigned long outBufAddr = requestBuffer->outBufAddr;
				int idx = requestBuffer->idx;

				// push new request into RQ
				bool breakLoop = false;
				while (!breakLoop){

					pthread_mutex_lock(&RQlock);
				
					#ifdef DEBUG	
					pthread_mutex_lock(&printLock);
					printf("RQ TAIL: move RQ tail\n");
					printf("RQ TAIL: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", RQhead, RQtail, RQcursor, AQhead, AQtail, *CPUside_AQcursor);					 
					pthread_mutex_unlock(&printLock);
					#endif

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

				#ifdef DEBUG
				pthread_mutex_lock(&printLock);
				printf("RQ TAIL: after moving RQ tail\n");
				printf("RQ TAIL: RQhead = %d, RQtail = %d, RQcursor = %d, AQhead = %d, AQtail = %d, AQcursor = %d\n", RQhead, RQtail, RQcursor, AQhead, AQtail, *CPUside_AQcursor);	
				pthread_mutex_unlock(&printLock);
				#endif
			
				// clean up the request buffer entry on GPU
				// and the doorbell register on FPGA
					
				countNum++;
			}

			killThread = true;
//			printf("sync 0 finishes\n");
			pthread_join(movingRQcursor, NULL);
//			printf("sync 1 finishes\n");
			pthread_join(movingRQhead,NULL);

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> diff = end - start;
			std::cout<<"it took me "<<diff.count()<<" seconds."<<std::endl;
			cuCtxSynchronize();
			munmap(va0, state->page_size);
		}
	}

	{
        	//const void* d_idata = (const void*)dptr;
	    	//cudaMemcpy(h_odata, d_idata, size, cudaMemcpyDeviceToHost);
    		//cudaDeviceSynchronize();

    		cuMemcpyDtoH( h_odata, dptr, size );
    		cuCtxSynchronize();

//		void* head = h_odata + AQsize * sizeof(struct AQentry) + sizeof(struct reqBuf) + 2 * MemBufferSize*m*n*sizeof(float);
//		float* floatHead = (float*)head;
		//float* h_odata_head = (float*)h_odata;
		
		//for (int i=0; i<3000; i++){
		//	printf("i=%d, %f\n", i, h_odata_head[i]);
		//}	
	
//		for (int i= 0; i<1500; i++){
			//printf("i=%d, %f\n", i, floatHead[i]);
//		}

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
