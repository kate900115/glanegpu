#include <stdio.h>

// for time measurement
#include <chrono>
#include <ctime>
#include <iostream>

__global__ void mult(float* a, float* b, float* c, int N, int BLOCKSIZE){
	__shared__ float A[8][8];
	__shared__ float B[8][8];
	
	int globalIdx_x = blockIdx.x*blockDim.x+threadIdx.x;
	int globalIdx_y = blockIdx.y*blockDim.y+threadIdx.y;
		
	float result =0;
	for (int k=0; k<N/BLOCKSIZE; k++){
		A[threadIdx.y][threadIdx.x] = a[globalIdx_y*N+k*BLOCKSIZE + threadIdx.x];
		B[threadIdx.y][threadIdx.x] = b[(k*BLOCKSIZE+threadIdx.y)*N + globalIdx_x];
		__syncthreads();
	
		for (int i=0; i<BLOCKSIZE; i++){
			result += A[threadIdx.y][i]*B[i][threadIdx.x];
		}
		
	}
	c[globalIdx_y*N+globalIdx_x]+=result;

}

int main(){
	float* h_a = NULL;
	float* h_b = NULL;
	float* h_c = NULL;
	float* d_a = NULL;
	float* d_b = NULL;
	float* d_c = NULL;

	int N = 64;
	int BLOCKSIZE = 16;

	h_a = (float*)malloc(N*N*sizeof(float));
	h_b = (float*)malloc(N*N*sizeof(float));
	h_c = (float*)malloc(N*N*sizeof(float));
	cudaMalloc((void**)&d_a, N*N*sizeof(float));
	cudaMalloc((void**)&d_b, N*N*sizeof(float));
	cudaMalloc((void**)&d_c, N*N*sizeof(float));

	if ((h_a==NULL)&&(h_b==NULL)&&(h_c==NULL)&&(d_a==NULL)&&(d_b==NULL)&&(d_c==NULL)){
		printf("cannot allocate memory.\n");
	}

	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			h_a[i*N+j]=i+j;
			h_b[i*N+j]=i/(j+1);
			h_c[i*N+j]=0;
	//		printf("%f\n",h_a[i*N+j]);
		}
	}
	dim3 grid((N+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE,1);
	dim3 block(BLOCKSIZE, BLOCKSIZE, 1);
	
	int count = 0;

	auto start = std::chrono::high_resolution_clock::now();

	while (count<100){
		cudaMemcpy(d_a, h_a, N*N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, N*N*sizeof(float), cudaMemcpyHostToDevice);
	
		mult<<<grid, block>>>(d_a, d_b, d_c, N, BLOCKSIZE);

		cudaMemcpy(h_c, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		count++;
	}

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> diff = end - start;
	std::cout<<"It took me "<<diff.count()<<" seconds."<<std::endl;

	for(int i=0; i<N; i++){
		for (int j=0; j<N; j++){
//			printf("c[%d][%d]=%f\n",i,j,h_c[i*N+j]);
		}
	}	
	return 0;
}
