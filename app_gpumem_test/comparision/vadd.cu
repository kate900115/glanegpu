#include <stdio.h>
// for time measurement
#include <chrono>
#include <ctime>
#include <iostream>

__global__ void vecAdd(int m, int n, float* A,  float* C ){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
//	printf("A[%d][%d]\n",i,j);
	if ((i<m)&&(j<n)) {
		C[i*n+j] = A[i*n+j]/7;
	//	printf("A[%d][%d]=%f\n",i,j,A[i*n+j]);
	}
}

int main(){
	int m = 16;
	int n = 16;
	float* h_a = NULL;
	float* h_b = NULL;
	float* h_c = NULL;
	float* d_a = NULL;
	float* d_b = NULL;
	float* d_c = NULL;
	h_a = (float*)malloc(m*n*sizeof(float));
	h_b = (float*)malloc(m*n*sizeof(float));
	h_c = (float*)malloc(m*n*sizeof(float));

	cudaMalloc((void**)&d_a, m*n*sizeof(float));
	cudaMalloc((void**)&d_b, m*n*sizeof(float));
	cudaMalloc((void**)&d_c, m*n*sizeof(float));

	if ((h_a==NULL)||(h_b==NULL)||(h_c==NULL)||(d_a==NULL)||(d_b==NULL)||(d_c==NULL)){
		printf("cannot allocate memory.\n");
	}
	
	//memset(h_c,0,m*n*sizeof(float));
	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			h_a[i*n+j]=i+j;
			h_b[i*n+j]=i+j;
			h_c[i*n+j]=0;
		//	printf("%f,%f,%f\n",h_a[i*n+j],h_b[i*n+j],h_c[i*n+j]);
		}
	}

	int count = 0;
	dim3 grid((n+15)/16, (m+15)/16,1);
	dim3 block(16, 16,1);
	
	auto start = std::chrono::high_resolution_clock::now();
	while (count<100){	
		cudaMemcpy(d_a, h_a, m*n*sizeof(float), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_b, h_b, m*n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_c, h_c, m*n*sizeof(float), cudaMemcpyHostToDevice);

		vecAdd<<<grid, block>>>	(m, n, d_a, d_c);
		cudaMemcpy(h_c, d_c, m*n*sizeof(float), cudaMemcpyDeviceToHost);
		count++;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout<<"it took me "<<diff.count()<<" seconds."<<std::endl;
	
	return 0;		
}
