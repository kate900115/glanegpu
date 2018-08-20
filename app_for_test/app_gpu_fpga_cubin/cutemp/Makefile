all:
#	nvcc -gencode=arch=compute_60,code=sm_60 -rdc=true -cubin vadd.cu
	nvcc -arch=sm_60 -ptx vadd.cu
#	nvcc -cubin vadd.cu
#	nvcc -ptx vadd.cu
clean:
	rm *.ptx
