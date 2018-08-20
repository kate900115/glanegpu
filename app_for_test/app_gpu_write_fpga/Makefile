 
PHONY = clean all
TARGET_NAME = gpu_direct

all: $(TARGET_NAME)

ROOT_DIR = $(shell pwd)

#CC = $(CROSS_COMPILE)gcc
CC = $(CROSS_COMPILE)gcc
LD = $(CROSS_COMPILE)gcc

#NVIDIA_DRIVER_PATH := $(HOME)/gpudma/nvidia
NVIDIA_DRIVER_PATH := ../nvidia


#INCDIR := . $(CUDADIR)/include ../module $(NVIDIA_DRIVER_PATH)/kernel/nvidia
#INCLUDE := $(addprefix -I, $(INCDIR))
#CFLAGS := -D__LINUX__ -g -Wall $(INCLUDE) 
CFLAGS := -D__LINUX__ -g -Wall -std=c++11 -I /usr/local/cuda-9.1/include -I /home/zyuxuan/gpudma/gpumem -I /home/zyuxuan/gpudma/v2p2v
#CFLAGS := -D__LINUX__ -g -I /usr/local/cuda-9.0/targets/x86_64-linux/include -I ~/zyuxuan/gpudma/module -std=c++11





#SRCFILE := $(wildcard *.cpp) 
#OBJFILE := $(patsubst %.cpp,%.o, $(SRCFILE))
SRCFILE := $(wildcard *.cpp) 
OBJFILE := $(patsubst %.cpp,%.o, $(SRCFILE))


#LDFLAGS := -Wl,-rpath,$(CUDADIR)/lib64/stubs -L"$(CUDADIR)/lib64/stubs" -lcuda
#LDFLAGS :=  -L/usr/local/cuda/lib64 -lcudart -L/usr/local/cuda/lib64/stubs -lcuda $(LIBS)
LDFLAGS +=  -L$(CUDADIR)/lib64 -lcuda -lpthread
#LDFLAGS +=  -L$(CUDADIR)/lib64 -lcudart
LDFLAGS +=  -lstdc++

$(TARGET_NAME): $(OBJFILE)
	$(LD) $(notdir $^) -o $(TARGET_NAME) $(LDFLAGS)
	cp $(TARGET_NAME) ../bin

%.o: %.cpp
	$(CC) $(CFLAGS) -c -MD $<
#	$(CC) $(CFLAGS) -c $<
	
include $(wildcard *.d)


clean:
	rm -f *.o *~ core
	rm -f *.d *~ core
	rm -f $(TARGET_NAME)
	
distclean:
	rm -f *.o *~ core
	rm -f *.d *~ core
	rm -f $(TARGET_NAME)

src:
	@echo $(SRCFILE)
	@echo $(OBJFILE)
