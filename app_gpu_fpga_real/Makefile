 
PHONY = clean all
TARGET_NAME = gpu_direct

all: $(TARGET_NAME)

ROOT_DIR = $(shell pwd)

CC = $(CROSS_COMPILE)gcc
LD = $(CROSS_COMPILE)gcc

NVIDIA_DRIVER_PATH := ../nvidia

CFLAGS := -D__LINUX__ -g -Wall -std=c++11 -I /usr/local/cuda-9.1/include -I /home/zyuxuan/gpudma/gpumem -I /home/zyuxuan/gpudma/v2p2v

SRCFILE := $(wildcard *.cpp) 
OBJFILE := $(patsubst %.cpp,%.o, $(SRCFILE))


LDFLAGS +=  -L$(CUDADIR)/lib64 -lcuda -lpthread
LDFLAGS +=  -lstdc++

$(TARGET_NAME): $(OBJFILE)
	$(LD) $(notdir $^) -o $(TARGET_NAME) $(LDFLAGS)
	cp $(TARGET_NAME) ../bin

%.o: %.cpp
	$(CC) $(CFLAGS) -c -MD $<
	
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
