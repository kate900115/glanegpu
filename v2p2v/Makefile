
KERNELVER := $(shell uname -r)

EXTRA_CFLAGS += -fno-stack-protector 
EXTRA_CFLAGS += -O2

ccflags-y += $(EXTRA_CFLAGS)

ifneq ($(KERNELRELEASE),)

obj-m += v2p2v.o
v2p2v-objs := v2p2vproc.o ioctlrw.o v2p2vdrv.o

else

endif

KERNELDIR ?= /lib/modules/$(shell uname -r)/build


#PWD := $(shell pwd)
#PWD := $(PWD)



all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules



distclean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions *.bak .*.cache *.d
clean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions *.bak .*.cache *.d *.markers *.symvers *.order
