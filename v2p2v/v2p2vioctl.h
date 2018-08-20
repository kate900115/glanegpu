
#ifndef __V2P2VIOTCL_H__
#define __V2P2VIOTCL_H__

//-----------------------------------------------------------------------------

#define V2P2V_DRIVER_NAME             "v2p2v"


//-----------------------------------------------------------------------------

#ifdef __linux__
#include <linux/types.h>
#ifndef __KERNEL__
#include <sys/ioctl.h>
#endif
#define GPUMEM_DEVICE_TYPE             'g'
#define GPUMEM_MAKE_IOCTL(c) _IO(GPUMEM_DEVICE_TYPE, (c))
#endif

#define IOCTL_V2P		GPUMEM_MAKE_IOCTL(10)
#define IOCTL_P2V		GPUMEM_MAKE_IOCTL(11)
//#define IOCTL_GPUMEM_STATE		GPUMEM_MAKE_IOCTL(12)

//-----------------------------------------------------------------------------
// for boundary alignment requirement
#define GPU_BOUND_SHIFT 16
#define GPU_BOUND_SIZE ((u64)1 << GPU_BOUND_SHIFT)
#define GPU_BOUND_OFFSET (GPU_BOUND_SIZE-1)
#define GPU_BOUND_MASK (~GPU_BOUND_OFFSET)


//zyuxuan
struct cpuaddr_t {
	uint64_t paddr;
};

//-----------------------------------------------------------------------------


#endif //_GPUDMAIOTCL_H_
