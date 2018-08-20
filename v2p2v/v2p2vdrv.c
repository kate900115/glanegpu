
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/version.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/ioport.h>
#include <linux/list.h>
#include <linux/pci.h>
#include <linux/proc_fs.h>
#include <linux/interrupt.h>
#include <linux/miscdevice.h>
#include <linux/platform_device.h>
//#include <linux/of.h>
//#include <linux/of_platform.h>
#include <asm/io.h>

#include <asm/uaccess.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/interrupt.h>

#include "v2p2vdrv.h"
#include "ioctlrw.h"
#include "v2p2vioctl.h"
#include "v2p2vproc.h"
#include <linux/memremap.h>
#include <linux/delay.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/errno.h>
#include <linux/ioctl.h>
#include <linux/cdev.h>


MODULE_AUTHOR("Yuxuan Zhang. v-yuxz@microsoft.com");
MODULE_LICENSE("GPL");

// This module has the following two functionalities:		//
// 	a) p2v: user calls ioctl() to pass physical addr 	//
// 		to kernel, and then calls mmap() to get 	//
// 		the corresponding virtual address.		//
// 	b) v2p: user calls mmap() to allocate a memory 		//
// 	        location with a virtual pointer. And then user  //
// 	        calls ioctl() to get the physical address.	//

static struct gpumem dev;

static struct gpumem *file_to_device( struct file *file ){
    return (struct gpumem*)file->private_data;
}

// file operations includes:					//
// (1) file open: 						//
//     static int v2p2v_open(struct inode*, struct file*)	//
// (2) file close: 						//
//     static int v2p2v_close(struct inode*, struct file*)	//
// (3) I/O control:						//
//     static long v2p2v_ioctl(struct file*, unsigned int,	//
//     unsigned long)						//
// (4) mmap: 							//
//     int v2p2v_mmap(struct file*, vm_struct_area*)		//
//  								//
//  these 4 operations will be registered into kernel. 		//

static int v2p2v_open( struct inode *inode, struct file *file ){
    file->private_data = (void*)&dev;
    return 0;
}

static int v2p2v_close( struct inode *inode, struct file *file ){
    file->private_data = 0;
    return 0;
}

static long v2p2v_ioctl( struct file *file, unsigned int cmd, unsigned long arg )
{
	// cmd has two operations: (1) IOCTL_V2P & (2) IOCTL_P2V //
	// (1) IOCTL_V2P: convert virtual address to physical 	 //
	//     address call ioctl_v2p_convert() and return error //
	//     code.						 //
	// (2) IOCTL_P2V: convert physical address to virtual 	 //
	//     address call ioctl_p2v_convert() and return error //
	//     code.						 //

	int error = 0;
	if (cmd==IOCTL_V2P) {
		#ifdef DEBUG
		pr_info("[gpumem_ioctl] The ioctl_v2p_convert(arg) is called.\n");
		#endif
		error = ioctl_v2p_convert(arg);
	}	
	if (cmd==IOCTL_P2V){
		#ifdef DEBUG
		pr_info("[gpumem_ioctl] The ioctl_p2v_convert(arg) is called.\n");
		#endif
		error = ioctl_p2v_convert(arg);
	}
	return error;
}


int v2p2v_mmap(struct file *file, struct vm_area_struct *vma)
{
	// When mmap() is called with the argument fd = v2p2v,	//
	// v2p2v_mmap will be called. OS will pass all argument //
	// to *vma.						//
	//							//
	// (1) set size for remap_pfn_range()			//
	// (2) set page property to be non-cached		//
	// (3) read saved operation type "savedAddr.op"		//
	size_t size = vma->vm_end - vma->vm_start;
	if (!(vma->vm_flags & VM_MAYSHARE)){
		return -EINVAL;
	}	
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

	struct savedAddress savedAddr;
	savedAddr = savedPhysAddr(0, 0, 1);

	// for debug
	#ifdef DEBUG
	pr_info("[mmap] I'm in mmap\n");
	pr_info("[mmap] read previously saved address from kernel\n");
	pr_info("[mmap] physical address = %ld\n", savedAddr.addr);
	#endif

	// (1) savedAddr.op == 1: p2v				//			
	//     This means there is function "ioctl(IOCTL_P2V)"	//
	//     that calls the savedPhysAddr() to save the 	//
	//     physical address before this mmap() is called.	//
	//     In this case, 					//
	//     a. mmap() uses remap_pfn_range() with physical 	//
	//        address to be savedPhysAddr to generate a user//
	//        space virtual address.			//
	//     b. And then it needs to clean up the value saved //
	//        in the static variable "savedAddr".		//
	//							//
	// (2) savedAddr.op == 0: v2p				//
	//     This means there is no previous functions that 	//
	//     calls the savedPhysAddr() to save the physcial 	//
	//     address.						//
	//     In this case, 					//
	//     a. We need to use kmalloc_area() to allocate a 	//
	//	  virtual address in kernel space. 		//
	//     b. use virt_to_phys() to convert the kernel space// 
	//        virtual pointer into physical address.	//
	//     c. save the physical address into static variable// 
	//        "savedAddr" for future usage [used by the 	//
	//        following ioctl() ]				//
	//     d. use remap_pfn_range(physAddr) to generate 	//
	//        user space virtual pointer.			//
	//     e. the 2nd argument for savePhysAddr() is 2, 	//
	//     	  which indicates that the static variable is 	//
	//     	  saved by mmap() and waiting for ioctl() to 	//
	//     	  copy it back to user space.			//

	if (savedAddr.op==1){
		#ifdef DEBUG
		pr_info("[mmap] this is p2v.\n");		
		#endif

		int errorCode = remap_pfn_range(vma, vma->vm_start, savedAddr.addr>>PAGE_SHIFT, size, PAGE_SHARED);  
		savedPhysAddr(savedAddr.addr, 0, 0);
		if (errorCode){
			return -ENXIO;
		}
		return 0;
	}

	else if (savedAddr.op==0){
		#ifdef DEBUG
		pr_info("[mmap] this is v2p.\n");
		#endif

		void* kmalloc_area = kmalloc(size, GFP_USER);
		uint64_t phyaddr = virt_to_phys(kmalloc_area);
		savedPhysAddr(phyaddr,2,0);

		if (kmalloc_area==NULL){
			pr_info("[mmap] kmalloc area failed\n");
			return -ENXIO;
		}
		
		int errorCode = remap_pfn_range(vma, vma->vm_start, phyaddr>>PAGE_SHIFT, size, PAGE_SHARED);  
		if (errorCode){
			return -ENXIO;
		}
		return 0;
	}
}

// register the ioctl(), file_open(), 		//
// file_release(), mmap() functions to kernel. 	//
// This functions called by user space will be  //
// replaced by our self-designed functions.	//	
	
struct file_operations gpumem_fops = {

    .owner = THIS_MODULE,
    .unlocked_ioctl = v2p2v_ioctl,
    .compat_ioctl = v2p2v_ioctl,
    .open = v2p2v_open,
    .release = v2p2v_close,
    .mmap = v2p2v_mmap,
};

static struct miscdevice gpumem_dev = {

    MISC_DYNAMIC_MINOR,
    V2P2V_DRIVER_NAME,
    &gpumem_fops
};

// for module initialization and clean up	//
static int __init gpumem_init(void)
{
    pr_info(V2P2V_DRIVER_NAME ": %s()\n", __func__);
    dev.proc = 0;
    sema_init(&dev.sem, 1);
    INIT_LIST_HEAD(&dev.table_list);
    gpumem_register_proc(V2P2V_DRIVER_NAME, 0, &dev);
    misc_register(&gpumem_dev);
    return 0;
}

static void __exit gpumem_cleanup(void)
{
    pr_info(V2P2V_DRIVER_NAME ": %s()\n", __func__);
    gpumem_remove_proc(V2P2V_DRIVER_NAME);
    misc_deregister(&gpumem_dev);
}

module_init(gpumem_init);
module_exit(gpumem_cleanup);

