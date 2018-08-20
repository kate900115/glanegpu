
#include <linux/kernel.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/pci.h>
#include <linux/pagemap.h>
#include <linux/interrupt.h>
#include <linux/proc_fs.h>
#include <asm/io.h>

#include "v2p2vdrv.h"
#include "v2p2vioctl.h"

	// savePhysAddr() is a function that saved physcial 	//
	// address which is passed from ioctl()			//

struct savedAddress savedPhysAddr(uint64_t addr, int op, bool isRead ){

	// for debug.						//
	// it shows how many times the savePhysAddr() is called.//
	
	#ifdef DEBUG
	static int times = 0;
	times++;
	pr_info("[savedPhysAddr] the function is called %d times\n",times);
	#endif

	// first need to figure out the type of operation:	//	
	// (1) read or (2) write				//
	// 							//
	// (1) If it is a read operation:			//
	//     a. It means that in user space, the programmer	// 
	//        want to convert physical address to virtual	// 
	//        address. The user calls mmap(), and mmap()	// 
	//        invokes savedPhysAddr() just to get the	// 
	//        physical address saved by ioctl().		//
	//     b. It means that in user space, the programmer 	//
	//        want to convert virtual address to physical	// 
	//        address. The user calls ioctl(), and ioctl()	//
	//        invokes savedPhysAddr() to get the physical	// 
	//        address saved by mmap(), and then return it	// 
	//        back to user space.				//
	//							//
	//							//
	// (2) If it is a write operation:			//
	//     a. It means that in user space, the programmer	// 
	//        want to convert virtual address to physical	//
	//        address. The user called mmap(), and mmap()	// 
	//        calls savedPhysAddr() to store the converted	//
	//        phsical address into a static variable 	//
	//        called in "SavedAddr" in kernel space.	//
	//        Later on, the ioctl() called by user will 	//
	//        read the "SavedAddr" to get the physical 	//
	//        address.					//
	//     b. It means that in user space, the programmer	//
	//        want to convert physical address to virtual	//
	//        address. The user calls ioctl(), and ioctl()	//
	//        calls the savedPhysAddr() to store the 	//
	//        physical address into a static variable 	//
	//        called "savedAddr" in kernel space. Later on,	//
	//        the mmap() called by user will read the 	//
	//        "SavedAddr" in kernel space, and convert it 	//
	//        into kernel space virtual pointer, and then	//
	//        send it back to user space.   		//
	

	static struct savedAddress SavedAddr;

	if (isRead){
		return SavedAddr;
	}
	else{
		SavedAddr.addr = addr;
		SavedAddr.op = op;
		return SavedAddr;
	}

}



int ioctl_v2p_convert(unsigned long arg){

	// Before kernel calls this function, it has already 	//
	// called v2p2v_mmap() to allocate virtual pointer to	//
	// physical pointer and stored that physical pointer 	//
	// to "savedAddr". User calls v2p2v_ioctl() just to     //
	// fetch the physical address back to user space.	//
	//							//
	// (1) copy user space variable void* arg into kernel	//
	//     space variable cpuaddr_t addr.			//
	// (2) read the address from saved address.		//
	// (3) store the address into struct cpuaddr_t addr	//
	//     and then clean up the static varialbe 		//
	//     "savedAddr".					//
	// (4) copy the kernel space variable cpuaddr_t addr 	//
	//     into user space variable void* arg.		//

	int error = 0;

	struct cpuaddr_t addr;
	if (copy_from_user(&addr, (void*)arg, sizeof(struct cpuaddr_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
		error = -EFAULT;
		return error;
	}
	
	struct savedAddress temp = savedPhysAddr(0,0,1);
	addr.paddr = temp.addr;	
	savedPhysAddr(0,0,0);

	if (copy_to_user((void*)arg, &addr, sizeof(struct cpuaddr_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n",__FUNCTION__);
		error = -EFAULT;
		return error;
	}

	// for debug
	#ifdef DEBUG
	pr_info("[ioctl_v2p] I'm ioctl_v2p_convert\n");		
 	pr_info("[ioctl_v2p] previously saved physical address = %ld\n", addr.paddr);
	pr_info("[ioctl_v2p] write 0 to saved address\n");
	#endif
	
	return error;
}


int ioctl_p2v_convert(unsigned long arg){

	// User calls this v2p2v_ioctl() to send the physical 	//
	// to kernel space for v2p2v_mmap() to convert it into	//
	// virtual address.					//
	//							//
	// (1) copy user space variable void* arg into kernel	//
	//     space variable cpuaddr_t addr.			//
	// (2) store the address into struct cpuaddr_t addr	//
	//     and then clean up the static varialbe 		//
	//     "savedAddr".					//
	//     The second argument of savedPhysAddr() indicates	//
	//     the saved address is used for p2v convertion.	// 

	int error = 0;
	struct cpuaddr_t addr;
	if (copy_from_user(&addr, (void*)arg, sizeof(struct cpuaddr_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
		error = -EFAULT;
		return error;
	}

	savedPhysAddr(addr.paddr, 1, 0);

	// for debug
	#ifdef DEBUG
	pr_info("[ioctl_p2v] I'm ioctl_p2v_convert\n");	
 	pr_info("[ioctl_p2v] now is saving physical address = %d\n", addr.paddr);
	#endif

	return error;
}


