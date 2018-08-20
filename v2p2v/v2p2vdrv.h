

#ifndef GPUMEM_H
#define GPUMEM_H

//-----------------------------------------------------------------------------

#include <linux/cdev.h>
#include <linux/sched.h>
#include <linux/version.h>
#include <linux/semaphore.h>

//-----------------------------------------------------------------------------

struct gpumem_t {
    struct list_head list;
    void *handle;
    u64 virt_start;
};

//-----------------------------------------------------------------------------

struct gpumem {
    struct semaphore         sem;
    struct proc_dir_entry*   proc;
    struct list_head         table_list;
};

//-----------------------------------------------------------------------------

int get_nv_page_size(int val);

//enum ioctlOp {
//	P2V, //1
//	V2P, //2
//	noop//0
//};

struct savedAddress{
	uint64_t addr;
	int op; // if isP2V=0 means it's V2P
};

//-----------------------------------------------------------------------------

#endif
