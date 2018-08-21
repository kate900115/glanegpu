#ifndef __VADD_H
#define __VADD_H

#define m 16
#define n 16

#define MemBufferSize 8
#define AQsize 16

#endif

struct AQentry{
	bool isInUse;
	int MemFreelistIdx;
};
