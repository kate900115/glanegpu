#ifndef __VADD_H
#define __VADD_H

#define m 16
#define n 16

#define iterationNum 100
#define MemBufferSize 8
#define AQsize 16
#define RQsize 200
#endif

struct AQentry{
	int isInUse;
	int MemFreelistIdx;
};

struct reqBuf{
	int isInUse;
	int OpType;
	int kernelID;
	unsigned long AQaddr;
	unsigned long inBufAddr;
	unsigned long outBufAddr;
	long length;
	int idx;
};

struct physAddr{
	unsigned long dptrPhyAddrOnGPU;
	int kernelID;
};
