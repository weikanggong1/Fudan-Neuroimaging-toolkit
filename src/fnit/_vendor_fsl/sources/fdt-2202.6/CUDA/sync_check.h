/*  sync_check.h

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include <fstream>

#define safecall(call) do{\
	cudaError_t err=call;\
	if (cudaSuccess != err){\
		printf("cuda error at %s:%d. %s\n",__FILE__,__LINE__,cudaGetErrorString(err));\
	}\
}while(0)

#define sync_check(message) do{;\
	safecall(cudaDeviceSynchronize());\
	cudaError_t error = cudaGetLastError();\
	if (cudaSuccess != error){\
		printf("ERROR: %s: %s\n",message,cudaGetErrorString(error));\
		exit(-1);\
	}\
}while(0)


