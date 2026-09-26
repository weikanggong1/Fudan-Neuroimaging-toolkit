#define THREADS_BLOCK 64
#define THREADS_BLOCK_UPDATE 128
#define THREADS_BLOCK_RAND 256

//dynamic
//#define MAX_THREADS 20000
//MAX_THREADS must be an even number

#define NSTREAMS 2 // do not change this !
#define MAX_TRI_SEED 12
#define PRECISION_SAME_VERTEX 1e-5f

//% of memory that must be free after allocating the masks and streamlines
#define FREEPERCENTAGE 0.8

// if 64 threads per block, each ROI takes 512 KB
#define maxNetsInShared 8

//Maximum number of target Masks in Shared memory... use global if more than this
#define maxTargsInShared 16
