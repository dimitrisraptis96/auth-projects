#ifndef GLOBAL_VARS_H_
#define GLOBAL_VARS_H_

// #ifndef DATASET_PATH
// #define DATASET_PATH	     "dataset/txt/1024_512.txt"
// #endif
#ifndef OUTPUT_PATH_SERIAL
#define OUTPUT_PATH_SERIAL	 "results/serial.csv"
#endif
#ifndef OUTPUT_PATH_PARALLEL
#define OUTPUT_PATH_PARALLEL "results/parallel.csv"
#endif

#ifndef VERBOSE
#define VERBOSE 1
#endif

// #ifndef EPSILON
// #define EPSILON 1e-4*BANDWIDTH
// #endif

// #ifndef BANDWIDTH
// #define BANDWIDTH 10.0
// #endif

// type of implementation
#ifndef TYPE_CPU
#define TYPE_CPU	1
#endif
#ifndef TYPE_GPU_SHARED
#define TYPE_GPU_SHARED	2
#endif
#ifndef TYPE_GPU_NON_SHARED
#define TYPE_GPU_NON_SHARED	3
#endif

// version of cuda implementation
#ifndef SPARSE_VERSION
#define SPARSE_VERSION		1
#endif
#ifndef EXHAUSTIVE_VERSION
#define EXHAUSTIVE_VERSION	2
#endif

extern int N;
extern int D;
extern double BANDWIDTH;
extern double EPSILON;
extern char *DATASET_PATH;

extern int USE_SHARED;



#endif