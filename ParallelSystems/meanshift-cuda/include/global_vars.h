#ifndef GLOBAL_VARS_H_
#define GLOBAL_VARS_H_

// print verbose messages if it's set
#ifndef VERBOSE
#define VERBOSE 1
#endif

// results paths
#ifndef OUTPUT_PATH_SERIAL
#define OUTPUT_PATH_SERIAL	 "results/serial.csv"
#endif
#ifndef OUTPUT_PATH_PARALLEL
#define OUTPUT_PATH_PARALLEL "results/parallel.csv"
#endif

// type of implementation
#ifndef TYPE_CPU
#define TYPE_CPU			1
#endif
#ifndef TYPE_GPU_SHARED
#define TYPE_GPU_SHARED		2
#endif
#ifndef TYPE_GPU_NON_SHARED
#define TYPE_GPU_NON_SHARED	3
#endif

// version of cuda implementation
#ifndef VERSION_SPARSE
#define VERSION_SPARSE		1
#endif
#ifndef VERSION_EXHAUSTIVE
#define VERSION_EXHAUSTIVE	2
#endif

// command line args
extern int N;
extern int D;
extern double BANDWIDTH;
extern double EPSILON;
extern char *DATASET_PATH;

extern int USE_SHARED;

#endif