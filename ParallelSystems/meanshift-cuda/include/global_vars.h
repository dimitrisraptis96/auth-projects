#ifndef GLOBAL_VARS_H_
#define GLOBAL_VARS_H_

#ifndef DATASET_PATH
#define DATASET_PATH     "data/5000_2.txt"
// #define DATASET_PATH     "data/600_2.txt"
#endif
#ifndef OUTPUT_PATH_SERIAL
#define OUTPUT_PATH_SERIAL	 	"data/serial.csv"
#endif
#ifndef OUTPUT_PATH_PARALLEL
#define OUTPUT_PATH_PARALLEL 	 "data/parallel.csv"
#endif

#ifndef VERBOSE
#define VERBOSE 1
#endif

#ifndef EPSILON
#define EPSILON 1e-4
#endif

#ifndef BANDWIDTH
#define BANDWIDTH 100000.0
#endif

#ifndef TYPE_CPU
#define TYPE_CPU	1
#endif
#ifndef TYPE_GPU
#define TYPE_GPU	2
#endif

extern int N;
extern int D;

extern int *nNBr;

extern double **x;
extern double **y;
extern double **y_new;
extern double **m;
extern double **d;
// extern int **id;

#endif