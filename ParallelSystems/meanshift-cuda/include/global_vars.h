#ifndef GLOBAL_VARS_H_
#define GLOBAL_VARS_H_

// error handlers
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

#define HANDLE_EOF( a ) {if (a == EOF) { \
                            printf( "File reading failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

#ifndef DATASET_PATH
#define DATASET_PATH     "data/S1.txt"
// #define DATASET_PATH     "data/dataset.txt"
#endif
#ifndef VALIDATION_PATH
#define VALIDATION_PATH  "data/..."
#endif
#ifndef OUTPUT_PATH_SERIAL
#define OUTPUT_PATH_SERIAL	 	"data/serial.txt"
// #define OUTPUT_PATH 	 "data/out-dataset.txt"
#endif
#ifndef OUTPUT_PATH_PARALLEL
#define OUTPUT_PATH_PARALLEL 	 "data/parallel.txt"
// #define OUTPUT_PATH 	 "data/out-dataset.txt"
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