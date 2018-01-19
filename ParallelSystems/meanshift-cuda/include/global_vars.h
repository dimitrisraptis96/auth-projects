#ifndef GLOBAL_VARS_H_
#define GLOBAL_VARS_H_


#ifndef DATASET_PATH
#define DATASET_PATH     "data/dataset.txt"
#endif
#ifndef VALIDATION_PATH
#define VALIDATION_PATH  "data/matlab-files/validation_mnist_train_svd.txt"
#endif
#ifndef OUTPUT_PATH
#define OUTPUT_PATH 	 "data/out.txt"
#endif

#ifndef VERBOSE
#define VERBOSE 1
#endif

#ifndef EPSILON
#define EPSILON 1e-4 * BANDWIDTH
#endif

#ifndef BANDWIDTH
#define BANDWIDTH 1
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
extern int **id;

#endif