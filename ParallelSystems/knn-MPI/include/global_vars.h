#ifndef GLOBAL_VARS_H_
#define GLOBAL_VARS_H_

#ifndef CORPUS_PATH
#define CORPUS_PATH     "data/mnist_train_svd.txt"
#endif
#ifndef VALIDATION_PATH
#define VALIDATION_PATH "data/validation_mnist_train_svd.txt"
#endif

//Set it for MPI communication debug messages and other messages
#ifndef VERBOSE
#define VERBOSE 1
#endif

#ifndef PRECISION
#define PRECISION 0.00001
#endif

//Define type of implementation
#ifndef TYPE_SERIAL
#define TYPE_SERIAL	1
#endif
#ifndef TYPE_BLOCK
#define TYPE_BLOCK	2
#endif
#ifndef TYPE_NO_BLOCK
#define TYPE_NO_BLOCK	3
#endif

extern int N;
extern int D;
extern int K;

extern double **in_buffer;
extern double **out_buffer;
extern double **array;
extern double **k_dist;
extern int **k_id;

//MPI implementation's data
extern double *in_data;
extern double *out_data;
extern double *array_data;
extern double *k_dist_data;
extern int    *k_id_data;

//openMP implementation
extern int N_THREADS;

#endif