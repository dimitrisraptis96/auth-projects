/*#ifndef PARALLEL_H_
#define PARALLEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include "mpi.h"
#include "omp.h"

#ifndef FIRST_REP
#define FIRST_REP 1
#endif

#ifndef OTHER_REP
#define OTHER_REP 0
#endif

int PID;
int MAX;
int CHUNK;
MPI_Status status;

double **in_buffer;
double **out_buffer;
double **array;
double **k_dist;
int    **k_id;

double *in_data;
double *out_data;
double *array_data;
double *k_dist_data;
int    *k_id_data;

void MPI_block(void);

void MPI_no_block(void);

void init_parallel(void);

void cont_memory_allocation(void);

void read_file_parallel(void);

void knn_block(void);

void knn_no_block(void);

void ring_block(int tag);

void ring_no_block(int tag);

void init_dist(void);

void calc_knn(int rep);

void copy_2D_arrays(double **arr1, double **arr2);

int validate_parallel(void);

int comp_result(FILE * fp);

#endif*/