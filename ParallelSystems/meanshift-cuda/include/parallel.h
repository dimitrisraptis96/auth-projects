#ifndef PARALLEL_H_
#define PARALLEL_H_

#include "../include/global_vars.h"
#include "../include/parallel.h"
#include "../include/helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>


// __global__ void init_arr(int *d_nNbr, long double **d_y_data, long double **d_m_data);
// void cuda_error_handler(cudaError_t err);

void parallel();
void init_parallel();
void cpu_malloc();
void gpu_malloc ();
void move_data_to_gpu();
void free_memory();
void read_file();
void write_csv_file (char *message, long double **a, const int ROW, const int COL);
void meanshift();
void rangesearch2sparse();
void matrix_mult();
void normalize();
long double sum_of_row(const int row_index);
long double frob_norm();
void calc_meanshift();
void copy_2Darray(long double **source, long double **destination, const int ROW, const int COL);
void print_2Darray(long double **a, const int ROW, const int COL);
long double gaussian_kernel(const long double dist);
long double euclidean_distance(const int first, const int second);

#endif