#ifndef PARALLEL_H_
#define PARALLEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <time.h>

// error handlers
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

#define HANDLE_EOF( a ) {if (a == EOF) { \
                            printf( "File reading failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

void init_parallel(int version);

int choose_version();

void read_file();

void cpu_malloc();

void gpu_malloc (int version);

void move_data_to_gpu();

void cuda_meanshift(int version);

void rangesearch2sparse();

double finish_reduction();

void gpu_free_memory(int version);

void cpu_free_memory();

void print_2Darray(double **a, const int ROW, const int COL);

void write_csv_file (char *message, double **a, const int ROW, const int COL);

#endif