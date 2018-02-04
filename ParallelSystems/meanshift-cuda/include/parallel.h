#ifndef PARALLEL_H_
#define PARALLEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <time.h>

void init_parallel();
void read_file();
void cpu_malloc();
void gpu_malloc ();
void move_data_to_gpu();
void meanshift();
void rangesearch2sparse();
double finish_reduction();
void gpu_free_memory();
void cpu_free_memory();
void print_2Darray(double **a, const int ROW, const int COL);
void write_csv_file (char *message, double **a, const int ROW, const int COL);

#endif