#ifndef SERIAL_H_
#define SERIAL_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <time.h>


void serial(void);

void init_serial(void);

void memory_allocation(void);

void free_memory(void);

void read_file(void);

void write_csv_file (char *message, double **a, const int ROW, const int COL);

int validate(void);

void init_arr(void);

void meanshift(void);

void rangesearch2sparse(void);

void matrix_mult(void);

void normalize(void);

double sum_of_row(const int row_index);

double frob_norm(void);

void calc_meanshift(void);

void copy_2Darray(double **source, double **destination, const int ROW, const int COL);

void print_2Darray(double **a, const int ROW, const int COL);

double gaussian_kernel(const double dist);

double euclidean_distance(const int first, const int second);

#endif