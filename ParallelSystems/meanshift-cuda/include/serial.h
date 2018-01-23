#ifndef SERIAL_H_
#define SERIAL_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>

#include "../include/global_vars.h"
#include "../include/serial.h"
#include "../include/helpers.h"

void serial(void);

void init_serial(void);

void memory_allocation(void);

void free_memory(void);

void read_file(void);

void write_csv_file (char *message, long double **a, const int ROW, const int COL);

int validate(void);

void init_arr(void);

void meanshift(void);

void rangesearch2sparse(void);

void matrix_mult(void);

void normalize(void);

long double sum_of_row(const int row_index);

long double frob_norm(void);

void calc_meanshift(void);

void copy_2Darray(long double **source, long double **destination, const int ROW, const int COL);

void print_2Darray(long double **a, const int ROW, const int COL);

long double gaussian_kernel(const long double dist);

long double euclidean_distance(const int first, const int second);

#endif