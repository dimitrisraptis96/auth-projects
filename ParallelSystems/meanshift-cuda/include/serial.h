#ifndef SERIAL_H_
#define SERIAL_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>


void serial();

void init_serial();

void memory_allocation();

void read_file();

void write_csv_file(char *message, long double **a, const int ROW, const int COL);

int validate();

void init_arr();

void meanshift();

void rangesearch();

void cpu_matrix_mult(long double **mult, long double **a, long double **b, 
		const int ROW1, const int COL1, const int COL2);

void normalize(long double **a, const int ROW, const int COL);

long double sum_of_row(long double **a, const int row, const int COL);

long double frob_norm(long double **a, const int ROW, const int COL) ;

void calc_meanshift(long double **a, long double **b, long double **c, const int ROW, const int COL);

void copy_2Darray(long double **source, long double **destination, const int ROW, const int COL);

void print_2Darray(long double **a, const int ROW, const int COL);

long double gaussian_kernel(const long double dist);

long double euclidean_distance(const int first, const int second);

long double euclidean_distance_sqr(const int first, const int second);

#endif