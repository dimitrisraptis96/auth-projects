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

void write_csv_file(char *message, double **a, const int ROW, const int COL);

int validate();

void init_arr();

void meanshift();

void rangesearch();

void cpu_matrix_mult(double **a, double **b,double **c, 
		const int ROW1, const int COL1, const int COL2);

void normalize(double **a, const int ROW, const int COL);

double sum_of_row(double **a, const int row, const int COL);

double eucl_norm(double **a, const int ROW, const int COL) ;

void calc_meanshift(double **a, double **b, double **c, const int ROW, const int COL);

void copy_2Darray(double **source, double **destination, const int ROW, const int COL);

void print_2Darray(double **a, const int ROW, const int COL);

double gaussian_kernel(const double dist);

double euclidean_distance(const int first, const int second);

double euclidean_distance_sqr(const int first, const int second);

#endif