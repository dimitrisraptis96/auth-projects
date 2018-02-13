#ifndef SERIAL_H_
#define SERIAL_H_

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

// declare functions                            
void serial(void);

void init_serial(void);

void memory_allocation(void);

void free_memory(void);

void read_file(void);

void write_csv_file (char *message, double **a, const int ROW, const int COL);

int validate(void);

static void init_arr(void);

static void meanshift(void);

static void rangesearch2sparse(void);

static void matrix_mult(void);

static void normalize(void);

static double sum_of_row(const int row_index);

static double frob_norm(void);

static void calc_meanshift(void);

static void copy_2Darray(double **source, double **destination, const int ROW, const int COL);

static void print_2Darray(double **a, const int ROW, const int COL);

static double gaussian_kernel(const double dist);

static double euclidean_distance(const int first, const int second);

#endif