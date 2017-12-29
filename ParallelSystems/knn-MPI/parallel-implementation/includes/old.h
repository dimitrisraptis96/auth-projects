#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <mpi.h>

void init(void);

void print(double** arr, int row, int col);

void print_id();

void copy_2D_arrays(double **arr1, double **arr2);

void read_file(void);

void ring_comm(int tag);

void knn(void);

void memory_allocation(void);

// void memory_deallocation(void);

int validate(void);

void check_args(void);

void init_dist(void);

void calc_distances (void);

void calc_knn(int rep);

void find_position(int i, double dist, int id);

void move(int i, int pos);

double euclidean_distance(int first, int second);

int main(int argc, char **argv);

#endif