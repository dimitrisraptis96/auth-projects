#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>

void init(void);

void print(double** arr, int row, int col);

void print_id();

void read_file(void);

void ring_comm(int tag);

void memory_allocation(void);

void memory_deallocation(void);

int test(void);

int cmp_func (const void * a, const void * b);

void check_args(void);

void calc_distances (void);

void calc_knn(void);

void find_position(int i, double dist, int id);

void move(int i, int pos);

double euclidean_distance(int first, int second);

int main(int argc, char **argv);

#endif