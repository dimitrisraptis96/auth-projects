#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

void init(void);

void print(float** arr, int row, int col);

void print_id();

int test(void);

int cmp_func (const void * a, const void * b);

void check_args(void);

void calc_distances (void);

void calc_knn(void);

void find_position(int i, float dist, int id);

void move(int i, int pos);

float euclidean_distance(int first, int second);

int main(int argc, char **argv);

#endif