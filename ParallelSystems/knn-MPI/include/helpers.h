#ifndef HELPERS_H_
#define HELPERS_H_

#include <stdio.h>
#include <stdlib.h>

double euclidean_distance(int first, int second, double **arr1, double **arr2);

void find_position(int i, double dist, int id);

void move(int i, int pos);

void print(double** arr, int row, int col);

void print_id();

#endif