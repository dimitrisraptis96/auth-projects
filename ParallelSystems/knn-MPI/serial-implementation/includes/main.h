#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdlib.h>

void init(void);

void print(int row, int col);

void check_args(void);

void calc_distances (void);

float euclidean_distance(int first, int second);

int main(int argc, char **argv);

#endif