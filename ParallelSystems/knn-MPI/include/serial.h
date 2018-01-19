#ifndef SERIAL_H_
#define SERIAL_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <omp.h>

void serial(void);

void init_serial(void);

void memory_allocation(void);

void read_file_serial(void);

void knn(void);

int validate_serial(void);

#endif