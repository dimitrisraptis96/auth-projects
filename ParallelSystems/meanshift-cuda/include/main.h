#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdlib.h>

void set_args(int argc, char **argv);

void check_args(void);

void choose_type(void);

int main(int argc, char **argv);

extern void parallel();

#endif