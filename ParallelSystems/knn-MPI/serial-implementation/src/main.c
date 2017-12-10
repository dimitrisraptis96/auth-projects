#include <stdio.h>
#include <stdlib.h>

#include "../includes/main.h"

//global variables
int N;
int D;
int K;

float **db;
float **dist_arr;

char CORPUS_FILENAME[] = "./data/corpus.txt";

int main (int argc, char **argv) {
  
  if (argc != 4) {
    printf("==============================================\n");
    printf("Usage: Serial implementation of knn algorithm.\n");
    printf("==============================================\n");
    printf("arg[1] = N ==> number of points\n");
    printf("arg[2] = D ==> point's dimensions\n");
    printf("arg[3] = K ==> k nearest points\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }

  N = atoi(argv[1]);
  D = atoi(argv[2]);
  K = atoi(argv[3]);

  check_args();
  init();
  print(N,D);
  printf("===============================================\n");
  printf("===============================================\n");
  calc_distances();
  // print(N,N);
  
  return(0);
}


//Dynamically allocate memory and create the 2D-array
void init(){
  int i,j;

  FILE * fp;
  fp = fopen (CORPUS_FILENAME, "r");

  db = (float **) malloc(N * sizeof(float *));
  if(db == NULL) { 
    printf("Memory allocation error!\n");
    exit(1);
  }

  for (i=0; i<N; i++) {
    db[i] = (float *) malloc(D * sizeof(float));
    if(db[i] == NULL) { 
      printf("Memory allocation error!\n");
      exit(1);
    }

    for (j=0; j<D; j++) {
      fscanf(fp, "%f", &db[i][j]);
    }
  }

  fclose(fp);
  return;
}

void print(int row, int col){
  int i, j;
  for (i=0; i<row; i++){
    for (j=0; j<col; j++){
      printf("%f ", db[i][j]);
    }
    printf("\n");
  }
  return;
}

void check_args() {
  if (N<=0 || K<=0 || D<=0){
    printf("Negative value for N, K or D.\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }
  if (N<=K){
    printf("K value is larger than N.\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }
  return;
}

float euclidean_distance(int first, int second){
  int j;
  float dist;

  for (j=0; j<D; j++)
    dist += (db[first][j] - db[second][j]) * (db[first][j] - db[second][j]);

  return dist;
}

void calc_distances (){
  int i, j;

  dist_arr = (float **) malloc(N * sizeof(float *));
  if(dist_arr == NULL) { 
    printf("Memory allocation error!\n");
    exit(1);
  }

  for (i=0; i<N; i++) {
    dist_arr[i] = (float *) malloc(N * sizeof(float));

    if(dist_arr[i] == NULL) { 
      printf("Memory allocation error!\n");
      exit(1);
    }

    for (j=0; j<N; j++) {

      if (i==j) {
          dist_arr[i][j] = -1.0;
          continue;
        }
        dist_arr[i][j] = euclidean_distance(i,j);
    }
  }

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      printf("%f ", dist_arr[i][j]);
    }
    printf("\n");
  }

  return;
}