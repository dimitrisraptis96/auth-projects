#include <stdio.h>
#include <stdlib.h>

#include "../includes/main.h"

//global variables
int N;
int D;
int K;

double **init_arr;
double **dist_arr;
double **k_dist;
int **k_id;

// char CORPUS_FILENAME[] = "./data/corpus.txt";
char CORPUS_FILENAME[]      = "../../corpus-files/corpus.txt";
char VALIDATION_FILENAME[]  = "../../corpus-files/validated.txt";

double PRECISION = 0.00001;

int main (int argc, char **argv) {

  struct timeval startwtime, endwtime;
  double seq_time;

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

  //Calculate kNN and measure time passed
  gettimeofday (&startwtime, NULL);
  //------------------------------
  calc_knn();
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);
  
  // printf("===============================================\n");  
  // printf("\t\tknn distances\n");
  // printf("===============================================\n");
  // print(k_dist,N,K);
  // printf("===============================================\n");  
  // printf("\t\tknn id's\n");
  // printf("===============================================\n");
  // print_id();
  // printf("===============================================\n");
  printf("Is test PASSed? %s\n\n", validate()?"YES":"NO");
  printf("===============================================\n\n");
  printf("Serial kNN wall clock time = %f\n", seq_time);

  return(0);
}


//Dynamically allocate memory and create the 2D-array
void init(){
  int i,j;

  FILE * fp;
  fp = fopen (CORPUS_FILENAME, "r");

  init_arr = (double **) malloc(N * sizeof(double *));
  if(init_arr == NULL) { 
    printf("Memory allocation error!\n");
    exit(1);
  }

  for (i=0; i<N; i++) {
    init_arr[i] = (double *) malloc(D * sizeof(double));
    if(init_arr[i] == NULL) { 
      printf("Memory allocation error!\n");
      exit(1);
    }

    for (j=0; j<D; j++) {
      fscanf(fp, "%lf", &init_arr[i][j]);
    }
  }

  fclose(fp);
  return;
}

//Print 2D double arr array
void print(double **arr, int row, int col){
  int i, j;
  for (i=0; i<row; i++){
    for (j=0; j<col; j++){
      printf("%lf ", arr[i][j]);
    }
    printf("\n");
  }
  return;
}

//Print 2d int arr array
void print_id(){
  int i, j;
  for (i=0; i<N; i++){
    for (j=0; j<K; j++){
      printf("%d ", k_id[i][j]);
    }
    printf("\n");
  }
  return;
}

//Quicksort
int cmp_func (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}


int test(){
  int i,j;
  calc_distances();
  //Change -1 with DBL_MAX 
  for (i=0;i<N;i++){
    for(j=0;j<N;j++){
      if (dist_arr[i][j] == -1){
        dist_arr[i][j] = DBL_MAX;
      }
    }
  }
  //Sort the dist_arr and do the testing
  for (i=0; i<N; i++){
    qsort(dist_arr[i], N, sizeof(double), cmp_func);
    for (j=0; j<K; j++){
      if (k_dist[i][j] != dist_arr[i][j]){
        printf("k_dist=%lf\n",k_dist[i][j]);
        printf("dist_arr=%lf\n",dist_arr[i][j]);
        printf("i=%d j=%d\n",i,j);
        return 0;
      }
    }
  }
  return 1;
}

int validate(){
  int i,j;
  double tmp;

  FILE * fp;
  fp = fopen (VALIDATION_FILENAME, "r");

  for (i=0; i<N; i++) {
    for (j=0; j<K; j++) {
      fscanf(fp, "%lf", &tmp);

      if(!(fabs(tmp - k_dist[i][j]) < PRECISION)){
        printf("k_dist=%lf\n",k_dist[i][j]);
        printf("validate_tmp=%lf\n",tmp);
        printf("i=%d j=%d\n",i,j);
        fclose(fp);
        return 0;
      }
    }
  }
  fclose(fp);
  return 1;
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


double euclidean_distance(int first, int second){
  int j;
  double dist = 0;
  for (j=0; j<D; j++)
    dist += (init_arr[first][j] - init_arr[second][j]) * (init_arr[first][j] - init_arr[second][j]);
  return dist;
}


void calc_distances (){
  int i, j;

  dist_arr = (double **) malloc(N * sizeof(double *));
  if(dist_arr == NULL) { 
    printf("Memory allocation error!\n");
    exit(1);
  }

  for (i=0; i<N; i++) {
    dist_arr[i] = (double *) malloc(N * sizeof(double));
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
  return;
}


void calc_knn(){
  int i, j;
  double dist;

  //TODO: USE 1 FOR QUICKER RESULTS!!! 

  //Memory allocation for k_dist[N][K] and k_id[N][K] array
  k_dist = (double **) malloc(N * sizeof(double *));
  k_id = (int **) malloc(N * sizeof(int *));
  if( (k_dist == NULL) || (k_id == NULL) ) { 
    printf("Memory allocation error!\n");
    exit(1);
  }
  for (i=0; i<N; i++) {
    k_dist[i] = (double *) malloc(K * sizeof(double));
    k_id[i] = (int *) malloc(K * sizeof(int));
    if( (k_dist[i] == NULL) || (k_id[i] == NULL) ) { 
      printf("Memory allocation error!\n");
      exit(1);
    }
    for (j=0; j<K; j++){
      k_dist[i][j] = DBL_MAX;
      k_id[i][j]   = -1;
    }
  }

  //Calculate k nearest points
  for(i=0; i<N; i++){
    for (j=0; j<N; j++) {
      if (i==j) continue;
        dist = euclidean_distance(i,j);
        if (dist < k_dist[i][K-1]){
          find_position(i,dist,j);  //Just found a new closer distance
        }
    }
  }
}

//Find the position of the new distance
void find_position(int i, double dist, int id){
  int j;
  for (j=0; j<K; j++){
    if (dist < k_dist[i][j]){
      //TODO: Check if qsort is quicker
      // if (i==j) continue;
      move(i,j);
      k_dist[i][j] = dist;
      k_id[i][j]   = id;
      return;
    }
  }
  return;
}

//Shift k_dist[i] & k_id[i] values one position in order to insert the new distance
void move(int i, int pos){
  int j;
  for (j=(K-1); j>pos; j--){
    if(j==0) continue;
    k_dist[i][j] = k_dist[i][j-1];
    k_id[i][j]   = k_id[i][j-1];
  }
  return;
}