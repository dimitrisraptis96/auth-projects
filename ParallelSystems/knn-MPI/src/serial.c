#include "../include/global_vars.h"
#include "../include/serial.h"
#include "../include/helpers.h"


double **array;
double **k_dist;
int **k_id;

void serial(){
  printf("[INFO]: SERIAL IMPLEMENTATION\n");
  printf("=============================\n");

  struct timeval startwtime, endwtime;
  double seq_time;

  init_serial();
  
  gettimeofday (&startwtime, NULL);
  //------------------------------
  knn();
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);
  
  printf("\n\nIs test PASSed? %s\n\n", validate_serial()?"YES":"NO");
  printf("===============================================\n\n");
  printf("Serial kNN wall clock time = %f\n", seq_time);
}

void init_serial(){
  memory_allocation();
  read_file_serial();
}

//Contiguous memory allocation for 2D arrays
void memory_allocation(){
  int i,j;
  
  printf("[INFO]: Allocating memory");

  //Memory allocation for k_dist[N][K] and k_id[N][K] array
  array   = (double **) malloc(N * sizeof(double *));
  k_dist  = (double **) malloc(N * sizeof(double *));
  k_id    = (int **)    malloc(N * sizeof(int *));
  if( (array == NULL) || (k_dist == NULL) || (k_id == NULL) ) { 
    printf("[ERROR]: Memory allocation error 1!\n");
    exit(1);
  }
  for (i=0; i<N; i++) {
    array[i]  = (double *) malloc(D * sizeof(double));
    k_dist[i] = (double *) malloc(K * sizeof(double));
    k_id[i]   = (int *)    malloc(K * sizeof(int));
    if( (array[i] == NULL) || (k_dist[i] == NULL) || (k_id[i] == NULL) ) { 
      printf("[ERROR]: Memory allocation error 2!\n");
      exit(1);
    }
    for (j=0; j<K; j++){
      k_dist[i][j] = DBL_MAX;
      k_id[i][j]   = -1;
    }
  }
}

void read_file_serial(){
  int i,j;

  FILE * fp;
  fp = fopen (CORPUS_PATH, "r");

  if (fp == NULL){
    printf("[ERROR]: Invalid path.\n");
    exit(1);
  }

  for (i=0; i<N; i++) 
    for (j=0; j<D; j++){
      //TODO
      if (EOF ==  fscanf(fp, "%lf", &array[i][j])){
        printf("[ERROR]: fscanf() failed.\n");
        exit(1);
      }
    }

  fclose(fp);
  return;
}

int validate_serial(){
  int i,j;
  double tmp;

  FILE * fp;
  fp = fopen (VALIDATION_PATH, "r");
  if (fp == NULL){
    printf("[ERROR]: Invalid path.\n");
    exit(1);
  }

  for (i=0; i<N; i++) {
    for (j=0; j<K; j++) {
      if(EOF==fscanf(fp, "%lf", &tmp)){
        printf("[ERROR]: fscanf() failed.\n");
        exit(1);
      }

      if(!(fabs(tmp - k_dist[i][j]) < PRECISION)){
        printf("[INFO]: Validation failed:\n");
        // printf("k_dist=%lf\n",k_dist[i][j]);
        // printf("validate_tmp=%lf\n",tmp);
        // printf("i=%d j=%d\n",i,j);
        fclose(fp);
        return 0;
      }
    }
  }
  fclose(fp);
  return 1;
}

void knn(){
  int i, j;
  double dist;

  int N_THREADS = 8;

  //Calculate k nearest points
  #pragma omp parallel for shared(i) private(j,dist) num_threads(N_THREADS)
  for(i=0; i<N; i++){
    for (j=0; j<N; j++) {
      if (i==j) continue;
        dist = euclidean_distance(i,j,array,array);
        if (dist < k_dist[i][K-1]){
          find_position(i,dist,j);  //Just found a new closer distance
        }
    }
  }
}