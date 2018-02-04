#include "../include/serial.h"
#include "../include/global_vars.h"

// TODO: inline optimization

int *nNbr;

double **x;
double **y;
double **y_new;
double **m;


typedef struct {
    int xid;
    double distance;
} SparseData;

SparseData **w;

void serial(){
  printf("[INFO]: SERIAL-CPU IMPLEMENTATION\n");
  printf("=============================\n");

  struct timeval startwtime, endwtime;
  double seq_time;

  init_serial();
  
  gettimeofday (&startwtime, NULL);
  //------------------------------
  meanshift();
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);

  /*printf("\n\nIs test PASSed? %s\n\n", validate_serial()?"YES":"NO");
  printf("===============================================\n\n");*/
  printf("\n\n[INFO]:Serial meanshift wall clock time = %f\n", seq_time);

}

void init_serial(){
  memory_allocation();
  read_file();
}

//Memory allocation for 2D arrays
void memory_allocation(){
  int i;
  
  if (VERBOSE) printf("[INFO]: Allocating memory...\n");

  x     = (double **) malloc(N * sizeof(double *));
  y     = (double **) malloc(N * sizeof(double *));
  y_new = (double **) malloc(N * sizeof(double *));
  m     = (double **) malloc(N * sizeof(double *));
  w     = (SparseData **)  malloc(N * sizeof(SparseData *));
  nNbr  = (int *)          malloc(N * sizeof(int));

  if (  (nNbr == NULL)  || (x == NULL) || (y == NULL) || 
        (y_new == NULL) || (m == NULL) || (w==NULL) ) { 
    perror("[ERROR]:"); exit(1);
  }

  for (i=0; i<N; i++) {
    x[i]      = (double *) malloc(D * sizeof(double));
    y[i]      = (double *) malloc(D * sizeof(double));
    y_new[i]  = (double *) malloc(D * sizeof(double));
    m[i]      = (double *) malloc(D * sizeof(double));

    if( (x[i] == NULL) || (y[i] == NULL) || (y_new[i] == NULL) || (m[i] == NULL) ) { 
      perror("[ERROR]:"); exit(1);
    }
  }
}

void free_memory(){
  int i;
  if (VERBOSE) printf("[INFO]: Deallocating memory...\n");
  //free() data
  for (i=0; i<N; i++){
    free(x[i]);
    free(y[i]);
    free(y_new[i]);
    free(m[i]);
    free(w[i]);
  }
  // free() pointers
  free(x);
  free(y);
  free(y_new);
  free(m);
  free(w);
  free(nNbr);
}

void read_file(){
  int i,j;

  FILE * fp;
  fp = fopen (DATASET_PATH, "r");

  if (fp == NULL) { perror("[ERROR]: "); exit(1); }

  for (i=0; i<N; i++) 
    for (j=0; j<D; j++)
      if (EOF ==  fscanf(fp, "%lf", &x[i][j])) { perror("[ERROR]:"); exit(1); }

  fclose(fp);
}

void write_csv_file (char *message, double **a, const int ROW, const int COL){
  int i,j;

  FILE * fp;
  fp = fopen (OUTPUT_PATH_SERIAL, "w");

  if (fp == NULL){ perror("[ERROR]: "); exit(1); }

  fprintf(fp,"%s",message);

  for (i=0; i<ROW; i++) {
    for (j=0; j<COL; j++)
      if (EOF ==  fprintf(fp, "%lf, ", a[i][j])) {
        perror("[ERROR]:"); exit(1);
      }
    fprintf(fp,"\n");
  }

  fclose(fp);
}

// TODO:
int validate(){

  return 1;
}

void init_arr(){
  int i,j;
  for (i=0; i<N; i++){
    nNbr[i] = 0;
    for (j=0; j<D; j++){
      y[i][j]  = x[i][j];
      m[i][j]  = LDBL_MAX;
    }
  }
}


void meanshift(){
  clock_t start, end;
  double cpu_time_used;

  int iter=0;
  double norm = LDBL_MAX;

  init_arr();

  while (norm > EPSILON){
    iter++;

    //=========================================
    start = clock();  

    // find distances between each row of y and the rows of x 
    // that are BANDWIDTH or less distant.
    // And calculate kernels for these distances.
    rangesearch2sparse();

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("search: %f\n", cpu_time_used);

    //=========================================
    start = clock();

    // compute new y vector
    matrix_mult();

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("mult: %f\n", cpu_time_used);
    
    //=========================================
    start = clock();

    // normalize vector
    normalize();    

    // calculate meanshift
    calc_meanshift();

    // update y
    copy_2Darray(y_new, y,N,D);

    // calculate Frobenius norm
    norm = frob_norm();

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("other: %f\n", cpu_time_used);
    
    printf("[INFO]: Iteration %d - error %lf\n", iter, norm);
  } 

  if (VERBOSE)  write_csv_file("",y_new,N,D);
  
  free_memory();
}

// ADDED: id array to prevent looping until N
void rangesearch2sparse(){
  int i,j,k;
  double dist;

  // clock_t start, end;
  // double cpu_time_used;

  // malloc buffer for sparse matrix's rows
  double *buffer = (double *) malloc(N*sizeof(double));
  if(buffer == NULL) { perror("[ERROR]:");exit(1); }

  // malloc temporary id's for each iteration of y rows
  int *id = (int *) malloc(N*sizeof(int));
  if(id == NULL) { perror("[ERROR]:");exit(1); }

  for (i=0; i<N; i++){
    nNbr[i] = 0;
    
    /*
    start = clock();
    */
    
    for (j=0; j<N; j++){
      // make sure diagonal elements are 1
      if (i==j) {
        buffer[j] = 1;  
        //add index of distance to id array
        id[nNbr[i]] = j; 
        nNbr[i]++;
        continue;
      }

      // find distances inside radius
      dist = euclidean_distance(i,j);
      if (dist < BANDWIDTH*BANDWIDTH){  // radius^2 because I don't use sqrt() at dist
        buffer[j]= gaussian_kernel(dist);
        //add index of distance
        id[nNbr[i]] = j; 
        nNbr[i]++;
      }
      // unnecessary points
      else{
        buffer[j]=0;
      }
    }
    
    /*
    end = clock();
    if (i==0){
      cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
      printf("inside loop: %f\n", cpu_time_used);
    }*/

    // now nNbr[i] contains the final number of x neighbours

    w[i]  = (SparseData *) malloc(nNbr[i] * sizeof(SparseData));
    if(w[i]==NULL) {perror("[ERROR]: "); exit(1);}

    // first nNbr[i] elements of id array have the distances
    for (j=0; j<nNbr[i]; j++){
        w[i][j].xid      = id[j];
        w[i][j].distance = buffer[id[j]];
    }
  }

  free(buffer); free(id); // free temporary arrays
}

double euclidean_distance(const int first, const int second){
  int j;
  double dist = 0;
  for (j=0; j<D; j++)
    dist += (y[first][j] - x[second][j]) * (y[first][j] - x[second][j]);
  return dist;
}

void matrix_mult() {
  int i,j,k;
  for(i=0; i<N; i++){
    for(j=0; j<D; j++){
      y_new[i][j] = 0;
      for(k=0; k<nNbr[i]; k++){
        y_new[i][j] += w[i][k].distance * x[ w[i][k].xid ][j];
      }
    }
  }
}

void normalize(){
  int i,j;
  double s=0;

  for (i=0;i<N;i++){
    s = sum_of_row(i);
    for (j=0; j<D; j++)
      y_new[i][j] /= s;       
  }
}

double sum_of_row(const int row_index){
  int j;
  double sum=0;
  
  for (j=0; j<nNbr[row_index]; j++)
    sum += w[row_index][j].distance;
  return sum;
}

double frob_norm(){
  int i,j;
  double norm=0;
  for (i=0; i<N; i++)
    for (j=0; j<D; j++)
      norm += m[i][j] * m[i][j];
  return sqrt(norm);
}

void calc_meanshift(){
  int i,j;
  for (i=0;i<N;i++)
    for (j=0; j<D; j++)
      m[i][j] = y_new[i][j] - y[i][j]; 

}

void copy_2Darray(double **source, 
                  double **destination, 
                  const int ROW, 
                  const int COL)
{
  int i,j;
  for (i=0;i<ROW;i++)
    for (j=0; j<COL; j++)
      destination[i][j] = source[i][j];     
}

void print_2Darray(double **a, const int ROW, const int COL){
  int i,j;
  for (i=0;i<ROW;i++){
    for (j=0; j<COL; j++){
      printf("%lf \t",a[i][j]);
    }
  printf("\n");
  }
}

double gaussian_kernel(const double dist){
    return exp(- dist / (2.0*BANDWIDTH*BANDWIDTH));
}

