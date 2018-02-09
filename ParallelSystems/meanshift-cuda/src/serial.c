#include "../include/serial.h"
#include "../include/global_vars.h"

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
  printf("=================================\n");
  printf("[INFO]: SERIAL-CPU IMPLEMENTATION\n");
  printf("=================================\n");
  printf("[INFO]: bandwidth=%lf\n",BANDWIDTH);
  printf("[INFO]: epsilon=%lf\n\n",EPSILON);

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

  printf("\n\n[FINAL]: serial meanshift wall clock time = %f\n\n", seq_time);
}

void init_serial(){
  memory_allocation();
  read_file();
}

//Memory allocation for 2D arrays
void memory_allocation(){
  int i;
  
  if (VERBOSE) printf("[INFO]: allocate memory...\n\n");

  HANDLE_NULL( (x     = (double **)     malloc(N * sizeof(double *))) );
  HANDLE_NULL( (y     = (double **)     malloc(N * sizeof(double *))) );
  HANDLE_NULL( (y_new = (double **)     malloc(N * sizeof(double *))) );
  HANDLE_NULL( (m     = (double **)     malloc(N * sizeof(double *))) );
  HANDLE_NULL( (w     = (SparseData **) malloc(N * sizeof(SparseData *))) );
  HANDLE_NULL( (nNbr  = (int *)         malloc(N * sizeof(int))) );

  for (i=0; i<N; i++) {
    HANDLE_NULL( (x[i]      = (double *) malloc(D * sizeof(double))) );
    HANDLE_NULL( (y[i]      = (double *) malloc(D * sizeof(double))) );
    HANDLE_NULL( (y_new[i]  = (double *) malloc(D * sizeof(double))) );
    HANDLE_NULL( (m[i]      = (double *) malloc(D * sizeof(double))) );
  }
}

void free_memory(){
  int i;
  if (VERBOSE) printf("\n[INFO]: deallocate memory...\n");
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
  HANDLE_NULL( (fp = fopen (DATASET_PATH, "r")) );

  for (i=0; i<N; i++) 
    for (j=0; j<D; j++)
      HANDLE_EOF( (fscanf(fp, "%lf", &x[i][j])) );

  HANDLE_EOF( (fclose(fp)) );
}

void write_csv_file (char *message, double **a, const int ROW, const int COL){
  int i,j;

  FILE * fp;
  HANDLE_NULL( (fp = fopen (OUTPUT_PATH_SERIAL, "w")) );

  HANDLE_EOF( (fprintf(fp,"%s",message)) );

  for (i=0; i<ROW; i++) {
    for (j=0; j<COL; j++){
      HANDLE_EOF( (fprintf(fp, "%lf, ", a[i][j])) ); 
    }
    HANDLE_EOF( (fprintf(fp,"\n")) );
  }

  HANDLE_EOF( (fclose(fp)) );
}

static void init_arr(){
  int i,j;
  for (i=0; i<N; i++){
    nNbr[i] = 0;
    for (j=0; j<D; j++){
      y[i][j]  = x[i][j];
      m[i][j]  = LDBL_MAX;
    }
  }
}


static void meanshift(){
  int iter=0;
  double norm = LDBL_MAX;

  init_arr();

  while (norm > EPSILON){
    iter++;

    // find distances and calculate sparse
    rangesearch2sparse();

    // compute new y vector
    matrix_mult();

    // normalize vector
    normalize();    
    
    // calculate meanshift
    calc_meanshift();

    // update y
    copy_2Darray(y_new, y,N,D);

    // calculate Frobenius norm
    norm = frob_norm();
    
    printf("[INFO]: iteration %d - error %lf\n", iter, norm);
  } 

  if (VERBOSE)  write_csv_file("",y_new,N,D);
  
  free_memory();
}

// ADDED: id array to prevent looping until N
static void rangesearch2sparse(){
  int i,j,k, *id;
  double dist, *buffer;

  // malloc buffer and id temporary arrays
  HANDLE_NULL( (buffer = (double *) malloc(N*sizeof(double))) );
  HANDLE_NULL( (id     = (int *)    malloc(N*sizeof(int)))    );

  for (i=0; i<N; i++){
    nNbr[i] = 0;
    
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
    // now nNbr[i] contains the final number of x neighbours

    // malloc sparse row i
    free(w[i]);
    HANDLE_NULL( (w[i]  = (SparseData *) malloc(nNbr[i] * sizeof(SparseData))) );

    // first nNbr[i] elements of id array have the distances
    for (j=0; j<nNbr[i]; j++){
        w[i][j].xid      = id[j];
        w[i][j].distance = buffer[id[j]];
    }

  }

  free(buffer); free(id); // free temporary arrays
}

static double euclidean_distance(const int first, const int second){
  int j;
  double dist = 0;
  for (j=0; j<D; j++)
    dist += (y[first][j] - x[second][j]) * (y[first][j] - x[second][j]);
  return dist;
}

static void matrix_mult() {
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

static void normalize(){
  int i,j;
  double s=0;

  for (i=0;i<N;i++){
    s = sum_of_row(i);
    for (j=0; j<D; j++)
      y_new[i][j] /= s;       
  }
}

static double sum_of_row(const int row_index){
  int j;
  double sum=0;
  
  for (j=0; j<nNbr[row_index]; j++)
    sum += w[row_index][j].distance;
  return sum;
}

static double frob_norm(){
  int i,j;
  double norm=0;
  for (i=0; i<N; i++)
    for (j=0; j<D; j++)
      norm += m[i][j] * m[i][j];
  return sqrt(norm);
}

static void calc_meanshift(){
  int i,j;
  for (i=0;i<N;i++)
    for (j=0; j<D; j++)
      m[i][j] = y_new[i][j] - y[i][j]; 

}

static void copy_2Darray(double **source, 
                  double **destination, 
                  const int ROW, 
                  const int COL)
{
  int i,j;
  for (i=0;i<ROW;i++)
    for (j=0; j<COL; j++)
      destination[i][j] = source[i][j];     
}

static void print_2Darray(double **a, const int ROW, const int COL){
  int i,j;
  for (i=0;i<ROW;i++){
    for (j=0; j<COL; j++){
      printf("%lf \t",a[i][j]);
    }
  printf("\n");
  }
}

static double gaussian_kernel(const double dist){
    return exp(- dist / (2.0*BANDWIDTH*BANDWIDTH));
}

