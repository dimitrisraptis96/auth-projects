#include "../include/global_vars.h"
#include "../include/parallel.h"
#include "../include/helpers.h"

// TODO: prepare_gpu()
// TODO: malloc continuous memory !!!

typedef struct {
    int j;
    long double distance;
} SparseData;

// host copies
  long double ** x,y;

// device copies
  long double ** d_x,d_y,d_y_new,d_m;
  long double *  d_x_data, d_y_data, d_y_new_data, d_m_data;
  int *d_nNbr;
  SparseData **d_w;   


void parallel(){
  printf("[INFO]: CUDA-GPU IMPLEMENTATION\n");
  printf("=============================\n");

  struct timeval startwtime, endwtime;
  double seq_time;

  init_prallel();
  
  gettimeofday (&startwtime, NULL);
  //------------------------------
  meanshift();
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (long double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);

  /*printf("\n\nIs test PASSed? %s\n\n", validate_parallel()?"YES":"NO");
  printf("===============================================\n\n");*/
  printf("\n\n[INFO]:Serial meanshift wall clock time = %f\n", seq_time);

}

// Check if cuda API calls returned successfully
void cuda_error_handler(cudaError_t err){
  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
    exit(1);
  }
}

void init_parallel(){
  cpu_malloc();
  gpu_malloc();
  read_file();
  move_data_to_gpu();
}

//Contiguous memory allocation for 2D arrays
void cpu_malloc(){
  int i;
  
  if(VERBOSE) printf("[INFO]: Allocating contiguous memory..\n");

  // malloc pointers to rows 
  x     = (long double **) malloc(N * sizeof(long double *));
  y     = (long double **) malloc(N * sizeof(long double *));

  if (x == NULL || y == NULL) {perror("[ERROR]:"); exit(1);} 

  // malloc data of the arrays
  x_data      = malloc(N * D * sizeof(long double));
  y_data      = malloc(N * D * sizeof(long double));

  if(x_data == NULL || y_data == NULL) {perror("[ERROR]:"); exit(1);}

  // assign pointers of data to arrays
  for(i=0; i < N; i++){
    x[i]      = x_data      + i * D;
    y[i]      = y_data      + i * D;
  }
}


// Allocate memory for devive variables and copy x host data to device
void gpu_malloc (){
  int size; 

  // malloc pointers of rows
  size = N * sizeof(long double *);
  cuda_error_handler( cudaMalloc(&d_x,    size) );
  cuda_error_handler( cudaMalloc(&d_y,    size) );
  cuda_error_handler( cudaMalloc(&d_y_new,size) );
  cuda_error_handler( cudaMalloc(&d_m,    size) );

  // malloc data of the arrays
  size = N * D * sizeof(long double);
  cuda_error_handler( cudaMalloc(&d_x_data,    size) );
  cuda_error_handler( cudaMalloc(&d_y_data,    size) );
  cuda_error_handler( cudaMalloc(&d_y_new_data,size) );
  cuda_error_handler( cudaMalloc(&d_m_data,    size) );

  //malloc nNbr
  size = N * sizeof(int);
  cuda_error_handler( cudaMalloc(&d_nNbr, size) );

  //malloc w indexes of rows
  size = N * sizeof(SparseData *);
  cuda_error_handler( cudaMalloc(&d_w, size) );
}

void move_data_to_gpu(){
  int size;
  size = N * sizeof(long double *);
  cuda_error_handler( cudaMemcpy(d_x,       x,      size,  cudaMemcpyHostToDevice) );
  size = N * D * sizeof(long double);
  cuda_error_handler( cudaMemcpy(d_x_data,  x_data, size,  cudaMemcpyHostToDevice) );
}

void free_memory(){
  int i;
  if (VERBOSE) printf("[INFO]: Deallocating memory...\n");
  //free() data
  for (i=0; i<N; i++){
    free(x[i]);
    free(y[i]);
  }
  // free() pointers
  free(x);
  free(y);
}

void read_file(){
  int i,j;

  FILE * fp;
  fp = fopen (DATASET_PATH, "r");

  if (fp == NULL) { perror("[ERROR]: "); exit(1); }

  for (i=0; i<N; i++) 
    for (j=0; j<D; j++)
      if (EOF ==  fscanf(fp, "%Lf", &x[i][j])) { perror("[ERROR]:"); exit(1); }

  fclose(fp);
}

void write_csv_file (char *message, long double **a, const int ROW, const int COL){
  int i,j;

  FILE * fp;
  fp = fopen (OUTPUT_PATH, "w");

  if (fp == NULL){ perror("[ERROR]: "); exit(1); }

  fprintf(fp,"%s",message);

  for (i=0; i<ROW; i++) {
    for (j=0; j<COL; j++)
      if (EOF ==  fprintf(fp, "%Lf, ", a[i][j])) {
        perror("[ERROR]:"); exit(1);
      }
    fprintf(fp,"\n");
  }

  fclose(fp);
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

__global__ void init_arr(int *d_nNbr, long double **d_y_data, long double **d_m_data)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  
  // TODO: shared memory: the data within the block

  if (id < N*D) {
    d_nNbr[id] = 0; // can be optized
    d_y_data[id] = d_x_data[id];
    d_m_data[id] = LDMX_MAX;
  }
}

/*void set_threads(int size){

  if ((size) < 32) {
    threads_num = size;
    blocks_num  = 1;
  }
  else {
    threads_num = size;
    blocks_num  = ceil(float(size / threads_num);
  }
}
*/


void meanshift(){
  int iter=0;
  int nblocks, nthreads;
  long double norm = LDBL_MAX;

  init_arr <<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> (d_nNbr, d_y, d_m); ///not ready!!!!!!!!

  while (norm > EPSILON){
    iter++;

    // find distances between each row of y and the rows of x that are BANDWIDTH or less distant.
    // And calculate kernels for these distances.
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

    if (VERBOSE){
      printf("[INFO]: Iteration %d - error %Lf\n", iter, norm);
    }
  } 
  if (VERBOSE)  write_csv_file("",y_new,N,D);
  
  free_memory();
}


void rangesearch2sparse(){
  int i,j, index;
  long double dist;

  // malloc buffer for sparse matrix's rows
  long double *buffer = (long double *) malloc(N*sizeof(long double));
  if(buffer == NULL) { perror("[ERROR]:");exit(1); }

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      // make sure diagonal elements are 1
      if (i==j) {
        buffer[j] = 1; nNbr[i]++; 
        continue;
      }

      // find distances inside radius
      dist = euclidean_distance(i,j);
      if (dist < BANDWIDTH*BANDWIDTH){  // radius^2 because I don't use sqrt() at dist
        buffer[j]= gaussian_kernel(dist);
        nNbr[i]++;
      }
      // unnecessary points
      else{
        buffer[j]=0;
      }
    }

    // malloc sparse matrix (w) rows
    w[i]  = (SparseData *) malloc(nNbr[i] * sizeof(SparseData));
    if(w[i]==NULL) {perror("[ERROR]: "); exit(1);}

    index = 0;
    for (j=0; j<N; j++){
      if (buffer[j] > 0){
        w[i][index].j        = j;
        w[i][index].distance = buffer[j]; 
        index++;
      }
    }
  }
}

/*__global__ void matrix_mult(int *d_nNbr, long double **d_y_new, SparseData **d_w)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int k;

  if((idx < N) && (idy < D)) {
    y_new[idx][idy] = 0;
    for(k=0; k<d_nNbr[i]; k++)
        d_y_new[idx][idy] += d+w[idx][k].distance * x[ w[idx][k].j ][idy];
  }
}*/

void matrix_mult() {
  int i,j,k;
  for(i=0; i<N; i++){
    for(j=0; j<D; j++){
      y_new[i][j] = 0;
      for(k=0; k<nNbr[i]; k++)
          y_new[i][j] += w[i][k].distance * x[ w[i][k].j ][j];
    }
  }
}

/*__global__ void normalize(int *d_nNbr, long double **d_y_new, SparseData **d_w)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  long double sum=0;  //shared within block for optimization

  if((idx < N) && (idy < D)) {
    if (threadIdx.x == 0) sum = sum_of_row(i);
    d_y_new[idx][idy] /= sum;
  }
}*/

/*
__device__ long double sum_of_row(const int row_index){
  // TODO call this from device
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  long double sum=0;
  
  if(idx < nNbr[row_index])
    sum += w[row_index][j].distance;
  __synchronized //wait all threads to sum the
  return sum; // make sure it returns the correct sum
}*/

void normalize(){
  int i,j;
  long double s=0;

  for (i=0;i<N;i++){
    s = sum_of_row(i);
    for (j=0; j<D; j++)
      y_new[i][j] /= s;       
  }
}

long double sum_of_row(const int row_index){
  int j;
  long double sum=0;
  
  for (j=0; j<nNbr[row_index]; j++)
    sum += w[row_index][j].distance;
  return sum;
}

/*__global__ long double frob_norm(int *d_nNbr, long double **d_y_new, SparseData **d_w)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  long double sum=0;  //shared within block for optimization

  if((idx < N) && (idy < D)) {
    if (threadIdx.x == 0) sum = sum_of_row(i);
    d_y_new[idx][idy] /= sum;
  }
}*/


long double frob_norm(){
  int i,j;
  long double norm=0;
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

void copy_2Darray(long double **source, long double **destination, const int ROW, const int COL){
  int i,j;
  for (i=0;i<ROW;i++)
    for (j=0; j<COL; j++)
      destination[i][j] = source[i][j];
}

void print_2Darray(long double **a, const int ROW, const int COL){
  int i,j;
  for (i=0;i<ROW;i++){
    for (j=0; j<COL; j++){
      printf("%Lf \t",a[i][j]);
    }
  printf("\n");
  }
}

long double gaussian_kernel(const long double dist){
    return exp(- dist / (2.0*BANDWIDTH*BANDWIDTH));
}

long double euclidean_distance(const int first, const int second){
  int j;
  long double dist = 0;
  for (j=0; j<D; j++)
    dist += (y[first][j] - x[second][j]) * (y[first][j] - x[second][j]);
  return dist;
}
