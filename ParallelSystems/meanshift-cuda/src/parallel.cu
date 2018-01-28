#include "../include/parallel.h"
#include "../include/global_vars.h"
#include "../include/cuda_helpers.h"

// constant device N and D
__device__ __constant__ int N_SIZE; 
__device__ __constant__ int D_SIZE;

// grid and block sizes
const int threads_per_block = 256;
const int blocks_per_grid = 32; 

typedef struct {
    int j;
    double distance;
} SparseData;

// host copies
double *x_data, *y_data, *buffer, **x, **y;
int *nNbr;
SparseData **w; 

// device copies
double *d_x_data,*d_y_data,*d_y_new_data,*d_m_data,*d_sum, *d_buffer;
double **d_x,**d_y,**d_y_new,**d_m;
int *d_nNbr;
SparseData **d_w; 

__global__ void gpu_matrix_mult(int *nNbr, double *x, double *y, SparseData **w);
__global__ void gpu_normalize(int *nNbr, SparseData **w, double *y_new);

extern "C"
void parallel(){
  printf("[INFO]: CUDA-GPU IMPLEMENTATION\n");
  printf("=============================\n");

  struct timeval startwtime, endwtime;
  double seq_time;

  init_parallel();
  
  gettimeofday (&startwtime, NULL);
  //------------------------------
  meanshift();
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);

  /*printf("\n\nIs test PASSed? %s\n\n", validate_parallel()?"YES":"NO");
  printf("===============================================\n\n");*/
  printf("\n\n[INFO]: Parallel meanshift wall clock time = %f\n", seq_time);

}

void init_parallel(){
  // blocks_per_grid = MIN(32, (N+threads_per_block-1) / threads_per_block);

  cpu_malloc();
  gpu_malloc();
  read_file();
  move_data_to_gpu();
}

//Contiguous memory allocation for 2D arrays
void cpu_malloc(){
  int i;
  
  if(VERBOSE) printf("[INFO]: Allocating cpu memory..\n");

  // malloc pointers to rows 
  HANDLE_NULL( (x = (double **) malloc(N * sizeof(double *))) );
  HANDLE_NULL( (y = (double **) malloc(N * sizeof(double *))) );
  HANDLE_NULL( (w = (SparseData **) malloc(N * sizeof(SparseData *))) );

  // malloc data of the arrays
  HANDLE_NULL( (x_data = (double *) malloc(N * D * sizeof(double))) );
  HANDLE_NULL( (y_data = (double *) malloc(N * D * sizeof(double))) );
  HANDLE_NULL( (nNbr = (int *) malloc(N * sizeof(int))) );

  // assign pointers of data to arrays
  for(i=0; i < N; i++){
    x[i]      = x_data      + i * D;
    y[i]      = y_data      + i * D;
  }
}


// Allocate memory for devive variables and copy x host data to device
void gpu_malloc (){
  int size; 

  if(VERBOSE) printf("[INFO]: Allocating device memory..\n");
  
  // malloc pointers of rows
  size = N * sizeof(double *);
  HANDLE_ERROR( cudaMalloc((void**)&d_x,    size) );
  HANDLE_ERROR( cudaMalloc((void**)&d_y,    size) );
  HANDLE_ERROR( cudaMalloc((void**)&d_y_new,size) );
  HANDLE_ERROR( cudaMalloc((void**)&d_m,    size) );


  // malloc data of the arrays
  size = N * D * sizeof(double);
  HANDLE_ERROR( cudaMalloc((void**)&d_x_data,    size) );
  HANDLE_ERROR( cudaMalloc((void**)&d_y_data,    size) );
  HANDLE_ERROR( cudaMalloc((void**)&d_y_new_data,size) );
  HANDLE_ERROR( cudaMalloc((void**)&d_m_data,    size) );

  // malloc d_sum
  size = threads_per_block * sizeof(double);
  HANDLE_ERROR( cudaMalloc((void**)&d_sum, size) );

  // malloc d_nNbr
  size = N * sizeof(int);
  HANDLE_ERROR( cudaMalloc((void**)&d_nNbr, size) );

  // malloc d_buffer
  size = N * sizeof(double);
  HANDLE_ERROR( cudaMalloc((void**)&d_buffer, size) );

  // malloc d_w indexes of rows
  size = N * sizeof(SparseData *);
  HANDLE_ERROR( cudaMalloc((void**)&d_w, size) );

  // get back indexes from device (need them in rangesearch2sparse)
  HANDLE_ERROR( cudaMemcpy(w, d_w, sizeof(SparseData**), cudaMemcpyDeviceToHost) );
}

void move_data_to_gpu(){
  if(VERBOSE) printf("[INFO]: Move data to device..\n");

  // move device constant variables N_SIZE and D_SIZE
  HANDLE_ERROR( cudaMemcpyToSymbol (N_SIZE, &N, sizeof(int)) );
  HANDLE_ERROR( cudaMemcpyToSymbol (D_SIZE, &D, sizeof(int)) );

  // move device global variable d_x and d_x_data
  HANDLE_ERROR( cudaMemcpy(d_x,      x,      N*sizeof(double *), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_x_data, x_data, N*D*sizeof(double), cudaMemcpyHostToDevice) );
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
      if (EOF ==  fscanf(fp, "%lf", &x[i][j])) { perror("[ERROR]:"); exit(1); }

  fclose(fp);
}

void write_csv_file (char *message, double **a, const int ROW, const int COL){
  int i,j;

  FILE * fp;
  fp = fopen (OUTPUT_PATH, "w");

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


void meanshift(){
  int iter=0;
  double norm = DBL_MAX;

  gpu_init_arr <<<blocks_per_grid, threads_per_block>>> (d_nNbr, d_x_data, d_y_data, d_m_data);

  
  while (norm > EPSILON){
    iter++;
    // find distances and calculate kernels
    rangesearch2sparse();
    printf("-1\n");

    // compute new y vector
    gpu_matrix_mult <<<blocks_per_grid, threads_per_block>>>(d_nNbr,d_x_data,d_y_data,d_w);

    // normalize vector
    gpu_normalize <<<blocks_per_grid, threads_per_block>>>(d_nNbr,d_w,d_y_new_data);    

    // calculate meanshift
    gpu_calc_meanshift <<<blocks_per_grid, threads_per_block>>>(d_m_data,d_y_new_data,d_y_data);

    // update y
    gpu_copy_2Darray <<<blocks_per_grid, threads_per_block>>>(d_y_new_data, d_y_data);

    // calculate Frobenius norm
    gpu_frob_norm_shared <<<blocks_per_grid, threads_per_block>>>(d_m_data,d_sum);
    
    //wait kernel calls to be executed
    HANDLE_ERROR( cudaDeviceSynchronize() );

    norm = sqrt ( finish_reduction() );

    if (VERBOSE){
      printf("[INFO]: Iteration %d - error %lf\n", iter, norm);
    }
  }

  // copy results back to host
  if (VERBOSE){  
    HANDLE_ERROR( cudaMemcpy(y_data, d_y_data, N*D*sizeof(double), cudaMemcpyDeviceToHost) );
    write_csv_file("",y,N,D);
  }

  // gpu_free_memory();
  // cpu_free_memory();
}

// TODO: shared memory: the data within the block
__global__
void gpu_init_arr(int *nNbr, double *x, double *y, double *m)
{
  int tid = threadIdx.x  + blockIdx.x*blockDim.x;

  while (tid < N_SIZE*D_SIZE) {
    nNbr[tid%N_SIZE] = 0;
    y[tid] = x[tid];
    m[tid] = DBL_MAX;
    
    tid += blockDim.x * gridDim.x;
  }
}

// TODO: reduction with shared memory
__global__ void gpu_find_neighbours
(int y_row, double h, double *buffer, double *y, double *x, double *n){

  int tid = threadIdx.x  + blockIdx.x*blockDim.x;
  int i,j, x_arr_offset, y_arr_offset, nbr=0;

  __shared__ int cache[threads_per_block];
  int cacheIndex = threadIdx.x;
  
  double dist;

  // i=tid;
  while (tid < N_SIZE) {
    i = tid/D_SIZE;
    // diagonal elements
    if (y_row == tid){
      buffer[tid] = 1; nbr++;
      tid += blockDim.x * gridDim.x;
      continue;
    }
    
    x_arr_offset = tid*D_SIZE;
    y_arr_offset = y_row*D_SIZE;  

    dist = 0; 
    for(j=0; j<D_SIZE; j++){
      dist += (y[y_arr_offset + j] - x[x_arr_offset + j])*(y[y_arr_offset + j] - x[x_arr_offset + j]);
    }

    // element inside radious
    if (dist < h*h){
      buffer[tid]= exp(-dist / (2.0*h*h));
      nbr++;
     }
    // unnecessary elements
    else{
      buffer[tid]=0;
    }

    tid += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[cacheIndex] = nbr;

  __syncthreads();

  /// reduction
  int k = blockDim.x/2;
    while (k != 0) {
        if (cacheIndex < k)
            cache[cacheIndex] += cache[cacheIndex + k];
        __syncthreads();
        k /= 2;
    }

    if (cacheIndex == 0)
        n[blockIdx.x] = cache[0];

}

void rangesearch2sparse(){
  int i,j, index, size;
  double *buffer;
  SparseData *tmp;

  // malloc buffer for sparse matrix's rows
  HANDLE_NULL( (buffer = (double *) malloc(N*sizeof(double))) );

  for (i=0; i<N; i++){
    // find neighbours of y[i] row
    gpu_find_neighbours <<<blocks_per_grid, threads_per_block>>>
        (i,BANDWIDTH,d_buffer,d_y_data,d_x_data,d_sum);

    // number of neighbours
    nNbr[i] = (int) finish_reduction(); printf("nNbr[%d]=%d\n",i,nNbr[i]);

    // get buffer from device
    HANDLE_ERROR( cudaMemcpy(buffer, d_buffer, N*sizeof(double), cudaMemcpyDeviceToHost) );
    
    // for (j=0; j<N; j++)
    //   printf("%lf ",buffer[j]);

    // malloc w[i] device memory
    HANDLE_ERROR( cudaMalloc((void**)&w[i], nNbr[i] * sizeof(SparseData)) );
    // move value of d_w[i] to device
    HANDLE_ERROR( cudaMemcpy(&d_w[i], &w[i], sizeof(SparseData*), cudaMemcpyHostToDevice) );

    index = 0;
    HANDLE_NULL( (tmp = (SparseData *) malloc(nNbr[i]*sizeof(SparseData))) );
    for (j=0; j<N; j++){
      if (buffer[j] > 0){
        tmp[index].j        = j;
        tmp[index].distance = buffer[j];
        index++;
      }
    // printf("index = %d\n",index );
    }

    // for (j=0; j<nNbr[i]; j++){
    //   printf("%lf ",tmp[j].distance);
    // }

// printf("d\n");
    HANDLE_ERROR( cudaMemcpy(&d_w[i], tmp, nNbr[i]*sizeof(SparseData), cudaMemcpyHostToDevice) );
        // printf("3\n");
    free (tmp);
    exit(1);
  }
  free(buffer);
}


__global__
void gpu_matrix_mult(int *nNbr, double *x, double *y, SparseData **w)
{
  int tid = threadIdx.x  + blockIdx.x*blockDim.x;
  int k, offset, i;
  
  i = tid/D_SIZE;

  while(tid < N_SIZE*D_SIZE){
    offset = i*D_SIZE;
    y[tid] = 0;
    for(k=0; k<nNbr[i]; k++)
      y[tid] += w[i][k].distance * x[ w[i][k].j + offset ];  
    tid += blockDim.x*gridDim.x;
  }
}

// TODO: reduction using shared memory
__global__ void gpu_normalize(int *nNbr, SparseData **w, double *y_new) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i,j;
  double sum=0;
  
  i = tid/D_SIZE;
  j = tid%D_SIZE;

  while(tid < N_SIZE*D_SIZE) {
    for (int k=0; k<nNbr[i]; k++)
      sum += w[i][j].distance; // not contiguous memory, [][] necessary

    y_new[tid] /= sum;
    tid += gridDim.x*blockDim.x;
  }
}

__global__ void gpu_calc_meanshift(double *m, double *y_new, double *y)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while(tid < N_SIZE*D_SIZE){
    m[tid] = y_new[tid] - y[tid];
    tid += gridDim.x+blockDim.x;
  }
}

__global__ void gpu_copy_2Darray(double *source, double *destination)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while(tid < N_SIZE*D_SIZE){
    destination[tid] = source[tid];
    tid += gridDim.x+blockDim.x;
  }
}

/*// TODO: non-shared implementation (use code from gpu_normalize)
__global__ void gpu_frob_norm(double *m, double *result){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  double norm = 0;
  while (tid < N_SIZE*D_SIZE) {
      norm += m[tid] * m[tid];
      tid += blockDim.x*gridDim.x;
  }
  
  // set the cache values
  cache[cacheIndex] = norm;
  
  // synchronize threads in this block
  __syncthreads();

  // for reductions, threads_per_block must be a power of 2
  int i = blockDim.x/2;
  while (i != 0) {
      if (cacheIndex < i)
          cache[cacheIndex] += cache[cacheIndex + i];
      __syncthreads();
      i /= 2;
  }

  if (cacheIndex == 0)
      result[blockIdx.x] = cache[0];
}*/

// TODO: non-shared implementation (use code from gpu_normalize)
__global__ void gpu_frob_norm_shared(double *m, double *result){
  __shared__ float cache[threads_per_block];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  double norm = 0;
  while (tid < N_SIZE*D_SIZE) {
      norm += m[tid] * m[tid];
      tid += blockDim.x*gridDim.x;
  }
  
  // set the cache values
  cache[cacheIndex] = norm;
  
  // synchronize threads in this block
  __syncthreads();

  // for reductions, threads_per_block must be a power of 2
  int i = blockDim.x/2;
  while (i != 0) {
      if (cacheIndex < i)
        cache[cacheIndex] += cache[cacheIndex + i];

      __syncthreads();
      i /= 2;
  }

  if (cacheIndex == 0)
      result[blockIdx.x] = cache[0];
}

// calculate last step of reduction on CPU because it's more efficient
double finish_reduction(){
  double *result, sum;

  // malloc result array
  HANDLE_NULL( (result = (double *) malloc(threads_per_block*sizeof(double))) );

  HANDLE_ERROR( cudaMemcpy( result, 
                            d_sum,
                            blocks_per_grid*sizeof(float),
                            cudaMemcpyDeviceToHost ) );
  sum = 0;
  for (int i=0; i<blocks_per_grid; i++)
      sum += result[i];
  return sum;
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
