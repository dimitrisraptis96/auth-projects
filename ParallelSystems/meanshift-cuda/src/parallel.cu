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
    int xid;
    double distance;
} SparseData;

// host copies 
double *x_data, *y_data, **x, **y;

// device copies
double *d_x_data,*d_y_data,*d_y_new_data,*d_m_data, *d_sum;
double *d_reduction;
int *d_nNbr;
SparseData *d_sparse; 


__global__ void gpu_matrix_mult(int *nNbr, double *x, double *y, SparseData *w);
__global__ void gpu_normalize(int *nNbr, SparseData *w, double *y_new, double *sum);

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
  cpu_malloc();
  gpu_malloc();
  read_file();
  move_data_to_gpu();
}

//Contiguous memory allocation for 2D arrays
void cpu_malloc(){
  if(VERBOSE) printf("[INFO]: Allocating cpu memory..\n");

  // malloc pointers to rows 
  HANDLE_NULL( (x = (double **)     malloc(N * sizeof(double *))) );
  HANDLE_NULL( (y = (double **)     malloc(N * sizeof(double *))) );

  // malloc data of the arrays
  HANDLE_NULL( (x_data = (double *) malloc(N * D * sizeof(double))) );
  HANDLE_NULL( (y_data = (double *) malloc(N * D * sizeof(double))) );

  // assign pointers of data to arrays
  int i;
  for(i=0; i < N; i++){
    x[i]      = x_data      + i * D;
    y[i]      = y_data      + i * D;
  }
}


// Allocate memory for devive arrays
void gpu_malloc (){
  int size; 

  if(VERBOSE) printf("[INFO]: Allocating device memory..\n");

  // malloc data of the arrays
  size = N * D * sizeof(double);
  HANDLE_ERROR( cudaMalloc((void**)&d_x_data,    size) );
  HANDLE_ERROR( cudaMalloc((void**)&d_y_data,    size) );
  HANDLE_ERROR( cudaMalloc((void**)&d_y_new_data,size) );
  HANDLE_ERROR( cudaMalloc((void**)&d_m_data,    size) );

  // malloc d_reduction
  size = blocks_per_grid * sizeof(double);
  HANDLE_ERROR( cudaMalloc((void**)&d_reduction, size) );

  // malloc d_sum
  size = N * sizeof(double);
  HANDLE_ERROR( cudaMalloc((void**)&d_sum, size) );

  // malloc d_nNbr
  size = N * sizeof(int);
  HANDLE_ERROR( cudaMalloc((void**)&d_nNbr, size) );
}


void move_data_to_gpu(){
  if(VERBOSE) printf("[INFO]: Move data to device..\n");

  // move to device constant variables N_SIZE and D_SIZE
  HANDLE_ERROR( cudaMemcpyToSymbol (N_SIZE, &N, sizeof(int)) );
  HANDLE_ERROR( cudaMemcpyToSymbol (D_SIZE, &D, sizeof(int)) );

  // move to device x array
  HANDLE_ERROR( cudaMemcpy(d_x_data, x_data, N*D*sizeof(double), cudaMemcpyHostToDevice) );
}

void cpu_free_memory(){
  if (VERBOSE) printf("[INFO]: Deallocating cpu memory...\n");
  
  // free global memory
  free(x);
  free(y);
  free(x_data);
  free(y_data);
}

void gpu_free_memory(){
  if (VERBOSE) printf("[INFO]: Deallocating gpu memory...\n");

  // free gpu memory
  HANDLE_ERROR( cudaFree(d_x_data) );
  HANDLE_ERROR( cudaFree(d_y_data) );
  HANDLE_ERROR( cudaFree(d_m_data) );
  HANDLE_ERROR( cudaFree(d_y_new_data) );
  HANDLE_ERROR( cudaFree(d_reduction) );
  HANDLE_ERROR( cudaFree(d_nNbr) );
  HANDLE_ERROR( cudaFree(d_sparse) );
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
  HANDLE_NULL( (fp = fopen (OUTPUT_PATH_PARALLEL, "w")) );

  HANDLE_EOF( (fprintf(fp,"%s",message)) );

  for (i=0; i<ROW; i++) {
    for (j=0; j<COL; j++){
      HANDLE_EOF( fprintf(fp, "%lf, ", a[i][j]) ); 
    }
    HANDLE_EOF( fprintf(fp,"\n") );
  }

  HANDLE_EOF( (fclose(fp)) );
}


void meanshift(){
  clock_t start;
  
  int iter=0;
  double norm = DBL_MAX;

  gpu_init_arr <<<blocks_per_grid, threads_per_block>>> (d_nNbr, d_x_data, d_y_data, d_m_data);

  
  while (norm > EPSILON){
    iter++;
    
    // find distances and calculate kernels
    start = clock();
    rangesearch2sparse();
    printf("\t\trangesearch2sparse: %f\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);

    // compute new y vector
    start = clock();
    gpu_matrix_mult <<<blocks_per_grid, threads_per_block>>>(d_nNbr,d_x_data,d_y_new_data,d_sparse);
    printf("\t\tmatrix_mult: %f\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);

    // normalize vector
    start = clock();
    gpu_normalize <<<blocks_per_grid, threads_per_block>>>(d_nNbr,d_sparse,d_y_new_data,d_sum);    
    printf("\t\tnormalize: %f\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);

    // calculate meanshift
    start = clock();
    gpu_calc_meanshift <<<blocks_per_grid, threads_per_block>>>(d_m_data,d_y_new_data,d_y_data);
    printf("\t\tmeanshift: %f\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);

    /*if (VERBOSE){  
      HANDLE_ERROR( cudaMemcpy(y_data, d_m_data, N*D*sizeof(double), cudaMemcpyDeviceToHost) );
      write_csv_file("",y,N,D);
    }
    exit(1);*/
    //at 4097 point meanshift vector is faulty
    // update y
    start = clock();
    gpu_copy_2Darray <<<blocks_per_grid, threads_per_block>>>(d_y_new_data, d_y_data);
    printf("\t\tcopy: %f\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);
    
    // calculate Frobenius norm
    start = clock();
    gpu_frob_norm_shared <<<blocks_per_grid, threads_per_block>>>(d_m_data,d_reduction);
    printf("\t\tnorm: %f\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);

    start = clock();
    norm = sqrt ( finish_reduction() );
    printf("\t\tnorm-serial: %f\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);

    if (VERBOSE) printf("[INFO]: Iteration %d - error %lf\n\n", iter, norm);   
      
  }

  // copy results back to host
  if (VERBOSE){  
    HANDLE_ERROR( cudaMemcpy(y_data, d_y_new_data, N*D*sizeof(double), cudaMemcpyDeviceToHost) );
    write_csv_file("",y,N,D);
  }

  gpu_free_memory();
  cpu_free_memory();
}

// TODO: shared memory: the data within the block
__global__
void gpu_init_arr(int *nNbr, double *x, double *y, double *m)
{
  int tid = threadIdx.x  + blockIdx.x*blockDim.x;

  while (tid < N_SIZE*D_SIZE) {
    // nNbr[tid%N_SIZE] = 0;
    y[tid] = x[tid];
    m[tid] = DBL_MAX;
    
    tid += blockDim.x * gridDim.x;
  }

}
// TODO: reduction with shared memory
__global__ void gpu_calc_distances
(int y_row, double h, double *buffer, double *y, double *x){

  int tid = threadIdx.x  + blockIdx.x*blockDim.x;
  int k, x_arr_offset, y_arr_offset;
  
  double dist;

  while (tid < N_SIZE) {
    // diagonal elements
    if (y_row == tid){
      buffer[tid] = 1;
      tid += blockDim.x * gridDim.x;
      continue;
    }
    
    x_arr_offset = tid*D_SIZE;
    y_arr_offset = y_row*D_SIZE;  

    dist = 0; 
    for(k=0; k<D_SIZE; k++){
      dist += (y[y_arr_offset + k] - x[x_arr_offset + k])*(y[y_arr_offset + k] - x[x_arr_offset + k]);
    }

    // element inside radious
    if (dist < h*h)
      buffer[tid]= exp(-dist / (2.0*h*h));
    // unnecessary elements
    else
      buffer[tid]=0;

    tid += blockDim.x * gridDim.x;
  }
}

void rangesearch2sparse(){
  int i,j, count=0;
  int *id, *nNbr;
  double *buffer, *sum, *d_buffer;
  SparseData **w;

  // malloc host arrays
  HANDLE_NULL( (buffer  = (double *) malloc(N * sizeof(double))) );
  HANDLE_NULL( (sum     = (double *) malloc(N * sizeof(double))) );
  HANDLE_NULL( (id      = (int *)    malloc(N * sizeof(int))) );
  HANDLE_NULL( (nNbr    = (int *)    malloc(N * sizeof(int))) );
  HANDLE_NULL( (w  = (SparseData **) malloc(N * sizeof(SparseData *))) );

  // malloc device arrays
  HANDLE_ERROR( cudaMalloc((void**)&d_buffer, N * sizeof(double)) );

  for (i=0; i<N; i++){
    // find neighbours of y[i] row
    gpu_calc_distances <<<blocks_per_grid, threads_per_block>>>(i,BANDWIDTH,d_buffer,d_y_data,d_x_data);

    // get buffer from device
    HANDLE_ERROR( cudaMemcpy(buffer, d_buffer, N*sizeof(double), cudaMemcpyDeviceToHost) );
    
    // find neighbours (including diagonal elements)
    sum[i]=0;
    nNbr[i] = 0; 
    for(j=0;j<N;j++)
      if(buffer[j]>0){
        sum[i] += buffer[j];  //total dist sum of y[i] row
        id[nNbr[i]] = j;
        nNbr[i]++;
        count++;    // total elements of final sparse array
      }
    // here all the neighbours are known!

    HANDLE_NULL( (w[i] = (SparseData *) malloc(nNbr[i]*sizeof(SparseData))) );

    // nNbr[i] << N
    for (j=0; j<nNbr[i]; j++){
        w[i][j].xid      = id[j];
        w[i][j].distance = buffer[id[j]];
    }
  }

  // here sparse is ready!!!
  // SparseData *tmp_sparse;
  // HANDLE_NULL( (tmp_sparse = (SparseData *) malloc(count*sizeof(SparseData))) );
  // int index=0;
  // // TODO: eliminate most of the per-transfer overhead
  // for(i=0;i<N;i++){
  //   for(j=0;j<nNbr[i];j++){
  //     tmp_sparse[index] = w[i][j];
  //   }
  // }

  // // move 2D host sparse to 1D device sparse
  // HANDLE_ERROR( cudaFree(d_sparse) );
  // HANDLE_ERROR( cudaMalloc((void**) &d_sparse, count*sizeof(SparseData)) );
  // HANDLE_ERROR( cudaMemcpy(d_sparse, tmp_sparse, count*sizeof(SparseData), cudaMemcpyHostToDevice) );




  // move 2D host sparse to 1D device sparse
  HANDLE_ERROR( cudaFree(d_sparse) ); 
  HANDLE_ERROR( cudaMalloc((void**) &d_sparse, count*sizeof(SparseData)) );

  int offset=0;
  for(i=0;i<N;i++){
    HANDLE_ERROR( cudaMemcpy(&d_sparse[offset/sizeof(SparseData)], w[i], nNbr[i]*sizeof(SparseData), cudaMemcpyHostToDevice) );
    offset += nNbr[i]*sizeof(SparseData);
  }

  // move nNbr to device
  HANDLE_ERROR( cudaMemcpy(d_nNbr, nNbr, N*sizeof(int), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_sum,  sum,  N*sizeof(double), cudaMemcpyHostToDevice) );
  
  // free memory
  for(i=0;i<N;i++)  free(w[i]);
  free(nNbr); free(sum); free(w); free(buffer); free(id); // free memory
}


__global__
void gpu_matrix_mult(int *nNbr, double *x, double *y, SparseData *w)
{
  int tid = threadIdx.x  + blockIdx.x*blockDim.x;
  int k, i,j,sparse_offset=0;
  

  while(tid < N_SIZE*D_SIZE){
    i = tid/D_SIZE;
    j = tid%D_SIZE;
    //find the dynamic offset of rows that depend on the number of previous neighbours
    for (k=0;k<i;k++)
      sparse_offset += nNbr[k];
    
    y[tid] = 0;
    // multiply only the sparse element (all the other are 0's)
    for(k=0; k<nNbr[i]; k++)
      y[tid] += w[sparse_offset + k].distance * x[ (w[sparse_offset + k].xid * D_SIZE)/*row offset of x*/ + j ];

    tid += blockDim.x*gridDim.x;
  }
}

// TODO: reduction using shared memory
__global__ void gpu_normalize(int *nNbr, SparseData *w, double *y_new, double *sum) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i,sparse_offset=0;
  // double sum=0;
  

  while(tid < N_SIZE*D_SIZE) {
    i = tid/D_SIZE;
    //find the dynamic offset of rows that depend on the number of previous neighbours
    for (int k=0;k<i;k++)
      sparse_offset += nNbr[k];

    // TODO: sum can be shared if D>>2

    // find sum of current row
    // for (int k=0; k<nNbr[i]; k++)
    //   sum += w[sparse_offset+k].distance; 

    y_new[tid] /= sum[i];
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

__global__ void gpu_copy_2Darray(double *src, double *dst)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while(tid < N_SIZE*D_SIZE){
    dst[tid] = src[tid];
    tid += gridDim.x+blockDim.x;
  }
}


// TODO: non-shared implementation (use code from gpu_normalize)
__global__ void gpu_frob_norm_shared(double *m, double *final){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ double cache[threads_per_block];
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

  // only 1rst thread of each block
  if (cacheIndex == 0)
    final[blockIdx.x] = cache[0];
}


// calculate last step of reduction on CPU because it's more efficient
double finish_reduction(){
  double  sum;
  // double result[blocks_per_grid];
  double *result;
  HANDLE_NULL( (result = (double *) malloc(blocks_per_grid * sizeof(double))) );
  // int *nNbr;
  // HANDLE_ERROR( cudaMemcpy(nNbr, d_nNbr, N*sizeof(int), cudaMemcpyDeviceToHost) );

  HANDLE_ERROR( cudaMemcpy( result,
                            d_reduction,
                            blocks_per_grid*sizeof(double),
                            cudaMemcpyDeviceToHost ) );
  exit(1);
  sum = 0;
  for (int i=0; i<blocks_per_grid; i++)
      sum += result[i];
  
  // free(result);
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
