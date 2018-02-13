#include "../include/parallel.h"
#include "../include/global_vars.h"
#include "../include/kernels.cuh"

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
double *arr_data, **arr;

// device copies
double *d_x_data,*d_y_data,*d_y_new_data,*d_m_data,*d_reduction, *d_sum, *d_Pdist;
int *d_nNbr;
SparseData *d_sparse; 

// declare them here because they include SparseData struct
__global__ void gpu_matrix_mult(int *nNbr, double *x, double *y, SparseData *w);
__global__ void gpu_normalize(int *nNbr, SparseData *w, double *y_new, double *sum);

extern "C"
void parallel(){
  printf("===============================\n");
  printf("[INFO]: CUDA-GPU IMPLEMENTATION\n");
  printf("===============================\n");
  printf("[INFO]: bandwidth=%lf\n",BANDWIDTH);
  printf("[INFO]: epsilon=%lf\n\n",EPSILON);

  struct timeval startwtime, endwtime;
  double seq_time;

  // choose exhaustive or sparse version
  int version = choose_version();
  init_parallel(version);
  
  gettimeofday (&startwtime, NULL);
  //------------------------------
  cuda_meanshift(version);
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);

  printf("\n\n[FINAL]: parallel meanshift wall clock time = %f\n\n", seq_time);
}

void init_parallel(int version){
  if (VERBOSE) 
    printf ("[INFO]: VERSION: %s & %s\n\n", 
          (version==VERSION_EXHAUSTIVE) ? "exhaustive": "sparse",
          (USE_SHARED)                  ? "shared"    : "non-shared");
  
  cpu_malloc();
  gpu_malloc(version);
  read_file();
  move_data_to_gpu();
}

// choose version according to N value and global memory size
int choose_version(){
  // get device props
  cudaDeviceProp  prop;
  HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
  
  double bytes = prop.totalGlobalMem / sizeof(double);

  return (N*N > bytes/2) ? VERSION_SPARSE : VERSION_EXHAUSTIVE; 
}

// ====================================================================
// ====================================================================
//                      MEMORY ALLOCATION
// ====================================================================
// ====================================================================

//Contiguous memory allocation for 2D array
void cpu_malloc(){
  if(VERBOSE) printf("[INFO]: allocate cpu memory..\n");

  // malloc pointers to rows 
  HANDLE_NULL( (arr = (double **)     malloc(N * sizeof(double *))) );

  // malloc data of the array
  HANDLE_NULL( (arr_data = (double *) malloc(N * D * sizeof(double))) );

  // assign pointers of data to array
  int i;
  for(i=0; i < N; i++){
    arr[i]      = arr_data      + i * D;
  }
}


// Allocate memory for devive arrays
void gpu_malloc (int version){
  int size; 

  if(VERBOSE) printf("[INFO]: allocate device memory..\n");

  if(version == VERSION_EXHAUSTIVE){
    // malloc d_Pdist
    size = N * N * sizeof(double);
    HANDLE_ERROR( cudaMalloc((void**)&d_Pdist, size) );
  }

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
  if(VERBOSE) printf("[INFO]: move data to device..\n\n");

  // move to device constant variables N_SIZE and D_SIZE
  HANDLE_ERROR( cudaMemcpyToSymbol (N_SIZE, &N, sizeof(int)) );
  HANDLE_ERROR( cudaMemcpyToSymbol (D_SIZE, &D, sizeof(int)) );

  // move to device x array
  HANDLE_ERROR( cudaMemcpy(d_x_data, arr_data, N*D*sizeof(double), cudaMemcpyHostToDevice) );
}

void cpu_free_memory(){
  if (VERBOSE) printf("[INFO]: deallocate cpu memory...\n");
  
  // free global memory
  free(arr);
  free(arr_data);
}

void gpu_free_memory(int version){
  if (VERBOSE) printf("\n[INFO]: deallocate gpu memory...\n");

  // free gpu memory
  HANDLE_ERROR( cudaFree(d_x_data) );
  HANDLE_ERROR( cudaFree(d_y_data) );
  HANDLE_ERROR( cudaFree(d_m_data) );
  HANDLE_ERROR( cudaFree(d_y_new_data) );
  HANDLE_ERROR( cudaFree(d_reduction) );
  HANDLE_ERROR( cudaFree(d_nNbr) );
  switch (version){
    case VERSION_EXHAUSTIVE:
      HANDLE_ERROR( cudaFree(d_Pdist) );
      break;

    case VERSION_SPARSE:
      HANDLE_ERROR( cudaFree(d_sparse) );
      break;
  }
}

// ====================================================================
// ====================================================================
//                      I/O OPERATIONS
// ====================================================================
// ====================================================================

void read_file(){
  int i,j;

  FILE * fp;
  HANDLE_NULL( (fp = fopen (DATASET_PATH, "r")) );

  for (i=0; i<N; i++) 
    for (j=0; j<D; j++)
      HANDLE_EOF( (fscanf(fp, "%lf", &arr[i][j])) );

  HANDLE_EOF( (fclose(fp)) );
}

void write_csv_file (char *message, double **a, const int ROW, const int COL){
  int i,j;

  FILE * fp;
  HANDLE_NULL( (fp = fopen (OUTPUT_PATH_PARALLEL, "w")) );

  if (message != NULL)  HANDLE_EOF( (fprintf(fp,"%s",message)) );

  for (i=0; i<ROW; i++) {
    for (j=0; j<COL; j++){
      if (j == COL-1){
        HANDLE_EOF( fprintf(fp, "%lf", a[i][j]) );
      } 
      else {
        HANDLE_EOF( fprintf(fp, "%lf, ", a[i][j]) ); 
      }
    }
    HANDLE_EOF( fprintf(fp,"\n") );
  }

  HANDLE_EOF( (fclose(fp)) );
}

// ====================================================================
// ====================================================================
//                      MEANSHIFT IMPLEMENTATION
// ====================================================================
// ====================================================================

void cuda_meanshift(int version){

  int iter=0;
  double norm = DBL_MAX;

  gpu_init_arr <<<blocks_per_grid, threads_per_block>>> (d_x_data, d_y_data, d_m_data);

  
  while (norm > EPSILON){
    iter++;

    switch(version){

      case VERSION_EXHAUSTIVE:

        // find distances and calculate kernels
        gpu_pdist<<<blocks_per_grid, threads_per_block>>>(BANDWIDTH,d_Pdist, d_y_data, d_x_data );

        // compute new y vector
        gpu_matrix_mult_exh <<<blocks_per_grid, threads_per_block>>>(d_x_data,d_y_new_data,d_Pdist);

        // normalize vector
        gpu_normalize_exh <<<blocks_per_grid, threads_per_block>>>(d_y_new_data,d_Pdist);    
        
        break;

      case VERSION_SPARSE:
        
        // find distances and calculate kernels
        rangesearch2sparse();

        // compute new y vector
        gpu_matrix_mult <<<blocks_per_grid, threads_per_block>>>(d_nNbr,d_x_data,d_y_new_data,d_sparse);

        // normalize vector
        gpu_normalize <<<blocks_per_grid, threads_per_block>>>(d_nNbr,d_sparse,d_y_new_data,d_sum);    
        
        break;
    }

    // calculate meanshift
    gpu_calc_meanshift <<<blocks_per_grid, threads_per_block>>>(d_m_data,d_y_new_data,d_y_data);
    
    // update y
    gpu_copy_2Darray <<<blocks_per_grid, threads_per_block>>>(d_y_new_data, d_y_data);
    
    // calculate Frobenius norm
    (USE_SHARED) ? 
          // shared norm
          gpu_frob_norm_shared <<<blocks_per_grid, threads_per_block>>>(d_m_data,d_reduction):
          // non-shared norm
          gpu_frob_norm        <<<blocks_per_grid, threads_per_block>>>(d_m_data,d_sum);

    norm = sqrt ( finish_reduction() );

    if (VERBOSE) printf("[INFO]: iteration %d - error %lf\n", iter, norm);   
  }

  // copy results back to host
  if (VERBOSE){  
    HANDLE_ERROR( cudaMemcpy(arr_data, d_y_new_data, N*D*sizeof(double), cudaMemcpyDeviceToHost) );
    write_csv_file(NULL,arr,N,D);
  }

  gpu_free_memory(version);
  cpu_free_memory();
}

// initialize the device arrays
__global__
void gpu_init_arr(double *x, double *y, double *m)
{
  int tid = threadIdx.x  + blockIdx.x*blockDim.x;

  while (tid < N_SIZE*D_SIZE) {
    y[tid] = x[tid];
    m[tid] = DBL_MAX;
    
    tid += blockDim.x * gridDim.x;
  }
}


// ====================================================================
// ====================================================================
//                      EXHAUSTIVE VERSION
// ====================================================================
// ====================================================================

// calculate the distances between x and y points
__global__ 
void gpu_pdist(double h, double *out, double *y, double *x)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  double tmp, dist;
  
  while (tid < N_SIZE) {

    for(int r=0; r<N_SIZE; r++){ //outer loop
      dist = 0;

     // calculate distances of y[r] point and all x[tid] points
     for(int i=0; i<D_SIZE; i++){
        tmp = y[r*D_SIZE +i] - x[tid*D_SIZE + i];
        dist += tmp*tmp;
      }

      // y is rows and x is columns
      if (dist == 0) {
        // diagonal elements
        out[r*N_SIZE+tid] = 1;
      }
      else if(dist < h*h){
        // distances inside radius
        out[r*N_SIZE+tid] = exp(-dist / (2.0*h*h));
      }
      else {
        // unnecessary distances
        out[r*N_SIZE+tid] = 0;
      }
    }

    tid += gridDim.x*blockDim.x;
  }
}

__global__
void gpu_matrix_mult_exh(double *x, double *y, double *dist)
{
  int tid = threadIdx.x  + blockIdx.x*blockDim.x;
  int k, i,j;
  
  while(tid < N_SIZE*D_SIZE){
    i = tid/D_SIZE; // i between [0,N_SIZE-1]
    j = tid%D_SIZE; // j between [0,D_SIZE-1]
    
    y[tid] = 0;

    for(k=0; k<N_SIZE; k++)
      y[tid] += dist[i*N_SIZE + k] * x[k*D_SIZE + j];

    tid += blockDim.x*gridDim.x;
  }
}


__global__ void gpu_normalize_exh(double *y_new, double *dist) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i;
  double sum;

  while(tid < N_SIZE*D_SIZE) {
    i = tid/D_SIZE;

    // calculate sum of the array
    sum = 0;
    for (int k=0; k<N_SIZE; k++)
      sum += dist[i*N_SIZE + k]; 
    // normalize here
    y_new[tid] /= sum;

    tid += gridDim.x*blockDim.x;
  }
}

// ====================================================================
// ====================================================================
//                      SPARSE VERSION
// ====================================================================
// ====================================================================

void rangesearch2sparse(){
  int i,j, count=0;
  int *id, *nNbr;
  double *buffer, *sum, *d_buffer;
  SparseData **w;

  // malloc host arrays
  HANDLE_NULL( (buffer  = (double *) malloc(N * sizeof(double))) ); // distances of y[i] row
  HANDLE_NULL( (sum     = (double *) malloc(N * sizeof(double))) ); // sum of i row
  HANDLE_NULL( (id      = (int *)    malloc(N * sizeof(int))) );    // id's of distances inside bandwidth
  HANDLE_NULL( (nNbr    = (int *)    malloc(N * sizeof(int))) );    // number of neighbours of y[i] row
  HANDLE_NULL( (w  = (SparseData **) malloc(N * sizeof(SparseData *))) );

  // malloc device array
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
  HANDLE_ERROR( cudaFree(d_buffer) );
  // here sparse is ready!!!
  
  // making contiguous sparse eliminates most of the per-transfer overhead
  SparseData *tmp_sparse;
  HANDLE_NULL( (tmp_sparse = (SparseData *) malloc(count*sizeof(SparseData))) );
  int index=0;
  for(i=0;i<N;i++){
    for(j=0;j<nNbr[i];j++){
      tmp_sparse[index] = w[i][j];
      index++;
    }
  }

  // move 2D host sparse to 1D device sparse
  HANDLE_ERROR( cudaFree(d_sparse) ); // need these for iterations after the 1rst one
  HANDLE_ERROR( cudaFree(d_sum) );
  HANDLE_ERROR( cudaFree(d_nNbr) );
  HANDLE_ERROR( cudaMalloc((void**) &d_sparse, count*sizeof(SparseData)) );
  HANDLE_ERROR( cudaMemcpy(d_sparse, tmp_sparse, count*sizeof(SparseData), cudaMemcpyHostToDevice) );

  // move nNbr to device
  HANDLE_ERROR( cudaMemcpy(d_nNbr, nNbr, N*sizeof(int),    cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_sum,  sum,  N*sizeof(double), cudaMemcpyHostToDevice) );
  
  // free host memory
  for(i=0;i<N;i++)
    free(w[i]);
  free(tmp_sparse);
  free(nNbr); free(sum); free(w); free(buffer); free(id);
}
            
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

    // calculate distances
    dist = 0; 
    for(k=0; k<D_SIZE; k++){
      dist += (y[y_arr_offset + k] - x[x_arr_offset + k])*(y[y_arr_offset + k] - x[x_arr_offset + k]);
    }

    // element inside radious
    if (dist < h*h){
      buffer[tid]= exp(-dist / (2.0*h*h));
    }
    // unnecessary elements
    else{
      buffer[tid]=0;
    }

    tid += blockDim.x * gridDim.x;
  }
}


__global__
void gpu_matrix_mult(int *nNbr, double *x, double *y, SparseData *w)
{
  int tid = threadIdx.x  + blockIdx.x*blockDim.x;
  int k, i,j,sparse_offset;
  

  while(tid < N_SIZE*D_SIZE){
    // i and j indexes of flattened 2D array x
    i = tid/D_SIZE;
    j = tid%D_SIZE;
    
    //find the dynamic offset of rows that depend on the number of previous neighbours
    sparse_offset=0;
    for (k=0;k<i;k++)
      sparse_offset += nNbr[k];
    
    y[tid] = 0;
    // multiply only the sparse element (all the other are 0's)
    for(k=0; k<nNbr[i]; k++)
      y[tid] += w[sparse_offset + k].distance * x[ (w[sparse_offset + k].xid * D_SIZE)/*row offset of x*/ + j ];

    tid += blockDim.x*gridDim.x;
  }
}


__global__ void gpu_normalize(int *nNbr, SparseData *w, double *y_new, double *sum) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i;

  while(tid < N_SIZE*D_SIZE) {
    i = tid/D_SIZE;

    y_new[tid] /= sum[i];
    tid += gridDim.x*blockDim.x;
  }
}

// ====================================================================
// ====================================================================
//                      SAME FOR BOTH VERSIONS
// ====================================================================
// ====================================================================

__global__ void gpu_calc_meanshift(double *m, double *y_new, double *y)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while(tid < N_SIZE*D_SIZE){
    m[tid] = y_new[tid] - y[tid];
    tid += gridDim.x*blockDim.x;
  }
}

__global__ void gpu_copy_2Darray(double *src, double *dst)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while(tid < N_SIZE*D_SIZE){
    dst[tid] = src[tid];
    tid += gridDim.x*blockDim.x;
  }
}

__global__ void gpu_frob_norm(double *m, double *final){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < N_SIZE){
    double sum = 0;
    for (int i=0;i<D_SIZE;i++){
      sum += m[tid*D_SIZE+i]*m[tid*D_SIZE+i];
    }
    final[tid] = sum; // here final's size is N

    tid += blockDim.x*gridDim.x;
  }
}

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
    final[blockIdx.x] = cache[0]; //here final's size is blocks_per_grid
}


// calculate last step of reduction on CPU because it's more efficient
double finish_reduction(){
  double  sum, *result, *device;
  int size;

  // decide the size and the device array according to shared/non-shared option
  if (USE_SHARED) {
    size   = blocks_per_grid;
    device = d_reduction;
  } 
  else {
    size   = N;
    device = d_sum;
  }

  HANDLE_NULL( (result = (double *) malloc(size * sizeof(double))) );

  HANDLE_ERROR( cudaMemcpy( result,
                            device,
                            size*sizeof(double),
                            cudaMemcpyDeviceToHost ) );
  sum = 0;
  for (int i=0; i<size; i++){
      sum += result[i];
  }
  free(result);
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