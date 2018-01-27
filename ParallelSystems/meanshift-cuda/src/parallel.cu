#include "../include/parallel.h"
#include "../include/global_vars.h"
#include "../include/cuda_helpers.h"

// constant device N and D
__device__ __constant__ int N_SIZE; 
__device__ __constant__ int D_SIZE;

// grid and block sizes
const int threads_per_block = 256;
const int blocks_per_grid  = MIN(32, (N+threads_per_block-1) / threads_per_block);

typedef struct {
    int j;
    double distance;
} SparseData;

// host copies
double **x, **y;
double *x_data, *y_data;

// device copies
double **d_x,**d_y,**d_y_new,**d_m;
double *d_x_data,*d_y_data,*d_y_new_data,*d_m_data;
int *d_nNbr;
SparseData **d_w; 


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
  int i;
  
  if(VERBOSE) printf("[INFO]: Allocating cpu memory..\n");

  // malloc pointers to rows 
  x     = (double **) malloc(N * sizeof(double *));
  y     = (double **) malloc(N * sizeof(double *));

  HANDLE_NULL( x );
  HANDLE_NULL( y );

  // malloc data of the arrays
  x_data      = (double *) malloc(N * D * sizeof(double));
  y_data      = (double *) malloc(N * D * sizeof(double));

  HANDLE_NULL( x_data );
  HANDLE_NULL( y_data );

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

  // malloc nNbr
  size = N * sizeof(int);
  HANDLE_ERROR( cudaMalloc((void**)&d_nNbr, size) );

  // malloc w indexes of rows
  size = N * sizeof(SparseData *);
  HANDLE_ERROR( cudaMalloc((void**)&d_w, size) );
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

__global__ void gpu_init_arr( int *d_nNbr, 
                              double *d_x_data, 
                              double *d_y_data, 
                              double *d_m_data)
{
  int tid = threadIdx.x  + blockIdx.x*blockDim.x;
  
  // TODO: shared memory: the data within the block

  while (tid < N_SIZE*D_SIZE) {
    if(tid%D_SIZE == 0) d_nNbr[tid/D_SIZE] = 0; // can be optized ???
    d_y_data[tid] = d_x_data[tid];
    d_m_data[tid] = DBL_MAX;
  }
}


void meanshift(){
  int iter=0;
  double norm = DBL_MAX;

  gpu_init_arr <<<blocks_per_grid, threads_per_block>>> (d_nNbr, d_x_data, d_y_data, d_m_data); ///not ready!!!!!!!!

  HANDLE_ERROR( cudaMemcpy(y_data, d_y_data, N*D*sizeof(double), cudaMemcpyDeviceToHost) );

  write_csv_file("",y,N,D);

  /*while (norm > EPSILON){
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
      printf("[INFO]: Iteration %d - error %lf\n", iter, norm);
    }
  } 
  if (VERBOSE)  write_csv_file("",y_new,N,D);*/
  
  // free_memory();
}


// void rangesearch2sparse(){
//   int i,j, index;
//   double dist;

//   // malloc buffer for sparse matrix's rows
//   double *buffer = (double *) malloc(N*sizeof(double));
//   if(buffer == NULL) { perror("[ERROR]:");exit(1); }

//   for (i=0; i<N; i++){
//     for (j=0; j<N; j++){
//       // make sure diagonal elements are 1
//       if (i==j) {
//         buffer[j] = 1; nNbr[i]++; 
//         continue;
//       }

//       // find distances inside radius
//       dist = euclidean_distance(i,j);
//       if (dist < BANDWIDTH*BANDWIDTH){  // radius^2 because I don't use sqrt() at dist
//         buffer[j]= gaussian_kernel(dist);
//         nNbr[i]++;
//       }
//       // unnecessary points
//       else{
//         buffer[j]=0;
//       }
//     }

//     //for (i=0)

//     // cudamalloc sparse matrix (w) rows
//     w[i]  = (SparseData *) malloc(nNbr[i] * sizeof(SparseData));
//     if(w[i]==NULL) {perror("[ERROR]: "); exit(1);}

//     index = 0;
//     for (j=0; j<N; j++){
//       if (buffer[j] > 0){
//         w[i][index].j        = j;
//         w[i][index].distance = buffer[j]; 
//         index++;
//       }
//     }
//   }
// }

// /*__global__ void matrix_mult(int *d_nNbr, double **d_y_new, SparseData **d_w)
// {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int idy = blockIdx.y * blockDim.y + threadIdx.y;
//   int k;

//   if((idx < N) && (idy < D)) {
//     y_new[idx][idy] = 0;
//     for(k=0; k<d_nNbr[i]; k++)
//         d_y_new[idx][idy] += d+w[idx][k].distance * x[ w[idx][k].j ][idy];
//   }
// }*/

// void matrix_mult() {
//   int i,j,k;
//   for(i=0; i<N; i++){
//     for(j=0; j<D; j++){
//       y_new[i][j] = 0;
//       for(k=0; k<nNbr[i]; k++)
//           y_new[i][j] += w[i][k].distance * x[ w[i][k].j ][j];
//     }
//   }
// }

// /*__global__ void normalize(int *d_nNbr, double **d_y_new, SparseData **d_w)
// {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int idy = blockIdx.y * blockDim.y + threadIdx.y;
//   double sum=0;  //shared within block for optimization

//   if((idx < N) && (idy < D)) {
//     if (threadIdx.x == 0) sum = sum_of_row(i);
//     d_y_new[idx][idy] /= sum;
//   }
// }*/


// __device__ double sum_of_row(const int row_index){
//   // TODO call this from device
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   double sum=0;
  
//   if(idx < nNbr[row_index])
//     sum += w[row_index][j].distance;
//   __synchronized //wait all threads to sum the
//   return sum; // make sure it returns the correct sum
// }

// void normalize(){
//   int i,j;
//   double s=0;

//   for (i=0;i<N;i++){
//     s = sum_of_row(i);
//     for (j=0; j<D; j++)
//       y_new[i][j] /= s;       
//   }
// }

// double sum_of_row(const int row_index){
//   int j;
//   double sum=0;
  
//   for (j=0; j<nNbr[row_index]; j++)
//     sum += w[row_index][j].distance;
//   return sum;
// }

// /*__global__ double frob_norm(int *d_nNbr, double **d_y_new, SparseData **d_w)
// {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int idy = blockIdx.y * blockDim.y + threadIdx.y;
//   double sum=0;  //shared within block for optimization

//   if((idx < N) && (idy < D)) {
//     if (threadIdx.x == 0) sum = sum_of_row(i);
//     d_y_new[idx][idy] /= sum;
//   }
// }*/


// double frob_norm(){
//   int i,j;
//   double norm=0;
//   for (i=0; i<N; i++)
//     for (j=0; j<D; j++)
//       norm += m[i][j] * m[i][j];
//   return sqrt(norm);
// }

// void calc_meanshift(){
//   int i,j;
//   for (i=0;i<N;i++)
//     for (j=0; j<D; j++)
//       m[i][j] = y_new[i][j] - y[i][j];       
// }

// void copy_2Darray(double **source, double **destination, const int ROW, const int COL){
//   int i,j;
//   for (i=0;i<ROW;i++)
//     for (j=0; j<COL; j++)
//       destination[i][j] = source[i][j];
// }

void print_2Darray(double **a, const int ROW, const int COL){
  int i,j;
  for (i=0;i<ROW;i++){
    for (j=0; j<COL; j++){
      printf("%lf \t",a[i][j]);
    }
  printf("\n");
  }
}

// double gaussian_kernel(const double dist){
//     return exp(- dist / (2.0*BANDWIDTH*BANDWIDTH));
// }

// double euclidean_distance(const int first, const int second){
//   int j;
//   double dist = 0;
//   for (j=0; j<D; j++)
//     dist += (y[first][j] - x[second][j]) * (y[first][j] - x[second][j]);
//   return dist;
// }
