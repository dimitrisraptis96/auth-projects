#ifndef __CUDA_HELPERS_H__
#define __CUDA_HELPERS_H__

// cuda error handler
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


// declare cuda kernels
__global__ void gpu_init_arr(double *x, double *y, double *m);

__global__ void gpu_calc_distances(int y_row, double h, double *buffer, double *y, double *x);

__global__ void gpu_calc_meanshift(double *m, double *y_new, double *y);

__global__ void gpu_copy_2Darray(double *src, double *dst);

__global__ void gpu_frob_norm(double *m, double *final);

__global__ void gpu_frob_norm_shared(double *m, double *final);

// exhaustive routines
__global__ void gpu_pdist(double h, double *out, double *y, double *x);

__global__ void gpu_matrix_mult_exh(double *x, double *y, double *dist);

__global__ void gpu_normalize_exh(double *y_new, double *dist) ;


#endif

