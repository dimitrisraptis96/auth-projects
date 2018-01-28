/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __CUDA_HELPERS_H__
#define __CUDA_HELPERS_H__

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


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


__global__ void gpu_init_arr(int *nNbr, double *x, double *y, double *m);
__global__ void gpu_calc_distances(int y_row, double h, int *nNbr, double *buffer, double **y, double **x);
// __global__ void gpu_matrix_mult(int *nNbr, double *x, double *y, SparseData **w);
// __global__ void gpu_normalize(int *nNbr, SparseData **w, double *y_new);
__global__ void gpu_calc_meanshift(double *m, double *y_new, double *y);
__global__ void gpu_copy_2Darray(double *source, double *destination);
__global__ void gpu_frob_norm_shared(double *m, double *result);

#endif

