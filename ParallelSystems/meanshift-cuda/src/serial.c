#include "../include/global_vars.h"
#include "../include/serial.h"
#include "../include/helpers.h"

int *nNbr;

long double **x;
long double **y;
long double **y_new;
long double **m;
long double **d;
int **id;

// TODO: free()

void serial(){
  printf("[INFO]: SERIAL IMPLEMENTATION\n");
  printf("=============================\n");

  struct timeval startwtime, endwtime;
  double seq_time;

  init_serial();
  
  gettimeofday (&startwtime, NULL);
  //------------------------------
  meanshift();
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (long double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);
  
  /*printf("\n\nIs test PASSed? %s\n\n", validate_serial()?"YES":"NO");
  printf("===============================================\n\n");*/
  printf("Serial meanshift wall clock time = %f\n", seq_time);
}

void init_serial(){
  memory_allocation();
  read_file();
}

//Memory allocation for 2D arrays
void memory_allocation(){
  int i;
  
  if (VERBOSE) printf("[INFO]: Allocating memory\n");

  nNbr  = (int *)     malloc(N * sizeof(int));
  x     = (long double **) malloc(N * sizeof(long double *));
  y     = (long double **) malloc(N * sizeof(long double *));
  y_new = (long double **) malloc(N * sizeof(long double *));
  m     = (long double **) malloc(N * sizeof(long double *));
  d     = (long double **) malloc(N * sizeof(long double *));
  id    = (int **)    malloc(N * sizeof(int *));

  if( (nNbr == NULL) || (x == NULL) || (y == NULL) || 
      (y_new == NULL) || (m == NULL) || (d == NULL) || 
      (id == NULL)) { 
    printf("[ERROR]: Memory allocation error 1!\n");
    exit(1);
  }

  for (i=0; i<N; i++) {
    x[i]      = (long double *) malloc(D * sizeof(long double));
    y[i]      = (long double *) malloc(D * sizeof(long double));
    y_new[i]  = (long double *) malloc(D * sizeof(long double));
    m[i]      = (long double *) malloc(D * sizeof(long double));
    d[i]      = (long double *) malloc(N * sizeof(long double));
    id[i]     = (int *) malloc   (N * sizeof(int));

    if( (x[i] == NULL) || (y[i] == NULL) || (y_new[i] == NULL) || (m[i] == NULL) ) { 
      printf("[ERROR]: Memory allocation error 2!\n");
      exit(1);
    }
  }
}

/*void read_file_bin()
{
    FILE *f;
    f = fopen(DATASET_PATH, "rb");
    fseek(f, 0L, SEEK_END);
    int pos = ftell(f);
    fseek(f, 0L, SEEK_SET);

    int number_elements = pos / sizeof(long double);
    long double *x = (long double *) malloc(sizeof *x * number_elements);
    int tmp = fread(x, sizeof *x, number_elements, f);
    fclose(f);
}*/

void read_file(){
  int i,j;

  FILE * fp;
  fp = fopen (DATASET_PATH, "r");

  if (fp == NULL){perror("[ERROR]: "); exit(1);}

  for (i=0; i<N; i++) 
    for (j=0; j<D; j++){
      if (EOF ==  fscanf(fp, "%Lf", &x[i][j])) {
        perror("[ERROR]:"); exit(1);
      }
      // x[i][j]/=10000;
    }

  fclose(fp);
}

void write_csv_file (char *message, long double **a, const int ROW, const int COL){
  int i,j;

  FILE * fp;
  fp = fopen (OUTPUT_PATH, "w+a");

  if (fp == NULL){ perror("[ERROR]: "); exit(1); }

  fprintf(fp,"%s",message);

  for (i=0; i<ROW; i++) {
    for (j=0; j<COL; j++)
      if (EOF ==  fprintf(fp, "%Lf, ", a[i][j])) {
        perror("[ERROR]:"); exit(1);
      }
    fprintf(fp,"\n");
  }
  fprintf(fp,"\n\n");

  fclose(fp);
}

// TODO
int validate(){
  /*int i,j;
  long double tmp;

  FILE * fp;
  fp = fopen (VALIDATION_PATH, "r");
  if (fp == NULL){ perror("[ERROR]:"); exit(1);}

  for (i=0; i<N; i++) {
    for (j=0; j<D; j++) {
      if(EOF == fscanf(fp, "%Lf", &tmp)) { perror("[ERROR]:"); exit(1);}

      if(!(fabs(tmp - k_dist[i][j]) < PRECISION)){
        printf("[INFO]: Validation failed:\n");
        // printf("k_dist=%Lf\n",k_dist[i][j]);
        // printf("validate_tmp=%Lf\n",tmp);
        // printf("i=%d j=%d\n",i,j);
        fclose(fp);
        return 0;
      }
    }
  }
  fclose(fp);
  return 1;*/
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
    for (j=0; j<N; j++){
      d[i][j]  = -1.0;
      id[i][j] = -1;
    }
  }
}

// Find the x points that are closer than EPSILON for each y point
void meanshift(){
  int iter=0;
  long double norm = LDBL_MAX;

  struct timeval startwtime, endwtime;
  double seq_time;
  
  init_arr();

  while (norm > EPSILON){
    iter++;

//==========================================================================
    // find distances between each row of y and the rows of x 
    // that are BANDWIDTH or less distant.
    // And calculate kernels for these distances.
    rangesearch();

    // compute new y vector
    cpu_matrix_mult(y_new,d,x,N,N,D);

    // normalize vector
    normalize(y_new,N,D);    

    // calculate meanshift
    calc_meanshift(m,y_new,y,N,D);

    // update y
    copy_2Darray(y_new, y,N,D);

    // calculate Frobenius norm
    norm = frob_norm(m,N,D);

    if (VERBOSE){
      printf("[INFO]: Iteration %d - error %Lf\n", iter, norm);
    }
  }

  // if (VERBOSE)  write_csv_file("",y_new,N,D);
}


void rangesearch(){
  int i,j;
  long double dist;

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      if (i==j) {d[i][j] = 1; continue;}

      dist = euclidean_distance(i,j);

      // (dist<BANDWIDTH*BANDWIDTH) ? (d[i][j]= gaussian_kernel(dist)) : (d[i][j]=0);
      if (dist<BANDWIDTH*BANDWIDTH){
        d[i][j]= gaussian_kernel(dist);
        nNbr[i]++;
        // count++;
      }
      else{
        d[i][j]=0;
      }
    }
    // exit(1);
  }
}

// a*b = c
void cpu_matrix_mult(long double **result, long double **a, long double **b, const int ROW1, const int COL1, const int COL2) {
  int i,j,k;
  for(i=0; i<ROW1; i++){
    for(j=0; j<COL2; j++){
      result[i][j] = 0;
      for(k=0; k<COL1; k++){
          result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void normalize(long double **a, const int ROW, const int COL){
  int i,j;
  long double s=0;

  for (i=0;i<ROW;i++){
    s = sum_of_row(d,i,N);
    // printf("%Lf", s);
    for (j=0; j<COL; j++)
      a[i][j] = a[i][j]/s;       
  }
}

long double sum_of_row(long double **a, const int row_index, const int COL){
  int j;
  long double sum=0;
  for (j=0; j<COL; j++)
    sum += a[row_index][j];
  return sum;
}

long double frob_norm(long double **a, const int ROW, const int COL) {
  int i,j;
  long double norm=0;
  for (i=0; i<ROW; i++)
    for (j=0; j<COL; j++)
      norm += a[i][j] * a[i][j];
  return sqrt(norm);
}

void calc_meanshift(long double **result, long double **a, long double **b, const int ROW, const int COL){
  int i,j;
  for (i=0;i<ROW;i++)
    for (j=0; j<COL; j++)
      result[i][j] = a[i][j] - b[i][j];       
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

//  W = spfun( @(x) exp( -x / (2*h^2) ), W );
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

/*typedef struct {
    int i,j,distance;
} SparseData;

SparseData * sparse(long double * d, int data_num){

  int i,j, index=0;

  SparseData *w  = (SparseData *)malloc(data_num * sizeof(struct   SparseData));
  if(w==NULL) {perror("Error: "); exit(1);}

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      if (d[i][j] > 0){
        w[index]->i        = i;
        w[index]->j        = j;
        w[index]->distance = d[i][j]; 
        index++;     
      }
    }
  }
  return w;

}*/