#include "../include/global_vars.h"
#include "../include/serial.h"
#include "../include/helpers.h"

int *nNbr;

double **x;
double **y;
double **y_new;
double **m;
double **d;
int **id;

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

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
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
  x     = (double **) malloc(N * sizeof(double *));
  y     = (double **) malloc(N * sizeof(double *));
  y_new = (double **) malloc(N * sizeof(double *));
  m     = (double **) malloc(N * sizeof(double *));
  d     = (double **) malloc(N * sizeof(double *));
  id    = (int **)    malloc(N * sizeof(int *));

  if( (nNbr == NULL) || (x == NULL) || (y == NULL) || 
      (y_new == NULL) || (m == NULL) || (d == NULL) || 
      (id == NULL)) { 
    printf("[ERROR]: Memory allocation error 1!\n");
    exit(1);
  }

  for (i=0; i<N; i++) {
    x[i]      = (double *) malloc(D * sizeof(double));
    y[i]      = (double *) malloc(D * sizeof(double));
    y_new[i]  = (double *) malloc(D * sizeof(double));
    m[i]      = (double *) malloc(D * sizeof(double));
    d[i]      = (double *) malloc(N * sizeof(double));
    id[i]     = (int *) malloc   (N * sizeof(int));

    if( (x[i] == NULL) || (y[i] == NULL) || (y_new[i] == NULL) || (m[i] == NULL) ) { 
      printf("[ERROR]: Memory allocation error 2!\n");
      exit(1);
    }
  }
}

void read_file(){
  int i,j;

  FILE * fp;
  fp = fopen (DATASET_PATH, "r");

  if (fp == NULL){perror("[ERROR]: "); exit(1);}

  for (i=0; i<N; i++) 
    for (j=0; j<D; j++){
      if (EOF ==  fscanf(fp, "%lf", &x[i][j])) {
        perror("[ERROR]:"); exit(1);
      }
    }

  fclose(fp);
}

void write_csv_file (char *message, double **a, const int ROW, const int COL){
  int i,j;

  FILE * fp;
  fp = fopen (OUTPUT_PATH, "w+a");

  if (fp == NULL){ perror("[ERROR]: "); exit(1); }

  fprintf(fp,"%s",message);

  for (i=0; i<ROW; i++) {
    for (j=0; j<COL; j++)
      if (EOF ==  fprintf(fp, "%lf, ", a[i][j])) {
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
  double tmp;

  FILE * fp;
  fp = fopen (VALIDATION_PATH, "r");
  if (fp == NULL){ perror("[ERROR]:"); exit(1);}

  for (i=0; i<N; i++) {
    for (j=0; j<D; j++) {
      if(EOF == fscanf(fp, "%lf", &tmp)) { perror("[ERROR]:"); exit(1);}

      if(!(fabs(tmp - k_dist[i][j]) < PRECISION)){
        printf("[INFO]: Validation failed:\n");
        // printf("k_dist=%lf\n",k_dist[i][j]);
        // printf("validate_tmp=%lf\n",tmp);
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
      m[i][j]  = DBL_MAX;
      d[i][j]  = -1.0;
      id[i][j] = -1;
    }
  }
}

// Find the x points that are closer than EPSILON for each y point
void meanshift(){
  int iter=0;
  double norm = DBL_MAX;
  
  // printf("HERE\n");
  init_arr();

  while (norm > EPSILON){
    iter++;
    //find neighbours with BANDWIDTH
    rangesearch();
    // print_2Darray(d,1,N);
    // write_csv_file("d test\n",d,N,N);

    // compute new y vector
    cpu_matrix_mult(d,x,y_new,N,N,D);
    
    // normalize vector
    normalize(y_new,N,D);

    // calculate meanshift
    calc_meanshift(y_new,y,m,N,D);

    //update y
    copy_2Darray(y_new, y,N,D);

    norm = eucl_norm(m,N,D);

    if (VERBOSE){
      printf("[INFO]: Iteration %d - error %lf\n", iter, norm);
      write_csv_file("y_new\n",y_new,N,D);
    }
  }

}


void rangesearch(){
  int i,j,count=0;
  double dist;
  // double **d;

  // d = (double **) malloc(N * sizeof(double *));
  // if(d==NULL) {perror("Error: "); exit(1);}
  for (i=0; i<N; i++){
    // d[i] = (double *) malloc(N * sizeof(double)); //malloc rows
    // if(d[i]==NULL) {perror("Error: "); exit(1);}

    for (j=0; j<N; j++){
      if (i==j) {d[i][j] = 1; continue;}

      dist = euclidean_distance(i,j);
      if (dist<BANDWIDTH){
        // printf("%lf \t",dist);
        d[i][j]= gaussian_kernel(dist);
        nNbr[i]++;
        count++;
      }
      else{
        d[i][j]=0;
      }
    }
  }
}

// a*b = c
void cpu_matrix_mult(double **a, double **b,double **c, const int ROW1, const int COL1, const int COL2) {
  int i,j,k;
  for(i=0; i<ROW1; i++){
    for(j=0; j<COL2; j++){
      c[i][j] = 0;
      for(k=0; k<COL1; k++){
          c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void normalize(double **a, const int ROW, const int COL){
  int i,j;
  double s=0;

  for (i=0;i<ROW;i++){
    s = sum_of_row(d,i,N);
    // printf("%lf", s);
    for (j=0; j<COL; j++)
      a[i][j] = a[i][j]/s;       
  }
}

double sum_of_row(double **a, const int row, const int COL){
  int j;
  double sum=0;
  for (j=0; j<COL; j++)
    sum += a[row][j];
  return sum;
}

double eucl_norm(double **a, const int ROW, const int COL) {
  int i,j;
  double norm=0;
  for (i=0; i<ROW; i++)
    for (j=0; j<COL; j++)
      norm += a[i][j] * a[i][j];
  return sqrt(norm);
}

void calc_meanshift(double **a, double **b, double **c, const int ROW, const int COL){
  int i,j;
  for (i=0;i<ROW;i++)
    for (j=0; j<COL; j++)
      c[i][j] = a[i][j] - b[i][j];       
}
void copy_2Darray(double **source, double **destination, const int ROW, const int COL){
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

//  W = spfun( @(x) exp( -x / (2*h^2) ), W );
double gaussian_kernel(const double dist){
    return exp(-1.0/2.0 * dist / (BANDWIDTH*BANDWIDTH));
}

double euclidean_distance(const int first, const int second){
  int j;
  double dist = 0;
  for (j=0; j<D; j++)
    dist += (y[first][j] - x[second][j]) * (y[first][j] - x[second][j]);
  return dist;
}

/*typedef struct {
    int i,j,distance;
} SparseData;

SparseData * sparse(double * d, int data_num){

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