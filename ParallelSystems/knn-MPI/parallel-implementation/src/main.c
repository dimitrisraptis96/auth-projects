#include "../includes/main.h"

//global variables
int N;
int D;
int K;

int PID;
int MAX;
int UNIT;
MPI_Status Stat;

double **buffer;
double **array;

double **k_dist;
int **k_id;

char CORPUS_FILENAME[]      = "../../corpus-files/corpus.txt";
char VALIDATION_FILENAME[]  = "../../corpus-files/validated.txt";

int main (int argc, char **argv) {
  
  if (argc != 4) {
    printf("==============================================\n");
    printf("Usage: Parallel implementation of knn algorithm.\n");
    printf("==============================================\n");
    printf("arg[1] = N ==> number of points\n");
    printf("arg[2] = D ==> point's dimensions\n");
    printf("arg[3] = K ==> k nearest points\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }

  N = atoi(argv[1]);
  D = atoi(argv[2]);
  K = atoi(argv[3]);

  MPI_Init(NULL, NULL);
  MPI_Comm_rank (MPI_COMM_WORLD, &PID);
  MPI_Comm_size(MPI_COMM_WORLD, &MAX);
  
  check_args();
  init();

  if(PID==1)
    printf("/////%lf",buffer[0][0]);
    // print(array,UNIT,D);

  MPI_Barrier(MPI_COMM_WORLD);
  // printf("\n\nafter\n\n");
  //Here every process has its array initialized
  //start measure of time
  // print(array,N,D);
  // calc_knn();
  // memory_deallocation();


  printf( "MPI_FINALIZE = %d\n\n",MPI_Finalize() )  ;
/*  printf("===============================================\n");
  printf("\t\tDistances\n");
  printf("===============================================\n");
  calc_distances();
  print(dist_arr,N,N);
  printf("===============================================\n");  
  printf("\t\tknn distances\n");
  printf("===============================================\n");
  calc_knn();
  print(k_dist,N,K);
  printf("===============================================\n");  
  printf("\t\tknn id's\n");
  printf("===============================================\n");
  print_id();
  printf("===============================================\n");
  printf("===============================================\n");
  printf ("\n\nIs test PASSed? %s\n\n", test()?"YES":"NO");*/

  return(0);
}

void check_args() {
  if (N<=0 || K<=0 || D<=0){
    printf("Negative value for N, K or D.\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }
  if (N<=K){
    printf("K value is larger than N.\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }
  if (N%MAX){
    printf("Number of processes is false.\n");
    printf("\n\nExiting program..\n");
    exit(1);
  } 
  return;
}

void read_file(){
  int i, j, source, dest;
 
  //The 1rst process handles the reading and sends to the other processes
  if (PID == 0){
    int i_buff;
    dest =1;

    FILE * fp;
    fp = fopen(CORPUS_FILENAME,"r");
    
    for (i=0; i<N; i++){      
      i_buff = i % UNIT;
      for (j=0; j<D; j++) 
        fscanf(fp, "%lf", &buffer[i_buff][j]);

      //The 1rs process keeps the last buffer for itself
      if (dest >= MAX)
        continue;
      // if (i == N -1)
        // break;

      MPI_Send(buffer[i_buff], D, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

      //Send buffer to the next process when buffer is full
      if (i_buff == UNIT-1){
        dest++;
        // if (MPI_SUCCESS == MPI_Send(&buffer[0][0], UNIT*(D+2), MPI_DOUBLE, ++dest, 0, MPI_COMM_WORLD))
          // printf("Process 0 sent buffer to process %d\n", dest); 
      }
    } 
    fclose(fp);
  }
  else {
    source = 0;
    for (i=0; i<UNIT; i++){
      MPI_Recv(buffer[i], D, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &Stat);
    }
    printf("Process %d received buffer from process 0\n", PID); 
    // if (MPI_SUCCESS == MPI_Recv(&buffer, UNIT*(D+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &Stat)){
    //   printf("Process %d received buffer from process 0\n", PID); 
    // }
  }
  
  array = buffer;
  MPI_Barrier(MPI_COMM_WORLD);
  return;
}

//Memory allocation for buffer, array, k_dist and k_id
void memory_allocation(){
  int i;

  buffer  = (double **) malloc(UNIT * sizeof(double *));
  array   = (double **) malloc(UNIT * sizeof(double *));
  k_dist  = (double **) malloc(UNIT * sizeof(double *));
  k_id    = (int **) malloc(UNIT * sizeof(int *));
  if(buffer == NULL || array == NULL || k_dist == NULL || k_id == NULL) { 
    printf("Memory allocation error!\n");
    exit(1);
  }
  for (i=0; i<D; i++){
    buffer[i] = (double *) malloc(D * sizeof(double));
    array[i]  = (double *) malloc(D * sizeof(double));
    if(buffer[i] == NULL || array[i] == NULL){ 
      printf("Memory allocation error!\n");
        exit(1);
    }
  }
  for (i=0; i<K; i++){
    k_dist[i] = (double *) malloc(D * sizeof(double));
    k_id[i]   = (int *) malloc(D * sizeof(int));
    if(k_dist[i] == NULL || k_id[i] == NULL){ 
      printf("Memory allocation error!\n");
        exit(1);
    }
  }
  return;
}

void memory_deallocation(){
  int i;

  for (i=0; i<D; i++){
    free (buffer[i]);
    free (array[i]);
  }
  for (i=0; i<K; i++){
    free (k_dist[i]);
    free (k_id[i]);
  }
  free (buffer);
  free (array);
  free (k_dist);
  free (k_id);
  return;
}

//Dynamically allocate memory and create the 2D-array
void init(){
  UNIT = N/MAX;
  memory_allocation();
  read_file();
  return;
}

//Print 2D double arr array
void print(double **arr, int row, int col){
  int i, j;
  for (i=0; i<row; i++){
    for (j=0; j<col; j++){
      printf("%lf ", arr[i][j]);
    }
    printf("\n");
  }
  return;
}

//Print 2d int arr array
void print_id(){
  int i, j;
  for (i=0; i<N; i++){
    for (j=0; j<K; j++){
      printf("%d ", k_id[i][j]);
    }
    printf("\n");
  }
  return;
}

//Quicksort
int cmp_func (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}


/*int test(){
  int i,j;
  //Change -1 with DBL_size 
  for (i=0;i<N;i++){
    for(j=0;j<N;j++){
      if (dist_arr[i][j] == -1){
        dist_arr[i][j] = DBL_MAX;
      }
    }
  }
  //Sort the dist_arr and do the testing
  for (i=0; i<N; i++){
    qsort(dist_arr[i], N, sizeof(double), cmp_func);
    for (j=0; j<K; j++){
      if (k_dist[i][j] != dist_arr[i][j]){
        printf("k_dist=%lf\n",k_dist[i][j]);
        printf("dist_arr=%lf\n",dist_arr[i][j]);
        printf("i=%d j=%d\n",i,j);
        return 0;
      }
    }
  }
  return 1;
}*/




double euclidean_distance(int first, int second){
  int j;
  double dist = 0;
  for (j=0; j<D; j++)
    dist += (buffer[first][j] - buffer[second][j]) * (buffer[first][j] - buffer[second][j]);
  return dist;
}


/*void calc_distances (){
  int i, j;

  dist_arr = (double **) malloc(N * sizeof(double *));
  if(dist_arr == NULL) { 
    printf("Memory allocation error!\n");
    exit(1);
  }

  for (i=0; i<N; i++) {
    dist_arr[i] = (double *) malloc(N * sizeof(double));
    if(dist_arr[i] == NULL) { 
      printf("Memory allocation error!\n");
      exit(1);
    }
    for (j=0; j<N; j++) {
      if (i==j) {
          dist_arr[i][j] = -1.0;
          continue;
        }
        dist_arr[i][j] = euclidean_distance(i,j);
    }
  }
  return;
}


void calc_knn(){
  int i, j;
  double dist;

  //Memory allocation for k_dist[N][K] array
  k_dist = (double **) malloc(N * sizeof(double *));
  k_id = (int **) malloc(N * sizeof(int *));
  if( (k_dist == NULL) || (k_id == NULL) ) { 
    printf("Memory allocation error!\n");
    exit(1);
  }
  for (i=0; i<N; i++) {
    k_dist[i] = (double *) malloc(K * sizeof(double));
    k_id[i] = (int *) malloc(K * sizeof(int));
    if( (k_dist[i] == NULL) || (k_id[i] == NULL) ) { 
      printf("Memory allocation error!\n");
      exit(1);
    }
    for (j=0; j<K; j++){
      //TODO: Set infinity as initial value
      k_dist[i][j] = DBL_MAX;
      k_id[i][j]   = -1;
    }
  }

  //Calculate k nearest points
  for(i=0; i<N; i++){
    for (j=0; j<N; j++) {
      if (i==j) continue;
        dist = euclidean_distance(i,j);
        if (dist < k_dist[i][K-1]){
          find_position(i,dist,j);  //Just found a new closer distance
        }
    }
  }
}

//Find the position of the new distance
void find_position(int i, double dist, int id){
  int j;
  for (j=0; j<K; j++){
    if (dist < k_dist[i][j]){
      //TODO: Check if qsort is quicker
      // if (i==j) continue;
      move(i,j);
      k_dist[i][j] = dist;
      k_id[i][j]   = id;
      return;
    }
  }
  return;
}

//Shift k_dist[i] & k_id[i] values one position in order to insert the new distance
void move(int i, int pos){
  int j;
  for (j=(K-1); j>pos; j--){
    if(j==0) continue;
    k_dist[i][j] = k_dist[i][j-1];
    k_id[i][j]   = k_id[i][j-1];
  }
  return;
}*/