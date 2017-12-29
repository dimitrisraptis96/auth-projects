//mpirun -npirun -n 4 ./bin/main 10000 100 5

#include "../includes/no_block.h"

#define CORPUS_PATH     "../../corpus-files/corpus.txt"

#define VALIDATION_PATH "../../corpus-files/validated.txt"

#define FIRST_REP 1

#define OTHER_REP 0

#define PRECISION 0.00001

//TODO: global variables file
//TODO: check MPI method's return values (== MPI_SUCCESS)
//TODO: Memory deallocation
//TODO: Array contiguous memory allocation
//        int (*arr)[cols] = malloc(sizeof *arr * rows);  // do stuff with arr[i][j]
//        int *arr = malloc(sizeof *arr * rows * cols);   // do stuff with arr[i * rows + j]


// http://www.cas.mcmaster.ca/~nedialk/COURSES/mpi/Lectures/lec2_1.pdf

int N;
int D;
int K;

int PID;
int MAX;
int CHUNK;
MPI_Status status;

double **in_buffer;
double **out_buffer;
double **array;
double **k_dist;
int    **k_id;

double *in_data;
double *out_data;
double *array_data;
double *k_dist_data;
int    *k_id_data;

int main (int argc, char **argv) {
  
  struct timeval startwtime, endwtime;
  double seq_time;

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

  // Calculate kNN and measure time passed
  gettimeofday (&startwtime, NULL);
  //------------------------------
  knn();
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);
  
  //Validation test
  int isValid = validate();
  if(PID == 1)  printf("\n\nIs test PASSed? %s\n\n", isValid?"YES":"NO");
  
  printf("MPI no-blocking kNN wall clock time = %f (PID=%d)\n", seq_time,PID);    
  
  MPI_Finalize();

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

//Dynamically allocate memory and create the 2D-array
void init(){
  CHUNK = N/MAX;
  memory_allocation();
  read_file();
}

void memory_allocation(){
  int i;

  printf("[INFO]: Allocating contiguous memory for process %d..\n", PID);

  array       = (double **) malloc(sizeof(double *) * CHUNK);
  in_buffer   = (double **) malloc(sizeof(double *) * CHUNK);
  out_buffer  = (double **) malloc(sizeof(double *) * CHUNK);
  k_dist      = (double **) malloc(sizeof(double *) * CHUNK);
  k_id        = (int **)    malloc(sizeof(double *) * CHUNK);

  if(in_buffer == NULL || out_buffer == NULL || array == NULL || k_dist == NULL || k_id == NULL) { 
    printf("[ERROR]: Memory allocation error 1!\n");
    exit(1);
  }

  array_data  = malloc(sizeof(double) * CHUNK * D);
  in_data     = malloc(sizeof(double) * CHUNK * D);
  out_data    = malloc(sizeof(double) * CHUNK * D);
  k_dist_data = malloc(sizeof(double) * CHUNK * K);
  k_id_data   = malloc(sizeof(double) * CHUNK * K);

  if(in_data == NULL || out_data == NULL || array_data == NULL || k_dist_data == NULL || k_id_data == NULL) { 
    printf("[ERROR]: Memory allocation error 2!\n");
    exit(1);
  }

  for(i=0; i < CHUNK; i++){
    array[i]      = array_data  + i * D;
    in_buffer[i]  = in_data     + i * D;
    out_buffer[i] = out_data    + i * D;
    k_dist[i]     = k_dist_data + i * K;
    k_id[i]       = k_id_data   + i * K;
  }
}

void read_file(){
  int i, j, source, dest;
 
  if (PID == 0){
    //MASTER process handles the reading and sends to the other processes
    int index_buff;

    FILE * fp;
    fp = fopen(CORPUS_PATH,"r");
    
    dest =1;

    for (i=0; i<N; i++){      
      index_buff = i % CHUNK;
      for (j=0; j<D; j++)
        fscanf(fp, "%lf", &out_buffer[index_buff][j]);
      
      //The process with PID=0 keeps the last buffer for itself
      if (dest >= MAX) continue;
      //When the buffer is full send it to the next process
      if (index_buff == CHUNK - 1)  
        MPI_Send(out_buffer[0], CHUNK*D, MPI_DOUBLE, dest++, 0, MPI_COMM_WORLD);
    }

    fclose(fp);
    copy_2D_arrays(array, out_buffer);
  }
  else {
    //SLAVE processes receive the data
    source = 0;
    MPI_Recv(array[0], CHUNK*D, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &status);
    printf("[INFO]: Process %d received array from process 0 (Initialization)\n", PID);
    copy_2D_arrays(out_buffer, array);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

//Change chunks between processes using ring topology
//http://www.cas.mcmaster.ca/~nedialk/COURSES/mpi/Lectures/lec2_1.pdf
//http://mpitutorial.com/tutorials/mpi-send-and-receive/
void ring_comm(int tag){
  int source, dest;

  MPI_Request reqs[2];
  MPI_Status stats[2];

  source  = (PID - 1);
  dest    = (PID + 1);

  if(PID == 0)      source  = MAX - 1;
  if(PID == MAX-1)  dest    = 0;

  double t1 = MPI_Wtime();
  MPI_Irecv(in_buffer[0], CHUNK*D, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &reqs[0]);
  MPI_Isend(out_buffer[0], CHUNK*D, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &reqs[1]);

  MPI_Waitall(2, reqs, stats);
  
  printf("[INFO]: Process %d communicated with procesess %d & %d (Ring state = %d)\n",PID , source , dest, tag);
  
  if (PID == 0) 
    printf( "\n[INFO]: Ring communication elapsed time is %f (PID = %d)\n\n", MPI_Wtime() - t1, PID );

  copy_2D_arrays(out_buffer, in_buffer);
}

//Calculate knn
void knn(){
  int i;

  init_dist();

  //Initial comparison with its own data
  calc_knn(FIRST_REP);

  for (i=0; i<MAX-1; i++){
    ring_comm(i);
    calc_knn(OTHER_REP);
  }
  //Now every process has its overall knn
}

double euclidean_distance(int first, int second){
  int j;
  double dist = 0;
  for (j=0; j<D; j++)
    dist += (array[first][j] - out_buffer[second][j]) * (array[first][j] - out_buffer[second][j]);
  return dist;
}

void init_dist(){
  int i, j;
  for (i=0; i<CHUNK; i++) {
    for (j=0; j<K; j++){
      k_dist[i][j] = DBL_MAX;
      k_id[i][j]   = -1;
    }
  }
}

void calc_knn(int rep){
  int i, j;
  double dist;

  //Calculate k nearest points
  for(i=0; i<CHUNK; i++){
    for (j=0; j<CHUNK; j++) {
      if (i==j && rep) continue;
        dist = euclidean_distance(i,j);
        if (dist < k_dist[i][K-1]){
          find_position(i,dist,j);  //Just found a new closer distance
        }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//Find the position of the new distance
void find_position(int i, double dist, int id){
  int j;
  for (j=0; j<K; j++){
    if (dist < k_dist[i][j]){
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
}

//Copy all arr2 elements to arr1
void copy_2D_arrays(double **arr1, double **arr2){
  int i,j;
  for (i=0; i<CHUNK; i++){
    for (j=0; j<D; j++){
      arr1[i][j] = arr2[i][j];
    }
  }
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

int validate(){
  int isValid = 1;
  int source, dest;

  if (PID == 1){
    //MASTER PID == 1 includes the begining of the file
    FILE * fp;
    fp = fopen (VALIDATION_PATH, "r");

    isValid = comp_result(fp);
    //get new k_dist and test again
    for (source=2; source < MAX+1; source++){
      MPI_Recv(k_dist[0], CHUNK*K, MPI_DOUBLE, source%MAX, 0, MPI_COMM_WORLD, &status);
      
      printf("[INFO]: Process %d  received k_dist from process %d (Validation)\n", PID, source%MAX);

      if (!isValid) continue;
      isValid = comp_result(fp);
    }

    fclose(fp);
    
  }
  else {
    //SLAVES receice the data
    dest = 1;
    MPI_Send(k_dist[0], CHUNK*K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

    printf("[INFO]: Process %d sent k_dist to process 1 (Validation)\n", PID);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return isValid;
}

int comp_result(FILE * fp){
  int i,j;
  double tmp;

  for (i=0; i<CHUNK; i++) {
    for (j=0; j<K; j++) {
      fscanf(fp, "%lf", &tmp);

      if(!(fabs(tmp - k_dist[i][j]) < PRECISION)){
        printf("k_dist=%lf\n",k_dist[i][j]);
        printf("validate_tmp=%lf\n",tmp);
        printf("i=%d j=%d\n",i,j);
        return 0;
      }
    }
  }
  return 1;      
}