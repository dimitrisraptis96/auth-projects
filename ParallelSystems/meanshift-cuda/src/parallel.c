/*#include "../include/global_vars.h"
#include "../include/parallel.h"
#include "../include/helpers.h"

int PID;
int MAX;
int CHUNK;
MPI_Status status;

int N_THREADS = 8;

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

void MPI_block() {

  MPI_Init(NULL, NULL);
  MPI_Comm_rank (MPI_COMM_WORLD, &PID);
  MPI_Comm_size(MPI_COMM_WORLD, &MAX);

  if (PID == 0){
    printf("[INFO]: MPI BLOCKING IMPLEMENTATION\n");
    printf("===================================\n");
    printf("\n[INFO]: Using %d processes.\n",MAX);
  }
  
  struct timeval startwtime, endwtime;
  double seq_time;

  init_parallel();

  // Calculate kNN and measure time passed
  gettimeofday (&startwtime, NULL);
  //------------------------------
  knn_block();
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);
  
  //Validation test
  int isValid = validate_parallel();  //all the processes should execute it
  if(PID == 1)  printf("\n\nIs test PASSed? %s\n\n", isValid?"YES":"NO");
  
  if(PID == 0)  printf("\n\nMPI blocking kNN wall clock time = %f (PID=%d)\n", seq_time,PID);    
  
  MPI_Finalize();
}

void MPI_no_block() {

  MPI_Init(NULL, NULL);
  MPI_Comm_rank (MPI_COMM_WORLD, &PID);
  MPI_Comm_size(MPI_COMM_WORLD, &MAX);
  
  if (PID == 0){
    printf("[INFO]: MPI NON-BLOCKING IMPLEMENTATION\n");
    printf("=======================================\n");
    printf("\n[INFO]: Using %d processes.\n",MAX);
  }

  struct timeval startwtime, endwtime;
  double seq_time;

  init_parallel();

  // Calculate kNN and measure time passed
  gettimeofday (&startwtime, NULL);
  //------------------------------
  knn_no_block();
  //------------------------------
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);
  
  //Validation test
  int isValid = validate_parallel();
  if(PID == 1)  printf("\n\nIs test PASSed? %s\n\n", isValid?"YES":"NO");
  
  if(PID == 0)  printf("\n\nMPI no blocking kNN wall clock time = %f (PID=%d)\n", seq_time,PID);    
  
  MPI_Finalize();
}

//Dynamically allocate memory and create the 2D-array
void init_parallel(){
  CHUNK = N/MAX;
  cont_memory_allocation();
  read_file_parallel();
}

//Contiguous memory allocation for 2D arrays
void cont_memory_allocation(){
  int i;
  
  if(VERBOSE) printf("[INFO]: Allocating contiguous memory for process %d..\n", PID);

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

void read_file_parallel(){
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
        if ( EOF == fscanf(fp, "%lf", &out_buffer[index_buff][j]) ){
          printf("[ERROR]: fscanf() failed.\n");
          exit(1);
        }
      
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
    if (VERBOSE) printf("[INFO]: Process %d received array from process 0 (Initialization)\n", PID);
    copy_2D_arrays(out_buffer, array);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

//Calculate knn
void knn_no_block(){
  int i;

  init_dist();

  //Initial comparison with its own data
  calc_knn(FIRST_REP);

  for (i=0; i<MAX-1; i++){
    ring_no_block(i);
    calc_knn(OTHER_REP);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  //Now every process has its overall knn
}

void knn_block(){
  int i;

  init_dist();

  //Initial comparison with its own data
  calc_knn(FIRST_REP);

  for (i=0; i<MAX-1; i++){
    ring_block(i);
    calc_knn(OTHER_REP);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  //Now every process has its overall knn
}

//Change chunks between processes using ring topology
//http://mpitutorial.com/tutorials/mpi-send-and-receive/
void ring_block(int tag){
  int source, dest;

  source  = (PID - 1);
  dest    = (PID + 1);

  double t1 = MPI_Wtime();

  if (PID != 0){
    MPI_Recv(in_buffer[0], CHUNK*D, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
    if (VERBOSE) printf("[INFO]: Process %d received buffer from process %d (Ring state = %d)\n", PID, source, tag);
  } 

  MPI_Send(out_buffer[0], CHUNK*D, MPI_DOUBLE, dest % MAX, tag, MPI_COMM_WORLD);

  // Now process 0 can receive from the last process.
  if (PID == 0){
    MPI_Recv(in_buffer[0], CHUNK*D, MPI_DOUBLE, MAX-1, tag, MPI_COMM_WORLD, &status);
    if (VERBOSE) printf("[INFO]: Process %d received buffer from process %d (Ring state = %d)\n", PID, MAX-1, tag);
  }

  copy_2D_arrays(out_buffer, in_buffer);
  MPI_Barrier(MPI_COMM_WORLD);

  if (PID == 0 && VERBOSE) printf( "\n[INFO]: Ring communication elapsed time is %f (PID = %d)\n\n", MPI_Wtime() - t1, PID );

}

void ring_no_block(int tag){
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
  
  if(VERBOSE) printf("[INFO]: Process %d communicated with procesess %d & %d (Ring state = %d)\n",PID , source , dest, tag);
  
  if (PID == 0 && VERBOSE)
    printf( "\n[INFO]: Ring communication elapsed time is %f (PID = %d)\n\n", MPI_Wtime() - t1, PID );

  copy_2D_arrays(out_buffer, in_buffer);
}

void init_dist(){
  int i, j;

  #pragma omp parallel for shared(i) private(j) num_threads(N_THREADS)  
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
  #pragma omp parallel for shared(i) private(j,dist) num_threads(N_THREADS)
  for(i=0; i<CHUNK; i++){
    for (j=0; j<CHUNK; j++) {
      if (i==j && rep) continue;
        dist = euclidean_distance(i,j,array,out_buffer);
        if (dist < k_dist[i][K-1]){
          find_position(i,dist,j);  //Just found a new closer distance
        }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//Copy all arr2 elements to arr1
void copy_2D_arrays(double **arr1, double **arr2){
  int i,j;

  #pragma omp parallel for shared(i) private(j) num_threads(N_THREADS)
  for (i=0; i<CHUNK; i++)
    for (j=0; j<D; j++)
      arr1[i][j] = arr2[i][j];
}

int validate_parallel(){
  int isValid = 1;
  int source, dest;

  if (PID == 1){
    //MASTER PID == 1 includes the begining of the file
    FILE * fp;
    fp = fopen (VALIDATION_PATH, "r");

    if (fp == NULL){
      printf("[ERROR]: Invalid path.\n");
      exit(1);
    }

    isValid = comp_result(fp);
    //get new k_dist and test again
    for (source=2; source < MAX+1; source++){
      MPI_Recv(k_dist[0], CHUNK*K, MPI_DOUBLE, source%MAX, 0, MPI_COMM_WORLD, &status);
      
      if (VERBOSE) printf("[INFO]: Process %d  received k_dist from process %d (Validation)\n", PID, source%MAX);

      if (!isValid) continue;
      isValid = comp_result(fp);
    }

    fclose(fp);
    
  }
  else {
    //SLAVES receice the data
    dest = 1;
    MPI_Send(k_dist[0], CHUNK*K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

    if (VERBOSE) printf("[INFO]: Process %d sent k_dist to process 1 (Validation)\n", PID);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return isValid;
}

int comp_result(FILE * fp){
  int i,j;
  double tmp;

  for (i=0; i<CHUNK; i++) {
    for (j=0; j<K; j++) {
      if (EOF == fscanf(fp, "%lf", &tmp)){
        printf("[ERROR]: fscanf() failed.\n");
        exit(1);
      }

      if(!(fabs(tmp - k_dist[i][j]) < PRECISION)){
        printf("[INFO]: Validation failed:\n");
        // printf("k_dist=%lf\n",k_dist[i][j]);
        // printf("validate_tmp=%lf\n",tmp);
        // printf("i=%d j=%d\n",i,j);
        return 0;
      }
    }
  }
  return 1;      
}*/