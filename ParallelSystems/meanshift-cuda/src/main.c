#include "../include/main.h"
#include "../include/serial.h"
#include "../include/parallel.h"
#include "../include/global_vars.h"

//global variables
int N;
int D;
int TYPE;
double BANDWIDTH;
double EPSILON;
char *DATASET_PATH;

int USE_SHARED;

int main (int argc, char **argv) {

  set_args(argc,argv);
  check_args();
  choose_type();

  return(0);
}

void set_args(int argc, char **argv){
  if (argc != 6) {
    printf("==============================================\n");
    printf("Usage: Serial implementation of knn algorithm.\n");
    printf("==============================================\n");
    printf("arg[1] = TYPE ==> type of implementation (1)CPU 2)GPU shared 3)GPU non-shared)\n");
    printf("arg[2] = PATH ==> path to dataset .txt\n");
    printf("arg[3] = BAND ==> bandwidth\n");
    printf("arg[4] = N    ==> number of points\n");
    printf("arg[5] = D    ==> point's dimensions\n");
    printf("\n\nExit program..\n");
    exit(1);
  }

  TYPE          = atoi(argv[1]);
  DATASET_PATH  = argv[2];
  BANDWIDTH     = (double) atoi(argv[3]);
  N             = atoi(argv[4]);
  D             = atoi(argv[5]);

  EPSILON       = 1e-4*BANDWIDTH;
}

void check_args() {
  if (N<=0  || D<=0){
    printf("[ERROR]: Negative value for N or D.\n");
    printf("\n\n[ERROR]: Exit program..\n");
    exit(1);
  }
  if (TYPE<1 || TYPE>3){
    printf("[ERROR]: TYPE value should be 1 or 2.\n");
    printf("\n\n[ERROR]: Exit program..\n");
    exit(1);
  }
  return;
}

//Choose implementation to execute
void choose_type(){
  switch(TYPE) {
    case TYPE_CPU:
      serial();
      break;  

    case TYPE_GPU_SHARED:
      USE_SHARED = 1;
      parallel();
      break;

    case TYPE_GPU_NON_SHARED:
      USE_SHARED = 0;
      parallel();
      break;

    default:
      printf("[ERROR]: TYPE error\n");
  }
}