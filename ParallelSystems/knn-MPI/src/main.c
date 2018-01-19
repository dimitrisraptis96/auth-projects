/**
    @author Raptis Dimitris
    @version 1.1 10/01/18 
*/

#include "../include/global_vars.h"
#include "../include/main.h"
#include "../include/serial.h"
#include "../include/parallel.h"

//global variables
int N;
int D;
int K;
int TYPE;

int main (int argc, char **argv) {

  set_args(argc,argv);
  check_args();
  choose_type();

  return(0);
}

void set_args(int argc, char **argv){
  if (argc != 5) {
    printf("==============================================\n");
    printf("Usage: Serial implementation of knn algorithm.\n");
    printf("==============================================\n");
    printf("arg[1] = TYPE ==> type of implementation (1->Serial 2->Blocking 3->No-blocking)\n");
    printf("arg[2] = N    ==> number of points\n");
    printf("arg[3] = D    ==> point's dimensions\n");
    printf("arg[4] = K    ==> k nearest points\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }

  TYPE  = atoi(argv[1]);
  N     = atoi(argv[2]);
  D     = atoi(argv[3]);
  K     = atoi(argv[4]);
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
  if (TYPE<1 || TYPE>3){
    printf("TYPE value should be 1,2 or 3.\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }
  return;
}

//Choose implementation to execute
void choose_type(){
  
  switch(TYPE) {
    case TYPE_SERIAL:
      serial();
      break;
    case TYPE_BLOCK:
      MPI_no_block();
      break;
    case TYPE_NO_BLOCK:
      MPI_block();
      break;
    default:
      printf("[ERROR]: TYPE error\n");
  }
}