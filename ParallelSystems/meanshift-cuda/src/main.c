#include "../include/main.h"
#include "../include/serial.h"
#include "../include/parallel.h"
#include "../include/global_vars.h"

//global variables
int N;
int D;
int TYPE;

int main (int argc, char **argv) {

  set_args(argc,argv);
  check_args();
  choose_type();

  return(0);
}

void set_args(int argc, char **argv){
  if (argc != 4) {
    printf("==============================================\n");
    printf("Usage: Serial implementation of knn algorithm.\n");
    printf("==============================================\n");
    printf("arg[1] = TYPE ==> type of implementation (1-->CPU 2-->GPU)\n");
    printf("arg[2] = N    ==> number of points\n");
    printf("arg[3] = D    ==> point's dimensions\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }

  TYPE  = atoi(argv[1]);
  N     = atoi(argv[2]);
  D     = atoi(argv[3]);
}

void check_args() {
  if (N<=0  || D<=0){
    printf("Negative value for N or D.\n");
    printf("\n\nExiting program..\n");
    exit(1);
  }
  if (TYPE<1 || TYPE>2){
    printf("TYPE value should be 1 or 2.\n");
    printf("\n\nExiting program..\n");
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
    case TYPE_GPU:
      parallel();
      break;
    default:
      printf("[ERROR]: TYPE error\n");
  }
}