#include <stdio.h>
#include <stdlib.h>

double **init_arr;

char CORPUS_FILENAME[] = "./data/bins/train_30.bin";

int main (int argc, char **argv) {
  
  int N,D,K;
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

  int i,j;

  FILE * in, out;
  in = fopen (CORPUS_FILENAME, "rb");
  out = fopen ("data/bins/out.txt", "w");
  
  double tmp[30];
  for(j=0;j<60000;j++){
    fread(tmp, sizeof(tmp), 1, in);
    for (i=0;i<30;i++){ 
      fwrite(tmp, sizeof(tmp),1, out);
    }
  }

  //   init_arr = (double **) malloc(N * sizeof(double *));
  //   if(init_arr == NULL) { 
  //     printf("Memory allocation error!\n");
  //     exit(1);
  //   }

  //   for (i=0; i<N; i++) {
  //     init_arr[i] = (double *) malloc(D * sizeof(double));
  //     if(init_arr[i] == NULL) { 
  //       printf("Memory allocation error!\n");
  //       exit(1);
  //     }

  //     for (j=0; j<D; j++) {
  //       fscanf(in, "%lf", &init_arr[i][j]);
  //     }
  //   }

  //   fclose(in);
  //   return 0;
}
