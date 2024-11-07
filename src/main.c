#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main() {
  
  printf("Max Threads: %d \n\n", omp_get_max_threads());

  size_t c, q, d, k;
  printf("Enter the number of corpus points: ");
  scanf("%zu", &c);
  printf("Enter the number of query points: ");
  scanf("%zu", &q);
  printf("Enter the number of dimensions: ");
  scanf("%zu", &d);
  printf("Enter the number of nearest neighbors (k): ");
  scanf("%zu", &k);
  printf("\n");

  Mat C, Q;
  random_data(&C, c, d);
  random_data(&Q, q, d);

  printf("This is Matrix Corpus: \n\n");
  print_matrix(&C);
  printf("\n");

  Mat newC;
  truncMat(&C, &newC, 1);
  printf("This is Matrix newC: \n\n");
  print_matrix(&newC);
  printf("\n");

  printf("This is Matrix Queries: \n\n");
  print_matrix(&Q);
  printf("\n");

  Mat N;
  N.rows = q;
  N.cols = k;
  N.data = (double*)malloc(q*k*sizeof(double));

  #pragma omp parallel for
  for(int i=0; i<q; i++) {

    double* D = (double*)malloc(newC.rows*sizeof(double));
    calculate_distances(&newC, &(Mat){.rows = 1, .cols = d, .data = Q.data + i*d}, D);

    quickSelect(D, 0, newC.rows-1, k, N.data + i*k);
    for(int j=0; j<k; j++) {
      N.data[i*k + j] = sqrt(N.data[i*k + j]);
    }

    free(D);
  }

  printf("This is Matrix Neighbors: \n\n");
  print_matrix(&N);
  printf("\n");

  free(C.data);
  free(newC.data);
  free(Q.data);
  free(N.data);

  return 0;
}
