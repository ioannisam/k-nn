#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

  if(k>c || k<=0) {
    printf("Error: invalid k value.\n");
    return 1;
  }

  Mat C, Q;
  random_data(&C, c, d);
  random_data(&Q, q, d);
  printf("This is Matrix Corpus: \n\n");
  print_matrix(&C);
  printf("\n");
  printf("This is Matrix Queries: \n\n");
  print_matrix(&Q);
  printf("\n");

  Mat D;
  D.rows = q;
  D.cols = c;
  D.data = (double*)malloc(q*c*sizeof(double));

  calculate_distances(&C, &Q, &D);
  printf("This is Matrix Distances: \n\n");
  print_matrix(&D);
  printf("\n");

  Mat N;
  N.rows = q;
  N.cols = k;
  N.data = (double*)malloc(q*k*sizeof(double));

  findKNN(&D, &N);
  printf("This is Matrix Neighbors: \n\n");
  print_matrix(&N);
  printf("\n");

  free(D.data);
  free(C.data);
  free(Q.data);
  free(N.data);

  return 0;
}
