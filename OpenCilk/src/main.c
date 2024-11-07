#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
  
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
  truncMat(&C, &newC, 0.7);
  free(C.data);
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

  findKNN(&newC, &Q, &N); 

  printf("This is Matrix Neighbors: \n\n");
  print_matrix(&N);
  printf("\n");

  free(newC.data);
  free(Q.data);
  free(N.data);

  return 0;
}
