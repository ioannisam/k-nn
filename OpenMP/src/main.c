#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

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
  random_input(&C, c, d);
  random_input(&Q, q, d);

  // load_hdf5("../../Fashion-MNIST.hdf5", "train", &C);
  // load_hdf5("../../Fashion-MNIST.hdf5", "test",  &Q);

  printf("This is Matrix Corpus: \n\n");
  print_matrix(&C);
  printf("\n");

  printf("This is Matrix Queries: \n\n");
  print_matrix(&Q);
  printf("\n");

  Neighbor* N = (Neighbor*)malloc(q*k*sizeof(Neighbor));

  findKNN(&C, &Q, N, k); 

  print_neighbors(N, q, k);

  free(C.data);
  free(Q.data);
  free(N);

  return 0;
}
