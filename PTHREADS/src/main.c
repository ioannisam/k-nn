#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <math.h>

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
  int num_threads;
  printf("Enter the number of threads to use: ");
  scanf("%d", &num_threads);
  printf("\n");

  Mat C, Q;
  random_input(&C, c, d);
  random_input(&Q, q, d);

  // load_hdf5("../../Fashion-MNIST.hdf5", "train", &C);
  // load_hdf5("../../Fashion-MNIST.hdf5", "test",  &Q);

  print_matrix(&C, "Corpus");
  print_matrix(&Q, "Queries");

  Neighbor* N = (Neighbor*)malloc(q*k*sizeof(Neighbor));
  memory_check(N);

  if((c*d>1000000 && d>100) || d>500) {
    
    Mat C_RP, Q_RP;
    double const e = 0.1;
    int    const t = 4*log((double)c) / (e*e);
    printf("Target dimension (t) for random projection: %d\n", t);
    random_projection(&C, t, &C_RP);
    random_projection(&Q, t, &Q_RP);

    print_matrix(&C_RP, "Corpus Projected");
    print_matrix(&Q_RP, "Queries Projected");

    findKNN(&C_RP, &Q_RP, N, k, num_threads); 
    print_neighbors(N, q, k);

    free(C_RP.data);
    free(Q_RP.data);
  } else {

    findKNN(&C, &Q, N, k, num_threads); 
    print_neighbors(N, q, k);
  }

  free(C.data);
  free(Q.data);
  free(N);

  return 0;
}
