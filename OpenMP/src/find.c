#include "../include/kNN.h"

#include <stdlib.h>
#include <omp.h>

void findKNN(Mat* C, Mat* Q, Neighbor* N, int k) {

  int const c = C->rows;
  int const q = Q->rows;
  int const d = C->cols;

  #pragma omp parallel for
  for(int i=0; i<q; i++) {

    long double* D = (long double*)malloc(c*sizeof(long double));
    memory_check(D);
    
    calculate_distances(C, &(Mat){.rows = 1, .cols = d, .data = Q->data + i*d}, D);

    int* indices = (int*)malloc(c*sizeof(int));
    memory_check(indices);

    for(int j=0; j<c; j++) {
      indices[j] = j;
    }

    quickSelect(D, indices, 0, c-1, k, N + i*k);
    free(D);
    free(indices);
  }
}
