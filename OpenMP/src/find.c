#include "../include/kNN.h"

#include <stdlib.h>
#include <omp.h>
#include <math.h>

void findKNN(Mat* C, Mat* Q, Neighbor* N, int k) {

  int const c = C->rows;
  int const q = Q->rows;
  int const d = C->cols;

  #pragma omp parallel for
  for(int i=0; i<q; i++) {

    double* D = (double*)malloc(c*sizeof(double));
    calculate_distances(C, &(Mat){.rows = 1, .cols = d, .data = Q->data + i*d}, D);

    int* indices = (int*)malloc(c*sizeof(int));
    for(int j=0; j<c; j++) {
      indices[j] = j;
    }

    quickSelect(D, indices, 0, c-1, k, N + i*k);
    for(int j=0; j<k; j++) {
      N[i*k + j].distance = sqrt(N[i*k + j].distance);
    }

    free(D);
  }
}
