#include "../include/kNN.h"

#include <stdlib.h>
#include <math.h>

void findKNN(Mat* C, Mat* Q, Mat* N) {

  int const c = C->rows;
  int const q = N->rows;
  int const d = C->cols;
  int const k = N->cols;

  for(int i=0; i<q; i++) {

    double* D = (double*)malloc(c*sizeof(double));
    calculate_distances(C, &(Mat){.rows = 1, .cols = d, .data = Q->data + i*d}, D);

    quickSelect(D, 0, c-1, k, N->data + i*k);
    for(int j=0; j<k; j++) {
      N->data[i*k + j] = sqrt(N->data[i*k + j]);
    }

    free(D);
  }
}
