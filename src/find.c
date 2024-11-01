#include "../include/kNN.h"

#include <float.h>

void findKNN(const Mat* D, Mat* N) {
  int const k = (int)N->cols; // Number of nearest neighbors
  int const c = (int)D->cols; // Number of corpus
  int const q = (int)N->rows; // Number of queries

  for(int i=0; i<q; i++) {
    quickSelect(D->data + i*c, 0, c-1, k, N->data + i*k);
  }
}
