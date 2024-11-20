#include "../include/kNN.h"

#include <stdlib.h>
#include <cblas.h>
#include <omp.h>

void calculate_distances(const Mat* C, const Mat* Q, int start_idx, int end_idx, long double* D) {

  int c = C->rows;
  int d = C->cols;
  int batch_size = end_idx - start_idx;

  double* C2 = (double*)malloc(c*sizeof(double));
  memory_check(C2);
  #pragma omp parallel for
  for(int i=0; i<c; i++) {
    double sum = 0.0;
    for(int j=0; j<d; j++) {
      sum += C->data[i*d + j]*C->data[i*d + j];
    }
    C2[i] = sum;
  }

  double* Q2 = (double*)malloc(batch_size*sizeof(double));
  memory_check(Q2);
  #pragma omp parallel for
  for(int i=0; i<batch_size; i++) {
    double sum = 0.0;
    for(int j=0; j<d; j++) {
      sum += Q->data[(start_idx+i)*d + j] * Q->data[(start_idx+i)*d + j];
    }
    Q2[i] = sum;
  }

  double* CQ = (double*)malloc(c*batch_size*sizeof(double));
  memory_check(CQ);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c, batch_size, d,
              1.0, C->data, d, Q->data + start_idx * d, d, 0.0, CQ, batch_size);

  #pragma omp parallel for collapse(2)
  for(int i=0; i<c; i++) {
    for(int j=0; j<batch_size; j++) {
      D[j*c + i] = C2[i] - 2.0*CQ[i*batch_size + j] + Q2[j];
      if(D[j*c + i] < 0.0) {
        D[j*c + i] = 0.0;
      }
    }
  }

  free(C2);
  free(Q2);
  free(CQ);
}
