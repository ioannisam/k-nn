#include "../include/kNN.h"

#include <cblas.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

void calculate_distances(const Mat *C, const Mat *Q, Mat *D) {

  int c = (int)C->rows;
  int q = (int)Q->rows;
  int d = (int)C->cols;

  double* C2 = (double*)malloc(c*sizeof(double));
  #pragma omp parallel for
  for(int i=0; i<c; i++) {
    double sum = 0.0;
    for(int j=0; j<d; j++) {
      sum += C->data[i*d + j]*C->data[i*d + j];
    }
    C2[i] = sum;
  }

  double* Q2 = (double*)malloc(q*sizeof(double));
  #pragma omp parallel for
  for(int i = 0; i<q; i++) {
    double sum = 0.0;
    for(int j=0; j<d; j++) {
      sum += Q->data[i*d + j]*Q->data[i*d + j];
    }
    Q2[i] = sum;
  }

  double* CQ = (double*)malloc(c*q*sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c, q, d, -2.0, C->data,
              d, Q->data, d, 0.0, CQ, q);

  #pragma omp parallel for collapse(2)
  for(int i=0; i<c; i++) {
    for(int j=0; j<q; j++) {
      CQ[i*q + j] += C2[i] + Q2[j];
      D->data[j*c + i] = CQ[i*q + j];
    }
  }

  free(C2);
  free(Q2);
  free(CQ);
}
