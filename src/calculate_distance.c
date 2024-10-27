#include "../include/kNN.h"

#include <stdlib.h>
#include <cblas.h>
#include <omp.h>

void calculate_distances(const DataSet* C, const DataSet* Q, double* D) {

  size_t n = C->rows;  // Number of points in C (and Q, since C == Q)
  size_t d = C->cols;  // Number of dimensions

  double* C_squared = (double*)malloc(n*sizeof(double));
  double* Q_squared = (double*)malloc(n*sizeof(double));

  for(size_t i=0; i<n; i++) {
    C_squared[i] = 0;
    for(size_t j=0; j<d; j++) {
      C_squared[i] += C->data[i*d + j]*C->data[i*d + j];
    }
  }

  for(size_t i=0; i<n; i++) {
    Q_squared[i] = 0;
    for(size_t j=0; j<d; j++) {
      Q_squared[i] += Q->data[i*d + j]*Q->data[i*d + j];
    }
  }

  double* C_QT = (double*)malloc(n*n*sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n, n, d,
              -2.0, C->data, d, Q->data, d,
              0.0, C_QT, n);

  #pragma omp parallel for
  for (size_t i=0; i<n; i++) {
    for (size_t j=0; j<n; j++) {
      D[i*n + j] = C_squared[i] + Q_squared[j] + C_QT[i*n + j];
    }
  }

  free(C_squared);
  free(Q_squared);
  free(C_QT);

  /*
  If matrix D is used only for comparisons (like finding the k-nn) sqrt is unnessecary
  #pragma omp parallel for
  for (size_t i=0; i<n*n; i++) {
    D[i] = sqrt(D[i]);
  }
  */
}
