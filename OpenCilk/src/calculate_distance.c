#include "../include/kNN.h"

#include <stdlib.h>
#include <cblas.h>
#include <math.h>

void calculate_distances(const Mat* C, const Mat* Q, double* D) {

  int c = (int)C->rows;
  int d = (int)C->cols;
  int q = (int)Q->rows;

  double* C2 = (double*)malloc(c*sizeof(double));
  double* Q2 = (double*)malloc(sizeof(double));

  for(int i=0; i<c; i++) {
    double sum = 0.0;
    for(int j=0; j<d; j++) {
      sum += C->data[i*d + j]*C->data[i*d + j];
    }
    C2[i] = sum;
  }

  double sumQ = 0.0;
  for(int j=0; j<d; j++) {
    sumQ += Q->data[j]*Q->data[j];
  }
  Q2[0]=sumQ; 

  double* CQ = (double*)malloc(c*sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, 1, d, 1.0, C->data, d, Q->data, d, 0.0, CQ, 1);

  for(int i=0; i<c; i++) {
    D[i] = fabs(C2[i] - 2*CQ[i] + Q2[0]);
  }

  free(C2);
  free(Q2);
  free(CQ);
}

