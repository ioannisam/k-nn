#include "../include/kNN.h"

#include <stdlib.h>
#include <cblas.h>

void calculate_distances(const Mat* C, const Mat* Q, long double* D) {

  int c = (int)C->rows;
  int d = (int)C->cols;

  double* C2 = (double*)malloc(c*sizeof(double));
  double  Q2;
  memory_check(C2); 

  for(int i=0; i<c; i++) {
    double sum = 0.0;
    for(int j=0; j<d; j++) {
      sum += C->data[i*d + j]*C->data[i*d + j];
    }
    C2[i] = sum;
  }

  double sum = 0.0;
  for(int j=0; j<d; j++) {
    sum += Q->data[j]*Q->data[j];
  }
  Q2 = sum; 

  double* CQ = (double*)malloc(c*sizeof(double));
  memory_check(CQ);
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, 1, d, 1.0, C->data, d, Q->data, 1, 0.0, CQ, 1);

  for(int i=0; i<c; i++) {

    D[i] = C2[i] - 2*CQ[i] + Q2;
    if(D[i] < 0.0) {
      D[i] = 0.0;
    }
  }

  free(C2);
  free(CQ);
}

