#include "../include/kNN.h"

#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <omp.h>

void random_projection(Mat* C, Mat* Q, int t, Mat* C_RP, Mat* Q_RP) {

  srand(time(NULL) + (uintptr_t)C);
  
  int c = (int)C->rows;
  int q = (int)Q->rows;
  int d = (int)C->cols;

  C_RP->rows = c;
  C_RP->cols = t;
  Q_RP->rows = q;
  Q_RP->cols = t;

  C_RP->data = (double*)malloc(c*t*sizeof(double));
  Q_RP->data = (double*)malloc(q*t*sizeof(double));

  double* R = (double*)malloc(d*t*sizeof(double));
  for (int i=0; i<d*t; i++) {
    // Rademacher distribution (-1 or +1)
    R[i] = (rand()%2 == 0 ? -1 : 1) / sqrt((double)t);
  }
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, t, d, 1.0, C->data, d, R, t, 0.0, C_RP->data, t);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, q, t, d, 1.0, Q->data, d, R, t, 0.0, Q_RP->data, t);

  free(R);
}

void truncMat(Mat* C, int r, Mat* C_TR) {

  int c = C->rows;
  int d = C->cols;

  C_TR->rows = r;
  C_TR->cols = d;
  C_TR->data = (double*)malloc(r*d*sizeof(double));

  srand(time(NULL));

  int first_idx = rand()%c;
  memcpy(C_TR->data, C->data + first_idx*d, d*sizeof(double));

  double* distances = (double*)malloc(c*sizeof(double));
  #pragma omp parallel for
  for(int i=0; i<c; i++) {
    double dist = 0.0;
    for(int j=0; j<d; j++) {
      double diff = C->data[i*d + j] - C_TR->data[j];
      dist += diff*diff;
    }
    distances[i] = dist;
  }

  for(int i=1; i<r; i++) {
    double total_dist = 0.0;

    #pragma omp parallel for reduction(+:total_dist)
    for(int j=0; j<c; j++) {
      total_dist += distances[j];
    }

    double rand_dist = ((double)rand()/RAND_MAX) * total_dist;
    double cumulative_dist = 0.0;
    int    next_idx = 0;

    for(int j=0; j<c; j++) {
      cumulative_dist += distances[j];
      if(cumulative_dist >= rand_dist) {
        next_idx = j;
        break;
      }
    }

    memcpy(C_TR->data + i*d, C->data + next_idx*d, d*sizeof(double));

    #pragma omp parallel for
    for(int j=0; j<c; j++) {
      double dist = 0.0;
      for(int k=0; k<d; k++) {
        double diff = C->data[j*d + k] - C_TR->data[i*d + k];
        dist += diff*diff;
      }
      if(dist < distances[j]) {
        distances[j] = dist;
      }
    }
  }

  free(distances);
}
