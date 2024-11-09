#include "../include/kNN.h"

#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

void random_projection(Mat* M, int t, Mat* RP) {

  srand(time(NULL) + (uintptr_t)M);
  
  int n = (int)M->rows;
  int d = (int)M->cols;

  RP->rows = n;
  RP->cols = t;
  RP->data = (double*)malloc(n*t*sizeof(double));

  double* R = (double*)malloc(d*t*sizeof(double));
  for (int i=0; i<d*t; i++) {
    // Rademacher distribution (-1 or +1)
    R[i] = (rand()%2 == 0 ? -1 : 1) / sqrt((double)t);
  }
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, t, d, 1.0, M->data, d, R, t, 0.0, RP->data, t);

  free(R);
}

void truncMat(Mat* src, Mat* target, double perc) {
  
  int const rows = src->rows;
  int const cols = src->cols;

  int newRows  = (int)rows*perc;
  target->data = (double*)malloc(newRows*cols*sizeof(double));
  target->rows = newRows;
  target->cols = cols;

  int* indices = (int*)malloc(rows*sizeof(int));
  for(int i=0; i<rows; i++) {
    indices[i] = i;
  }

  srand(time(NULL));
  for(int i=rows-1; i>0; i--) {
    int j = rand()%(i+1);

    int temp   = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  for(int i=0; i<newRows; i++) {
    int rowIndex = indices[i];
    memcpy(target->data + i*cols, src->data + rowIndex*cols, cols*sizeof(double));
  }

  free(indices);
}
