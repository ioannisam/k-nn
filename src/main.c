#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cblas.h>

void random_data(DataSet *dataset, size_t num_points, size_t dimensions) {
  dataset->data = (double *)malloc(num_points * dimensions * sizeof(double));
  dataset->rows = num_points;
  dataset->cols = dimensions;
    
  for (size_t i = 0; i < num_points * dimensions; ++i) {
    dataset->data[i] = ((double)rand()/RAND_MAX)*100.0;
  }
}

void print_matrix(const double* matrix, size_t rows, size_t cols) {
  
  for (size_t i=0; i<rows; i++) {
    for (size_t j=0 ; j<cols; j++) {
      printf("%.2f ", matrix[i*cols+j]);
    }
    printf("\n");
  }
  printf("\n");
}

void calculate_distances(const DataSet* C, const DataSet* Q, double* distances) {

  size_t n = C->rows;  // Number of points in C (and Q, since C == Q)
  size_t d = C->cols;  // Number of dimensions

  double* C_squared = (double*)malloc(n*sizeof(double));
  double* Q_squared = (double*)malloc(n*sizeof(double));

  for (size_t i = 0; i < n; ++i) {
    C_squared[i] = 0;
    for (size_t j = 0; j < d; ++j) {
      C_squared[i] += C->data[i * d + j] * C->data[i * d + j];
    }
  }

  for (size_t i = 0; i < n; ++i) {
    Q_squared[i] = C_squared[i]; // Same values since C == Q
  }

  double *C_QT = (double *)malloc(n * n * sizeof(double)); // C * Q^T result
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n, n, d,
              -2.0, C->data, d, Q->data, d,
              0.0, C_QT, n);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      distances[i * n + j] = C_squared[i] + Q_squared[j] + C_QT[i * n + j];
    }
  }

  free(C_squared);
  free(Q_squared);
  free(C_QT);
}

int main() {

  srand(time(NULL));

  size_t points, dimensions;
  printf("Enter the number of points: ");
  scanf("%zu", &points);
  printf("Enter the number of dimensions: ");
  scanf("%zu", &dimensions);

  DataSet C, Q;
  random_data(&C, points, dimensions);
  random_data(&Q, points, dimensions);

  double* distances = (double*)malloc(points*points*sizeof(double));
  calculate_distances(&C, &Q, distances);

  printf("Distance Matrix D (first 10x10 subset):\n");
  size_t subset = points < 10 ? points : 10;
  print_matrix(distances, subset, subset);

  free(distances);
  free(C.data);
  free(Q.data);

  return 0;
}
