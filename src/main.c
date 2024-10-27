#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>

int main() {

  size_t points, dimensions;
  printf("Enter the number of points: ");
  scanf("%zu", &points);
  printf("Enter the number of dimensions: ");
  scanf("%zu", &dimensions);

  DataSet C, Q;
  random_data(&C, points, dimensions);
  random_data(&Q, points, dimensions);
  print_matrix(C.data, points, dimensions);
  printf("\n");

  double* D = (double*)malloc(points*points*sizeof(double));
  calculate_distances(&C, &Q, D);

  printf("Squared Distance Matrix D (first 10x10 subset):\n");
  size_t subset = points < 10 ? points : 10;
  print_matrix(D, subset, subset);

  free(D);
  free(C.data);
  free(Q.data);

  return 0;
}
