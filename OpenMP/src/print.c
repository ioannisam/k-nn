#include "../include/kNN.h"

#include <stdio.h>

void print_matrix(const Mat* matrix) {

  size_t rows = matrix->rows;
  size_t cols = matrix->cols;

  printf("        ");
  for(size_t j=0; j<cols; j++) {
    printf("    Point %zu  ", j+1);
  }
  printf("\n");
    
  printf("   ");
  for(size_t j=0; j<cols; j++) {
    printf("───────────────");
  }
  printf("\n");

  for(size_t i=0; i<rows; i++) {
    printf("Point %zu |", i+1);
    for(size_t j=0; j<cols; j++) {
      printf(" %10.2f |", matrix->data[i*cols + j]);
    }
    printf("\n");

    printf("   ");
    for(size_t j=0; j<cols; j++) {
      printf("───────────────");
    }
    printf("\n");
  }

  printf("\n");
}

void print_neighbors(Neighbor* N, int q, int k) {

  for(int i=0; i<q; i++) {
    printf("Query Point %d Neighbors:\n", i+1);
    for(int j=0; j<k; j++) {
      Neighbor* neighbor = &N[i*k + j];
      printf("  Neighbor %d -> Index: %d, Distance: %.2f\n", j+1, neighbor->index, neighbor->distance);
    }
    printf("\n");
  }
} 
