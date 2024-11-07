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
