#include "../include/kNN.h"

#include <stdio.h>

void print_matrix(const double* matrix, size_t rows, size_t cols) {

  printf("        ");
  for (size_t j=0; j<cols; j++) {
    printf("    Point %zu  ", j+1);
  }
  printf("\n");
    
  printf("   ");
  for (size_t j=0; j<cols; j++) {
    printf("───────────────");
  }
  printf("\n");

  for (size_t i=0; i<rows; i++) {
    printf("Point %zu |", i+1); // Row label
    for (size_t j=0; j<cols; j++) {
      printf(" %10.2f |", matrix[i * cols + j]); // Align numbers
    }
    printf("\n");

    printf("   ");
    for (size_t j=0; j<cols; j++) {
      printf("───────────────");
    }
    printf("\n");
  }

  printf("\n");
}

