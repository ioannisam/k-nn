#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>

void print_matrix(const Mat* matrix, const char* name) {

  size_t rows = matrix->rows;
  size_t cols = matrix->cols;

  printf("This is matrix %s \n\n", name);
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

  printf("\n\n");
}

void print_neighbors(Neighbor* N, int q, int k) {

  printf("These are the %d Nearest Neighbors for each query: \n\n", k);
  for(int i=0; i<q; i++) {
    printf("Query Point %d Neighbors:\n", i+1);
    for(int j=0; j<k; j++) {
      Neighbor* neighbor = &N[i*k + j];
      printf("  Neighbor %d -> Index: %d, Distance: %.2f\n", j+1, neighbor->index, neighbor->distance);
    }
    printf("\n");
  }
} 

void memory_check(void* ptr) {
  if(ptr == NULL) {
    fprintf(stderr, "Memory Allocation Failed!\n");
    free(ptr);
    exit(1);
  }
}
