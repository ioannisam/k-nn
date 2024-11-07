#include "../include/kNN.h"

#include <stdlib.h>
#include <time.h>
#include <stdint.h>

void random_data(Mat* matrix, size_t points, size_t dimensions) {

  srand(time(NULL) + (uintptr_t)matrix);

  matrix->data = (double*)malloc(points*dimensions*sizeof(double));
  matrix->rows = points;
  matrix->cols = dimensions;
   
  for(size_t i=0; i<points*dimensions; i++) {
    matrix->data[i] = ((double)rand()/RAND_MAX)*100.0;
  }
}
