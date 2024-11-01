#include "../include/kNN.h"

#include <stdlib.h>
#include <time.h>
#include <stdint.h>

void random_data(Mat* dataset, size_t points, size_t dimensions) {

  srand(time(NULL) + (uintptr_t)dataset);

  dataset->data = (double*)malloc(points*dimensions*sizeof(double));
  dataset->rows = points;
  dataset->cols = dimensions;
   
  for(size_t i=0; i<points*dimensions; i++) {
    dataset->data[i] = ((double)rand()/RAND_MAX)*100.0;
  }
}
