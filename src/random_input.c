#include "../include/kNN.h"

#include <stdlib.h>
#include <time.h>

void random_data(DataSet* dataset, size_t points, size_t dimensions) {

  srand(time(NULL));

  dataset->data = (double*)malloc(points*dimensions*sizeof(double));
  dataset->rows = points;
  dataset->cols = dimensions;
    
  for(size_t i=0; i<points*dimensions; i++) {
    dataset->data[i] = ((double)rand()/RAND_MAX)*100.0;
  }
}
