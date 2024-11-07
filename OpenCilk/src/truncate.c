#include "../include/kNN.h"

#include <stdlib.h>
#include <time.h>
#include <string.h>

void truncMat(Mat* src, Mat* target, double perc) {
  
  int const rows = src->rows;
  int const cols = src->cols;

  int newRows = (int)rows*perc;
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
