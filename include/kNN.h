#ifndef KNN_H
#define KNN_H

#include <stddef.h>

typedef struct {
  double* data; // Pointer to the data points
  size_t  rows; // Number of data points
  size_t  cols; // Number of dimensions
} DataSet;

void calculate_distances(const DataSet* C, const DataSet* Q, double* distances);
void quick_select(double* array, int left, int right, int k);
void knn(const DataSet *C, const DataSet* Q, int k, size_t* indices);

#endif
