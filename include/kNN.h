#ifndef KNN_H
#define KNN_H

#include <stddef.h> // For size_t

typedef struct {
    double* data;
    size_t  rows;
    size_t  cols;
} DataSet;

void random_data(DataSet* dataset, size_t points, size_t dimensions);
void print_matrix(const double* matrix, size_t rows, size_t cols);
void calculate_distances(const DataSet* C, const DataSet* Q, double* D);

#endif
