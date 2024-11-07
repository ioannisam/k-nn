#ifndef KNN_H
#define KNN_H

#include <stddef.h>

// Struct to represent a dataset with data points stored in a row-major format
typedef struct {
  double* data;
  size_t  rows;
  size_t  cols;
} Mat;

// Generates random data points for a dataset
void random_data(Mat* dataset, size_t points, size_t dimensions);

// Prints a matrix with given number of rows and columns
void print_matrix(const Mat* matrix);

// Trunicates a matrix keeping only perc% of its rows
void truncMat(Mat* src, Mat* target, double perc);

// Calculates the distance matrix D between two datasets C and Q
void calculate_distances(const Mat* C, const Mat* Q, double* D);

// Finds the k-nearest neighbors for each point in the dataset
// Parameters:
// - D: Distance matrix (flattened 1D array) of size n x n (where n is the number of points in C == Q)
// - n: Number of data points
// - k: Number of neighbors to find
// - knn_indices: Output array of k nearest neighbor indices for each point
// - knn_distances: Output array of distances for the k nearest neighbors
void findKNN(Mat* C, Mat* Q, Mat* N);

// Helper functions for quick-select algorithm
int  partition  (double* arr, int left, int right);
void quickSelect(double* arr, int left, int right, int k, double* neighbors);

#endif // KNN_H
