#ifndef KNN_H
#define KNN_H

#include <stddef.h>
#include <pthread.h>

// Struct to represent a dataset with data points stored in a row-major format
typedef struct {
  double* data;
  size_t  rows;
  size_t  cols;
} Mat;

// Struct to represent a neighbor
typedef struct {
  double distance;
  int    index;
} Neighbor;

// Struct for thread arguments
typedef struct {
  Mat* C;
  Mat* Q;          
  Neighbor* N;     
  int k;
  int start;
  int end;
} ThreadArgs;

// Loads hdf5 or mat files and directs their output to a Mat struct
int file_input(Mat* C, Mat* Q, size_t* c, size_t* q, size_t* d, size_t* k, size_t* num_threads);
int load_hdf5(const char* filename, const char* matname, Mat* matrix);
int load_mat (const char* filename, const char* matname, Mat* matrix);

// Prints a matrix with given number of rows and columns
void print_matrix(const Mat* matrix, const char* name);
void print_neighbors(Neighbor* N, int q, int k);
void memory_check(void* ptr);

// Generates random data points for a dataset
void random_input(Mat* C, Mat* Q, size_t* c, size_t* q, size_t* d, size_t* k, size_t* num_threads); 
void random_mat(Mat* dataset, size_t points, size_t dimensions);

// Preforms random projection on matrix M, t is the target dimensionality
void random_projection(Mat* M, int t, Mat* RP);
// Trunicates a matrix keeping only perc% of its rows
void truncMat(Mat* src, Mat* target, double perc);

// Calculates the distance matrix D between two datasets C and Q
void calculate_distances(const Mat* C, const Mat* Q, long double* D);

// Finds the k-nearest neighbors for each point in the dataset
// Parameters:
// - D: Distance matrix (flattened 1D array) of size n x n (where n is the number of points in C == Q)
// - n: Number of data points
// - k: Number of neighbors to find
// - knn_indices: Output array of k nearest neighbor indices for each point
// - knn_distances: Output array of distances for the k nearest neighbors
void  findKNN(Mat* C, Mat* Q, Neighbor* N, int k, int num_threads);
void* threadKNN(void* args);

// Helper functions for quick-select algorithm
void swap       (long double* arr, int* indices, int i, int j);
int  partition  (long double* arr, int* indices, int left, int right);
void quickSelect(long double* arr, int* indices, int left, int right, int k, Neighbor* neighbors);

#endif // KNN_H

