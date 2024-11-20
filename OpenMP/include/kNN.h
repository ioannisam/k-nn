#ifndef KNN_H
#define KNN_H

#include <stddef.h>
#include <time.h>

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

/* input.c*/

// User selects to either load file or to generate random data
int  file_input  (Mat* C, Mat* Q, size_t* c, size_t* q, size_t* d, size_t* k);
void random_input(Mat* C, Mat* Q, size_t* c, size_t* q, size_t* d, size_t* k); 

// Loads hdf5 or mat files and directs their output to a Mat struct
int load_hdf5(const char* filename, const char* matname, Mat* matrix);
int load_mat (const char* filename, const char* matname, Mat* matrix);

// Generates random data points for a dataset
void random_mat(Mat* dataset, size_t points, size_t dimensions);

/* print.c*/

// Prints a matrix with given number of rows and columns
void print_matrix(const Mat* matrix, const char* name);

// Prints the kNN and their respective indices
void print_neighbors(Neighbor* N, int q, int k);

// Routine to ensure memory allocation was successful
void memory_check(void* ptr);

/* minimize.c */

// Preforms random projection on matrix M, t is the target dimensionality
void random_projection(Mat* C, Mat* Q, int t, Mat* C_RP, Mat* Q_RP);

// Trunicates a matrix keeping t representative rows
void truncMat(Mat* C, int r, Mat* C_RP);

/* distance.c */

// Calculates the distance matrix D between two datasets C and Q
void calculate_distances(const Mat* C, const Mat* Q, int start_idx, int end_idx, long double* D);

/* find.c */

// Finds the k-nearest neighbors for each point in the dataset
void findKNN(Mat* C, Mat* Q, Neighbor* N, int k);

/* select.c */

// Implementation of the quick select algorithm
void quickSelect(long double* arr, int* indices, int left, int right, int k, Neighbor* neighbors);

// Helper functions for quick-select algorithm
void swap     (long double* arr, int* indices, int i, int j);
int  partition(long double* arr, int* indices, int left, int right);

/* test.c */

// Calculates execution time of findKNN
double duration(clock_t start);

// Calculates queries per second
double qps(clock_t start, size_t q);

// Calculates recall
double recall(Mat* C, Mat* Q, Neighbor* N, int k);
int    compare(const void* a, const void* b);

#endif // KNN_H
