#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <matio.h>
#include <hdf5.h>
#include <time.h>
#include <string.h>

int file_input(Mat* C, Mat* Q, size_t* c, size_t* q, size_t* d, size_t* k, size_t* num_threads) {

  char filename[256];

  printf("Enter the filename (in the test folder): ");
  scanf("%s", filename);

  char filepath[300];
  snprintf(filepath, sizeof(filepath), "../../test/%s", filename);

  FILE* file = fopen(filepath, "r");
  if(!file) {
    fprintf(stderr, "File does not exist: %s\n", filepath);
    return -1;
  } else {
    fclose(file);
  }

  if(strstr(filename, ".mat") != NULL) {
    if(load_mat(filepath, "CORPUS", C) == -1) {
      fprintf(stderr, "Error loading .mat file.\n");
      return -1;
    }
    if(load_mat(filepath, "QUERY", Q) == -1) {
      fprintf(stderr, "Error loading .mat file.\n");
      free(C->data);
      return -1;
    }
  } else if(strstr(filename, ".hdf5") != NULL) {
    if(load_hdf5(filepath, "CORPUS", C) == -1) {
      fprintf(stderr, "Error loading .hdf5 file.\n");
      return -1;
    }
    if(load_hdf5(filepath, "QUERY", Q) == -1) {
      fprintf(stderr, "Error loading .hdf5 file.\n");
      free(C->data);
      return -1;
    }
  } else {
    fprintf(stderr, "Unsupported file format. Use .mat or .hdf5.\n");
    return -1;
  }

  *c = C->rows;
  *q = Q->rows;
  *d = C->cols;

  printf("Enter the number of nearest neighbors (k): ");
  scanf("%zu", k);
  printf("Enter the number of threads to use: ");
  scanf("%zu", num_threads);

  return 0;
}

int load_mat(const char* filename, const char* matname, Mat* matrix) {

  mat_t* file = Mat_Open(filename, MAT_ACC_RDONLY);
  if(file == NULL) {
    printf("Opening file failed...\n");
    return -1;
  }
  printf("File opened successfully!\n");

  matvar_t* arrayPtr = Mat_VarRead(file, matname);
  if(arrayPtr == NULL || arrayPtr->data == NULL) {
    printf("Array pointer or data pointer in array is NULL.\n");
    if(arrayPtr) {
      Mat_VarFree(arrayPtr);
    }
    Mat_Close(file);
    return -1;
  }

  matrix->rows = (size_t)arrayPtr->dims[0];
  matrix->cols = (size_t)arrayPtr->dims[1];

  matrix->data = (double*)malloc(matrix->rows*matrix->cols*sizeof(double));
  if(matrix->data == NULL) {
    printf("Memory allocation for matrix data failed.\n");
    Mat_VarFree(arrayPtr);
    Mat_Close(file);
    return -1;
  }

  double* dataPtr = (double*)arrayPtr->data;
  for(size_t i=0; i<matrix->rows*matrix->cols; i++) {
    matrix->data[i] = dataPtr[i];
  }

  Mat_VarFree(arrayPtr);
  Mat_Close(file);

  printf("Matrix %s loaded successfully!\n", matname);
  return 0;  
}

int load_hdf5(const char* filename, const char* matname, Mat* matrix) {

  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if(file_id < 0) {
    printf("Error opening HDF5 file: %s\n", filename);
    return -1;
  }

  hid_t dataset_id = H5Dopen(file_id, matname, H5P_DEFAULT);
  if(dataset_id < 0) {
    printf("Error opening dataset: %s\n", matname);
    H5Fclose(file_id);
    return -1;
  }

  hid_t   space_id = H5Dget_space(dataset_id);
  int     ndims    = H5Sget_simple_extent_ndims(space_id);
  hsize_t dims[2];
  H5Sget_simple_extent_dims(space_id, dims, NULL);

  matrix->rows = dims[0];
  matrix->cols = dims[1];

  matrix->data = (double*)malloc(matrix->rows*matrix->cols*sizeof(double));
  if(matrix->data == NULL) {
    printf("Memory allocation failed\n");
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return -1;
  }

  H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix->data);

  H5Sclose(space_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);

  printf("Matrix %s loaded successfully!\n", matname);
  return 0;
}

void random_input(Mat* C, Mat* Q, size_t* c, size_t* q, size_t* d, size_t* k, size_t* num_threads) {

  printf("Enter the number of corpus points: ");
  scanf("%zu", c);
  printf("Enter the number of query points: ");
  scanf("%zu", q);
  printf("Enter the number of dimensions: ");
  scanf("%zu", d);
  printf("Enter the number of nearest neighbors (k): ");
  scanf("%zu", k);
  printf("Enter the number of threads to use: ");
  scanf("%zu", num_threads);
  printf("\n");

  random_mat(C, *c, *d);
  random_mat(Q, *q, *d);
} 

void random_mat(Mat* matrix, size_t points, size_t dimensions) {

  srand(time(NULL) + (uintptr_t)matrix);

  matrix->data = (double*)malloc(points*dimensions*sizeof(double));
  memory_check(matrix->data);

  matrix->rows = points;
  matrix->cols = dimensions;
   
  for (size_t i = 0; i < points * dimensions; i++) {
    // Scale to [0, 200], then shift to [-100, 100]
    matrix->data[i] = ((double)rand()/RAND_MAX)*200.0 - 100.0;
  }
}
