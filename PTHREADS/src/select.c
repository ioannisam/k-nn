#include "../include/kNN.h"

#include <math.h>

void swap(long double* arr, int* indices, int i, int j) {
  double temp = arr[i];
  arr[i] = arr[j];
  arr[j] = temp;

  int tempIdx = indices[i];
  indices[i] = indices[j];
  indices[j] = tempIdx;
}

int partition(long double* arr, int* indices, int left, int right) {
  double pivot = arr[right];
  int pivotIndex = indices[right];
  int i = left;

  for(int j=left; j<right; j++) {
    if(arr[j] < pivot) {
      swap(arr, indices, i, j);
      i++;
    }
  }
  swap(arr, indices, i, right);
  return i;
}

void quickSelect(long double* arr, int* indices, int left, int right, int k, Neighbor* result) {

  if(left <= right) {
    int pivotIndex = partition(arr, indices, left, right);

    if(pivotIndex == k-1) {
      // Copy k nearest distances and indices to result
      for(int i=0; i<k; i++) {
        result[i].distance = sqrt(arr[i]);
        result[i].index = indices[i];
      }
      return;
    } else if(k-1 < pivotIndex) {
      quickSelect(arr, indices, left, pivotIndex-1, k, result);
    } else {
      quickSelect(arr, indices, pivotIndex+1, right, k, result);
    }
  }
}
