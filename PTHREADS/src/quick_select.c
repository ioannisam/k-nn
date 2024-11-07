#include "../include/kNN.h"

void swap(double* arr, int i, int j) {
  double temp = arr[i];
  arr[i] = arr[j];
  arr[j] = temp;
}

int partition(double* arr, int left, int right) {

  double pivot = arr[right]; 
  int i = left;

  for(int j = left; j < right; j++) {
    if(arr[j] < pivot) {
      swap(arr, i, j);
      i++;
    }
  }
  swap(arr, i, right);
  return i;
}

void quickSelect(double* arr, int left, int right, int k, double* result) {

  if(left <= right) {
    int pivotIndex = partition(arr, left, right);

    if(pivotIndex == k-1) {
      for(int i=0; i<k; i++) {
        result[i] = arr[i]; 
      }
      return;
    } 
    else if(k-1 < pivotIndex) {
      quickSelect(arr, left, pivotIndex-1, k, result);
    } 
    else {
      quickSelect(arr, pivotIndex + 1, right, k, result);
    }
  }
}
