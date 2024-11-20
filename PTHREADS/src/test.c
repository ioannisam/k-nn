#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define SAMPLE 100

double duration(clock_t start) {
  
  clock_t end = clock();
  double elapsed = ((double)end-start) / CLOCKS_PER_SEC;
  return elapsed; 
}

double qps(clock_t start, size_t q) {

  clock_t end = clock();
  double  elapsed = ((double)(end-start)) / CLOCKS_PER_SEC;
  return q/elapsed;
}

int compare(const void* a, const void* b) {
  const Neighbor* n1 = (const Neighbor*)a;
  const Neighbor* n2 = (const Neighbor*)b;

  if(n1->distance < n2->distance) {
    return -1;
  } else if(n1->distance > n2->distance) {
    return 1;
  } else {
    return 0;
  }
}

double recall(Mat* C, Mat* Q, Neighbor* N, int k, int num_threads) {

  double accuracy = 0.0;

  Mat Q_TEST;
  Q_TEST.rows = SAMPLE;
  Q_TEST.cols = Q->cols;
  Q_TEST.data = (double*)malloc(SAMPLE*Q->cols*sizeof(double));
  memory_check(Q_TEST.data);
  memcpy(Q_TEST.data, Q->data, SAMPLE*Q->cols*sizeof(double));

  Neighbor* N_TEST = (Neighbor*)malloc(SAMPLE*k*sizeof(Neighbor));
  memory_check(N_TEST);
  findKNN(C, &Q_TEST, N_TEST, k, num_threads);

  for(int i=0; i<SAMPLE; i++) {
    qsort(N      + i*k, k, sizeof(Neighbor), compare);
    qsort(N_TEST + i*k, k, sizeof(Neighbor), compare);
  }

  double error = 0.0;
  for(int i=0; i<SAMPLE; i++) {
    for(int j=0; j<k; j++) {
      double approx = N     [i*k + j].distance;
      double exact  = N_TEST[i*k + j].distance;

      if(exact != 0) {
        error += fabs(approx-exact) / fabs(exact);
      }
    }
  }

  double average = error / (SAMPLE*k);
  accuracy = 1.0 - average;

  free(Q_TEST.data);
  free(N_TEST);

  return accuracy;
}
