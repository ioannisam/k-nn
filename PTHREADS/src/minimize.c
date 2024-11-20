#include "../include/kNN.h"

#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <pthread.h>

void random_projection(Mat* C, Mat* Q, int t, Mat* C_RP, Mat* Q_RP) {

  srand(time(NULL) + (uintptr_t)C);
  
  int c = (int)C->rows;
  int q = (int)Q->rows;
  int d = (int)C->cols;

  C_RP->rows = c;
  C_RP->cols = t;
  Q_RP->rows = q;
  Q_RP->cols = t;

  C_RP->data = (double*)malloc(c*t*sizeof(double));
  Q_RP->data = (double*)malloc(q*t*sizeof(double));

  double* R = (double*)malloc(d*t*sizeof(double));
  for (int i=0; i<d*t; i++) {
    // Rademacher distribution (-1 or +1)
    R[i] = (rand()%2 == 0 ? -1 : 1) / sqrt((double)t);
  }
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, t, d, 1.0, C->data, d, R, t, 0.0, C_RP->data, t);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, q, t, d, 1.0, Q->data, d, R, t, 0.0, Q_RP->data, t);

  free(R);
}

// Thread data structure
typedef struct {
  Mat* C;
  double* distances;
  double* ref_point;
  int start;
  int end;
  int d;
} ThreadData;

void* compute_distances(void* arg) {

  ThreadData* data = (ThreadData*)arg;
  Mat* C = data->C;
  double* distances = data->distances;
  double* ref_point = data->ref_point;
  int d = data->d;

  for(int i=data->start; i<data->end; i++) {
    double dist = 0.0;
    for(int j=0; j<d; j++) {
      double diff = C->data[i*d + j] - ref_point[j];
      dist += diff*diff;
    }
    distances[i] = dist;
  }
  return NULL;
}

// Thread data structure for reduction
typedef struct {
  double* distances;
  double total_dist;
  int start;
  int end;
} ReductionData;

void* compute_total_distance(void* arg) {

  ReductionData* data = (ReductionData*)arg;
  double sum = 0.0;

  for (int i = data->start; i < data->end; i++) {
    sum += data->distances[i];
  }
  data->total_dist = sum;
  return NULL;
}

void truncMat(Mat* C, int r, Mat* C_TR, int num_threads) {

  int c = C->rows;
  int d = C->cols;

  C_TR->rows = r;
  C_TR->cols = d;
  C_TR->data = (double*)malloc(r*d*sizeof(double));

  srand(time(NULL));

  int first_idx = rand()%c;
  memcpy(C_TR->data, C->data + first_idx*d, d*sizeof(double));

  double* distances = (double*)malloc(c*sizeof(double));

  pthread_t  threads[num_threads];
  ThreadData thread_data[num_threads];

  int rows_per_thread = c/num_threads;
  for(int t=0; t<num_threads; t++) {
    thread_data[t] = (ThreadData){
      .C = C,
      .distances = distances,
      .ref_point = C_TR->data,
      .start = t*rows_per_thread,
      .end = (t == num_threads-1) ? c : (t+1)*rows_per_thread,
      .d = d
    };
    pthread_create(&threads[t], NULL, compute_distances, &thread_data[t]);
  }
  for(int t=0; t<num_threads; t++) {
    pthread_join(threads[t], NULL);
  }

  for(int i=1; i<r; i++) {
    double total_dist = 0.0;

    ReductionData reduction_data[num_threads];
    for(int t=0; t<num_threads; t++) {
      reduction_data[t] = (ReductionData){
        .distances = distances,
        .total_dist = 0.0,
        .start = t*rows_per_thread,
        .end = (t == num_threads-1) ? c : (t+1)*rows_per_thread
      };
      pthread_create(&threads[t], NULL, compute_total_distance, &reduction_data[t]);
    }
    for(int t=0; t<num_threads; t++) {
      pthread_join(threads[t], NULL);
      total_dist += reduction_data[t].total_dist;
    }

    double rand_dist = ((double)rand() / RAND_MAX)*total_dist;
    double cumulative_dist = 0.0;
    int next_idx = 0;
    for(int j=0; j< c; j++) {
      cumulative_dist += distances[j];
      if(cumulative_dist >= rand_dist) {
        next_idx = j;
        break;
      }
    }

    memcpy(C_TR->data + i*d, C->data + next_idx*d, d*sizeof(double));

    for(int t=0; t<num_threads; t++) {
      thread_data[t].ref_point = C_TR->data + i*d;
      pthread_create(&threads[t], NULL, compute_distances, &thread_data[t]);
    }
    for(int t=0; t<num_threads; t++) {
      pthread_join(threads[t], NULL);
    }
  }

  free(distances);
}
