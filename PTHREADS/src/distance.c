#include "../include/kNN.h"

#include <stdlib.h>
#include <cblas.h>
#include <pthread.h>

// Struct for thread arguments
typedef struct {
  const Mat* C;
  const Mat* Q;
  int start_idx;
  int end_idx;
  int c;
  int d;
  int batch_size;
  double* C2;
  double* Q2;
  double* CQ;
  long double* D;
  int thread_start;
  int thread_end;
} ThreadArgs;

void* compute_C2(void* args) {

  ThreadArgs* data = (ThreadArgs*)args;
  const Mat* C = data->C;
  int d = data->d;
  double* C2 = data->C2;

  for(int i=data->thread_start; i<data->thread_end; i++) {
    double sum = 0.0;
    for(int j=0; j<d; j++) {
      sum += C->data[i*d + j]*C->data[i*d + j];
    }
    C2[i] = sum;
  }

  pthread_exit(NULL);
}

void* compute_Q2(void* args) {

  ThreadArgs* data = (ThreadArgs*)args;
  const Mat* Q = data->Q;
  int d = data->d;
  double* Q2 = data->Q2;
  int start_idx = data->start_idx;

  for(int i=data->thread_start; i<data->thread_end; i++) {
    double sum = 0.0;
    for(int j=0; j<d; j++) {
      sum += Q->data[(start_idx+i)*d + j]*Q->data[(start_idx+i)*d + j];
    }
    Q2[i] = sum;
  }

  pthread_exit(NULL);
}

void* compute_D(void* args) {

  ThreadArgs* data = (ThreadArgs*)args;
  int c = data->c;
  int batch_size = data->batch_size;
  double* C2 = data->C2;
  double* Q2 = data->Q2;
  double* CQ = data->CQ;
  long double* D = data->D;

  for(int i=data->thread_start; i<data->thread_end; i++) {
    for(int j=0; j<batch_size; j++) {
      D[j*c + i] = C2[i] - 2.0*CQ[i*batch_size + j] + Q2[j];
      if(D[j*c + i] < 0.0) {
        D[j*c + i] = 0.0;
      }
    }
  }

  pthread_exit(NULL);
}

void calculate_distances(const Mat* C, const Mat* Q, int start_idx, int end_idx, long double* D) {

  int c = C->rows;
  int d = C->cols;
  int batch_size = end_idx - start_idx;

  double* C2 = (double*)malloc(c*sizeof(double));
  memory_check(C2);

  double* Q2 = (double*)malloc(batch_size*sizeof(double));
  memory_check(Q2);

  double* CQ = (double*)malloc(c*batch_size*sizeof(double));
  memory_check(CQ);

  int num_threads = 4;
  pthread_t threads[num_threads];
  ThreadArgs args[num_threads];

  int chunk_size = (c + num_threads-1)/num_threads;
  for(int t=0; t<num_threads; t++) {
    args[t] = (ThreadArgs){
        .C = C,
        .c = c,
        .d = d,
        .C2 = C2,
        .thread_start = t*chunk_size,
        .thread_end = (t+1)*chunk_size > c ? c : (t+1)*chunk_size};
    pthread_create(&threads[t], NULL, compute_C2, &args[t]);
  }

  for(int t=0; t<num_threads; t++) {
    pthread_join(threads[t], NULL);
  }

  chunk_size = (batch_size + num_threads-1) / num_threads;
  for(int t=0; t<num_threads; t++) {
    args[t] = (ThreadArgs){
        .Q = Q,
        .d = d,
        .start_idx = start_idx,
        .batch_size = batch_size,
        .Q2 = Q2,
        .thread_start = t*chunk_size,
        .thread_end = (t+1)*chunk_size > batch_size ? batch_size : (t+1)*chunk_size};
    pthread_create(&threads[t], NULL, compute_Q2, &args[t]);
  }

  for(int t=0; t<num_threads; t++) {
    pthread_join(threads[t], NULL);
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c, batch_size, d,
              1.0, C->data, d, Q->data + start_idx * d, d, 0.0, CQ, batch_size);

  chunk_size = (c + num_threads - 1) / num_threads;
  for(int t=0; t<num_threads; t++) {
    args[t] = (ThreadArgs){
        .c = c,
        .batch_size = batch_size,
        .C2 = C2,
        .Q2 = Q2,
        .CQ = CQ,
        .D = D,
        .thread_start = t*chunk_size,
        .thread_end = (t+1)*chunk_size > c ? c : (t+1)*chunk_size};
    pthread_create(&threads[t], NULL, compute_D, &args[t]);
  }

  for(int t=0; t<num_threads; t++) {
    pthread_join(threads[t], NULL);
  }

  free(C2);
  free(Q2);
  free(CQ);
}
