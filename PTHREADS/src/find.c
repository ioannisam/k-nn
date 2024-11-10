#include "../include/kNN.h"

#include <stdlib.h>
#include <pthread.h>

void findKNN(Mat* C, Mat* Q, Neighbor* N, int k, int num_threads) {

  pthread_t  threads[num_threads];
  ThreadArgs args[num_threads];

  int q = Q->rows;
  int queries_per_thread = q / num_threads;

  for(int i=0; i<num_threads; i++) {

    args[i].C = C;
    args[i].Q = Q;
    args[i].N = N;
    args[i].k = k;
    args[i].start = i*queries_per_thread;
    args[i].end = (i == num_threads-1) ? q : (i+1)*queries_per_thread;

    pthread_create(&threads[i], NULL, threadKNN, &args[i]);
  }

  for(int i=0; i<num_threads; i++) {
    pthread_join(threads[i], NULL);
  }
}

void* threadKNN(void* args) {

  ThreadArgs* data = (ThreadArgs*)args;
  Mat*        C = data->C;
  Mat*        Q = data->Q;
  Neighbor*   N = data->N;
  int         k = data->k;
  int     start = data->start;
  int       end = data->end;

  int const c = C->rows;
  int const d = C->cols;

  for(int i=start; i<end; i++) {

    long double* D = (long double*)malloc(c*sizeof(long double));
    memory_check(D);

    calculate_distances(C, &(Mat){.rows = 1, .cols = d, .data = Q->data + i*d}, D);

    int* indices = (int*)malloc(c*sizeof(int));
    memory_check(indices);

    for(int j=0; j<c; j++) {
      indices[j] = j;
    }

    quickSelect(D, indices, 0, c-1, k, N + i*k);
    free(D);
    free(indices);
  }

  pthread_exit(NULL);
}
