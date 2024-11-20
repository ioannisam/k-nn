#include "../include/kNN.h"

#include <stdlib.h>
#include <pthread.h>

// Struct for thread arguments
typedef struct {
  Mat* C;           
  Mat* Q;
  Neighbor* N;
  int k;
  int c;
  int chunk_size;
  int start_chunk;
  int end_chunk;
} ThreadArgs;

void* threadKNN(void* args) {

  ThreadArgs* data = (ThreadArgs*)args;
  Mat* C = data->C;
  Mat* Q = data->Q;
  Neighbor* N = data->N;
  int k = data->k;
  int c = data->c;
  int chunk_size = data->chunk_size;

  for(int chunk=data->start_chunk; chunk<data->end_chunk; chunk++) {

    int start_idx = chunk*chunk_size;
    int end_idx = (start_idx+chunk_size > Q->rows) ? Q->rows : start_idx+chunk_size;

    long double* D = (long double*)malloc((end_idx-start_idx)*c*sizeof(long double));
    memory_check(D);

    calculate_distances(C, Q, start_idx, end_idx, D);

    for(int i=start_idx; i<end_idx; i++) {
      int query_idx = i - start_idx;

      int* indices = (int*)malloc(c*sizeof(int));
      memory_check(indices);

      for(int j=0; j<c; j++) {
        indices[j] = j;
      }

      quickSelect(D + query_idx*c, indices, 0, c-1, k, N + i*k);

      free(indices);
    }

    free(D);
  }

  pthread_exit(NULL);
}

void findKNN(Mat* C, Mat* Q, Neighbor* N, int k, int num_threads) {

  int c = C->rows;
  int q = Q->rows;
  int chunk_size = 300;
  int num_chunks = (q + chunk_size-1) / chunk_size;

  pthread_t threads[num_threads];
  ThreadArgs args[num_threads];

  int chunks_per_thread = (num_chunks + num_threads-1) / num_threads;

  for(int t=0; t<num_threads; t++) {
    args[t] = (ThreadArgs){
      .C = C,
      .Q = Q,
      .N = N,
      .k = k,
      .c = c,
      .chunk_size = chunk_size,
      .start_chunk = t*chunks_per_thread,
      .end_chunk = (t+1)*chunks_per_thread > num_chunks ? num_chunks : (t+1)*chunks_per_thread,
    };

    pthread_create(&threads[t], NULL, threadKNN, &args[t]);
  }

  for(int t=0; t<num_threads; t++) {
    pthread_join(threads[t], NULL);
  }
}
