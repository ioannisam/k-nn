#include "../include/kNN.h"

#include <stdlib.h>
#include <cilk/cilk.h>

void findKNN(Mat* C, Mat* Q, Neighbor* N, int k) {

  int const c = C->rows;
  int const q = Q->rows;
  int const d = C->cols;

  int const chunk_size = 300;
  int num_chunks = (q + chunk_size-1) / chunk_size;

  cilk_for(int chunk = 0; chunk<num_chunks; chunk++) {

    int start_idx = chunk*chunk_size;
    int end_idx   = (start_idx+chunk_size > q) ? q : start_idx+chunk_size;

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
}
