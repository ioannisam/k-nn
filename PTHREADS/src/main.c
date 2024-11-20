#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

int main() {
  
  Mat    C, Q;
  size_t c, q, d, k, num_threads;

  char choice;
  do {
  printf("Do you want to use random input or load from file? (r for random, f for file): ");
  scanf(" %c", &choice);
  } while(choice != 'r' && choice != 'R' && choice != 'f' && choice != 'F');

  if(choice == 'r' || choice == 'R') {
    random_input(&C, &Q, &c, &q, &d, &k, &num_threads);
  } else if(choice == 'f' || choice == 'F') {
    if(file_input(&C, &Q, &c, &q, &d, &k, &num_threads) != 0) {
      printf("Failed to load file, exiting program.\n");
      return -1;
    }
  }

  Neighbor* N = (Neighbor*)malloc(q*k*sizeof(Neighbor));
  memory_check(N);

  double const e = 0.3;
  int    const t = log((double)c) / (e*e);
  double elapsed = 0, speed = 0, accuracy = 0;
  if(t<d && (c>1000 && d>50)) {
    
    Mat C_RP, Q_RP;
    printf("Target dimension (t) for random projection: %d\n", t);
    random_projection(&C, &Q, t, &C_RP, &Q_RP);

    clock_t start = clock();
    findKNN(&C_RP, &Q_RP, N, k, num_threads); 
    elapsed  = duration(start);
    speed    = qps(start, q);
    accuracy = recall(&C, &Q, N, k, num_threads);
    print_neighbors(N, q, k);

    free(C_RP.data);
    free(Q_RP.data);
  } else if(c>100000) {

    Mat C_TR;
    int const r = (int)(100*log(c)) + 10*d;
    printf("Representative rows (r): %d\n", r);
    truncMat(&C, r, &C_TR, num_threads);

    clock_t start = clock();
    findKNN(&C_TR, &Q, N, k, num_threads);
    elapsed  = duration(start);
    speed    = qps(start, q);
    accuracy = recall(&C, &Q, N, k, num_threads);
    print_neighbors(N, q, k);

    free(C_TR.data);
  } else {

    printf("Exact calculation\n");

    clock_t start = clock();
    findKNN(&C, &Q, N, k, num_threads); 
    elapsed  = duration(start);
    speed    = qps(start, q);
    accuracy = 1;
    print_neighbors(N, q, k);
  }

  printf("Execution Time: %.2f seconds\n", elapsed);
  printf("Queries per Second: %.1f queries\n", speed);
  printf("Accuracy: %.2f%%\n", 100*accuracy);

  free(C.data);
  free(Q.data);
  free(N);

  return 0;
}
