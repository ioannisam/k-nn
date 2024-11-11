#include "../include/kNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main() {
  
  Mat    C, Q;
  size_t c, q, d, k;

  char choice;
  do {
  printf("Do you want to use random input or load from file? (r for random, f for file): ");
  scanf(" %c", &choice);
  } while(choice != 'r' && choice != 'R' && choice != 'f' && choice != 'F');

  if(choice == 'r' || choice == 'R') {
    random_input(&C, &Q, &c, &q, &d, &k);
  } else if(choice == 'f' || choice == 'F') {
    if(file_input(&C, &Q, &c, &q, &d, &k) != 0) {
      printf("Failed to load file, exiting program.\n");
      return -1;
    }
  }

  print_matrix(&C, "Corpus");
  print_matrix(&Q, "Queries");

  Neighbor* N = (Neighbor*)malloc(q*k*sizeof(Neighbor));
  memory_check(N);

  if((c*d>1000000 && d>100) || d>500) {
    
    Mat C_RP, Q_RP;
    double const e = 0.1;
    int    const t = 4*log((double)c) / (e*e);
    printf("Target dimension (t) for random projection: %d\n", t);
    random_projection(&C, t, &C_RP);
    random_projection(&Q, t, &Q_RP);

    print_matrix(&C_RP, "Corpus Projected");
    print_matrix(&Q_RP, "Queries Projected");

    findKNN(&C_RP, &Q_RP, N, k); 
    print_neighbors(N, q, k);

    free(C_RP.data);
    free(Q_RP.data);
  } else {

    findKNN(&C, &Q, N, k); 
    print_neighbors(N, q, k);
  }

  free(C.data);
  free(Q.data);
  free(N);

  return 0;
}
