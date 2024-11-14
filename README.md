# k-Nearest Neighbors (k-NN) Implementation

**Authors:** Ioannis Michalainas, Iasonas Lamprinidis  
**Date:** October/November 2024

## Abstract
This C project implements the k-NN algorithm to identify the nearest neighbors of a set of query points \( Q \) relative to a corpus set \( C \) in a high-dimensional space. 
Given a corpus of \( c \) points and \( q \) query points, both in \( d \)-dimensional space, the algorithm efficiently determines the k-nearest neighbors for each query point. 
The implementation leverages optimized matrix operations, parallel processing, and approximation techniques to handle large datasets and high-dimensional distance calculations efficiently.

## Testing/Benchmarks
PC specs: Intel Pentium Silver N5030 (4 cores, 4 threads), RAM (4GB), Arch Linux
Test data: mnist-784-euclidean(1), sift-128-euclidean(2)

|                | Pthreads (1) | Pthreads (2) | OpenMP (1) | OpenMP (2) | OpenCilk (1) | OpenCilk (2) |
|----------------|--------------|--------------|------------|------------|--------------|--------------|
| Threads        | 4            | 4            | 4          | 4          | 4            | 4            |
| Recall         |              |              |            |            |              |              |
| Queries/second |              |              |            |            |              |              |

## Appendix
For more in-depth information about the project, please refer to the k_nn.pdf document.
