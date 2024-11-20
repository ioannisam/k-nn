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
[ANN Benchmarks](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file)

|                | Pthreads (1) | Pthreads (2) | OpenCilk (1) | OpenCilk (2) | OpenMP (1) | OpenMP (2) |
|----------------|--------------|--------------|--------------|--------------|------------|------------|
| Threads Used   | 4            | 4            | 4            | 4            | 4          | 4          |
| Accuracy (%)   | 97.03        | 54.59        | 96.88        | 51.79        | 96.72      | 47.78      |
| Queries/second | 378.0        | 12.2         | 388.8        | 13.3         | 391.8      | 14.1       |
| Execution Time | 26.46 secs   | 819.99       | 25.72        | 750.32       | 25.52      | 707.58     |

## Appendix
For more in-depth information about the project, please refer to the report.pdf document.
