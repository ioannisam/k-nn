**Authors:** Ioannis Michalainas  
**Date:** October/November 2024

## Abstract
This C project implements the k-NN algorithm to identify the nearest neighbors of a set of query points **Q** relative to a corpus set **C** in a high-dimensional space. 
Given a corpus of **c** points and **q** query points, both in **d**-dimensional space, the algorithm efficiently determines the k-nearest neighbors for each query point. 
The implementation leverages optimized matrix operations, parallel processing, and approximation techniques to handle large datasets and high-dimensional distance calculations efficiently.

## Testing/Benchmarks
PC specs: Intel Pentium Silver N5030 (4 cores, 4 threads), RAM (4GB), Arch Linux

Test data: mnist-784-euclidean(1), sift-128-euclidean(2)
[ANN Benchmarks](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file)

|                    | Pthreads (1) | Pthreads (2) | OpenCilk (1) | OpenCilk (2) | OpenMP (1) | OpenMP (2) |
|--------------------|--------------|--------------|--------------|--------------|------------|------------|
| Threads Used       | 4            | 4            | 4            | 4            | 4          | 4          |
| Accuracy (%)       | 97.03        | 54.59        | 96.88        | 51.79        | 96.72      | 47.78      |
| Queries/second     | 378.0        | 12.2         | 388.8        | 13.3         | 391.8      | 14.1       |
| Execution Time (s) | 26.46        | 819.99       | 25.72        | 750.32       | 25.52      | 707.58     |

## Installation

1. **Clone this repository:**
```
git clone https://github.com/ioannisam/k-nn.git
```
2. **Navigate to the src folder of the implementation you want to run:**
```
cd OpenMP/src
```
3. **Compile and run:**
```
make
```
```
./kNN
```

The libraries used in this project include **MATIO** and **HDF5** for file reading, and **OpenMP**, **OpenCilk**, and **Pthreads** for parallelism. 
If any required libraries are missing, the compiler will notify you accordingly. 
You may need to adjust the Makefile to reflect the correct locations of these libraries on your system.
To use custom test files (.mat and .hdf5 supported) you need to place them in the **test** folder.

**Note:** When using OpenCilk, ensure the version compatibility of the library, as mismatches can occur.

## Appendix
For more in-depth information about the project, please refer to the `report.pdf` document.
