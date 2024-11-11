# k-Nearest Neighbors (k-NN) Implementation

**Authors:** Ioannis Michalainas, Iasonas Lamprinidis  
**Date:** October/November 2024

## Abstract
This C project implements the k-NN algorithm to identify the nearest neighbors of a set of query points \( Q \) relative to a corpus set \( C \) in a high-dimensional space. 
Given a corpus of \( c \) points and \( q \) query points, both in \( d \)-dimensional space, the algorithm efficiently determines the k-nearest neighbors for each query point. 
The implementation leverages optimized matrix operations, parallel processing, and approximation techniques to handle large datasets and high-dimensional distance calculations efficiently.

## Problem Statement
The goal of this project is to develop a subroutine that computes the **k-nearest neighbors (k-NN)** for each query point in \( Q \) based on their distances to the points in \( C \). 
The algorithm uses the quickselect method to retrieve the k smallest distances and their corresponding indices in \( O(n) \) time.

To compute the distances, we use the following formula:

\[
D = \sqrt{C^2 - 2 C Q^T + (Q^2)^T}
\]

where:
- \( C \) is the set of corpus points.
- \( Q \) is the set of query points.
- \( D \) is the distance matrix containing the distances between each pair of points from \( C \) and \( Q \).

Each row of the \( N \times M \) matrix \( D \) (queries × corpus) contains the distances from a query point to all corpus points. 
We then apply the quickselect algorithm to identify the k nearest neighbors for each query point.

## Challenges
The k-NN computation becomes increasingly challenging when dealing with large datasets or high-dimensional spaces. Some of the key challenges include:
- The distance matrix \( D \) may not fit entirely into memory.
- Computational complexity increases dramatically as the number of dimensions grows (the curse of dimensionality).

To address these issues, we employ techniques such as segmenting and computing matrix \( D \) in slices, as well as using Random Projection to reduce dimensionality. 
These approaches introduce an approximation to the results, but they significantly improve computational efficiency and scalability for large-scale datasets.

## Appendix
For more in-depth information about the project, please refer to the k_nn.pdf document.
