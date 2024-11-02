# k-Nearest Neighbors (k-NN) Implementation

**Authors:** Ioannis Michalainas, Iasonas Lamprinidis  
**Date:** October 2024

## Abstract
This project involves implementing the k-NN algorithm to find the nearest neighbors of a set of query points \( Q \) relative to a corpus set \( C \) in a high-dimensional space. Given a set of \( M \)-point corpus data and \( N \)-point query data, both in \( D \)-dimensional space, the algorithm identifies the k-nearest neighbors of each query point. Using optimized matrix operations and the CBLAS library, our implementation efficiently handles high-dimensional distance calculations.

## Problem Statement
The objective of this project is to implement a subroutine that computes the **k-nearest neighbors (k-NN)** of each query point in \( Q \) based on their distances to the data points in \( C \).

To calculate the distances, we use the following formula:

\[ D = \sqrt{C^2 - 2 C Q^T + (Q^2)^T} \]

where:
- \( C \) is the set of data points (corpus).
- \( Q \) is the set of query points.
- \( D \) is the distance matrix containing distances between each pair of points from \( C \) and \( Q \).

Each row of the \( N \times M \) (queries × corpus) matrix \( D \) contains distances from a query point to all corpus points. We then use the quickselect algorithm to retrieve the k smallest distances in \( O(n) \) time.
