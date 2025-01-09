# Traveling-Salesperson-Circle-Heuristic
This repository showcases progressive attempts at creating algorithmic solutions to the Traveling Salesperson Problem (TSP) using Python.

The underlying idea combines human intuition—drawing a circle to approximate the problem—with computational power to refine the circle into (hopefully) a global minimum. While this approach is unlikely to formally prove P = NP, the probabilistic shuffling method aims to produce high-quality approximations. By leveraging the geometric properties of a circle, which maximizes the area-to-perimeter ratio, this algorithm narrows the phase space, eliminating many invalid solutions with crossovers.

If successful, this heuristic may significantly reduce the time needed to find excellent approximations, particularly for higher-dimensional datasets where circles generalize to spheres, hyperspheres, etc.

## TSP5 Overview

TSP5 implements a heuristic approach to solving the Traveling Salesperson Problem (TSP). This version integrates multiple strategies, including geometric heuristics, angular sweeps, and path optimization, to approximate efficient solutions. Key features include:

File Loading:

City Data: Loads coordinates from a .tsp file (e.g., berlin52.tsp).
Optimal Tour: Optionally loads an optimal tour from a .tour file for comparison.
Matrix Computation:

Calculates distance matrices and angle matrices for all city pairs using vectorized operations for speed.
Edge Node Identification:

Identifies "edge nodes" (cities farthest from the centroid) as candidates for optimizing initial paths.
Path Generation:

Constructs an initial path using a heuristic that balances:
Distance minimization.
Angular alignment.
Edge prioritization.
Incorporates probabilistic selection for robustness.
Path Optimization:

Refines the generated path using the 2-opt algorithm, which iteratively removes crossovers to reduce path length.
Visualization:

Plots the generated, optimized, and (if available) optimal tours for comparison.
Performance Metrics:

Measures and prints path lengths for the generated and optimized paths. If an optimal tour is provided, it calculates its length as a benchmark.
Core Functionality:
Geometric Heuristic: Uses angular alignment and distance weighting to prioritize city selection, avoiding crossovers.
2-Opt Optimization: Implements the 2-opt algorithm to iteratively improve the generated path.
Visualization: Employs matplotlib to create visual representations of the tours.
