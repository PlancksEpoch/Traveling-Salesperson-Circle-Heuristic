# Traveling-Salesperson-Circle-Heuristic

This repository showcases progressive attempts at creating algorithmic solutions to the Traveling Salesperson Problem (TSP) using Python. 

The underlying idea combines human intuition—drawing a circle to approximate the problem—with computational power to refine the circle into (hopefully) a global minimum. While this approach is unlikely to prove P = NP formally, the probabilistic shuffling method aims to produce high-quality approximations. By leveraging the geometric properties of a circle, which maximizes the area-to-perimeter ratio, this algorithm narrows the phase space, eliminating many invalid solutions with crossovers. 

If successful, this heuristic may significantly reduce the time needed to find excellent approximations, particularly for higher-dimensional datasets where circles generalize to spheres and hyperspheres.



## TSP5 Overview

`TSP5` implements a heuristic approach to solving the Traveling Salesperson Problem (TSP). This version integrates multiple strategies, including:

- Geometric heuristics
- Angular sweeps
- Path optimization

### Key Features
1. **File Loading**
   - Load `.tsp` files containing city coordinates.
   - Load `.tour` files containing optimal solutions.

2. **Matrix Computation**
   - Compute distance and angle matrices using NumPy.

3. **Path Optimization**
   - Refine paths with the **2-opt algorithm**.
