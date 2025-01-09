# Traveling-Salesperson-Circle-Heuristic

This repository showcases progressive attempts at creating algorithmic solutions to the Traveling Salesperson Problem (TSP) using Python.

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
