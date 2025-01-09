# Traveling-Salesperson-Circle-Heuristic

This repository showcases progressive attempts at creating algorithmic solutions to the Traveling Salesperson Problem (TSP) using Python. 

The underlying idea combines human intuition—drawing a circle to approximate the solution—with computational power to refine the circle into (hopefully) a global minimum. While this approach is unlikely to prove P = NP formally, this heuristic beginning plus probabilistic shuffling ending aims to produce high-quality approximations. By leveraging the geometric properties of a circle, which maximizes the area-to-perimeter ratio, this algorithm, like a human, first tries to narrow the phase space by eliminating many invalid solutions with crossovers. Then looks to utilize the processing power of a computer to find the smallest perimeter circle via a method like 2-opt.

If successful, this heuristic may significantly reduce the time needed to find excellent approximations, particularly for higher-dimensional datasets where circles generalize to spheres and hyperspheres.

---

## TSP5 Overview

`TSP5` implements a heuristic approach to solving the Traveling Salesperson Problem (TSP). This version integrates multiple strategies, including:

- Geometric heuristics
- Angular sweeps
- Path optimization

### Key Features
1. **File Loading**
   - *City Data:* Loads coordinates from a `.tsp` file (e.g., `berlin52.tsp`).
   - *Optimal Tour:* Optionally loads an optimal tour from a `.tour` file for comparison.

2. **Matrix Computation**
   - Calculates the **distance** and **angle matrices** for all city pairs using vectorized operations from Numpy for speed.
  
3. **Edge Node Identification:**
   - Identifies "edge nodes" (cities farthest from the centroid) as candidates for optimizing initial paths.
   - This portion is incorrect, as this edge node identifier is a tunable variable, meaning what classifies an edge node will be different given different starting positions and node numbers. However, this portion was implemented to add weight to nodes around the exterior of the mass, which is likely a necessary component of a successful solution.

4. **Path Generation**
   - Constructs an initial path using a heuristic that balances:
      - **Distance minimization**
      - **Angular alignment**
      - **Edge prioritization**
   - Incorporates probabilistic selection for robustness.
  
5. **Path Optimization:**
   - Refines the generated path using the 2-opt algorithm, which iteratively removes crossovers to reduce path length.
   - This portion is also incorrect, as I have not dug deep enough into it to figure out why crossovers are not being eliminated.

6. **Visualization:**
   - Plots the generated, optimized, and (if available) optimal tours for comparison.
  
7. **Performance Metrics:**
   - Measures and prints path lengths for the generated and optimized paths. If an optimal tour is provided, it calculates its length as a benchmark.

---

## Eventual Core Functionality

### Geometric Heuristic
Uses angular alignment and distance weighting to prioritize city selection, avoiding crossovers.

### 2-Opt Optimization
Implements the 2-opt algorithm to iteratively improve the generated path by removing crossovers.

### Visualization
Uses `matplotlib` to create visual representations of:
   - The generated tour.
   - The optimized tour.
   - The optimal tour (if available).

---

## How to Run

1. Clone the repository
   - `git clone https://github.com/your-username/Traveling-Salesperson-Circle-Heuristic.git`
   - `cd Traveling-Salesperson-Circle-Heuristic`
2. Place your `.tsp` file (e.g., `berlin52.tsp`) in the project directory.
3. Run the script: `python TSP5.py`
4. View the output:
   - Path length of the generated and optimized tours.
   - Plots of the tours.
