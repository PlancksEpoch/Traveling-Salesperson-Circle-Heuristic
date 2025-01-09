Traveling-Salesperson-Circle-Heuristic
This repository showcases progressive attempts at creating algorithmic solutions to the Traveling Salesperson Problem (TSP) using Python.

The underlying idea combines human intuition—drawing a circle to approximate the problem—with computational power to refine the circle into (hopefully) a global minimum. While this approach is unlikely to formally prove 
𝑃
=
𝑁
𝑃
P=NP, the probabilistic shuffling method aims to produce high-quality approximations. By leveraging the geometric properties of a circle, which maximizes the area-to-perimeter ratio, this algorithm narrows the phase space, eliminating many invalid solutions with crossovers.

If successful, this heuristic may significantly reduce the time needed to find excellent approximations, particularly for higher-dimensional datasets where circles generalize to spheres and hyperspheres.

TSP5 Overview
TSP5 implements a heuristic approach to solving the Traveling Salesperson Problem (TSP). This version integrates multiple strategies, including geometric heuristics, angular sweeps, and path optimization, to approximate efficient solutions.

Key Features
File Loading:

City Data: Loads coordinates from a .tsp file (e.g., berlin52.tsp).
Optimal Tour (Optional): Loads an optimal tour from a .tour file for comparison.
Matrix Computation:

Calculates distance matrices and angle matrices for all city pairs using vectorized operations for speed.
Edge Node Identification:

Identifies "edge nodes" (cities farthest from the centroid) as candidates for optimizing initial paths.
Path Generation:

Constructs an initial path using a heuristic that balances:
Distance minimization
Angular alignment
Edge prioritization
Incorporates probabilistic selection for robustness.
Path Optimization:

Refines the generated path using the 2-opt algorithm, which iteratively removes crossovers to reduce path length.
Visualization:

Plots the generated, optimized, and (if available) optimal tours for comparison.
Performance Metrics:

Measures and prints path lengths for the generated and optimized paths. If an optimal tour is provided, it calculates its length as a benchmark.
Core Functionality
Geometric Heuristic
Uses angular alignment and distance weighting to prioritize city selection, avoiding crossovers.

2-Opt Optimization
Implements the 2-opt algorithm to iteratively improve the generated path by removing crossovers.

Visualization
Uses matplotlib to create visual representations of:

The generated tour.
The optimized tour.
The optimal tour (if available).
How to Run
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/Traveling-Salesperson-Circle-Heuristic.git
cd Traveling-Salesperson-Circle-Heuristic
Place your .tsp file (e.g., berlin52.tsp) in the project directory.

Run the script:

bash
Copy code
python TSP5.py
View the output:

Path length of the generated and optimized tours.
Plots of the tours.
File Descriptions
TSP5.py: Main script implementing the heuristic and optimization.
README.md: Documentation for the repository.
Example .tsp and .tour Files: Test datasets to verify functionality.
Future Enhancements
Explore additional heuristics for initial path generation.
Parallelize the optimization process for faster results.
Test the algorithm on higher-dimensional datasets.
