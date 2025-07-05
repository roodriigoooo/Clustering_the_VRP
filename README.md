# Vehicle Routing Problem (VRP) Solution Framework

#### *Lorena Pinillos*, *Laura Rodríguez* and *Rodrigo Sastré*

This repository contains an solution framework for the Vehicle Routing Problem (VRP) with a focus on optimizing the cluster-route strategy and applying it to Kelly's instances.

## Overview

The Vehicle Routing Problem (VRP) is a combinatorial optimization problem that involves finding the optimal set of routes for a fleet of vehicles to deliver goods to a set of customers. This implementation focuses on the Capacitated VRP (CVRP), where each vehicle has a limited capacity.

The solution framework includes:

1. **Clarke-Wright Savings Algorithm**: A heuristic algorithm for solving the VRP.
2. **Clustering Algorithms**: Various clustering methods to group customers before routing.
3. **Local Search Improvements**: 2-opt and 3-opt local search to improve routes.
4. **Evaluation Framework**: Tools for comparing different solution methods.

## Components

### Data Handling

- `data_handler.py`: Handles loading and saving VRP instances, including Kelly's instances.

### Core VRP Objects

- `vrp_objects.py`: Defines the core objects used in the VRP solution (Node, Edge, Route, Solution).

### Solution Methods

- `clarke_wright.py`: Implements the Clarke-Wright Savings algorithm.
- `clustering.py`: Implements various clustering algorithms (K-means, DBSCAN, hierarchical, etc.).
- `local_search.py`: Implements local search improvements (2-opt, 3-opt).

### Evaluation

- `evaluation.py`: Provides tools for evaluating and comparing VRP solutions.

### Main Scripts

- `main.py`: Main script to run experiments and compare different solution methods.
- `test_vrp.py`: Unit tests for the VRP solution framework.

## Kelly's Instances

Kelly's instances are a set of benchmark instances for the VRP characterized by clustered customers. This implementation includes:

- Generation of Kelly's instances with varying numbers of customers and vehicles.
- Solving Kelly's instances using different solution methods.
- Comparing solutions with and without clustering.

## References

- Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. Operations Research, 12(4), 568-581.
- Kelly, J. P., & Xu, J. (1999). A set-partitioning-based heuristic for the vehicle routing problem. INFORMS Journal on Computing, 11(2), 161-172.
- Laporte, G. (1992). The vehicle routing problem: An overview of exact and approximate algorithms. European Journal of Operational Research, 59(3), 345-358.
