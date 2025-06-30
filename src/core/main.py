"""
Main script to run the improved VRP solution framework with enhanced DBSCAN clustering
and complementary metaheuristics.
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#local
from src.data.data_handler import DataHandler
from src.utilities.clarke_wright import ClarkeWrightSolver
from src.utilities.clustering import VRPClusterer
from local_search import LocalSearch
from metaheuristics import VariableNeighborhoodSearch, GuidedLocalSearch, DBSCANTabuSearch
from src.evaluation import VRPEvaluator
from src.viz.advanced_visualization import AdvancedVRPVisualizer
from vrp_objects import Solution

def solve_without_clustering(nodes, vehicle_capacity):
    """
    Solve the VRP without clustering.

    Args:
        nodes: List of Node objects
        vehicle_capacity: Vehicle capacity

    Returns:
        Solution object
    """
    # Solve using Clarke-Wright
    cw_solver = ClarkeWrightSolver(nodes, vehicle_capacity)
    solution = cw_solver.solve()

    return solution

def solve_with_clustering(nodes, vehicle_capacity, clustering_method='kmeans', verbose=False):
    """
    Solve the VRP with clustering.

    Args:
        nodes: List of Node objects
        vehicle_capacity: Vehicle capacity
        clustering_method: Clustering method to use
        verbose: Whether to print detailed information

    Returns:
        Solution object, cluster_labels (if DBSCAN is used, otherwise None)
    """
    # Create clusterer
    clusterer = VRPClusterer(nodes, vehicle_capacity=vehicle_capacity)

    # Apply clustering
    if clustering_method == 'kmeans':
        if verbose:
            print("Applying K-means clustering...")
        clusters = clusterer.kmeans_clustering()
        cluster_labels = None
    elif clustering_method == 'dbscan':
        if verbose:
            print("Applying enhanced DBSCAN clustering with auto-tuning...")
        clusters = clusterer.dbscan_clustering(auto_tune=True, verbose=verbose)

        # Extract DBSCAN labels for later use with DBSCAN-Tabu Search
        cluster_labels = []
        for i, node in enumerate(nodes[1:]):  # Skip depot
            # Find which cluster this node belongs to
            for cluster_idx, cluster in enumerate(clusters):
                if node in cluster:
                    cluster_labels.append(cluster_idx)
                    break
            else:
                # Node not found in any cluster (should not happen)
                cluster_labels.append(-1)
    elif clustering_method == 'hierarchical':
        if verbose:
            print("Applying hierarchical clustering...")
        clusters = clusterer.hierarchical_clustering()
        cluster_labels = None
    elif clustering_method == 'capacity':
        if verbose:
            print("Applying capacity-based clustering...")
        clusters = clusterer.capacity_based_clustering()
        cluster_labels = None
    elif clustering_method == 'balanced_kmeans':
        if verbose:
            print("Applying balanced K-means clustering...")
        clusters = clusterer.balanced_kmeans_clustering()
        cluster_labels = None
    elif clustering_method == 'kmeans_dbscan_hybrid':
        if verbose:
            print("Applying K-means+DBSCAN hybrid clustering...")
        clusters = clusterer.kmeans_dbscan_hybrid_clustering(auto_tune=True, verbose=verbose)

        # Extract cluster labels for later use with DBSCAN-Tabu Search
        cluster_labels = []
        for i, node in enumerate(nodes[1:]):  # Skip depot
            # Find which cluster this node belongs to
            for cluster_idx, cluster in enumerate(clusters):
                if node in cluster:
                    cluster_labels.append(cluster_idx)
                    break
            else:
                # Node not found in any cluster (should not happen)
                cluster_labels.append(-1)
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")

    if verbose:
        print(f"Clustering complete. Found {len(clusters)} clusters.")
        for i, cluster in enumerate(clusters):
            cluster_demand = sum(node.demand for node in cluster)
            print(f"  Cluster {i}: {len(cluster)} nodes, total demand: {cluster_demand:.2f}")

    # Solve each cluster separately
    depot = nodes[0]
    combined_solution = Solution()

    for cluster_idx, cluster in enumerate(clusters):
        # Skip empty clusters
        if not cluster:
            if verbose:
                print(f"Skipping empty cluster {cluster_idx}")
            continue

        if verbose:
            print(f"Solving cluster {cluster_idx} with {len(cluster)} nodes...")

        # Create a new list of nodes with the depot and the cluster
        cluster_nodes = [depot] + cluster

        # Solve using Clarke-Wright
        cw_solver = ClarkeWrightSolver(cluster_nodes, vehicle_capacity)
        cluster_solution = cw_solver.solve()

        if verbose:
            print(f"  Cluster {cluster_idx} solution: {len(cluster_solution.routes)} routes, cost: {cluster_solution.cost:.2f}")

        # Add routes to combined solution
        for route in cluster_solution.routes:
            combined_solution.add_route(route)

    if verbose:
        print(f"Combined solution: {len(combined_solution.routes)} routes, cost: {combined_solution.cost:.2f}")

    return combined_solution, cluster_labels

def solve_with_local_search(nodes, vehicle_capacity, use_clustering=True, clustering_method='kmeans', verbose=False):
    """
    Solve the VRP with local search improvements.

    Args:
        nodes: List of Node objects
        vehicle_capacity: Vehicle capacity
        use_clustering: Whether to use clustering
        clustering_method: Clustering method to use
        verbose: Whether to print detailed information

    Returns:
        Solution object
    """
    # Get initial solution
    if use_clustering:
        if verbose:
            print("Getting initial solution with clustering...")
        initial_solution, _ = solve_with_clustering(nodes, vehicle_capacity, clustering_method, verbose)
    else:
        if verbose:
            print("Getting initial solution without clustering...")
        initial_solution = solve_without_clustering(nodes, vehicle_capacity)

    if verbose:
        print(f"Initial solution cost: {initial_solution.cost:.2f}")
        print("Applying local search improvements...")

    # Apply local search
    local_search = LocalSearch(initial_solution, nodes, vehicle_capacity)

    # Apply 2-opt
    start_time = time.time()
    improved_solution = local_search.apply_2opt()
    end_time = time.time()

    if verbose:
        print(f"2-opt improvement: {initial_solution.cost - improved_solution.cost:.2f} "
              f"({(initial_solution.cost - improved_solution.cost) / initial_solution.cost * 100:.2f}%)")
        print(f"2-opt runtime: {end_time - start_time:.2f} seconds")

    # Apply 3-opt
    start_time = time.time()
    final_solution = local_search.apply_3opt()
    end_time = time.time()

    if verbose:
        print(f"3-opt improvement: {improved_solution.cost - final_solution.cost:.2f} "
              f"({(improved_solution.cost - final_solution.cost) / improved_solution.cost * 100:.2f}%)")
        print(f"3-opt runtime: {end_time - start_time:.2f} seconds")
        print(f"Total improvement: {initial_solution.cost - final_solution.cost:.2f} "
              f"({(initial_solution.cost - final_solution.cost) / initial_solution.cost * 100:.2f}%)")
        print(f"Final solution cost: {final_solution.cost:.2f}")

    return final_solution

def solve_with_vns(nodes, vehicle_capacity, use_clustering=True, clustering_method='kmeans', verbose=False):
    """
    Solve the VRP with Variable Neighborhood Search.

    Args:
        nodes: List of Node objects
        vehicle_capacity: Vehicle capacity
        use_clustering: Whether to use clustering
        clustering_method: Clustering method to use
        verbose: Whether to print detailed information

    Returns:
        Solution object
    """
    # Get initial solution
    if use_clustering:
        if verbose:
            print("Getting initial solution with clustering...")
        initial_solution, _ = solve_with_clustering(nodes, vehicle_capacity, clustering_method, verbose)
    else:
        if verbose:
            print("Getting initial solution without clustering...")
        initial_solution = solve_without_clustering(nodes, vehicle_capacity)

    if verbose:
        print(f"Initial solution cost: {initial_solution.cost:.2f}")
        print("Applying Variable Neighborhood Search...")

    # Apply VNS
    start_time = time.time()
    vns = VariableNeighborhoodSearch(initial_solution, nodes, vehicle_capacity, verbose=verbose)
    improved_solution = vns.solve()
    end_time = time.time()

    if verbose:
        print(f"VNS improvement: {initial_solution.cost - improved_solution.cost:.2f} "
              f"({(initial_solution.cost - improved_solution.cost) / initial_solution.cost * 100:.2f}%)")
        print(f"VNS runtime: {end_time - start_time:.2f} seconds")
        print(f"Final solution cost: {improved_solution.cost:.2f}")

    return improved_solution

def solve_with_gls(nodes, vehicle_capacity, use_clustering=True, clustering_method='kmeans', verbose=False):
    """
    Solve the VRP with Guided Local Search.

    Args:
        nodes: List of Node objects
        vehicle_capacity: Vehicle capacity
        use_clustering: Whether to use clustering
        clustering_method: Clustering method to use
        verbose: Whether to print detailed information

    Returns:
        Solution object
    """
    # Get initial solution
    if use_clustering:
        if verbose:
            print("Getting initial solution with clustering...")
        initial_solution, _ = solve_with_clustering(nodes, vehicle_capacity, clustering_method, verbose)
    else:
        if verbose:
            print("Getting initial solution without clustering...")
        initial_solution = solve_without_clustering(nodes, vehicle_capacity)

    if verbose:
        print(f"Initial solution cost: {initial_solution.cost:.2f}")
        print("Applying Guided Local Search...")

    # Apply GLS
    start_time = time.time()
    gls = GuidedLocalSearch(initial_solution, nodes, vehicle_capacity, verbose=verbose)
    improved_solution = gls.solve()
    end_time = time.time()

    if verbose:
        print(f"GLS improvement: {initial_solution.cost - improved_solution.cost:.2f} "
              f"({(initial_solution.cost - improved_solution.cost) / initial_solution.cost * 100:.2f}%)")
        print(f"GLS runtime: {end_time - start_time:.2f} seconds")
        print(f"Final solution cost: {improved_solution.cost:.2f}")

    return improved_solution

def solve_with_dbscan_tabu(nodes, vehicle_capacity, verbose=False):
    """
    Solve the VRP with DBSCAN-Tabu Search hybrid.

    Args:
        nodes: List of Node objects
        vehicle_capacity: Vehicle capacity
        verbose: Whether to print detailed information

    Returns:
        Solution object
    """
    if verbose:
        print("Getting initial solution with DBSCAN clustering...")

    # Get initial solution with DBSCAN clustering
    initial_solution, dbscan_labels = solve_with_clustering(nodes, vehicle_capacity,
                                                          clustering_method='dbscan',
                                                          verbose=verbose)

    if verbose:
        print(f"Initial solution cost: {initial_solution.cost:.2f}")
        print("Applying DBSCAN-Tabu Search hybrid...")

    # Apply DBSCAN-Tabu Search
    start_time = time.time()
    dbscan_tabu = DBSCANTabuSearch(initial_solution, nodes, vehicle_capacity,
                                  dbscan_labels, verbose=verbose)
    improved_solution = dbscan_tabu.solve()
    end_time = time.time()

    if verbose:
        print(f"DBSCAN-Tabu improvement: {initial_solution.cost - improved_solution.cost:.2f} "
              f"({(initial_solution.cost - improved_solution.cost) / initial_solution.cost * 100:.2f}%)")
        print(f"DBSCAN-Tabu runtime: {end_time - start_time:.2f} seconds")
        print(f"Final solution cost: {improved_solution.cost:.2f}")

    return improved_solution

def run_experiments(instance_names=None, verbose=True):
    """
    Run experiments on the specified instances.

    Args:
        instance_names: List of instance names to run experiments on
        verbose: Whether to print detailed information
    """
    # Create data handler
    print("Creating data handler...")
    data_handler = DataHandler()

    # Generate Kelly's instances if they don't exist
    if not os.path.exists('../../data'):
        print("Creating data directory and generating Kelly's instances...")
        os.makedirs('../../data')
        data_handler.generate_all_kellys_instances()

    # If no instance names provided, use a range of Kelly's instances from small to large
    if instance_names is None:
        instance_names = [
            "Kelly_n20_k3",   # Small instance
            "Kelly_n30_k4",   # Small-medium instance
            "Kelly_n50_k5",   # Medium instance
            "Kelly_n75_k8",   # Medium-large instance
            "Kelly_n100_k10"  # Large instance
        ]

    # Create evaluator
    print("Creating evaluator...")
    evaluator = VRPEvaluator()

    # Create results directory if it doesn't exist
    if not os.path.exists('../../results'):
        os.makedirs('../../results')

    # Run experiments for each instance
    for instance_name in instance_names:
        print(f"\n{'='*80}")
        print(f"Running experiments for instance {instance_name}")
        print(f"{'='*80}")

        # Load instance
        print(f"Loading instance {instance_name}...")
        nodes, vehicle_capacity = data_handler.load_instance(instance_name)
        print(f"Loaded {len(nodes)} nodes with vehicle capacity {vehicle_capacity}")

        # Define methods to compare
        methods = {
            # Basic methods
            "CW": lambda n, vc: solve_without_clustering(n, vc),

            # DBSCAN-based methods (our focus)
            "DBSCAN+CW": lambda n, vc: solve_with_clustering(n, vc, clustering_method='dbscan', verbose=verbose)[0],
            "DBSCAN+CW+2opt+3opt": lambda n, vc: solve_with_local_search(n, vc, use_clustering=True, clustering_method='dbscan', verbose=verbose),
            "DBSCAN+CW+VNS": lambda n, vc: solve_with_vns(n, vc, use_clustering=True, clustering_method='dbscan', verbose=verbose),
            "DBSCAN+CW+GLS": lambda n, vc: solve_with_gls(n, vc, use_clustering=True, clustering_method='dbscan', verbose=verbose),
            "DBSCAN-TabuSearch": lambda n, vc: solve_with_dbscan_tabu(n, vc, verbose=verbose),

            # Other clustering methods for comparison
            "KMeans+CW": lambda n, vc: solve_with_clustering(n, vc, clustering_method='kmeans', verbose=verbose)[0],
            "BalancedKMeans+CW": lambda n, vc: solve_with_clustering(n, vc, clustering_method='balanced_kmeans', verbose=verbose)[0]
        }

        # Run benchmark
        solutions = evaluator.run_benchmark(instance_name, nodes, vehicle_capacity, methods)

        # Visualize all solutions
        for method_name, solution in solutions.items():
            fig = evaluator.visualize_solution(solution, nodes, instance_name, method_name)
            plt.savefig(f"results/{instance_name}_{method_name.replace('+', '_')}.png")
            plt.close(fig)

        # Visualize best solution
        best_method = min(solutions.keys(), key=lambda k: solutions[k].cost)
        best_solution = solutions[best_method]

        print(f"\nBest method for {instance_name}: {best_method}")
        print(f"Best solution cost: {best_solution.cost:.2f}")

        fig = evaluator.visualize_solution(best_solution, nodes, instance_name, best_method)
        plt.savefig(f"results/BEST_{instance_name}_{best_method.replace('+', '_')}.png")
        plt.close(fig)

    # Save results
    evaluator.save_results("results/vrp_results.csv")

    # Plot comparisons
    fig = evaluator.plot_comparison(metric='total_cost')
    plt.savefig("results/comparison_total_cost.png")
    plt.close(fig)

    fig = evaluator.plot_comparison(metric='runtime')
    plt.savefig("results/comparison_runtime.png")
    plt.close(fig)

    fig = evaluator.plot_comparison(metric='num_routes')
    plt.savefig("results/comparison_num_routes.png")
    plt.close(fig)

    # Create detailed report
    print("\nGenerating detailed report...")
    df = evaluator.get_results_dataframe()

    # Calculate improvement over basic CW
    instance_methods = df.groupby(['instance_name', 'method_name']).agg({
        'total_cost': 'mean',
        'num_routes': 'mean',
        'runtime': 'mean'
    }).reset_index()

    # Create a pivot table for easier comparison
    pivot_cost = pd.pivot_table(
        instance_methods,
        values='total_cost',
        index='instance_name',
        columns='method_name'
    )

    # Calculate improvement over CW
    for method in pivot_cost.columns:
        if method != 'CW':
            pivot_cost[f'{method}_improvement'] = (pivot_cost['CW'] - pivot_cost[method]) / pivot_cost['CW'] * 100

    # Save detailed report
    pivot_cost.to_csv("results/detailed_cost_comparison.csv")

    # Print summary
    print("\nSummary of results:")
    comparison = evaluator.compare_methods()
    print(comparison)

    # Print detailed comparison of DBSCAN methods
    print("\nDetailed comparison of DBSCAN-based methods:")
    dbscan_methods = [m for m in comparison['method_name'] if 'DBSCAN' in m]
    dbscan_comparison = comparison[comparison['method_name'].isin(dbscan_methods)]
    print(dbscan_comparison)

    # Print improvement over basic CW
    print("\nImprovement over basic CW (%):")
    for method in pivot_cost.columns:
        if 'improvement' in method:
            method_name = method.replace('_improvement', '')
            improvement = pivot_cost[method].mean()
            print(f"{method_name}: {improvement:.2f}%")

def run_small_test():
    """Run a small test with just one instance and a subset of methods."""
    # Create data handler
    print("Creating data handler...")
    data_handler = DataHandler()

    # Generate Kelly's instances if they don't exist
    if not os.path.exists('../../data'):
        print("Creating data directory and generating Kelly's instances...")
        os.makedirs('../../data')
        data_handler.generate_all_kellys_instances()

    # Use the smallest instance
    instance_name = "Kelly_n20_k3"

    # Create evaluator
    print("Creating evaluator...")
    evaluator = VRPEvaluator()

    # Create results directory if it doesn't exist
    if not os.path.exists('../../results'):
        os.makedirs('../../results')

    print(f"\n{'='*80}")
    print(f"Running small test for instance {instance_name}")
    print(f"{'='*80}")

    # Load instance
    print(f"Loading instance {instance_name}...")
    nodes, vehicle_capacity = data_handler.load_instance(instance_name)
    print(f"Loaded {len(nodes)} nodes with vehicle capacity {vehicle_capacity}")

    # Define a subset of methods to test
    methods = {
        "CW": lambda n, vc: solve_without_clustering(n, vc),
        "DBSCAN+CW": lambda n, vc: solve_with_clustering(n, vc, clustering_method='dbscan', verbose=True)[0],
        "DBSCAN-TabuSearch": lambda n, vc: solve_with_dbscan_tabu(n, vc, verbose=True)
    }

    # Run benchmark
    solutions = evaluator.run_benchmark(instance_name, nodes, vehicle_capacity, methods)

    # Visualize all solutions
    for method_name, solution in solutions.items():
        fig = evaluator.visualize_solution(solution, nodes, instance_name, method_name)
        plt.savefig(f"results/{instance_name}_{method_name.replace('+', '_')}.png")
        plt.close(fig)

    # Visualize best solution
    best_method = min(solutions.keys(), key=lambda k: solutions[k].cost)
    best_solution = solutions[best_method]

    print(f"\nBest method for {instance_name}: {best_method}")
    print(f"Best solution cost: {best_solution.cost:.2f}")

    fig = evaluator.visualize_solution(best_solution, nodes, instance_name, best_method)
    plt.savefig(f"results/BEST_{instance_name}_{best_method.replace('+', '_')}.png")
    plt.close(fig)

    # Save results
    evaluator.save_results("results/small_test_results.csv")

    # Print summary
    print("\nSummary of results:")
    comparison = evaluator.compare_methods()
    print(comparison)

def run_dbscan_focused_experiments(instance_names=None, verbose=True):
    """
    Run focused experiments on DBSCAN with different parameters and metaheuristics.
    This function is specifically designed to evaluate DBSCAN performance on larger instances.

    Args:
        instance_names: List of instance names to run experiments on
        verbose: Whether to print detailed information
    """
    # Create data handler
    print("Creating data handler...")
    data_handler = DataHandler()

    # Generate Kelly's instances if they don't exist
    if not os.path.exists('../../data'):
        print("Creating data directory and generating Kelly's instances...")
        os.makedirs('../../data')
        data_handler.generate_all_kellys_instances()

    # If no instance names provided, use medium to large instances
    if instance_names is None:
        # For testing, use just one medium instance to save time
        instance_names = [
            "Kelly_n50_k5",   # Medium instance
            # Uncomment for full experiments
            # "Kelly_n75_k8",   # Medium-large instance
            # "Kelly_n100_k10"  # Large instance
        ]

    # Create evaluator
    print("Creating evaluator...")
    evaluator = VRPEvaluator()

    # Create results directory if it doesn't exist
    if not os.path.exists('../../results'):
        os.makedirs('../../results')

    # Create a directory for DBSCAN-focused results
    dbscan_results_dir = 'results/dbscan_focused'
    if not os.path.exists(dbscan_results_dir):
        os.makedirs(dbscan_results_dir)

    # Run experiments for each instance
    for instance_name in instance_names:
        print(f"\n{'='*80}")
        print(f"Running DBSCAN-focused experiments for instance {instance_name}")
        print(f"{'='*80}")

        # Load instance
        print(f"Loading instance {instance_name}...")
        nodes, vehicle_capacity = data_handler.load_instance(instance_name)
        print(f"Loaded {len(nodes)} nodes with vehicle capacity {vehicle_capacity}")

        # Define DBSCAN-focused methods to compare
        methods = {
            # Baseline for comparison
            "CW": lambda n, vc: solve_without_clustering(n, vc),

            # DBSCAN with different parameters
            "DBSCAN-Auto+CW": lambda n, vc: solve_with_clustering(n, vc, clustering_method='dbscan', verbose=verbose)[0],

            # DBSCAN with metaheuristics - choose one for faster testing
            "DBSCAN-TabuSearch": lambda n, vc: solve_with_dbscan_tabu(n, vc, verbose=verbose)

            # Uncomment for full experiments
            # "DBSCAN+CW+2opt+3opt": lambda n, vc: solve_with_local_search(n, vc, use_clustering=True, clustering_method='dbscan', verbose=verbose),
            # "DBSCAN+CW+VNS": lambda n, vc: solve_with_vns(n, vc, use_clustering=True, clustering_method='dbscan', verbose=verbose),
        }

        # Run benchmark
        solutions = evaluator.run_benchmark(instance_name, nodes, vehicle_capacity, methods)

        # Create advanced visualizations
        advanced_visualizer = AdvancedVRPVisualizer()

        # 1. Visualize customer density
        density_fig = advanced_visualizer.visualize_cluster_density(
            nodes,
            title=f"Customer Density Heatmap - {instance_name}",
            save_path=f"{dbscan_results_dir}/{instance_name}_density_heatmap.png"
        )
        plt.close(density_fig)

        # 2. Run DBSCAN with different parameters for analysis
        print("\nAnalyzing DBSCAN parameters...")

        # Extract customer coordinates
        customer_coords = np.array([[node.x, node.y] for node in nodes[1:]])

        # Determine a range of eps values based on data
        distances = np.sqrt(np.sum((customer_coords[:, np.newaxis, :] - customer_coords[np.newaxis, :, :]) ** 2, axis=2))
        distances = distances[distances > 0]  # Remove self-distances

        # Use percentiles of distances for eps values
        eps_percentiles = [10, 25, 50, 75, 90]
        eps_values = [np.percentile(distances, p) for p in eps_percentiles]
        eps_values = [round(eps, 2) for eps in eps_values]

        # Define min_samples values
        min_samples_values = [3, 5, 10]

        # Visualize DBSCAN parameter analysis
        dbscan_params_fig, dbscan_heatmaps_fig = advanced_visualizer.visualize_dbscan_parameter_analysis(
            nodes,
            eps_values,
            min_samples_values,
            title=f"DBSCAN Parameter Analysis - {instance_name}",
            save_path=f"{dbscan_results_dir}/{instance_name}_dbscan_parameter_analysis.png"
        )
        plt.close(dbscan_params_fig)
        plt.close(dbscan_heatmaps_fig)

        # 3. Run DBSCAN with optimal parameters and visualize clusters
        print("\nRunning DBSCAN with optimal parameters...")

        # Use auto-tuned parameters from our enhanced DBSCAN implementation
        clusterer = VRPClusterer(nodes, vehicle_capacity=vehicle_capacity)
        solution, _ = solve_with_clustering(nodes, vehicle_capacity, clustering_method='dbscan', verbose=verbose)

        # Get clusters directly from the clusterer
        clusters = clusterer.dbscan_clustering(auto_tune=True, verbose=verbose)

        # Extract DBSCAN labels
        customer_labels = []
        for i, node in enumerate(nodes[1:]):  # Skip depot
            # Find which cluster this node belongs to
            for cluster_idx, cluster in enumerate(clusters):
                if node in cluster:
                    customer_labels.append(cluster_idx)
                    break
            else:
                # Node not found in any cluster (should not happen)
                customer_labels.append(-1)

        # Visualize clusters with convex hulls
        clusters_fig = advanced_visualizer.visualize_clusters_with_convex_hulls(
            nodes,
            customer_labels,
            title=f"DBSCAN Clusters with Convex Hulls - {instance_name}",
            save_path=f"{dbscan_results_dir}/{instance_name}_dbscan_clusters.png"
        )
        plt.close(clusters_fig)

        # 4. Create solution visualizations for each method
        for method_name, solution in solutions.items():
            # Basic solution visualization
            fig = evaluator.visualize_solution(solution, nodes, instance_name, method_name)
            plt.savefig(f"{dbscan_results_dir}/{instance_name}_{method_name.replace('+', '_')}.png")
            plt.close(fig)

        # 5. Create comparative visualization of all solutions
        comparison_fig = advanced_visualizer.visualize_solution_comparison(
            nodes,
            list(solutions.values()),
            list(solutions.keys()),
            title=f"Solution Comparison - {instance_name}",
            save_path=f"{dbscan_results_dir}/{instance_name}_solution_comparison.png"
        )
        plt.close(comparison_fig)

        # Visualize best solution
        best_method = min(solutions.keys(), key=lambda k: solutions[k].cost)
        best_solution = solutions[best_method]

        print(f"\nBest method for {instance_name}: {best_method}")
        print(f"Best solution cost: {best_solution.cost:.2f}")

        fig = evaluator.visualize_solution(best_solution, nodes, instance_name, best_method)
        plt.savefig(f"{dbscan_results_dir}/BEST_{instance_name}_{best_method.replace('+', '_')}.png")
        plt.close(fig)

    # Save results
    evaluator.save_results(f"{dbscan_results_dir}/dbscan_focused_results.csv")

    # Plot comparisons
    fig = evaluator.plot_comparison(metric='total_cost')
    plt.savefig(f"{dbscan_results_dir}/comparison_total_cost.png")
    plt.close(fig)

    fig = evaluator.plot_comparison(metric='runtime')
    plt.savefig(f"{dbscan_results_dir}/comparison_runtime.png")
    plt.close(fig)

    # Create detailed report
    print("\nGenerating detailed report...")
    df = evaluator.get_results_dataframe()

    # Calculate improvement over basic CW
    instance_methods = df.groupby(['instance_name', 'method_name']).agg({
        'total_cost': 'mean',
        'num_routes': 'mean',
        'runtime': 'mean'
    }).reset_index()

    # Create a pivot table for easier comparison
    pivot_cost = pd.pivot_table(
        instance_methods,
        values='total_cost',
        index='instance_name',
        columns='method_name'
    )

    # Calculate improvement over CW
    for method in pivot_cost.columns:
        if method != 'CW':
            pivot_cost[f'{method}_improvement'] = (pivot_cost['CW'] - pivot_cost[method]) / pivot_cost['CW'] * 100

    # Save detailed report
    pivot_cost.to_csv(f"{dbscan_results_dir}/dbscan_detailed_comparison.csv")

    # Print summary
    print("\nSummary of DBSCAN-focused results:")
    comparison = evaluator.compare_methods()
    print(comparison)

    # Print improvement over basic CW
    print("\nImprovement over basic CW (%):")
    for method in pivot_cost.columns:
        if 'improvement' in method:
            method_name = method.replace('_improvement', '')
            improvement = pivot_cost[method].mean()
            print(f"{method_name}: {improvement:.2f}%")

    # Create a summary table showing how DBSCAN performance scales with instance size
    print("\nDBSCAN performance scaling with instance size:")
    scaling_data = []
    for instance in instance_names:
        instance_size = int(instance.split('_n')[1].split('_')[0])
        for method in [m for m in comparison['method_name'] if 'DBSCAN' in m]:
            method_data = comparison[(comparison['instance_name'] == instance) &
                                    (comparison['method_name'] == method)]
            if not method_data.empty:
                scaling_data.append({
                    'instance': instance,
                    'size': instance_size,
                    'method': method,
                    'cost': method_data['total_cost'].values[0],
                    'runtime': method_data['runtime'].values[0]
                })

    scaling_df = pd.DataFrame(scaling_data)
    print(scaling_df)
    scaling_df.to_csv(f"{dbscan_results_dir}/dbscan_scaling.csv")

def run_augerat_experiments(verbose=True):
    """
    Run experiments on the Augerat instances.

    Args:
        verbose: Whether to print detailed information
    """
    # Create data handler
    print("Creating data handler...")
    data_handler = DataHandler()

    # Import Augerat instances if they don't exist in the data directory
    augerat_dir = '/Users/rodrigo/Downloads/A-VRP'

    # Check if any Augerat instances are already imported
    augerat_instances = [f.split('_')[0] for f in os.listdir('../../data') if f.endswith('_augerat_info.txt')]
    augerat_instances.sort() # Ensure consistent order

    if not augerat_instances:
        print("Importing Augerat instances...")
        augerat_instances = data_handler.import_all_augerat_instances(augerat_dir)
        augerat_instances.sort() # Ensure consistent order after import
    else:
        print(f"Found {len(augerat_instances)} Augerat instances already imported.")

    # Select every second Augerat instance for faster testing
    selected_instances = augerat_instances[::2]

    if not selected_instances:
        print("No Augerat instances found to select from. Please ensure Augerat data files (e.g., A-nXX-kY_augerat_info.txt) are in the 'data/' directory.")
        return # Exit if no instances

    # Create evaluator
    print("Creating evaluator...")
    evaluator = VRPEvaluator()

    # Create results directory if it doesn't exist
    if not os.path.exists('../../results'):
        os.makedirs('../../results')

    # Create a directory for Augerat results
    augerat_results_dir = '../../results/augerat'
    if not os.path.exists(augerat_results_dir):
        os.makedirs(augerat_results_dir)

    # Run experiments for each instance
    for instance_name in selected_instances:
        print(f"\n{'='*80}")
        print(f"Running experiments for Augerat instance {instance_name}")
        print(f"{'='*80}")

        # Load instance
        print(f"Loading instance {instance_name}...")
        nodes, vehicle_capacity = data_handler.load_instance(instance_name)
        print(f"Loaded {len(nodes)} nodes with vehicle capacity {vehicle_capacity}")

        # Define methods to compare
        methods = {
            # Basic methods
            "CW": lambda n, vc: solve_without_clustering(n, vc),
            # DBSCAN-based methods
            "DBSCAN+CW": lambda n, vc: solve_with_clustering(n, vc, clustering_method='dbscan', verbose=verbose)[0],
            # New hybrid method
            "KMeans-DBSCAN-Hybrid+CW": lambda n, vc: solve_with_clustering(n, vc, clustering_method='kmeans_dbscan_hybrid', verbose=verbose)[0],
        }

        # Run benchmark
        solutions = evaluator.run_benchmark(instance_name, nodes, vehicle_capacity, methods)

        # Create advanced visualizations
        advanced_visualizer = AdvancedVRPVisualizer()

        # 1. Visualize customer density
        density_fig = advanced_visualizer.visualize_cluster_density(
            nodes,
            title=f"Customer Density Heatmap - {instance_name}",
            save_path=f"{augerat_results_dir}/{instance_name}_density_heatmap.png"
        )
        plt.close(density_fig)

        # 2. Run DBSCAN with optimal parameters and visualize clusters
        print("\nRunning DBSCAN with optimal parameters...")

        # Use auto-tuned parameters from our enhanced DBSCAN implementation
        clusterer = VRPClusterer(nodes, vehicle_capacity=vehicle_capacity)
        solution, _ = solve_with_clustering(nodes, vehicle_capacity, clustering_method='dbscan', verbose=verbose)

        # Get clusters directly from the clusterer
        clusters = clusterer.dbscan_clustering(auto_tune=True, verbose=verbose)

        # Extract DBSCAN labels
        customer_labels = []
        for i, node in enumerate(nodes[1:]):  # Skip depot
            # Find which cluster this node belongs to
            for cluster_idx, cluster in enumerate(clusters):
                if node in cluster:
                    customer_labels.append(cluster_idx)
                    break
            else:
                # Node not found in any cluster (should not happen)
                customer_labels.append(-1)

        # Visualize clusters with convex hulls
        clusters_fig = advanced_visualizer.visualize_clusters_with_convex_hulls(
            nodes,
            customer_labels,
            title=f"DBSCAN Clusters with Convex Hulls - {instance_name}",
            save_path=f"{augerat_results_dir}/{instance_name}_dbscan_clusters.png"
        )
        plt.close(clusters_fig)

        # 3. Create solution visualizations for each method
        for method_name, solution in solutions.items():
            # Basic solution visualization
            fig = evaluator.visualize_solution(solution, nodes, instance_name, method_name)
            plt.savefig(f"{augerat_results_dir}/{instance_name}_{method_name.replace('+', '_')}.png")
            plt.close(fig)

        # 4. Create comparative visualization of all solutions
        comparison_fig = advanced_visualizer.visualize_solution_comparison(
            nodes,
            list(solutions.values()),
            list(solutions.keys()),
            title=f"Solution Comparison - {instance_name}",
            save_path=f"{augerat_results_dir}/{instance_name}_solution_comparison.png"
        )
        plt.close(comparison_fig)

        # Visualize best solution
        best_method = min(solutions.keys(), key=lambda k: solutions[k].cost)
        best_solution = solutions[best_method]

        print(f"\nBest method for {instance_name}: {best_method}")
        print(f"Best solution cost: {best_solution.cost:.2f}")

        fig = evaluator.visualize_solution(best_solution, nodes, instance_name, best_method)
        plt.savefig(f"{augerat_results_dir}/BEST_{instance_name}_{best_method.replace('+', '_')}.png")
        plt.close(fig)

    # Save results
    evaluator.save_results(f"{augerat_results_dir}/augerat_results.csv")

    # Plot comparisons
    fig = evaluator.plot_comparison(metric='total_cost')
    plt.savefig(f"{augerat_results_dir}/comparison_total_cost.png")
    plt.close(fig)

    fig = evaluator.plot_comparison(metric='runtime')
    plt.savefig(f"{augerat_results_dir}/comparison_runtime.png")
    plt.close(fig)

    # Create detailed report
    print("\nGenerating detailed report...")
    df = evaluator.get_results_dataframe()

    # Calculate improvement over basic CW
    instance_methods = df.groupby(['instance_name', 'method_name']).agg({
        'total_cost': 'mean',
        'num_routes': 'mean',
        'runtime': 'mean'
    }).reset_index()

    # Pivot table for easier comparison
    pivot_cost = pd.pivot_table(
        instance_methods,
        values='total_cost',
        index='instance_name',
        columns='method_name'
    )

    # Calculate improvement percentages
    for method in pivot_cost.columns:
        if method != 'CW':
            pivot_cost[f"{method}_improvement"] = (pivot_cost['CW'] - pivot_cost[method]) / pivot_cost['CW'] * 100

    # Save detailed report
    pivot_cost.to_csv(f"{augerat_results_dir}/augerat_detailed_comparison.csv")

    # Print summary
    print("\nSummary of Augerat results:")
    comparison = evaluator.compare_methods()
    print(comparison)

    # Print improvement over basic CW
    print("\nImprovement over basic CW (%):")
    for method in pivot_cost.columns:
        if 'improvement' in method:
            method_name = method.replace('_improvement', '')
            improvement = pivot_cost[method].mean()
            print(f"{method_name}: {improvement:.2f}%")

    # Plot std dev comparisons for each instance
    target_methods_for_std_dev = ["CW", "DBSCAN+CW", "KMeans-DBSCAN-Hybrid+CW"]
    print(f"\nGenerating standard deviation comparison plots for methods: {target_methods_for_std_dev}...")
    for instance_name_plot in selected_instances: # Use the same list of instances processed
        std_dev_fig = evaluator.plot_std_dev_comparison(
            instance_name=instance_name_plot,
            target_methods=target_methods_for_std_dev,
            save_path=f"{augerat_results_dir}/{instance_name_plot}_std_dev_comparison.png"
        )
        if std_dev_fig:
            plt.close(std_dev_fig)
        else:
            print(f"Could not generate std dev plot for {instance_name_plot}")
    
    # Plot overall metric trends across instances
    print(f"\nGenerating overall metric trend plots for methods: {target_methods_for_std_dev}...")
    trend_length_fig = evaluator.plot_metric_trend_across_instances(
        metric_name='length_std',
        target_methods=target_methods_for_std_dev,
        save_path=f"{augerat_results_dir}/overall_length_std_trend.png"
    )
    if trend_length_fig:
        plt.close(trend_length_fig)

    trend_demand_fig = evaluator.plot_metric_trend_across_instances(
        metric_name='demand_std',
        target_methods=target_methods_for_std_dev,
        save_path=f"{augerat_results_dir}/overall_demand_std_trend.png"
    )
    if trend_demand_fig:
        plt.close(trend_demand_fig)
            
    print("\nFinished Augerat experiments.")

if __name__ == "__main__":
    # Run a small test instead of full experiments
    # run_small_test()

    # Run DBSCAN-focused experiments on larger instances
    # run_dbscan_focused_experiments()

    # Run experiments with Augerat instances
    run_augerat_experiments()

    # Uncomment to run full experiments
    # run_experiments()
