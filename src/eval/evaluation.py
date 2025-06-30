"""
Evaluation metrics and visualization for VRP solutions.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from collections import defaultdict

class VRPEvaluator:
    """
    Class for evaluating and comparing VRP solutions.
    """
    def __init__(self):
        """Initialize the evaluator."""
        self.results = defaultdict(list)
    
    def evaluate_solution(self, solution, instance_name, method_name, runtime=None):
        """
        Evaluate a solution and store the results.
        
        Args:
            solution: Solution object
            instance_name: Name of the instance
            method_name: Name of the solution method
            runtime: Runtime in seconds (if None, not recorded)
        """
        # Calculate metrics
        total_cost = solution.cost
        num_routes = len(solution.routes)
        
        # Calculate route statistics
        route_lengths = [route.cost for route in solution.routes]
        route_demands = [route.demand for route in solution.routes]
        
        min_route_length = min(route_lengths) if route_lengths else 0
        max_route_length = max(route_lengths) if route_lengths else 0
        avg_route_length = sum(route_lengths) / num_routes if num_routes > 0 else 0
        
        min_route_demand = min(route_demands) if route_demands else 0
        max_route_demand = max(route_demands) if route_demands else 0
        avg_route_demand = sum(route_demands) / num_routes if num_routes > 0 else 0
        
        # Calculate route balance metrics
        length_std = np.std(route_lengths) if len(route_lengths) > 1 else 0
        demand_std = np.std(route_demands) if len(route_demands) > 1 else 0
        
        # Store results
        self.results['instance_name'].append(instance_name)
        self.results['method_name'].append(method_name)
        self.results['total_cost'].append(total_cost)
        self.results['num_routes'].append(num_routes)
        self.results['min_route_length'].append(min_route_length)
        self.results['max_route_length'].append(max_route_length)
        self.results['avg_route_length'].append(avg_route_length)
        self.results['min_route_demand'].append(min_route_demand)
        self.results['max_route_demand'].append(max_route_demand)
        self.results['avg_route_demand'].append(avg_route_demand)
        self.results['length_std'].append(length_std)
        self.results['demand_std'].append(demand_std)
        self.results['runtime'].append(runtime if runtime is not None else float('nan'))
    
    def get_results_dataframe(self):
        """
        Get the results as a pandas DataFrame.
        
        Returns:
            pandas DataFrame with results
        """
        return pd.DataFrame(self.results)
    
    def save_results(self, filename):
        """
        Save the results to a CSV file.
        
        Args:
            filename: Name of the file to save to
        """
        df = self.get_results_dataframe()
        df.to_csv(filename, index=False)
    
    def compare_methods(self, instance_name=None):
        """
        Compare different solution methods.
        
        Args:
            instance_name: Name of the instance to compare (if None, compare all)
            
        Returns:
            pandas DataFrame with comparison results
        """
        df = self.get_results_dataframe()
        
        if instance_name is not None:
            df = df[df['instance_name'] == instance_name]
        
        # Group by instance and method
        grouped = df.groupby(['instance_name', 'method_name']).agg({
            'total_cost': 'mean',
            'num_routes': 'mean',
            'avg_route_length': 'mean',
            'length_std': 'mean',
            'avg_route_demand': 'mean',
            'demand_std': 'mean',
            'runtime': 'mean'
        }).reset_index()
        
        return grouped
    
    def plot_comparison(self, metric='total_cost', instance_name=None, figsize=(10, 6)):
        """
        Plot a comparison of different methods.
        
        Args:
            metric: Metric to compare ('total_cost', 'num_routes', etc.)
            instance_name: Name of the instance to compare (if None, compare all)
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        df = self.get_results_dataframe()
        
        if instance_name is not None:
            df = df[df['instance_name'] == instance_name]
            title = f'Comparison of {metric} for instance {instance_name}'
        else:
            title = f'Comparison of {metric} across all instances'
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot data
        if instance_name is not None:
            # Bar chart for single instance
            methods = df['method_name'].unique()
            values = [df[df['method_name'] == method][metric].mean() for method in methods]
            
            ax.bar(methods, values)
            ax.set_xlabel('Method')
            ax.set_ylabel(metric)
            ax.set_title(title)
            
            # Add values on top of bars
            for i, v in enumerate(values):
                ax.text(i, v + 0.01 * max(values), f'{v:.2f}', ha='center')
        else:
            # Line chart for multiple instances
            instances = df['instance_name'].unique()
            methods = df['method_name'].unique()
            
            for method in methods:
                method_data = df[df['method_name'] == method]
                values = [method_data[method_data['instance_name'] == instance][metric].mean() 
                         for instance in instances]
                ax.plot(instances, values, marker='o', label=method)
            
            ax.set_xlabel('Instance')
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def visualize_solution(self, solution, nodes, instance_name, method_name, figsize=(10, 8)):
        """
        Visualize a solution.
        
        Args:
            solution: Solution object
            nodes: List of Node objects
            instance_name: Name of the instance
            method_name: Name of the solution method
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot depot
        depot = nodes[0]
        ax.scatter(depot.x, depot.y, c='red', s=100, marker='s', label='Depot')
        
        # Plot customers
        customers_x = [node.x for node in nodes[1:]]
        customers_y = [node.y for node in nodes[1:]]
        ax.scatter(customers_x, customers_y, c='blue', s=50, alpha=0.5, label='Customers')
        
        # Plot routes
        colors = plt.cm.tab10.colors
        for i, route in enumerate(solution.routes):
            color = colors[i % len(colors)]
            
            # Get nodes in the route
            route_nodes = []
            if route.edges:
                route_nodes.append(route.edges[0].origin)
                for edge in route.edges:
                    route_nodes.append(edge.end)
            
            # Plot route
            if route_nodes:
                x = [node.x for node in route_nodes]
                y = [node.y for node in route_nodes]
                ax.plot(x, y, c=color, linewidth=2, alpha=0.7, label=f'Route {i+1}')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Solution for {instance_name} using {method_name}\nTotal Cost: {solution.cost:.2f}')
        
        # Add legend (only show depot, customers, and a few routes)
        handles, labels = ax.get_legend_handles_labels()
        max_routes_in_legend = min(5, len(solution.routes))
        legend_items = handles[:2 + max_routes_in_legend]
        legend_labels = labels[:2 + max_routes_in_legend]
        if len(solution.routes) > max_routes_in_legend:
            legend_items.append(plt.Line2D([0], [0], color='gray', linewidth=2))
            legend_labels.append(f'+ {len(solution.routes) - max_routes_in_legend} more routes')
        
        ax.legend(legend_items, legend_labels, loc='best')
        
        plt.tight_layout()
        return fig
    
    def run_benchmark(self, instance_name, nodes, vehicle_capacity, methods_dict):
        """
        Run a benchmark comparing different methods on the same instance.
        
        Args:
            instance_name: Name of the instance
            nodes: List of Node objects
            vehicle_capacity: Vehicle capacity
            methods_dict: Dictionary mapping method names to functions that return a solution
            
        Returns:
            Dictionary mapping method names to solutions
        """
        solutions = {}
        
        for method_name, method_func in methods_dict.items():
            print(f"Running {method_name} on {instance_name}...")
            start_time = time.time()
            solution = method_func(nodes, vehicle_capacity)
            end_time = time.time()
            runtime = end_time - start_time
            
            self.evaluate_solution(solution, instance_name, method_name, runtime)
            solutions[method_name] = solution
            
            print(f"  Cost: {solution.cost:.2f}")
            print(f"  Routes: {len(solution.routes)}")
            print(f"  Runtime: {runtime:.2f} seconds")
        
        return solutions

    def plot_std_dev_comparison(self, instance_name, target_methods, figsize=(12, 7), save_path=None):
        """
        Plot a comparison of route length and demand standard deviations for specific methods for a single instance.

        Args:
            instance_name: Name of the instance to compare.
            target_methods: List of method names to include in the comparison.
            figsize: Figure size.
            save_path: Path to save the figure (if None, figure is not saved).

        Returns:
            matplotlib figure
        """
        df = self.get_results_dataframe()
        
        # Filter for the specific instance and methods
        df_filtered = df[(df['instance_name'] == instance_name) & (df['method_name'].isin(target_methods))]

        if df_filtered.empty:
            print(f"No data found for instance '{instance_name}' and methods '{target_methods}'. Skipping plot.")
            return None

        # Set up the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        n_methods = len(target_methods)
        bar_width = 0.35
        index = np.arange(n_methods)

        # Get std dev values
        length_stds = [df_filtered[df_filtered['method_name'] == method]['length_std'].mean() for method in target_methods]
        demand_stds = [df_filtered[df_filtered['method_name'] == method]['demand_std'].mean() for method in target_methods]

        bar1 = ax.bar(index - bar_width/2, length_stds, bar_width, label='Route Length Std Dev')
        bar2 = ax.bar(index + bar_width/2, demand_stds, bar_width, label='Route Demand Std Dev')

        ax.set_xlabel('Method', fontsize=14)
        ax.set_ylabel('Standard Deviation', fontsize=14)
        ax.set_title(f'Route Length & Demand Std. Dev. Comparison\nInstance: {instance_name}', fontsize=16)
        ax.set_xticks(index)
        ax.set_xticklabels(target_methods, rotation=45, ha="right")
        ax.legend(fontsize=12)

        # Add values on top of bars
        for bar in bar1 + bar2:
            yval = bar.get_height()
            if yval > 0: # Only add text if value is positive
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(max(length_stds, default=0), max(demand_stds, default=0)),
                        f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
            elif yval == 0: # Add text for zero values as well
                 ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(max(length_stds, default=1), max(demand_stds, default=1)), # adjust offset if max is 0
                        f'{yval:.2f}', ha='center', va='bottom', fontsize=10)


        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved std dev comparison plot to {save_path}")

        return fig

    def plot_metric_trend_across_instances(self, metric_name, target_methods, figsize=(15, 8), save_path=None):
        """
        Plot a trend of a specific metric across all processed instances for target methods.

        Args:
            metric_name: The name of the metric to plot (e.g., 'length_std', 'demand_std').
            target_methods: List of method names to include in the comparison.
            figsize: Figure size.
            save_path: Path to save the figure (if None, figure is not saved).

        Returns:
            matplotlib figure
        """
        df = self.get_results_dataframe()

        # Filter for the target methods
        df_filtered = df[df['method_name'].isin(target_methods)]

        if df_filtered.empty:
            print(f"No data found for methods '{target_methods}'. Skipping trend plot for {metric_name}.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique instances, try to sort them naturally if they have numbers like A-n32-k5
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
            
        instance_names = sorted(df_filtered['instance_name'].unique(), key=natural_sort_key)

        for method in target_methods:
            method_data = df_filtered[df_filtered['method_name'] == method]
            
            metric_means_for_method = []
            valid_instances_for_method = []
            for inst_name in instance_names:
                instance_data_for_method = method_data[method_data['instance_name'] == inst_name]
                if not instance_data_for_method.empty:
                    metric_means_for_method.append(instance_data_for_method[metric_name].mean())
                    valid_instances_for_method.append(inst_name)
            
            if valid_instances_for_method:
                 ax.plot(valid_instances_for_method, metric_means_for_method, marker='o', linestyle='-', label=method)

        ax.set_xlabel('Instance Name', fontsize=14)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=14)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Trend Across Instances', fontsize=16)
        ax.legend(fontsize=12)
        plt.xticks(rotation=70, ha="right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved {metric_name} trend plot to {save_path}")

        return fig
