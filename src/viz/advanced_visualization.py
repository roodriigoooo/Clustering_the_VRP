"""
Advanced visualization tools for VRP solutions with a focus on DBSCAN clustering.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

class AdvancedVRPVisualizer:
    """
    Advanced visualization tools for VRP solutions with a focus on DBSCAN clustering.
    """
    def __init__(self, figsize=(12, 10), dpi=100):
        """
        Initialize the visualizer.

        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi

    def visualize_clusters_with_convex_hulls(self, nodes, labels, title="DBSCAN Clusters with Convex Hulls",
                                           save_path=None, show=False):
        """
        Visualize clusters with convex hulls around each cluster.

        Args:
            nodes: List of Node objects
            labels: Cluster labels for each node (excluding depot)
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            show: Whether to show the figure

        Returns:
            matplotlib figure
        """
        # Extract coordinates
        coords = np.array([[node.x, node.y] for node in nodes])
        depot_coords = coords[0]
        customer_coords = coords[1:]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot depot
        ax.scatter(depot_coords[0], depot_coords[1], c='black', s=200, marker='s', label='Depot')

        # Get unique labels
        unique_labels = set(labels)
        labels_np = np.array(labels) # Convert to NumPy array for boolean masking

        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        # Plot clusters with convex hulls
        for i, label_value in enumerate(unique_labels): # Use label_value to avoid conflict with labels argument
            if label_value == -1:
                # Noise points (black)
                noise_mask = labels_np == -1 # Use NumPy array for masking
                noise_points = customer_coords[noise_mask]
                ax.scatter(noise_points[:, 0], noise_points[:, 1], c='black', s=50, alpha=0.5, label='Noise')
            else:
                # Cluster points
                cluster_mask = labels_np == label_value # Use NumPy array for masking
                cluster_points = customer_coords[cluster_mask]
                color = colors[i % len(colors)]

                # Plot cluster points
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], s=80,
                          alpha=0.7, label=f'Cluster {label_value}')

                # Create convex hull if cluster has at least 3 points
                if len(cluster_points) >= 3:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    hull_polygon = Polygon(hull_points, alpha=0.2, color=color)
                    ax.add_patch(hull_polygon)

                    # Add cluster center
                    center = np.mean(cluster_points, axis=0)
                    ax.scatter(center[0], center[1], c=[color], s=200, marker='*',
                              edgecolor='black', linewidth=1.5)

        # Set labels and title
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_title(title, fontsize=16)

        # Add legend
        ax.legend(loc='best', fontsize=12)

        # Set equal aspect ratio
        ax.set_aspect('equal')

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)

        # Show figure if requested
        if show:
            plt.show()

        return fig

    def visualize_cluster_density(self, nodes, title="Customer Density Heatmap",
                                save_path=None, show=False, sigma=5):
        """
        Visualize customer density using a heatmap.

        Args:
            nodes: List of Node objects
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            show: Whether to show the figure
            sigma: Standard deviation for Gaussian kernel

        Returns:
            matplotlib figure
        """
        # Extract coordinates
        coords = np.array([[node.x, node.y] for node in nodes])
        depot_coords = coords[0]
        customer_coords = coords[1:]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create grid for heatmap
        x_min, x_max = np.min(coords[:, 0]) - 5, np.max(coords[:, 0]) + 5
        y_min, y_max = np.min(coords[:, 1]) - 5, np.max(coords[:, 1]) + 5

        grid_size = 100
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Create density map
        density = np.zeros((grid_size, grid_size))

        for x, y in customer_coords:
            # Find closest grid point
            i = int((x - x_min) / (x_max - x_min) * (grid_size - 1))
            j = int((y - y_min) / (y_max - y_min) * (grid_size - 1))

            # Ensure indices are within bounds
            i = max(0, min(i, grid_size - 1))
            j = max(0, min(j, grid_size - 1))

            density[j, i] += 1

        # Apply Gaussian filter for smoothing
        density = gaussian_filter(density, sigma=sigma)

        # Plot heatmap
        im = ax.imshow(density, extent=[x_min, x_max, y_min, y_max], origin='lower',
                      cmap='viridis', alpha=0.7)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Customer Density', fontsize=12)

        # Plot depot
        ax.scatter(depot_coords[0], depot_coords[1], c='red', s=200, marker='s',
                  edgecolor='black', linewidth=1.5, label='Depot')

        # Plot customers
        ax.scatter(customer_coords[:, 0], customer_coords[:, 1], c='white', s=50,
                  edgecolor='black', linewidth=1, alpha=0.7, label='Customers')

        # Set labels and title
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_title(title, fontsize=16)

        # Add legend
        ax.legend(loc='best', fontsize=12)

        # Set equal aspect ratio
        ax.set_aspect('equal')

        plt.tight_layout()

        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)

        # Show figure if requested
        if show:
            plt.show()

        return fig

    def visualize_solution_comparison(self, nodes, solutions, method_names,
                                    title="Solution Comparison", save_path=None, show=False):
        """
        Visualize multiple solutions side by side for comparison.

        Args:
            nodes: List of Node objects
            solutions: List of Solution objects
            method_names: List of method names corresponding to solutions
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            show: Whether to show the figure

        Returns:
            matplotlib figure
        """
        # Determine grid size
        n_solutions = len(solutions)
        n_cols = min(3, n_solutions)
        n_rows = (n_solutions + n_cols - 1) // n_cols

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0] * n_cols, self.figsize[1] * n_rows),
                               dpi=self.dpi)

        # Flatten axes if there's more than one subplot
        if n_solutions > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Extract coordinates
        coords = np.array([[node.x, node.y] for node in nodes])
        depot_coords = coords[0]
        customer_coords = coords[1:]

        # Plot each solution
        for i, (solution, method_name, ax) in enumerate(zip(solutions, method_names, axes)):
            # Plot depot
            ax.scatter(depot_coords[0], depot_coords[1], c='black', s=100, marker='s', label='Depot')

            # Plot customers
            ax.scatter(customer_coords[:, 0], customer_coords[:, 1], c='gray', s=30, alpha=0.3)

            # Plot routes with different colors
            colors = plt.cm.tab10(np.linspace(0, 1, len(solution.routes)))

            for j, route in enumerate(solution.routes):
                color = colors[j % len(colors)]

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
                    ax.plot(x, y, c=color, linewidth=2, alpha=0.7)

                    # Highlight route nodes
                    # Find indices by matching node IDs instead of object identity
                    route_customer_indices = []
                    for node in route_nodes:
                        if node.ID != 0:  # Skip depot
                            # Find the index of the node with the same ID in the nodes list
                            for i, n in enumerate(nodes):
                                if n.ID == node.ID:
                                    route_customer_indices.append(i)
                                    break

                    if route_customer_indices:
                        route_customer_coords = coords[route_customer_indices]
                        ax.scatter(route_customer_coords[:, 0], route_customer_coords[:, 1],
                                 c=[color], s=50, edgecolor='black', linewidth=1)

            # Set title and labels
            ax.set_title(f"{method_name}\nCost: {solution.cost:.2f}, Routes: {len(solution.routes)}",
                        fontsize=12)
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)

            # Set equal aspect ratio
            ax.set_aspect('equal')

            # Add grid
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n_solutions, len(axes)):
            axes[i].axis('off')

        # Set overall title
        fig.suptitle(title, fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle

        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)

        # Show figure if requested
        if show:
            plt.show()

        return fig

    def visualize_dbscan_parameter_analysis(self, nodes, eps_values, min_samples_values,
                                          title="DBSCAN Parameter Analysis",
                                          save_path=None, show=False):
        """
        Visualize the effect of different DBSCAN parameters on clustering.

        Args:
            nodes: List of Node objects
            eps_values: List of eps values to test
            min_samples_values: List of min_samples values to test
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            show: Whether to show the figure

        Returns:
            matplotlib figure
        """
        # Extract coordinates (excluding depot)
        coords = np.array([[node.x, node.y] for node in nodes[1:]])

        # Determine grid size
        n_eps = len(eps_values)
        n_min_samples = len(min_samples_values)

        # Create figure
        fig, axes = plt.subplots(n_min_samples, n_eps,
                               figsize=(self.figsize[0] * n_eps / 2, self.figsize[1] * n_min_samples / 2),
                               dpi=self.dpi)

        # Ensure axes is a 2D array
        if n_min_samples == 1 and n_eps == 1:
            axes = np.array([[axes]])
        elif n_min_samples == 1:
            axes = np.array([axes])
        elif n_eps == 1:
            axes = np.array([[ax] for ax in axes])

        # Store metrics for heatmap
        metrics = np.zeros((n_min_samples, n_eps, 3))  # [n_clusters, n_noise, silhouette]

        # Run DBSCAN with different parameters
        for i, min_samples in enumerate(min_samples_values):
            for j, eps in enumerate(eps_values):
                # Run DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(coords)

                # Get unique labels
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = list(labels).count(-1)

                # Calculate silhouette score if there are at least 2 clusters and no noise points
                silhouette = 0
                if n_clusters >= 2 and n_noise < len(coords):
                    try:
                        silhouette = silhouette_score(coords, labels)
                    except:
                        silhouette = 0

                # Store metrics
                metrics[i, j, 0] = n_clusters
                metrics[i, j, 1] = n_noise
                metrics[i, j, 2] = silhouette

                # Plot clusters
                ax = axes[i, j]

                # Create colormap
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

                # Plot clusters
                for k, label in enumerate(unique_labels):
                    if label == -1:
                        # Noise points (black)
                        noise_mask = labels == -1
                        noise_points = coords[noise_mask]
                        ax.scatter(noise_points[:, 0], noise_points[:, 1], c='black', s=10, alpha=0.5)
                    else:
                        # Cluster points
                        cluster_mask = labels == label
                        cluster_points = coords[cluster_mask]
                        color = colors[k % len(colors)]

                        # Plot cluster points
                        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], s=20, alpha=0.7)

                # Set title with parameters and metrics
                ax.set_title(f"eps={eps}, min_samples={min_samples}\n"
                           f"Clusters: {n_clusters}, Noise: {n_noise}\n"
                           f"Silhouette: {silhouette:.2f}", fontsize=8)

                # Remove axis ticks for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])

                # Set equal aspect ratio
                ax.set_aspect('equal')

        # Set overall title
        fig.suptitle(title, fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle

        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)

        # Show figure if requested
        if show:
            plt.show()

        # Create heatmaps for metrics
        fig_heatmaps, axes_heatmaps = plt.subplots(1, 3, figsize=(self.figsize[0] * 3, self.figsize[1]),
                                                 dpi=self.dpi)

        # Heatmap for number of clusters
        sns.heatmap(metrics[:, :, 0], annot=True, fmt=".0f", cmap="viridis",
                   xticklabels=eps_values, yticklabels=min_samples_values, ax=axes_heatmaps[0])
        axes_heatmaps[0].set_title("Number of Clusters")
        axes_heatmaps[0].set_xlabel("eps")
        axes_heatmaps[0].set_ylabel("min_samples")

        # Heatmap for number of noise points
        sns.heatmap(metrics[:, :, 1], annot=True, fmt=".0f", cmap="viridis",
                   xticklabels=eps_values, yticklabels=min_samples_values, ax=axes_heatmaps[1])
        axes_heatmaps[1].set_title("Number of Noise Points")
        axes_heatmaps[1].set_xlabel("eps")
        axes_heatmaps[1].set_ylabel("min_samples")

        # Heatmap for silhouette score
        sns.heatmap(metrics[:, :, 2], annot=True, fmt=".2f", cmap="viridis",
                   xticklabels=eps_values, yticklabels=min_samples_values, ax=axes_heatmaps[2])
        axes_heatmaps[2].set_title("Silhouette Score")
        axes_heatmaps[2].set_xlabel("eps")
        axes_heatmaps[2].set_ylabel("min_samples")

        # Set overall title for heatmaps
        fig_heatmaps.suptitle(f"{title} - Metrics Heatmaps", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle

        # Save heatmaps if path is provided
        if save_path:
            heatmap_path = save_path.replace('.png', '_heatmaps.png')
            plt.savefig(heatmap_path)

        return fig, fig_heatmaps
