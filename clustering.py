"""
Clustering algorithms for the Vehicle Routing Problem.
"""
import numpy as np
import math
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from vrp_objects import Node

class VRPClusterer:
    """
    Class for clustering customers in the VRP problem.
    """
    def __init__(self, nodes, num_vehicles=None, vehicle_capacity=None):
        """
        Initialize the clusterer.

        Args:
            nodes: List of Node objects
            num_vehicles: Number of vehicles (used for determining number of clusters)
            vehicle_capacity: Vehicle capacity (used for capacity-based clustering)
        """
        self.nodes = nodes
        self.depot = nodes[0]
        self.customers = nodes[1:]  # Exclude depot
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity

        # Extract customer coordinates and demands
        self.coordinates = np.array([[node.x, node.y] for node in self.customers])
        self.demands = np.array([node.demand for node in self.customers])

    def _determine_num_clusters(self):
        """
        Determine the optimal number of clusters.

        Returns:
            Optimal number of clusters
        """
        if self.num_vehicles is not None:
            # Use number of vehicles as a starting point
            return self.num_vehicles

        # If vehicle capacity is provided, estimate based on capacity
        if self.vehicle_capacity is not None:
            total_demand = sum(node.demand for node in self.customers)
            return max(2, math.ceil(total_demand / (self.vehicle_capacity * 0.8)))

        # Default: use silhouette method to find optimal number
        max_score = -1
        best_k = 2  # Minimum 2 clusters

        # Try different numbers of clusters
        for k in range(2, min(10, len(self.customers) // 5 + 1)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.coordinates)

            # Skip if any cluster is empty
            if len(np.unique(labels)) < k:
                continue

            # Calculate silhouette score
            score = silhouette_score(self.coordinates, labels)

            if score > max_score:
                max_score = score
                best_k = k

        return best_k

    def kmeans_clustering(self, num_clusters=None):
        """
        Cluster customers using K-means algorithm.

        Args:
            num_clusters: Number of clusters (if None, determined automatically)

        Returns:
            List of lists of Node objects, one list per cluster
        """
        if num_clusters is None:
            num_clusters = self._determine_num_clusters()

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.coordinates)

        # Group customers by cluster
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(self.customers[i])

        return clusters

    def dbscan_clustering(self, eps=None, min_samples=3, auto_tune=True, verbose=False,
                          use_grid_search=True, eps_values=None, min_samples_values=None):
        """
        Cluster customers using enhanced DBSCAN algorithm with auto-tuning, spatial constraints,
        and optional grid search for parameter optimization.

        Args:
            eps: Maximum distance between samples (if None, determined automatically or by grid search)
            min_samples: Minimum number of samples in a cluster (default 3)
            auto_tune: Whether to automatically tune DBSCAN parameters (if not using grid search)
            verbose: Whether to print detailed information
            use_grid_search: Whether to perform grid search for optimal eps and min_samples
            eps_values: List of eps values for grid search (if None, a default range is used)
            min_samples_values: List of min_samples values for grid search (if None, a default range is used)

        Returns:
            List of lists of Node objects, one list per cluster
        """
        if verbose:
            print("Starting DBSCAN clustering...")

        if len(self.coordinates) == 0:
            if verbose: print("No customer coordinates for DBSCAN.")
            return []

        best_eps = eps
        best_min_samples = min_samples

        if use_grid_search:
            if verbose:
                print("Performing grid search for DBSCAN parameters...")

            if eps_values is None:
                # Generate a default range for eps based on pairwise distances
                if len(self.coordinates) > 1:
                    pairwise_distances = cdist(self.coordinates, self.coordinates)
                    # Get unique non-zero distances
                    unique_distances = np.unique(pairwise_distances[pairwise_distances > 1e-5])
                    generated_eps_values = []
                    if len(unique_distances) > 5: # Ensure enough unique distances to form a range
                        generated_eps_values = np.percentile(unique_distances, [10, 25, 50, 75, 90])
                        generated_eps_values = [round(e, 2) for e in generated_eps_values if e > 0.01]
                        if not generated_eps_values and len(unique_distances) > 0: # Fallback if all percentiles are too small
                             generated_eps_values = [np.mean(unique_distances) * f for f in [0.5, 1.0, 1.5]]
                    elif len(unique_distances) > 0: # Fewer than 5 unique distances
                        generated_eps_values = [np.mean(unique_distances) * f for f in [0.5, 1.0, 1.5]]
                    else: # All points might be identical
                        generated_eps_values = [0.1, 0.5, 1.0]
                else: # Single point
                    generated_eps_values = [20.0] # Default for single point
                
                # Add user-requested fixed eps values
                fixed_eps_values = [3.0, 5.0, 7.0, 15.0]
                combined_eps_values = list(generated_eps_values) + fixed_eps_values
                
                eps_values = sorted(list(set(e for e in combined_eps_values if e > 0.01))) # Ensure positive, unique, and sorted
                if not eps_values: eps_values = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0] # Absolute fallback
                if verbose: print(f"  Generated eps_values for grid search: {eps_values}")

            if min_samples_values is None:
                # Add 2 to the list of candidates for min_samples
                base_min_samples = [2, 3, 5]
                # Add larger values based on dataset size, ensuring they are not too large
                if len(self.coordinates) > 10: # Only add larger min_samples if there are enough points
                    base_min_samples.append(max(7, len(self.coordinates) // 20))
                if len(self.coordinates) > 20:
                    base_min_samples.append(max(10, len(self.coordinates) // 10))
                
                # Filter out values that are too large (>= number of samples) or non-positive, then sort and unique
                min_samples_values = sorted(list(set(ms for ms in base_min_samples if ms > 0 and (ms < len(self.coordinates) if len(self.coordinates) > 1 else True) )))
                if not min_samples_values and len(self.coordinates) == 1: min_samples_values = [1] # For a single point
                elif not min_samples_values: min_samples_values = [2,3] # Fallback if all generated values were invalid

                if verbose: print(f"  Generated min_samples_values for grid search: {min_samples_values}")


            best_score = -1
            
            for current_eps in eps_values:
                for current_min_samples in min_samples_values:
                    if current_min_samples >= len(self.coordinates) and len(self.coordinates) > 1 : # min_samples cannot be >= n_samples
                        if verbose: print(f"  Skipping min_samples={current_min_samples} as it's >= n_samples ({len(self.coordinates)})")
                        continue
                    dbscan_gs = DBSCAN(eps=current_eps, min_samples=current_min_samples)
                    labels_gs = dbscan_gs.fit_predict(self.coordinates)
                    
                    num_clusters = len(set(labels_gs)) - (1 if -1 in labels_gs else 0)
                    
                    score = -1 # Default for invalid clustering
                    if num_clusters >= 2 and num_clusters < len(self.coordinates):
                        try:
                            score = silhouette_score(self.coordinates, labels_gs)
                        except ValueError: # Can happen if all points are noise or single cluster
                            score = -1
                    
                    if verbose:
                        print(f"  Grid Search: eps={current_eps:.2f}, min_samples={current_min_samples} -> "
                              f"Clusters={num_clusters}, Silhouette={score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_eps = current_eps
                        best_min_samples = current_min_samples
            
            if verbose:
                print(f"Grid search best parameters: eps={best_eps:.2f}, min_samples={best_min_samples} "
                      f"with Silhouette Score: {best_score:.3f}")
            eps = best_eps # Use the best found eps
            min_samples = best_min_samples # Use the best found min_samples

        elif eps is None or auto_tune: # Original auto-tuning logic if not using grid search
            if verbose:
                print("Auto-tuning DBSCAN parameters (grid search not used)...")
            # Estimate eps using k-distance graph method
            if len(self.coordinates) == 1:
                auto_eps = 20.0  # Default eps for a single customer
                if verbose: print(f"Single customer point. Using default auto_eps: {auto_eps}")
            else: # More than 1 customer
                distances = cdist(self.coordinates, self.coordinates)
                k_for_plot = min(min_samples, len(self.coordinates) - 1)
                k_for_plot = max(1, k_for_plot)

                k_distances = np.sort(distances, axis=1)[:, k_for_plot]
                sorted_k_distances = np.sort(k_distances)
                diffs = np.diff(sorted_k_distances)
                
                if len(diffs) > 0:
                    elbow_index = np.argmax(diffs) + 1
                    if elbow_index < len(sorted_k_distances):
                        auto_eps = sorted_k_distances[elbow_index]
                    else:
                        auto_eps = sorted_k_distances[-1]
                    auto_eps *= 1.15
                elif len(sorted_k_distances) > 0:
                    auto_eps = sorted_k_distances[-1] * 1.2
                else:
                    auto_eps = 20.0

                min_coord_std = np.std(self.coordinates)
                if auto_eps < 0.01 or (auto_eps < 0.1 and min_coord_std > 1e-4):
                    if min_coord_std > 1e-4:
                        all_pairwise_dist = distances[np.triu_indices_from(distances, k=1)]
                        positive_pairwise_dist = all_pairwise_dist[all_pairwise_dist > 1e-5]
                        if len(positive_pairwise_dist) > 0:
                            sensible_min_eps = np.percentile(positive_pairwise_dist, 5)
                            auto_eps = max(auto_eps, sensible_min_eps, 0.01)
                        else:
                            auto_eps = max(auto_eps, 0.01)
                    else:
                         auto_eps = max(auto_eps, 0.01)
                auto_eps = max(auto_eps, 0.01)
            
            if eps is not None and not auto_tune: # eps provided, auto_tune is False
                 pass # Use the provided eps
            elif eps is not None and auto_tune: # eps provided, auto_tune is True
                eps = max(eps, auto_eps)
            else: # eps is None
                eps = auto_eps

            # Adjust min_samples based on data density if auto_tuning and not set by grid search
            if auto_tune and not use_grid_search: # Only if auto_tune is on and grid search didn't set it
                if len(self.coordinates) > 1: # Need distances for this
                    distances_for_density = cdist(self.coordinates, self.coordinates)
                    avg_neighbors = np.mean((distances_for_density < eps).sum(axis=1)) - 1
                    min_samples = max(3, int(avg_neighbors * 0.5))
                else: # Single point, min_samples doesn't really matter but set to 1
                    min_samples = 1


            if verbose:
                print(f"Using parameters: eps={eps:.2f}, min_samples={min_samples}")
        
        # Ensure min_samples is not too large for the number of points
        if len(self.coordinates) > 1 and min_samples >= len(self.coordinates):
            min_samples = max(1, len(self.coordinates) -1)
            if verbose: print(f"Adjusted min_samples to {min_samples} as it was too large.")
        elif len(self.coordinates) == 1:
             min_samples = 1 # For a single point, min_samples must be 1

        # Apply DBSCAN clustering
        # We add depot coordinates as a feature weight to encourage clusters to form around the depot
        depot_coords = np.array([[self.depot.x, self.depot.y]])
        depot_distances = cdist(self.coordinates, depot_coords).flatten()

        # Normalize depot distances to [0, 1]
        if np.max(depot_distances) > 0:
            normalized_depot_distances = depot_distances / np.max(depot_distances)
        else:
            normalized_depot_distances = depot_distances

        # Create spatially-constrained features by adding depot distance as a feature
        # The weight parameter controls how much influence the depot has on clustering
        # weight = 0.3  # Adjust this to control depot influence # REMOVING DEPOT INFLUENCE FOR PURE PROXIMITY
        # spatial_features = np.column_stack((
        #     self.coordinates,
        #     normalized_depot_distances * weight
        # ))

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # labels = dbscan.fit_predict(spatial_features) # FIT ON ORIGINAL COORDINATES
        labels = dbscan.fit_predict(self.coordinates)

        # Calculate cluster statistics for reporting
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_noise = list(labels).count(-1)

        if verbose:
            print(f"DBSCAN found {num_clusters} clusters and {num_noise} noise points")

            # Calculate average distance to cluster center for each cluster
            if num_clusters > 0:
                cluster_stats = []
                for label in unique_labels:
                    if label == -1:
                        continue

                    # Get points in this cluster
                    cluster_points = self.coordinates[labels == label]
                    cluster_center = np.mean(cluster_points, axis=0)

                    # Calculate average distance to center
                    avg_distance = np.mean(np.sqrt(np.sum((cluster_points - cluster_center)**2, axis=1)))

                    # Calculate total demand in cluster
                    cluster_demand = sum(self.demands[labels == label])

                    cluster_stats.append({
                        'label': label,
                        'size': len(cluster_points),
                        'avg_distance': avg_distance,
                        'demand': cluster_demand
                    })

                print("\nCluster Statistics:")
                for stat in cluster_stats:
                    print(f"Cluster {stat['label']}: {stat['size']} points, "
                          f"Avg distance to center: {stat['avg_distance']:.2f}, "
                          f"Total demand: {stat['demand']:.2f}")

        # Group customers by cluster
        clusters = []

        # First add non-noise clusters
        for label in unique_labels:
            if label == -1:
                continue
            cluster = [self.customers[i] for i, l in enumerate(labels) if l == label]
            clusters.append(cluster)

        # Handle noise points based on their proximity to existing clusters or depot
        noise_indices = [i for i, l in enumerate(labels) if l == -1]

        if noise_indices:
            if verbose:
                print(f"\nAssigning {len(noise_indices)} noise points to nearest clusters or creating new clusters...")

            # If we have clusters, try to assign noise points to the nearest cluster
            if clusters:
                for i in noise_indices:
                    # Get the noise point
                    noise_point = self.coordinates[i]
                    noise_node = self.customers[i]

                    # Calculate distance to each cluster center
                    min_distance = float('inf')
                    nearest_cluster_idx = -1

                    for cluster_idx, cluster in enumerate(clusters):
                        # Calculate cluster center
                        cluster_coords = np.array([[node.x, node.y] for node in cluster])
                        cluster_center = np.mean(cluster_coords, axis=0)

                        # Calculate distance to cluster center
                        distance = np.sqrt(np.sum((noise_point - cluster_center)**2))

                        # Check if this is the nearest cluster so far
                        if distance < min_distance:
                            min_distance = distance
                            nearest_cluster_idx = cluster_idx

                    # Calculate distance to depot
                    depot_distance = np.sqrt(np.sum((noise_point - np.array([self.depot.x, self.depot.y]))**2))

                    # If the noise point is closer to a cluster than to the depot, add it to that cluster
                    # Otherwise, create a new cluster for it
                    if min_distance < depot_distance * 1.5:  # Use a threshold to decide
                        clusters[nearest_cluster_idx].append(noise_node)
                        if verbose:
                            print(f"  Assigned noise point {i} to cluster {nearest_cluster_idx}")
                    else:
                        clusters.append([noise_node])
                        if verbose:
                            print(f"  Created new cluster for noise point {i}")
            else:
                # If no clusters were found, create individual clusters for noise points
                for i in noise_indices:
                    clusters.append([self.customers[i]])

        # Balance clusters if needed
        if self.vehicle_capacity is not None:
            clusters = self._balance_dbscan_clusters(clusters, verbose)

        if verbose:
            print(f"\nFinal DBSCAN clustering: {len(clusters)} clusters")
            for i, cluster in enumerate(clusters):
                cluster_demand = sum(node.demand for node in cluster)
                print(f"  Cluster {i}: {len(cluster)} points, Total demand: {cluster_demand:.2f}")

        return clusters

    def _balance_dbscan_clusters(self, clusters, verbose=False):
        """
        Balance DBSCAN clusters to respect vehicle capacity constraints.

        Args:
            clusters: List of clusters (each a list of Node objects)
            verbose: Whether to print detailed information

        Returns:
            Balanced list of clusters
        """
        if verbose:
            print("\nBalancing DBSCAN clusters for capacity constraints...")

        # Check if any cluster exceeds vehicle capacity
        overloaded_clusters = []
        for i, cluster in enumerate(clusters):
            cluster_demand = sum(node.demand for node in cluster)
            if cluster_demand > self.vehicle_capacity:
                overloaded_clusters.append((i, cluster, cluster_demand))

        if not overloaded_clusters:
            if verbose:
                print("  No overloaded clusters found, no balancing needed")
            return clusters

        if verbose:
            print(f"  Found {len(overloaded_clusters)} overloaded clusters")

        # Process overloaded clusters
        balanced_clusters = [c for i, c in enumerate(clusters) if i not in [idx for idx, _, _ in overloaded_clusters]]

        for cluster_idx, cluster, cluster_demand in overloaded_clusters:
            if verbose:
                print(f"  Processing overloaded cluster {cluster_idx} with demand {cluster_demand:.2f}")

            # Sort nodes by distance from cluster center
            cluster_coords = np.array([[node.x, node.y] for node in cluster])
            cluster_center = np.mean(cluster_coords, axis=0)

            # Calculate distances from center
            distances = [np.sqrt(np.sum(((node.x, node.y) - cluster_center)**2)) for node in cluster]

            # Sort nodes by distance (farthest first, to split from the edges)
            sorted_nodes = [x for _, x in sorted(zip(distances, cluster), key=lambda pair: -pair[0])]

            # Create new clusters by splitting the overloaded cluster
            new_cluster = []
            current_demand = 0

            for node in sorted_nodes:
                if current_demand + node.demand <= self.vehicle_capacity:
                    new_cluster.append(node)
                    current_demand += node.demand
                else:
                    # This node would exceed capacity, start a new cluster
                    if new_cluster:  # Only add non-empty clusters
                        balanced_clusters.append(new_cluster)
                        if verbose:
                            print(f"    Created new cluster with {len(new_cluster)} nodes and demand {current_demand:.2f}")

                    new_cluster = [node]
                    current_demand = node.demand

            # Add the last cluster if not empty
            if new_cluster:
                balanced_clusters.append(new_cluster)
                if verbose:
                    print(f"    Created new cluster with {len(new_cluster)} nodes and demand {current_demand:.2f}")

        if verbose:
            print(f"  After balancing: {len(balanced_clusters)} clusters")

        return balanced_clusters

    def hierarchical_clustering(self, num_clusters=None):
        """
        Cluster customers using hierarchical clustering.

        Args:
            num_clusters: Number of clusters (if None, determined automatically)

        Returns:
            List of lists of Node objects, one list per cluster
        """
        if num_clusters is None:
            num_clusters = self._determine_num_clusters()

        # Apply hierarchical clustering
        hc = AgglomerativeClustering(n_clusters=num_clusters)
        labels = hc.fit_predict(self.coordinates)

        # Group customers by cluster
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(self.customers[i])

        return clusters

    def capacity_based_clustering(self):
        """
        Cluster customers based on vehicle capacity constraints.

        Returns:
            List of lists of Node objects, one list per cluster
        """
        if self.vehicle_capacity is None:
            raise ValueError("Vehicle capacity must be provided for capacity-based clustering")

        # Sort customers by distance from depot
        depot_coords = np.array([[self.depot.x, self.depot.y]])
        distances = cdist(self.coordinates, depot_coords).flatten()
        sorted_indices = np.argsort(distances)

        clusters = []
        current_cluster = []
        current_demand = 0

        for idx in sorted_indices:
            customer = self.customers[idx]

            # If adding this customer would exceed capacity, start a new cluster
            if current_demand + customer.demand > self.vehicle_capacity:
                if current_cluster:  # Only add non-empty clusters
                    clusters.append(current_cluster)
                current_cluster = [customer]
                current_demand = customer.demand
            else:
                current_cluster.append(customer)
                current_demand += customer.demand

        # Add the last cluster if not empty
        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def balanced_kmeans_clustering(self, num_clusters=None):
        """
        Cluster customers using K-means with balanced cluster sizes.

        Args:
            num_clusters: Number of clusters (if None, determined automatically)

        Returns:
            List of lists of Node objects, one list per cluster
        """
        if num_clusters is None:
            num_clusters = self._determine_num_clusters()

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.coordinates)

        # Calculate cluster centers
        centers = kmeans.cluster_centers_

        # Calculate total demand per cluster
        cluster_demands = [0] * num_clusters
        for i, label in enumerate(labels):
            cluster_demands[label] += self.demands[i]

        # Identify overloaded clusters
        avg_demand = sum(self.demands) / num_clusters
        overloaded = [i for i, demand in enumerate(cluster_demands) if demand > self.vehicle_capacity]

        # If no overloaded clusters, return regular K-means result
        if not overloaded and self.vehicle_capacity is not None:
            # Group customers by cluster
            clusters = [[] for _ in range(num_clusters)]
            for i, label in enumerate(labels):
                clusters[label].append(self.customers[i])
            return clusters

        # Balance clusters by moving points from overloaded to underloaded clusters
        # Calculate distances to all cluster centers
        all_distances = cdist(self.coordinates, centers)

        # Sort customers by their distance to their second-closest center
        second_closest = []
        for i, dists in enumerate(all_distances):
            label = labels[i]
            dists[label] = float('inf')  # Exclude current cluster
            second_closest.append((i, np.argmin(dists), np.min(dists)))

        # Sort by distance (ascending)
        second_closest.sort(key=lambda x: x[2])

        # Move customers from overloaded to underloaded clusters
        for i, new_label, _ in second_closest:
            old_label = labels[i]

            # Skip if old cluster is not overloaded
            if old_label not in overloaded:
                continue

            # Skip if new cluster would become overloaded
            if cluster_demands[new_label] + self.demands[i] > self.vehicle_capacity:
                continue

            # Move customer to new cluster
            cluster_demands[old_label] -= self.demands[i]
            cluster_demands[new_label] += self.demands[i]
            labels[i] = new_label

            # Check if old cluster is no longer overloaded
            if cluster_demands[old_label] <= self.vehicle_capacity:
                overloaded.remove(old_label)

            # If no more overloaded clusters, break
            if not overloaded:
                break

    def kmeans_dbscan_hybrid_clustering(self, num_clusters=None, eps=None, min_samples=None,
                                       auto_tune=True, verbose=False):
        """
        Apply a hybrid of K-means and DBSCAN clustering.

        This hybrid approach:
        1. Uses K-means for initial clustering to create well-balanced clusters
        2. Applies DBSCAN within each K-means cluster to refine boundaries and identify outliers
        3. Reassigns outliers to their nearest suitable cluster
        4. Balances the final clusters to respect vehicle capacity constraints

        Args:
            num_clusters: Number of initial K-means clusters (if None, use num_vehicles)
            eps: DBSCAN eps parameter (if None and auto_tune is False, use heuristic)
            min_samples: DBSCAN min_samples parameter (if None, use heuristic)
            auto_tune: Whether to auto-tune DBSCAN parameters
            verbose: Whether to print detailed information

        Returns:
            List of lists of Node objects, one list per cluster
        """
        if verbose:
            print("Starting K-means+DBSCAN hybrid clustering...")

        # Step 1: Initial K-means clustering
        if num_clusters is None:
            # Use slightly more clusters than vehicles to allow for refinement
            num_clusters = self.num_vehicles + 2 if self.num_vehicles else max(2, len(self.customers) // 15)

        if verbose:
            print(f"Performing initial K-means clustering with {num_clusters} clusters...")

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        customer_coords = np.array([[node.x, node.y] for node in self.customers])
        kmeans_labels = kmeans.fit_predict(customer_coords)

        # Create initial K-means clusters
        kmeans_clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(kmeans_labels):
            kmeans_clusters[label].append(self.customers[i])

        # Step 2: Apply DBSCAN within each K-means cluster
        refined_clusters = []
        outliers = []

        for cluster_idx, cluster in enumerate(kmeans_clusters):
            if len(cluster) <= 2:  # Too small for DBSCAN
                refined_clusters.append(cluster)
                continue

            # Extract coordinates for this cluster
            cluster_coords = np.array([[node.x, node.y] for node in cluster])

            # Auto-tune DBSCAN parameters for this specific cluster
            if auto_tune:
                # Calculate distances between points in this cluster
                distances = np.sqrt(np.sum((cluster_coords[:, np.newaxis, :] -
                                          cluster_coords[np.newaxis, :, :]) ** 2, axis=2))
                distances = distances[distances > 0]  # Remove self-distances

                if len(distances) > 0:
                    # Use percentile of distances for eps
                    local_eps = np.percentile(distances, 50)  # Median distance
                    local_min_samples = max(2, len(cluster) // 5)  # 20% of cluster size
                else:
                    # Fallback for very small clusters
                    local_eps = 20.0
                    local_min_samples = 2
            else:
                # Use provided parameters or defaults
                local_eps = eps if eps is not None else 20.0
                local_min_samples = min_samples if min_samples is not None else max(2, len(cluster) // 5)

            if verbose:
                print(f"  Applying DBSCAN to K-means cluster {cluster_idx} with "
                      f"eps={local_eps:.2f}, min_samples={local_min_samples}")

            # Apply DBSCAN to this cluster
            dbscan = DBSCAN(eps=local_eps, min_samples=local_min_samples)
            dbscan_labels = dbscan.fit_predict(cluster_coords)

            # Process DBSCAN results
            dbscan_clusters = {}
            for i, label in enumerate(dbscan_labels):
                if label == -1:  # Noise point
                    outliers.append(cluster[i])
                else:
                    if label not in dbscan_clusters:
                        dbscan_clusters[label] = []
                    dbscan_clusters[label].append(cluster[i])

            # Add refined clusters
            for label, nodes in dbscan_clusters.items():
                refined_clusters.append(nodes)

        if verbose:
            print(f"After DBSCAN refinement: {len(refined_clusters)} clusters and {len(outliers)} outliers")

        # Step 3: Reassign outliers to nearest cluster
        for outlier in outliers:
            best_cluster_idx = None
            best_distance = float('inf')

            for i, cluster in enumerate(refined_clusters):
                if not cluster:  # Skip empty clusters
                    continue

                # Calculate centroid of this cluster
                centroid = np.mean([[node.x, node.y] for node in cluster], axis=0)

                # Calculate distance from outlier to centroid
                distance = np.sqrt((outlier.x - centroid[0])**2 + (outlier.y - centroid[1])**2)

                if distance < best_distance:
                    best_distance = distance
                    best_cluster_idx = i

            # Assign outlier to best cluster
            if best_cluster_idx is not None:
                refined_clusters[best_cluster_idx].append(outlier)
                if verbose:
                    print(f"  Assigned outlier node {outlier.ID} to cluster {best_cluster_idx}")
            else:
                # Create a new cluster for this outlier if no suitable cluster found
                refined_clusters.append([outlier])
                if verbose:
                    print(f"  Created new cluster for outlier node {outlier.ID}")

        # Step 4: Balance clusters for vehicle capacity
        if self.vehicle_capacity:
            if verbose:
                print("Balancing clusters for capacity constraints...")

            # Check if any cluster exceeds vehicle capacity
            overloaded_clusters = []
            for i, cluster in enumerate(refined_clusters):
                cluster_demand = sum(node.demand for node in cluster)
                if cluster_demand > self.vehicle_capacity:
                    overloaded_clusters.append(i)

            if overloaded_clusters:
                if verbose:
                    print(f"  Found {len(overloaded_clusters)} overloaded clusters")

                # Process each overloaded cluster
                new_clusters = []
                for i, cluster in enumerate(refined_clusters):
                    if i in overloaded_clusters:
                        # Split this cluster
                        cluster_demand = sum(node.demand for node in cluster)
                        num_subclusters = math.ceil(cluster_demand / self.vehicle_capacity)

                        if verbose:
                            print(f"  Splitting cluster {i} with demand {cluster_demand:.2f} "
                                  f"into {num_subclusters} subclusters")

                        # Use K-means to split the cluster
                        if len(cluster) > num_subclusters:
                            subcluster_coords = np.array([[node.x, node.y] for node in cluster])
                            kmeans = KMeans(n_clusters=num_subclusters, random_state=42, n_init=10)
                            subcluster_labels = kmeans.fit_predict(subcluster_coords)

                            # Create subclusters
                            subclusters = [[] for _ in range(num_subclusters)]
                            for j, label in enumerate(subcluster_labels):
                                subclusters[label].append(cluster[j])

                            # Add subclusters
                            for subcluster in subclusters:
                                new_clusters.append(subcluster)
                        else:
                            # If we have fewer nodes than subclusters, create one node per cluster
                            for node in cluster:
                                new_clusters.append([node])
                    else:
                        # Keep this cluster as is
                        new_clusters.append(cluster)

                refined_clusters = new_clusters

        # Ensure we have at least as many clusters as vehicles
        if self.num_vehicles and len(refined_clusters) < self.num_vehicles:
            if verbose:
                print(f"Need to create more clusters to match {self.num_vehicles} vehicles")

            # Find largest clusters to split
            cluster_sizes = [(i, len(cluster)) for i, cluster in enumerate(refined_clusters)]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)

            # Split largest clusters until we have enough
            clusters_to_add = self.num_vehicles - len(refined_clusters)
            for i in range(min(clusters_to_add, len(cluster_sizes))):
                cluster_idx = cluster_sizes[i][0]
                cluster = refined_clusters[cluster_idx]

                if len(cluster) <= 1:
                    continue  # Can't split a cluster with only one node

                # Split this cluster in two
                if verbose:
                    print(f"  Splitting cluster {cluster_idx} with {len(cluster)} nodes")

                cluster_coords = np.array([[node.x, node.y] for node in cluster])
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                split_labels = kmeans.fit_predict(cluster_coords)

                # Create two new clusters
                cluster1 = [cluster[j] for j in range(len(cluster)) if split_labels[j] == 0]
                cluster2 = [cluster[j] for j in range(len(cluster)) if split_labels[j] == 1]

                # Replace original cluster with first new cluster
                refined_clusters[cluster_idx] = cluster1

                # Add second new cluster
                refined_clusters.append(cluster2)

        # Final check for empty clusters
        refined_clusters = [cluster for cluster in refined_clusters if cluster]

        if verbose:
            print(f"Final hybrid clustering: {len(refined_clusters)} clusters")
            for i, cluster in enumerate(refined_clusters):
                cluster_demand = sum(node.demand for node in cluster)
                print(f"  Cluster {i}: {len(cluster)} points, Total demand: {cluster_demand:.2f}")

        return refined_clusters
