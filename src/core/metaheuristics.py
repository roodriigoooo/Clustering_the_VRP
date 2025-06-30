"""
Metaheuristics to complement and improve DBSCAN clustering for VRP.

This module contains advanced metaheuristics that work particularly well
with DBSCAN clustering for solving the Vehicle Routing Problem:

1. Variable Neighborhood Search (VNS): A metaheuristic that systematically
   explores different neighborhood structures to escape local optima.

2. Guided Local Search (GLS): A metaheuristic that uses penalties to guide
   the search away from local optima.

3. DBSCAN-Tabu Search: A specialized tabu search that leverages DBSCAN
   clustering information to guide the search.
"""
import math
import copy
import random
from collections import defaultdict
from vrp_objects import Edge, Route

class VariableNeighborhoodSearch:
    """
    Variable Neighborhood Search (VNS) metaheuristic for improving VRP solutions.
    Particularly effective when combined with DBSCAN clustering.
    """
    def __init__(self, solution, nodes, vehicle_capacity, max_iterations=15, max_no_improve=3, verbose=False):
        """
        Initialize the VNS.

        Args:
            solution: Initial Solution object
            nodes: List of Node objects
            vehicle_capacity: Vehicle capacity
            max_iterations: Maximum number of iterations
            max_no_improve: Maximum number of iterations without improvement
            verbose: Whether to print detailed information
        """
        self.solution = copy.deepcopy(solution)
        self.best_solution = copy.deepcopy(solution)
        self.nodes = nodes
        self.depot = nodes[0]
        self.vehicle_capacity = vehicle_capacity
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.verbose = verbose

        # Define neighborhood structures
        self.neighborhoods = [
            self._swap_within_route,
            self._swap_between_routes,
            self._relocate_within_route,
            self._relocate_between_routes,
            self._two_opt_move,
            self._three_opt_move
        ]

    def _calculate_distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes."""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def _get_route_nodes(self, route):
        """Get list of nodes in a route."""
        if not route.edges:
            return []

        nodes = [route.edges[0].origin]
        for edge in route.edges:
            nodes.append(edge.end)

        return nodes

    def _rebuild_route(self, nodes):
        """Rebuild a route from a list of nodes."""
        route = Route()

        for i in range(len(nodes) - 1):
            node1 = nodes[i]
            node2 = nodes[i + 1]

            edge = Edge(node1, node2)
            edge.cost = self._calculate_distance(node1, node2)

            route.add_edge(edge)

            # Update node properties
            node1.isInterior = (i > 0)
            node1.inRoute = route

        # Update last node
        nodes[-1].isInterior = False
        nodes[-1].inRoute = route

        return route

    def _swap_within_route(self, solution):
        """Swap two non-adjacent nodes within a route."""
        new_solution = copy.deepcopy(solution)

        # Randomly select a route
        if not new_solution.routes:
            return new_solution

        route_idx = random.randint(0, len(new_solution.routes) - 1)
        route = new_solution.routes[route_idx]

        # Get nodes in the route
        route_nodes = self._get_route_nodes(route)

        # Need at least 4 nodes (depot, 2 customers, depot) for a swap
        if len(route_nodes) < 4:
            return new_solution

        # Select two non-adjacent customer positions to swap (excluding depot)
        positions = list(range(1, len(route_nodes) - 1))
        if len(positions) < 2:
            return new_solution

        pos1 = random.choice(positions)
        positions.remove(pos1)

        # Ensure non-adjacent
        adjacent_positions = [pos1 - 1, pos1 + 1]
        valid_positions = [p for p in positions if p not in adjacent_positions]

        if not valid_positions:
            return new_solution

        pos2 = random.choice(valid_positions)

        # Swap nodes
        route_nodes[pos1], route_nodes[pos2] = route_nodes[pos2], route_nodes[pos1]

        # Rebuild route
        new_route = self._rebuild_route(route_nodes)
        new_solution.routes[route_idx] = new_route

        # Update solution cost
        new_solution.cost = sum(r.cost for r in new_solution.routes)

        return new_solution

    def _swap_between_routes(self, solution):
        """Swap a node from one route with a node from another route."""
        new_solution = copy.deepcopy(solution)

        # Need at least 2 routes
        if len(new_solution.routes) < 2:
            return new_solution

        # Randomly select two different routes
        route_indices = list(range(len(new_solution.routes)))
        route1_idx = random.choice(route_indices)
        route_indices.remove(route1_idx)
        route2_idx = random.choice(route_indices)

        route1 = new_solution.routes[route1_idx]
        route2 = new_solution.routes[route2_idx]

        # Get nodes in the routes
        route1_nodes = self._get_route_nodes(route1)
        route2_nodes = self._get_route_nodes(route2)

        # Need at least 3 nodes in each route (depot, customer, depot)
        if len(route1_nodes) < 3 or len(route2_nodes) < 3:
            return new_solution

        # Select a customer from each route (excluding depot)
        pos1 = random.randint(1, len(route1_nodes) - 2)
        pos2 = random.randint(1, len(route2_nodes) - 2)

        # Check capacity constraints
        node1 = route1_nodes[pos1]
        node2 = route2_nodes[pos2]

        new_route1_demand = route1.demand - node1.demand + node2.demand
        new_route2_demand = route2.demand - node2.demand + node1.demand

        if new_route1_demand > self.vehicle_capacity or new_route2_demand > self.vehicle_capacity:
            return new_solution

        # Swap nodes
        route1_nodes[pos1], route2_nodes[pos2] = route2_nodes[pos2], route1_nodes[pos1]

        # Rebuild routes
        new_route1 = self._rebuild_route(route1_nodes)
        new_route2 = self._rebuild_route(route2_nodes)

        new_solution.routes[route1_idx] = new_route1
        new_solution.routes[route2_idx] = new_route2

        # Update solution cost
        new_solution.cost = sum(r.cost for r in new_solution.routes)

        return new_solution

    def _relocate_within_route(self, solution):
        """Move a node to a different position within the same route."""
        new_solution = copy.deepcopy(solution)

        # Randomly select a route
        if not new_solution.routes:
            return new_solution

        route_idx = random.randint(0, len(new_solution.routes) - 1)
        route = new_solution.routes[route_idx]

        # Get nodes in the route
        route_nodes = self._get_route_nodes(route)

        # Need at least 4 nodes for relocation to be meaningful
        if len(route_nodes) < 4:
            return new_solution

        # Select a customer to relocate (excluding depot)
        pos1 = random.randint(1, len(route_nodes) - 2)

        # Select a different position to move to (excluding depot)
        positions = list(range(1, len(route_nodes) - 1))
        positions.remove(pos1)

        if not positions:
            return new_solution

        pos2 = random.choice(positions)

        # Relocate node
        node = route_nodes.pop(pos1)
        route_nodes.insert(pos2, node)

        # Rebuild route
        new_route = self._rebuild_route(route_nodes)
        new_solution.routes[route_idx] = new_route

        # Update solution cost
        new_solution.cost = sum(r.cost for r in new_solution.routes)

        return new_solution

    def _relocate_between_routes(self, solution):
        """Move a node from one route to another route."""
        new_solution = copy.deepcopy(solution)

        # Need at least 2 routes
        if len(new_solution.routes) < 2:
            return new_solution

        # Randomly select two different routes
        route_indices = list(range(len(new_solution.routes)))
        route1_idx = random.choice(route_indices)
        route_indices.remove(route1_idx)
        route2_idx = random.choice(route_indices)

        route1 = new_solution.routes[route1_idx]
        route2 = new_solution.routes[route2_idx]

        # Get nodes in the routes
        route1_nodes = self._get_route_nodes(route1)
        route2_nodes = self._get_route_nodes(route2)

        # Need at least 3 nodes in source route
        if len(route1_nodes) < 3:
            return new_solution

        # Select a customer from route1 (excluding depot)
        pos1 = random.randint(1, len(route1_nodes) - 2)
        node = route1_nodes[pos1]

        # Check capacity constraint for route2
        if route2.demand + node.demand > self.vehicle_capacity:
            return new_solution

        # Select position in route2 (excluding depot)
        pos2 = random.randint(1, len(route2_nodes) - 1)

        # Relocate node
        route1_nodes.pop(pos1)
        route2_nodes.insert(pos2, node)

        # Rebuild routes
        new_route1 = self._rebuild_route(route1_nodes)
        new_route2 = self._rebuild_route(route2_nodes)

        new_solution.routes[route1_idx] = new_route1
        new_solution.routes[route2_idx] = new_route2

        # Update solution cost
        new_solution.cost = sum(r.cost for r in new_solution.routes)

        return new_solution

    def _two_opt_move(self, solution):
        """Apply a random 2-opt move to a random route."""
        new_solution = copy.deepcopy(solution)

        # Randomly select a route
        if not new_solution.routes:
            return new_solution

        route_idx = random.randint(0, len(new_solution.routes) - 1)
        route = new_solution.routes[route_idx]

        # Get nodes in the route
        route_nodes = self._get_route_nodes(route)

        # Need at least 4 nodes for 2-opt
        if len(route_nodes) < 4:
            return new_solution

        # Select two positions for 2-opt (excluding depot)
        i = random.randint(1, len(route_nodes) - 3)
        j = random.randint(i + 1, len(route_nodes) - 2)

        # Apply 2-opt: reverse the segment between i and j
        route_nodes[i:j+1] = reversed(route_nodes[i:j+1])

        # Rebuild route
        new_route = self._rebuild_route(route_nodes)
        new_solution.routes[route_idx] = new_route

        # Update solution cost
        new_solution.cost = sum(r.cost for r in new_solution.routes)

        return new_solution

    def _three_opt_move(self, solution):
        """Apply a random 3-opt move to a random route."""
        new_solution = copy.deepcopy(solution)

        # Randomly select a route
        if not new_solution.routes:
            return new_solution

        route_idx = random.randint(0, len(new_solution.routes) - 1)
        route = new_solution.routes[route_idx]

        # Get nodes in the route
        route_nodes = self._get_route_nodes(route)

        # Need at least 6 nodes for 3-opt
        if len(route_nodes) < 6:
            return new_solution

        # Select three positions for 3-opt (excluding depot)
        i = random.randint(1, len(route_nodes) - 5)
        j = random.randint(i + 1, len(route_nodes) - 3)
        k = random.randint(j + 1, len(route_nodes) - 2)

        # Apply a random 3-opt move
        # There are 7 possible ways to reconnect the segments
        case = random.randint(0, 6)

        if case == 0:
            # Reverse segment [i, j]
            route_nodes[i:j+1] = reversed(route_nodes[i:j+1])
        elif case == 1:
            # Reverse segment [j+1, k]
            route_nodes[j+1:k+1] = reversed(route_nodes[j+1:k+1])
        elif case == 2:
            # Reverse segments [i, j] and [j+1, k]
            route_nodes[i:j+1] = reversed(route_nodes[i:j+1])
            route_nodes[j+1:k+1] = reversed(route_nodes[j+1:k+1])
        elif case == 3:
            # Move segment [j+1, k] to after i-1
            segment = route_nodes[j+1:k+1].copy()
            del route_nodes[j+1:k+1]
            for idx, node in enumerate(segment):
                route_nodes.insert(i + idx, node)
        elif case == 4:
            # Move segment [i, j] to after k
            segment = route_nodes[i:j+1].copy()
            del route_nodes[i:j+1]
            for idx, node in enumerate(segment):
                route_nodes.insert(k - (j - i + 1) + 1 + idx, node)
        elif case == 5:
            # Move segment [i, j] to after k and reverse it
            segment = list(reversed(route_nodes[i:j+1]))
            del route_nodes[i:j+1]
            for idx, node in enumerate(segment):
                route_nodes.insert(k - (j - i + 1) + 1 + idx, node)
        elif case == 6:
            # Move segment [j+1, k] to after i-1 and reverse it
            segment = list(reversed(route_nodes[j+1:k+1]))
            del route_nodes[j+1:k+1]
            for idx, node in enumerate(segment):
                route_nodes.insert(i + idx, node)

        # Rebuild route
        new_route = self._rebuild_route(route_nodes)
        new_solution.routes[route_idx] = new_route

        # Update solution cost
        new_solution.cost = sum(r.cost for r in new_solution.routes)

        return new_solution

    def solve(self):
        """
        Apply VNS to improve the solution.

        Returns:
            Improved Solution object
        """
        if self.verbose:
            print("\nStarting Variable Neighborhood Search...")
            print(f"Initial solution cost: {self.solution.cost:.2f}")

        current_solution = copy.deepcopy(self.solution)
        best_solution = copy.deepcopy(self.solution)
        best_cost = best_solution.cost

        iteration = 0
        no_improve = 0

        while iteration < self.max_iterations and no_improve < self.max_no_improve:
            # Shaking phase: apply a random neighborhood
            k = random.randint(0, len(self.neighborhoods) - 1)
            neighborhood = self.neighborhoods[k]

            # Generate a solution in the kth neighborhood
            new_solution = neighborhood(current_solution)

            # Local search phase
            # Apply a sequence of neighborhoods to improve the solution
            for neighborhood in self.neighborhoods:
                improved_solution = neighborhood(new_solution)
                if improved_solution.cost < new_solution.cost:
                    new_solution = improved_solution

            # Move or not
            if new_solution.cost < current_solution.cost:
                current_solution = new_solution

                if new_solution.cost < best_cost:
                    best_solution = copy.deepcopy(new_solution)
                    best_cost = best_solution.cost
                    no_improve = 0

                    if self.verbose:
                        print(f"Iteration {iteration}: Improved solution cost to {best_cost:.2f}")
                else:
                    no_improve += 1
            else:
                # With some probability, accept worse solution to escape local optima
                probability = math.exp(-(new_solution.cost - current_solution.cost) / (iteration + 1))
                if random.random() < probability:
                    current_solution = new_solution
                    if self.verbose:
                        print(f"Iteration {iteration}: Accepted worse solution with cost {new_solution.cost:.2f}")

                no_improve += 1

            iteration += 1

        if self.verbose:
            print(f"VNS completed after {iteration} iterations")
            print(f"Best solution cost: {best_cost:.2f}")
            print(f"Improvement: {(self.solution.cost - best_cost) / self.solution.cost * 100:.2f}%")

        return best_solution


class GuidedLocalSearch:
    """
    Guided Local Search (GLS) metaheuristic for improving VRP solutions.
    Uses penalties to guide the search away from local optima.
    """
    def __init__(self, solution, nodes, vehicle_capacity, max_iterations=15,
                 lambda_param=0.1, max_no_improve=3, verbose=False):
        """
        Initialize the GLS.

        Args:
            solution: Initial Solution object
            nodes: List of Node objects
            vehicle_capacity: Vehicle capacity
            max_iterations: Maximum number of iterations
            lambda_param: Parameter controlling the influence of penalties
            max_no_improve: Maximum number of iterations without improvement
            verbose: Whether to print detailed information
        """
        self.solution = copy.deepcopy(solution)
        self.best_solution = copy.deepcopy(solution)
        self.nodes = nodes
        self.depot = nodes[0]
        self.vehicle_capacity = vehicle_capacity
        self.max_iterations = max_iterations
        self.lambda_param = lambda_param
        self.max_no_improve = max_no_improve
        self.verbose = verbose

        # Initialize penalties for all possible edges
        self.penalties = defaultdict(int)

        # Initialize features (edges) in the current solution
        self.features = self._extract_features(self.solution)

    def _calculate_distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes."""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def _extract_features(self, solution):
        """Extract features (edges) from a solution."""
        features = []

        for route in solution.routes:
            for edge in route.edges:
                features.append((edge.origin.ID, edge.end.ID))

        return features

    def _augmented_cost(self, solution):
        """Calculate the augmented cost of a solution (cost + penalties)."""
        cost = solution.cost

        # Add penalties
        for route in solution.routes:
            for edge in route.edges:
                feature = (edge.origin.ID, edge.end.ID)
                cost += self.lambda_param * self.penalties[feature]

        return cost

    def _get_route_nodes(self, route):
        """Get list of nodes in a route."""
        if not route.edges:
            return []

        nodes = [route.edges[0].origin]
        for edge in route.edges:
            nodes.append(edge.end)

        return nodes

    def _rebuild_route(self, nodes):
        """Rebuild a route from a list of nodes."""
        route = Route()

        for i in range(len(nodes) - 1):
            node1 = nodes[i]
            node2 = nodes[i + 1]

            edge = Edge(node1, node2)
            edge.cost = self._calculate_distance(node1, node2)

            route.add_edge(edge)

            # Update node properties
            node1.isInterior = (i > 0)
            node1.inRoute = route

        # Update last node
        nodes[-1].isInterior = False
        nodes[-1].inRoute = route

        return route

    def _local_search(self, solution):
        """Apply local search to improve a solution."""
        improved = True
        current_solution = copy.deepcopy(solution)

        while improved:
            improved = False

            # Try 2-opt moves for each route
            for route_idx, route in enumerate(current_solution.routes):
                route_nodes = self._get_route_nodes(route)

                # Skip routes with less than 4 nodes
                if len(route_nodes) < 4:
                    continue

                # Try all possible 2-opt moves
                for i in range(1, len(route_nodes) - 2):
                    for j in range(i + 1, len(route_nodes) - 1):
                        # Create a new route by reversing the segment [i, j]
                        new_route_nodes = route_nodes.copy()
                        new_route_nodes[i:j+1] = reversed(new_route_nodes[i:j+1])

                        # Rebuild the route
                        new_route = self._rebuild_route(new_route_nodes)

                        # Calculate the change in augmented cost
                        old_aug_cost = route.cost
                        for edge in route.edges:
                            feature = (edge.origin.ID, edge.end.ID)
                            old_aug_cost += self.lambda_param * self.penalties[feature]

                        new_aug_cost = new_route.cost
                        for edge in new_route.edges:
                            feature = (edge.origin.ID, edge.end.ID)
                            new_aug_cost += self.lambda_param * self.penalties[feature]

                        # If the move improves the augmented cost, apply it
                        if new_aug_cost < old_aug_cost:
                            # Update the solution
                            new_solution = copy.deepcopy(current_solution)
                            new_solution.routes[route_idx] = new_route
                            new_solution.cost = sum(r.cost for r in new_solution.routes)

                            current_solution = new_solution
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            # Try relocate moves between routes
            if not improved and len(current_solution.routes) > 1:
                for route1_idx in range(len(current_solution.routes)):
                    for route2_idx in range(len(current_solution.routes)):
                        if route1_idx == route2_idx:
                            continue

                        route1 = current_solution.routes[route1_idx]
                        route2 = current_solution.routes[route2_idx]

                        route1_nodes = self._get_route_nodes(route1)
                        route2_nodes = self._get_route_nodes(route2)

                        # Skip routes with less than 3 nodes
                        if len(route1_nodes) < 3:
                            continue

                        # Try relocating each customer from route1 to route2
                        for i in range(1, len(route1_nodes) - 1):
                            node = route1_nodes[i]

                            # Check capacity constraint
                            if route2.demand + node.demand > self.vehicle_capacity:
                                continue

                            # Try inserting at each position in route2
                            for j in range(1, len(route2_nodes)):
                                # Create new routes
                                new_route1_nodes = route1_nodes.copy()
                                new_route2_nodes = route2_nodes.copy()

                                # Relocate node
                                new_route1_nodes.pop(i)
                                new_route2_nodes.insert(j, node)

                                # Rebuild routes
                                new_route1 = self._rebuild_route(new_route1_nodes)
                                new_route2 = self._rebuild_route(new_route2_nodes)

                                # Calculate the change in augmented cost
                                old_aug_cost = route1.cost + route2.cost
                                for edge in route1.edges + route2.edges:
                                    feature = (edge.origin.ID, edge.end.ID)
                                    old_aug_cost += self.lambda_param * self.penalties[feature]

                                new_aug_cost = new_route1.cost + new_route2.cost
                                for edge in new_route1.edges + new_route2.edges:
                                    feature = (edge.origin.ID, edge.end.ID)
                                    new_aug_cost += self.lambda_param * self.penalties[feature]

                                # If the move improves the augmented cost, apply it
                                if new_aug_cost < old_aug_cost:
                                    # Update the solution
                                    new_solution = copy.deepcopy(current_solution)
                                    new_solution.routes[route1_idx] = new_route1
                                    new_solution.routes[route2_idx] = new_route2
                                    new_solution.cost = sum(r.cost for r in new_solution.routes)

                                    current_solution = new_solution
                                    improved = True
                                    break

                            if improved:
                                break

                        if improved:
                            break

                    if improved:
                        break

        return current_solution

    def _update_penalties(self, solution):
        """Update penalties based on the current solution."""
        # Extract features from the current solution
        features = self._extract_features(solution)

        # Calculate utility for each feature
        utilities = {}

        for route in solution.routes:
            for edge in route.edges:
                feature = (edge.origin.ID, edge.end.ID)

                # Utility = cost / (1 + penalty)
                utilities[feature] = edge.cost / (1 + self.penalties[feature])

        # Find features with maximum utility
        max_utility = max(utilities.values()) if utilities else 0
        max_utility_features = [f for f, u in utilities.items() if u == max_utility]

        # Increase penalties for features with maximum utility
        for feature in max_utility_features:
            self.penalties[feature] += 1

    def solve(self):
        """
        Apply GLS to improve the solution.

        Returns:
            Improved Solution object
        """
        if self.verbose:
            print("\nStarting Guided Local Search...")
            print(f"Initial solution cost: {self.solution.cost:.2f}")

        current_solution = copy.deepcopy(self.solution)
        best_solution = copy.deepcopy(self.solution)
        best_cost = best_solution.cost

        iteration = 0
        no_improve = 0

        while iteration < self.max_iterations and no_improve < self.max_no_improve:
            # Apply local search to find a local optimum
            current_solution = self._local_search(current_solution)

            # Update best solution if improved
            if current_solution.cost < best_cost:
                best_solution = copy.deepcopy(current_solution)
                best_cost = best_solution.cost
                no_improve = 0

                if self.verbose:
                    print(f"Iteration {iteration}: Improved solution cost to {best_cost:.2f}")
            else:
                no_improve += 1

            # Update penalties to escape local optima
            self._update_penalties(current_solution)

            iteration += 1

        if self.verbose:
            print(f"GLS completed after {iteration} iterations")
            print(f"Best solution cost: {best_cost:.2f}")
            print(f"Improvement: {(self.solution.cost - best_cost) / self.solution.cost * 100:.2f}%")

        return best_solution


class DBSCANTabuSearch:
    """
    DBSCAN-Tabu Search hybrid metaheuristic for improving VRP solutions.
    Leverages DBSCAN clustering information to guide the tabu search.
    """
    def __init__(self, solution, nodes, vehicle_capacity, dbscan_labels,
                 tabu_tenure=4, max_iterations=15, max_no_improve=3, verbose=False):
        """
        Initialize the DBSCAN-Tabu Search.

        Args:
            solution: Initial Solution object
            nodes: List of Node objects
            vehicle_capacity: Vehicle capacity
            dbscan_labels: DBSCAN cluster labels for each customer
            tabu_tenure: Number of iterations a move remains tabu
            max_iterations: Maximum number of iterations
            max_no_improve: Maximum number of iterations without improvement
            verbose: Whether to print detailed information
        """
        self.solution = copy.deepcopy(solution)
        self.best_solution = copy.deepcopy(solution)
        self.nodes = nodes
        self.depot = nodes[0]
        self.vehicle_capacity = vehicle_capacity
        self.dbscan_labels = dbscan_labels  # Cluster label for each customer
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.verbose = verbose

        # Initialize tabu list
        self.tabu_list = {}

        # Create a mapping from node ID to cluster label
        self.node_to_cluster = {}
        for i, label in enumerate(dbscan_labels):
            # Add 1 to i because nodes[0] is depot
            if i + 1 < len(nodes):
                self.node_to_cluster[nodes[i + 1].ID] = label

    def _calculate_distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes."""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def _get_route_nodes(self, route):
        """Get list of nodes in a route."""
        if not route.edges:
            return []

        nodes = [route.edges[0].origin]
        for edge in route.edges:
            nodes.append(edge.end)

        return nodes

    def _rebuild_route(self, nodes):
        """Rebuild a route from a list of nodes."""
        route = Route()

        for i in range(len(nodes) - 1):
            node1 = nodes[i]
            node2 = nodes[i + 1]

            edge = Edge(node1, node2)
            edge.cost = self._calculate_distance(node1, node2)

            route.add_edge(edge)

            # Update node properties
            node1.isInterior = (i > 0)
            node1.inRoute = route

        # Update last node
        nodes[-1].isInterior = False
        nodes[-1].inRoute = route

        return route

    def _get_cluster_coherence(self, route):
        """
        Calculate the cluster coherence of a route.
        Higher values indicate that the route contains nodes from the same cluster.
        """
        route_nodes = self._get_route_nodes(route)

        # Skip depot
        customer_nodes = [node for node in route_nodes if node.ID != 0]

        if len(customer_nodes) <= 1:
            return 1.0  # Perfect coherence for routes with 0 or 1 customer

        # Count nodes from each cluster
        cluster_counts = defaultdict(int)
        for node in customer_nodes:
            cluster = self.node_to_cluster.get(node.ID, -1)
            cluster_counts[cluster] += 1

        # Calculate coherence as the ratio of nodes from the most common cluster
        max_count = max(cluster_counts.values()) if cluster_counts else 0
        coherence = max_count / len(customer_nodes)

        return coherence

    def _get_neighborhood(self, solution):
        """
        Generate neighborhood of solutions by applying various moves.
        Prioritizes moves that improve cluster coherence.
        """
        neighbors = []

        # 1. Swap nodes between routes to improve cluster coherence
        for i in range(len(solution.routes)):
            for j in range(i + 1, len(solution.routes)):
                route1 = solution.routes[i]
                route2 = solution.routes[j]

                route1_nodes = self._get_route_nodes(route1)
                route2_nodes = self._get_route_nodes(route2)

                # Skip routes with less than 3 nodes
                if len(route1_nodes) < 3 or len(route2_nodes) < 3:
                    continue

                # Try swapping each pair of customers
                for pos1 in range(1, len(route1_nodes) - 1):
                    for pos2 in range(1, len(route2_nodes) - 1):
                        node1 = route1_nodes[pos1]
                        node2 = route2_nodes[pos2]

                        # Check if the move is tabu
                        move_key = (node1.ID, node2.ID, "swap")
                        if move_key in self.tabu_list and self.tabu_list[move_key] > 0:
                            continue

                        # Check capacity constraints
                        new_route1_demand = route1.demand - node1.demand + node2.demand
                        new_route2_demand = route2.demand - node2.demand + node1.demand

                        if new_route1_demand > self.vehicle_capacity or new_route2_demand > self.vehicle_capacity:
                            continue

                        # Create new routes
                        new_route1_nodes = route1_nodes.copy()
                        new_route2_nodes = route2_nodes.copy()

                        # Swap nodes
                        new_route1_nodes[pos1] = node2
                        new_route2_nodes[pos2] = node1

                        # Rebuild routes
                        new_route1 = self._rebuild_route(new_route1_nodes)
                        new_route2 = self._rebuild_route(new_route2_nodes)

                        # Create new solution
                        new_solution = copy.deepcopy(solution)
                        new_solution.routes[i] = new_route1
                        new_solution.routes[j] = new_route2
                        new_solution.cost = sum(r.cost for r in new_solution.routes)

                        # Calculate cluster coherence improvement
                        old_coherence = self._get_cluster_coherence(route1) + self._get_cluster_coherence(route2)
                        new_coherence = self._get_cluster_coherence(new_route1) + self._get_cluster_coherence(new_route2)
                        coherence_improvement = new_coherence - old_coherence

                        # Add to neighbors with priority based on cost and coherence
                        neighbors.append((new_solution, move_key, coherence_improvement))

        # 2. Relocate nodes between routes to improve cluster coherence
        for i in range(len(solution.routes)):
            for j in range(len(solution.routes)):
                if i == j:
                    continue

                route1 = solution.routes[i]
                route2 = solution.routes[j]

                route1_nodes = self._get_route_nodes(route1)
                route2_nodes = self._get_route_nodes(route2)

                # Skip routes with less than 3 nodes
                if len(route1_nodes) < 3:
                    continue

                # Try relocating each customer from route1 to route2
                for pos1 in range(1, len(route1_nodes) - 1):
                    node = route1_nodes[pos1]

                    # Check if the move is tabu
                    move_key = (node.ID, i, j, "relocate")
                    if move_key in self.tabu_list and self.tabu_list[move_key] > 0:
                        continue

                    # Check capacity constraint
                    if route2.demand + node.demand > self.vehicle_capacity:
                        continue

                    # Try inserting at each position in route2
                    for pos2 in range(1, len(route2_nodes)):
                        # Create new routes
                        new_route1_nodes = route1_nodes.copy()
                        new_route2_nodes = route2_nodes.copy()

                        # Relocate node
                        new_route1_nodes.pop(pos1)
                        new_route2_nodes.insert(pos2, node)

                        # Rebuild routes
                        new_route1 = self._rebuild_route(new_route1_nodes)
                        new_route2 = self._rebuild_route(new_route2_nodes)

                        # Create new solution
                        new_solution = copy.deepcopy(solution)
                        new_solution.routes[i] = new_route1
                        new_solution.routes[j] = new_route2
                        new_solution.cost = sum(r.cost for r in new_solution.routes)

                        # Calculate cluster coherence improvement
                        old_coherence = self._get_cluster_coherence(route1) + self._get_cluster_coherence(route2)
                        new_coherence = self._get_cluster_coherence(new_route1) + self._get_cluster_coherence(new_route2)
                        coherence_improvement = new_coherence - old_coherence

                        # Add to neighbors with priority based on cost and coherence
                        neighbors.append((new_solution, move_key, coherence_improvement))

        # 3. Apply 2-opt moves within routes
        for i in range(len(solution.routes)):
            route = solution.routes[i]
            route_nodes = self._get_route_nodes(route)

            # Skip routes with less than 4 nodes
            if len(route_nodes) < 4:
                continue

            # Try all possible 2-opt moves
            for pos1 in range(1, len(route_nodes) - 2):
                for pos2 in range(pos1 + 1, len(route_nodes) - 1):
                    # Check if the move is tabu
                    move_key = (i, pos1, pos2, "2opt")
                    if move_key in self.tabu_list and self.tabu_list[move_key] > 0:
                        continue

                    # Create new route
                    new_route_nodes = route_nodes.copy()
                    new_route_nodes[pos1:pos2+1] = reversed(new_route_nodes[pos1:pos2+1])

                    # Rebuild route
                    new_route = self._rebuild_route(new_route_nodes)

                    # Create new solution
                    new_solution = copy.deepcopy(solution)
                    new_solution.routes[i] = new_route
                    new_solution.cost = sum(r.cost for r in new_solution.routes)

                    # Calculate cluster coherence improvement
                    old_coherence = self._get_cluster_coherence(route)
                    new_coherence = self._get_cluster_coherence(new_route)
                    coherence_improvement = new_coherence - old_coherence

                    # Add to neighbors
                    neighbors.append((new_solution, move_key, coherence_improvement))

        return neighbors

    def _update_tabu_list(self):
        """Update the tabu list by decrementing tabu tenure."""
        for move in list(self.tabu_list.keys()):
            self.tabu_list[move] -= 1
            if self.tabu_list[move] <= 0:
                del self.tabu_list[move]

    def solve(self):
        """
        Apply DBSCAN-Tabu Search to improve the solution.

        Returns:
            Improved Solution object
        """
        if self.verbose:
            print("\nStarting DBSCAN-Tabu Search...")
            print(f"Initial solution cost: {self.solution.cost:.2f}")

            # Calculate initial cluster coherence
            total_coherence = sum(self._get_cluster_coherence(route) for route in self.solution.routes)
            avg_coherence = total_coherence / len(self.solution.routes) if self.solution.routes else 0
            print(f"Initial average cluster coherence: {avg_coherence:.2f}")

        current_solution = copy.deepcopy(self.solution)
        best_solution = copy.deepcopy(self.solution)
        best_cost = best_solution.cost

        iteration = 0
        no_improve = 0

        while iteration < self.max_iterations and no_improve < self.max_no_improve:
            # Generate neighborhood
            neighbors = self._get_neighborhood(current_solution)

            if not neighbors:
                if self.verbose:
                    print(f"Iteration {iteration}: No valid neighbors found")
                break

            # Sort neighbors by cost and coherence improvement
            # We want to minimize cost and maximize coherence
            neighbors.sort(key=lambda x: (x[0].cost, -x[2]))

            # Select best non-tabu neighbor or best tabu neighbor that satisfies aspiration criterion
            best_neighbor = None
            for neighbor, move_key, coherence_improvement in neighbors:
                # Check if this is the best neighbor so far
                if best_neighbor is None:
                    best_neighbor = (neighbor, move_key)

                # Check aspiration criterion: accept tabu move if it gives the best solution so far
                if neighbor.cost < best_cost:
                    best_neighbor = (neighbor, move_key)
                    break

            if best_neighbor is None:
                if self.verbose:
                    print(f"Iteration {iteration}: No valid neighbors found")
                break

            # Update current solution
            current_solution, move_key = best_neighbor

            # Add move to tabu list
            self.tabu_list[move_key] = self.tabu_tenure

            # Update best solution if improved
            if current_solution.cost < best_cost:
                best_solution = copy.deepcopy(current_solution)
                best_cost = best_solution.cost
                no_improve = 0

                if self.verbose:
                    print(f"Iteration {iteration}: Improved solution cost to {best_cost:.2f}")

                    # Calculate cluster coherence
                    total_coherence = sum(self._get_cluster_coherence(route) for route in current_solution.routes)
                    avg_coherence = total_coherence / len(current_solution.routes) if current_solution.routes else 0
                    print(f"  Average cluster coherence: {avg_coherence:.2f}")
            else:
                no_improve += 1

            # Update tabu list
            self._update_tabu_list()

            iteration += 1

        if self.verbose:
            print(f"DBSCAN-Tabu Search completed after {iteration} iterations")
            print(f"Best solution cost: {best_cost:.2f}")
            print(f"Improvement: {(self.solution.cost - best_cost) / self.solution.cost * 100:.2f}%")

            # Calculate final cluster coherence
            total_coherence = sum(self._get_cluster_coherence(route) for route in best_solution.routes)
            avg_coherence = total_coherence / len(best_solution.routes) if best_solution.routes else 0
            print(f"Final average cluster coherence: {avg_coherence:.2f}")

        return best_solution