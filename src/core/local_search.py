"""
Local search improvements for VRP solutions.
"""
import math
import copy
from vrp_objects import Edge, Route, Solution

class LocalSearch:
    """
    Class for applying local search improvements to VRP solutions.
    """
    def __init__(self, solution, nodes, vehicle_capacity):
        """
        Initialize the local search.

        Args:
            solution: Solution object to improve
            nodes: List of Node objects
            vehicle_capacity: Vehicle capacity
        """
        self.solution = solution
        self.nodes = nodes
        self.depot = nodes[0]
        self.vehicle_capacity = vehicle_capacity

    def _calculate_edge_cost(self, node1, node2):
        """
        Calculate the cost (distance) between two nodes.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Euclidean distance between the nodes
        """
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def _get_route_nodes(self, route):
        """
        Get the list of nodes in a route.

        Args:
            route: Route object

        Returns:
            List of Node objects in the route
        """
        if not route.edges:
            return []

        nodes = [route.edges[0].origin]
        for edge in route.edges:
            nodes.append(edge.end)

        return nodes

    def _rebuild_route(self, nodes):
        """
        Rebuild a route from a list of nodes.

        Args:
            nodes: List of Node objects

        Returns:
            Route object
        """
        route = Route()

        for i in range(len(nodes) - 1):
            node1 = nodes[i]
            node2 = nodes[i + 1]

            edge = Edge(node1, node2)
            edge.cost = self._calculate_edge_cost(node1, node2)

            route.add_edge(edge)

            # Update node properties
            node1.isInterior = (i > 0)
            node1.inRoute = route

        # Update last node
        nodes[-1].isInterior = False
        nodes[-1].inRoute = route

        return route

    def apply_2opt(self, max_iterations=100):
        """
        Apply 2-opt local search to improve each route.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Improved Solution object
        """
        improved_solution = copy.deepcopy(self.solution)

        for route_idx, route in enumerate(improved_solution.routes):
            # Get nodes in the route
            route_nodes = self._get_route_nodes(route)

            # Skip routes with less than 4 nodes (depot -> customer -> depot)
            if len(route_nodes) < 4:
                continue

            # Apply 2-opt
            improved = True
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                best_improvement = 0
                best_i, best_j = -1, -1

                # Try all possible 2-opt moves
                for i in range(1, len(route_nodes) - 2):
                    for j in range(i + 1, len(route_nodes) - 1):
                        # Calculate current cost
                        current_cost = (self._calculate_edge_cost(route_nodes[i-1], route_nodes[i]) +
                                       self._calculate_edge_cost(route_nodes[j], route_nodes[j+1]))

                        # Calculate new cost after 2-opt
                        new_cost = (self._calculate_edge_cost(route_nodes[i-1], route_nodes[j]) +
                                   self._calculate_edge_cost(route_nodes[i], route_nodes[j+1]))

                        # Calculate improvement
                        improvement = current_cost - new_cost

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i, best_j = i, j

                # Apply best move if it improves the solution
                if best_improvement > 0:
                    # Reverse the segment [i, j]
                    route_nodes[best_i:best_j+1] = reversed(route_nodes[best_i:best_j+1])
                    improved = True

                iteration += 1

            # Rebuild the route
            new_route = self._rebuild_route(route_nodes)
            improved_solution.routes[route_idx] = new_route

        # Recalculate solution cost
        improved_solution.cost = sum(route.cost for route in improved_solution.routes)

        return improved_solution

    def apply_3opt(self, max_iterations=50):
        """
        Apply 3-opt local search to improve each route.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Improved Solution object
        """
        improved_solution = copy.deepcopy(self.solution)

        for route_idx, route in enumerate(improved_solution.routes):
            # Get nodes in the route
            route_nodes = self._get_route_nodes(route)

            # Skip routes with less than 5 nodes
            if len(route_nodes) < 5:
                continue

            # Apply 3-opt
            improved = True
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                best_improvement = 0
                best_move = None

                # Try all possible 3-opt moves
                for i in range(1, len(route_nodes) - 4):
                    for j in range(i + 1, len(route_nodes) - 2):
                        for k in range(j + 1, len(route_nodes) - 1):
                            # Calculate current cost
                            current_cost = (self._calculate_edge_cost(route_nodes[i-1], route_nodes[i]) +
                                           self._calculate_edge_cost(route_nodes[j], route_nodes[j+1]) +
                                           self._calculate_edge_cost(route_nodes[k], route_nodes[k+1]))

                            # Try all possible reconnections
                            for case in range(7):  # 7 possible ways to reconnect
                                new_route = route_nodes.copy()

                                if case == 0:
                                    # Reverse segment [i, j]
                                    new_route[i:j+1] = reversed(new_route[i:j+1])
                                elif case == 1:
                                    # Reverse segment [j+1, k]
                                    new_route[j+1:k+1] = reversed(new_route[j+1:k+1])
                                elif case == 2:
                                    # Reverse segments [i, j] and [j+1, k]
                                    new_route[i:j+1] = reversed(new_route[i:j+1])
                                    new_route[j+1:k+1] = reversed(new_route[j+1:k+1])
                                elif case == 3:
                                    # Move segment [j+1, k] to after i-1
                                    segment = new_route[j+1:k+1].copy()
                                    del new_route[j+1:k+1]
                                    for idx, node in enumerate(segment):
                                        new_route.insert(i + idx, node)
                                elif case == 4:
                                    # Move segment [i, j] to after k
                                    segment = new_route[i:j+1].copy()
                                    del new_route[i:j+1]
                                    new_route.insert(k - (j - i + 1) + 1, segment[0])
                                    for idx in range(1, len(segment)):
                                        new_route.insert(k - (j - i + 1) + 1 + idx, segment[idx])
                                elif case == 5:
                                    # Move segment [i, j] to after k and reverse it
                                    segment = list(reversed(new_route[i:j+1]))
                                    del new_route[i:j+1]
                                    for idx, node in enumerate(segment):
                                        new_route.insert(k - (j - i + 1) + 1 + idx, node)
                                elif case == 6:
                                    # Move segment [j+1, k] to after i-1 and reverse it
                                    segment = list(reversed(new_route[j+1:k+1]))
                                    del new_route[j+1:k+1]
                                    for idx, node in enumerate(segment):
                                        new_route.insert(i + idx, node)

                                # Calculate new cost
                                new_cost = (self._calculate_edge_cost(new_route[i-1], new_route[i]) +
                                           self._calculate_edge_cost(new_route[j], new_route[j+1]) +
                                           self._calculate_edge_cost(new_route[k], new_route[k+1]))

                                # Calculate improvement
                                improvement = current_cost - new_cost

                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_move = (case, i, j, k)

                # Apply best move if it improves the solution
                if best_improvement > 0:
                    case, i, j, k = best_move

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
                        segment = route_nodes[j+1:k+1]
                        route_nodes[i:k+1] = route_nodes[i:j+1] + segment
                    elif case == 4:
                        # Move segment [i, j] to after k
                        segment = route_nodes[i:j+1]
                        del route_nodes[i:j+1]
                        route_nodes.insert(k - (j - i + 1) + 1, segment)
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

                    improved = True

                iteration += 1

            # Rebuild the route
            new_route = self._rebuild_route(route_nodes)
            improved_solution.routes[route_idx] = new_route

        # Recalculate solution cost
        improved_solution.cost = sum(route.cost for route in improved_solution.routes)

        return improved_solution
