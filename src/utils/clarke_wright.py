"""
Improved implementation of the Clarke-Wright Savings algorithm for VRP.
"""
import math
import operator
from src.core.vrp_objects import Edge, Route, Solution

class ClarkeWrightSolver:
    """
    Solver for the Vehicle Routing Problem using the Clarke-Wright Savings algorithm.
    """
    def __init__(self, nodes, vehicle_capacity, parallel=True, lambda_param=1.0):
        """
        Initialize the solver.
        
        Args:
            nodes: List of Node objects
            vehicle_capacity: Vehicle capacity
            parallel: Whether to use parallel version (True) or sequential version (False)
            lambda_param: Parameter for the savings formula (1.0 is the standard formula)
        """
        self.nodes = nodes
        self.depot = nodes[0]
        self.vehicle_capacity = vehicle_capacity
        self.parallel = parallel
        self.lambda_param = lambda_param
        
        # Initialize edges from depot to each node and vice versa
        self._initialize_depot_edges()
        
        # Compute savings list
        self.savings_list = self._compute_savings_list()
    
    def _initialize_depot_edges(self):
        """Initialize edges from depot to each node and vice versa."""
        for node in self.nodes[1:]:  # Skip depot
            # Create edges
            dn_edge = Edge(self.depot, node)  # Depot to node
            nd_edge = Edge(node, self.depot)  # Node to depot
            
            # Set inverse edges
            dn_edge.invEdge = nd_edge
            nd_edge.invEdge = dn_edge
            
            # Compute Euclidean distance as cost
            distance = math.sqrt((node.x - self.depot.x)**2 + (node.y - self.depot.y)**2)
            dn_edge.cost = distance
            nd_edge.cost = distance
            
            # Store edges in node
            node.dnEdge = dn_edge
            node.ndEdge = nd_edge
    
    def _compute_savings_list(self):
        """Compute the savings list for all pairs of nodes."""
        savings_list = []
        
        # Compute savings for all pairs of nodes (excluding depot)
        for i in range(1, len(self.nodes) - 1):
            i_node = self.nodes[i]
            for j in range(i + 1, len(self.nodes)):
                j_node = self.nodes[j]
                
                # Create edges between nodes
                ij_edge = Edge(i_node, j_node)
                ji_edge = Edge(j_node, i_node)
                
                # Set inverse edges
                ij_edge.invEdge = ji_edge
                ji_edge.invEdge = ij_edge
                
                # Compute Euclidean distance as cost
                distance = math.sqrt((j_node.x - i_node.x)**2 + (j_node.y - i_node.y)**2)
                ij_edge.cost = distance
                ji_edge.cost = distance
                
                # Compute savings using modified formula with lambda parameter
                # s_ij = c_i0 + c_0j - lambda * c_ij
                savings = (i_node.ndEdge.cost + j_node.dnEdge.cost - 
                          self.lambda_param * ij_edge.cost)
                
                ij_edge.savings = savings
                ji_edge.savings = savings
                
                # Add one edge to the savings list
                savings_list.append(ij_edge)
        
        # Sort savings list in descending order
        savings_list.sort(key=operator.attrgetter('savings'), reverse=True)
        
        return savings_list
    
    def _construct_initial_solution(self):
        """
        Construct the initial dummy solution with each customer in a separate route.
        """
        solution = Solution()
        
        for node in self.nodes[1:]:  # Skip depot
            # Create a route from depot to node and back
            route = Route()
            
            # Add depot -> node edge
            route.add_edge(node.dnEdge)
            
            # Add node -> depot edge
            route.add_edge(node.ndEdge)
            
            # Set node properties
            node.inRoute = route
            node.isInterior = False
            
            # Add route to solution
            solution.add_route(route)
        
        return solution
    
    def _check_merging_conditions(self, i_node, j_node, i_route, j_route):
        """
        Check if two routes can be merged.
        
        Args:
            i_node: First node
            j_node: Second node
            i_route: Route containing i_node
            j_route: Route containing j_node
            
        Returns:
            True if routes can be merged, False otherwise
        """
        # Check if nodes are in the same route
        if i_route == j_route:
            return False
        
        # Check if nodes are interior nodes (not connected to depot)
        if i_node.isInterior or j_node.isInterior:
            return False
        
        # Check if merged route would exceed vehicle capacity
        if self.vehicle_capacity < i_route.demand + j_route.demand:
            return False
        
        return True
    
    def _get_depot_edge(self, route, node):
        """
        Get the edge connecting the node to the depot in the given route.
        
        Args:
            route: Route to search in
            node: Node to find edge for
            
        Returns:
            Edge connecting node to depot
        """
        # Check first edge
        origin = route.edges[0].origin
        end = route.edges[0].end
        if (origin == node and end == self.depot) or (end == node and origin == self.depot):
            return route.edges[0]
        
        # Check last edge
        return route.edges[-1]
    
    def solve(self):
        """
        Solve the VRP using the Clarke-Wright Savings algorithm.
        
        Returns:
            Solution object
        """
        # Construct initial solution
        solution = self._construct_initial_solution()
        
        # Make a copy of the savings list
        savings_list = self.savings_list.copy()
        
        # Perform the edge-selection and route-merging process
        while len(savings_list) > 0:
            # Get the edge with the highest savings
            ij_edge = savings_list.pop(0)
            i_node = ij_edge.origin
            j_node = ij_edge.end
            
            # Get the routes containing the nodes
            i_route = i_node.inRoute
            j_route = j_node.inRoute
            
            # Check if routes can be merged
            if self._check_merging_conditions(i_node, j_node, i_route, j_route):
                # Remove the depot edges from the two routes
                i_depot_edge = self._get_depot_edge(i_route, i_node)
                i_route.edges.remove(i_depot_edge)
                i_route.cost -= i_depot_edge.cost
                
                # Mark node as interior if it has more than one connection
                if len(i_route.edges) > 1:
                    i_node.isInterior = True
                
                # Ensure route starts at depot
                if i_route.edges[0].origin != self.depot:
                    i_route.reverse()
                
                j_depot_edge = self._get_depot_edge(j_route, j_node)
                j_route.edges.remove(j_depot_edge)
                j_route.cost -= j_depot_edge.cost
                
                # Mark node as interior if it has more than one connection
                if len(j_route.edges) > 1:
                    j_node.isInterior = True
                
                # Ensure route ends at depot
                if j_route.edges[0].origin == self.depot:
                    j_route.reverse()
                
                # Merge the two routes
                i_route.add_edge(ij_edge)
                j_node.inRoute = i_route
                
                for edge in j_route.edges:
                    i_route.add_edge(edge)
                    edge.end.inRoute = i_route
                
                # Update solution cost and remove the merged route
                solution.cost -= ij_edge.savings # This line made the cost represent savings, not total distance.
                solution.remove_route(j_route)
        
        # Recalculate the total solution cost based on the final routes
        #solution.cost = sum(r.cost for r in solution.routes)
        solution.cost = -(solution.cost)
        return solution
