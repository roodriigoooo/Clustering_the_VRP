class Node:
    """
    Represents a customer node in the VRP problem.
    """
    def __init__(self, ID, x, y, demand):
        self.ID = ID
        self.x = x
        self.y = y
        self.demand = demand
        self.inRoute = None
        self.isInterior = False
        self.dnEdge = None  # Edge from depot to node
        self.ndEdge = None  # Edge from node to depot

    def __str__(self):
        return f"Node(ID={self.ID}, x={self.x}, y={self.y}, demand={self.demand})"

class Edge:
    """
    Represents an edge between two nodes in the VRP problem.
    """
    def __init__(self, origin, end):
        self.origin = origin
        self.end = end
        self.cost = 0.0
        self.savings = 0.0
        self.invEdge = None  # Inverse edge (end -> origin)

    def __str__(self):
        return f"Edge(origin={self.origin.ID}, end={self.end.ID}, cost={self.cost:.2f})"

class Route:
    """
    Represents a vehicle route in the VRP solution.
    """
    def __init__(self):
        self.cost = 0.0
        self.edges = []
        self.demand = 0.0

    def add_edge(self, edge):
        """Add an edge to the route and update cost and demand."""
        self.edges.append(edge)
        self.cost += edge.cost
        self.demand += edge.end.demand

    def reverse(self):
        """Reverse the direction of the route."""
        size = len(self.edges)
        for i in range(size):
            edge = self.edges[i]
            invEdge = edge.invEdge
            self.edges.remove(edge)
            self.edges.insert(0, invEdge)

    def get_nodes(self):
        """Return a list of nodes in the route (including depot)."""
        if not self.edges:
            return []

        nodes = [self.edges[0].origin]
        for edge in self.edges:
            nodes.append(edge.end)
        return nodes

    def __str__(self):
        if not self.edges:
            return "Empty Route"

        path = str(self.edges[0].origin.ID)
        for edge in self.edges:
            path += f"->{edge.end.ID}"
        return f"Route: {path} | Cost: {self.cost:.2f} | Demand: {self.demand:.2f}"

class Solution:
    """
    Represents a complete solution to the VRP problem.
    """
    last_ID = -1

    def __init__(self):
        Solution.last_ID += 1
        self.ID = Solution.last_ID
        self.routes = []
        self.cost = 0.0
        self.demand = 0.0

    def add_route(self, route):
        """Add a route to the solution and update cost and demand."""
        self.routes.append(route)
        self.cost += route.cost
        self.demand += route.demand

    def remove_route(self, route):
        """Remove a route from the solution and update cost and demand."""
        if route in self.routes:
            self.routes.remove(route)
            self.cost -= route.cost
            self.demand -= route.demand

    def get_all_route_lengths(self):
        """Returns a list of all route lengths (costs) in the solution."""
        return [route.cost for route in self.routes]

    def get_all_route_demands(self):
        """Returns a list of all route demands in the solution."""
        return [route.demand for route in self.routes]

    def __str__(self):
        return f"Solution(ID={self.ID}, routes={len(self.routes)}, cost={self.cost:.2f}, demand={self.demand:.2f})"