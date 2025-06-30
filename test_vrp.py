"""
Test script for the VRP solution framework.
"""
import unittest
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from data_handler import DataHandler
from clarke_wright import ClarkeWrightSolver
from clustering import VRPClusterer
from local_search import LocalSearch
from evaluation import VRPEvaluator
from vrp_objects import Node, Edge, Route, Solution

class TestVRPFramework(unittest.TestCase):
    """Test cases for the VRP solution framework."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a small test instance
        self.depot = Node(0, 50, 50, 0)
        self.nodes = [
            self.depot,
            Node(1, 30, 70, 10),
            Node(2, 70, 30, 15),
            Node(3, 30, 30, 20),
            Node(4, 70, 70, 5),
            Node(5, 40, 60, 10),
            Node(6, 60, 40, 15),
            Node(7, 40, 40, 20),
            Node(8, 60, 60, 5)
        ]
        self.vehicle_capacity = 50

    def test_data_handler(self):
        """Test the DataHandler class."""
        handler = DataHandler()

        # Test generating a Kelly's instance
        instance_name = "Test_n10_k2"
        nodes = handler.generate_kellys_instance(instance_name, 10, 2, save=True)

        # Check if files were created
        self.assertTrue(os.path.exists(f"data/{instance_name}_input_nodes.txt"))
        self.assertTrue(os.path.exists(f"data/{instance_name}_info.txt"))

        # Test loading the instance
        loaded_nodes, loaded_capacity = handler.load_instance(instance_name)

        # Check if the number of nodes is correct
        self.assertEqual(len(loaded_nodes), 11)  # 10 customers + 1 depot

        # Clean up
        os.remove(f"data/{instance_name}_input_nodes.txt")
        os.remove(f"data/{instance_name}_info.txt")

    def test_clarke_wright(self):
        """Test the ClarkeWrightSolver class."""
        solver = ClarkeWrightSolver(self.nodes, self.vehicle_capacity)
        solution = solver.solve()

        # Check if solution is valid
        self.assertIsNotNone(solution)
        self.assertGreater(len(solution.routes), 0)

        # Check if all customers are assigned to a route
        assigned_nodes = set()
        for route in solution.routes:
            for edge in route.edges:
                if edge.end != self.depot:
                    assigned_nodes.add(edge.end.ID)

        self.assertEqual(len(assigned_nodes), len(self.nodes) - 1)  # All customers except depot

        # Check if capacity constraints are satisfied
        for route in solution.routes:
            self.assertLessEqual(route.demand, self.vehicle_capacity)

    def test_clustering(self):
        """Test the VRPClusterer class."""
        clusterer = VRPClusterer(self.nodes, vehicle_capacity=self.vehicle_capacity)

        # Test K-means clustering
        kmeans_clusters = clusterer.kmeans_clustering(num_clusters=2)
        self.assertEqual(len(kmeans_clusters), 2)

        # Test DBSCAN clustering
        dbscan_clusters = clusterer.dbscan_clustering()
        self.assertGreater(len(dbscan_clusters), 0)

        # Test hierarchical clustering
        hierarchical_clusters = clusterer.hierarchical_clustering(num_clusters=2)
        self.assertEqual(len(hierarchical_clusters), 2)

        # Test capacity-based clustering
        capacity_clusters = clusterer.capacity_based_clustering()
        self.assertGreater(len(capacity_clusters), 0)

        # Test balanced K-means clustering
        balanced_clusters = clusterer.balanced_kmeans_clustering(num_clusters=2)
        self.assertEqual(len(balanced_clusters), 2)

    def test_local_search(self):
        """Test the LocalSearch class."""
        # Create a solution with a known suboptimal route
        solution = Solution()

        # Create a deliberately suboptimal route (a "zigzag" pattern)
        route = Route()

        # Add edges to create a zigzag pattern
        prev_node = self.depot
        for i in [1, 3, 5, 7, 2, 4, 6, 8]:
            node = self.nodes[i]
            edge = Edge(prev_node, node)
            edge.cost = math.sqrt((prev_node.x - node.x)**2 + (prev_node.y - node.y)**2)
            route.add_edge(edge)
            prev_node = node

        # Add edge back to depot
        edge = Edge(prev_node, self.depot)
        edge.cost = math.sqrt((prev_node.x - self.depot.x)**2 + (prev_node.y - self.depot.y)**2)
        route.add_edge(edge)

        # Add route to solution
        solution.add_route(route)
        solution.cost = route.cost

        # Apply local search
        local_search = LocalSearch(solution, self.nodes, self.vehicle_capacity)

        # Test 2-opt
        improved_2opt = local_search.apply_2opt()
        self.assertLessEqual(improved_2opt.cost, solution.cost)

        # Test 3-opt
        improved_3opt = local_search.apply_3opt()
        self.assertLessEqual(improved_3opt.cost, solution.cost)

    def test_evaluation(self):
        """Test the VRPEvaluator class."""
        evaluator = VRPEvaluator()

        # Create two solutions
        solver1 = ClarkeWrightSolver(self.nodes, self.vehicle_capacity)
        solution1 = solver1.solve()

        solver2 = ClarkeWrightSolver(self.nodes, self.vehicle_capacity, lambda_param=0.8)
        solution2 = solver2.solve()

        # Evaluate solutions
        evaluator.evaluate_solution(solution1, "Test", "Method1", 1.0)
        evaluator.evaluate_solution(solution2, "Test", "Method2", 1.5)

        # Get results
        df = evaluator.get_results_dataframe()
        self.assertEqual(len(df), 2)

        # Compare methods
        comparison = evaluator.compare_methods("Test")
        self.assertEqual(len(comparison), 2)

        # Test visualization (just check if it runs without errors)
        fig = evaluator.visualize_solution(solution1, self.nodes, "Test", "Method1")
        plt.close(fig)

        fig = evaluator.plot_comparison(metric='total_cost', instance_name="Test")
        plt.close(fig)

if __name__ == "__main__":
    unittest.main()
