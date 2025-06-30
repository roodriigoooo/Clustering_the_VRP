"""
Data handler for VRP instances, including Kelly's instances.
"""
import os
import math
import numpy as np
from vrp_objects import Node, Edge, Solution, Route

class DataHandler:
    """
    Class to handle loading and saving VRP data instances.
    """
    def __init__(self):
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')

    def generate_kellys_instance(self, instance_name, num_customers, num_vehicles,
                                seed=42, save=True):
        """
        Generate a Kelly's instance with the given parameters.
        Kelly's instances are characterized by clustered customers.

        Args:
            instance_name: Name of the instance
            num_customers: Number of customers
            num_vehicles: Number of vehicles
            seed: Random seed for reproducibility
            save: Whether to save the instance to a file

        Returns:
            List of Node objects representing the instance
        """
        np.random.seed(seed)

        # Define depot at the center
        depot_x, depot_y = 50, 50

        # Create clusters
        num_clusters = min(5, num_vehicles)
        cluster_centers = [
            (30, 70),  # Top left
            (70, 30),  # Bottom right
            (30, 30),  # Bottom left
            (70, 70),  # Top right
            (50, 50),  # Center
        ][:num_clusters]

        # Distribute customers among clusters
        customers_per_cluster = num_customers // num_clusters
        remainder = num_customers % num_clusters

        # Generate customer coordinates and demands
        nodes = [Node(0, depot_x, depot_y, 0)]  # Depot is always node 0 with 0 demand

        node_id = 1
        for i in range(num_clusters):
            center_x, center_y = cluster_centers[i]
            # Add extra customer to first 'remainder' clusters
            cluster_size = customers_per_cluster + (1 if i < remainder else 0)

            # Generate clustered customers
            for _ in range(cluster_size):
                # Customer coordinates with normal distribution around cluster center
                x = np.random.normal(center_x, 10)
                y = np.random.normal(center_y, 10)

                # Ensure coordinates are within bounds (0-100)
                x = max(0, min(100, x))
                y = max(0, min(100, y))

                # Generate demand (uniform between 5 and 25)
                demand = np.random.randint(5, 26)

                # Create node and add to list
                nodes.append(Node(node_id, x, y, demand))
                node_id += 1

        # Save instance to file if requested
        if save:
            self._save_instance(instance_name, nodes, num_vehicles)

        return nodes

    def _save_instance(self, instance_name, nodes, num_vehicles):
        """
        Save an instance to files in the data directory.

        Args:
            instance_name: Name of the instance
            nodes: List of Node objects
            num_vehicles: Number of vehicles
        """
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')

        # Save nodes to file
        with open(f'data/{instance_name}_input_nodes.txt', 'w') as f:
            for node in nodes:
                f.write(f"{node.x} {node.y} {node.demand}\n")

        # Save instance info to file
        with open(f'data/{instance_name}_info.txt', 'w') as f:
            f.write(f"Instance: {instance_name}\n")
            f.write(f"Number of customers: {len(nodes) - 1}\n")
            f.write(f"Number of vehicles: {num_vehicles}\n")

            # Calculate total demand
            total_demand = sum(node.demand for node in nodes)
            f.write(f"Total demand: {total_demand}\n")

            # Calculate average demand per vehicle
            avg_demand_per_vehicle = total_demand / num_vehicles
            f.write(f"Average demand per vehicle: {avg_demand_per_vehicle:.2f}\n")

            # Calculate vehicle capacity (with 20% buffer)
            vehicle_capacity = math.ceil(avg_demand_per_vehicle * 1.2)
            f.write(f"Recommended vehicle capacity: {vehicle_capacity}\n")

    def load_instance(self, instance_name, vehicle_capacity=None):
        """
        Load an instance from files in the data directory.

        Args:
            instance_name: Name of the instance
            vehicle_capacity: Vehicle capacity (if None, read from info file)

        Returns:
            nodes: List of Node objects
            vehicle_capacity: Vehicle capacity
        """
        # Check if instance exists
        nodes_file = f'data/{instance_name}_input_nodes.txt'
        info_file = f'data/{instance_name}_info.txt'

        if not os.path.exists(nodes_file):
            raise FileNotFoundError(f"Instance {instance_name} not found")

        # Load nodes
        nodes = []
        with open(nodes_file, 'r') as f:
            node_id = 0
            for line in f:
                data = [float(x) for x in line.split()]
                nodes.append(Node(node_id, data[0], data[1], data[2]))
                node_id += 1

        # Load vehicle capacity if not provided
        if vehicle_capacity is None and os.path.exists(info_file):
            with open(info_file, 'r') as f:
                for line in f:
                    if line.startswith("Recommended vehicle capacity:"):
                        vehicle_capacity = int(line.split(":")[-1].strip())
                        break

        # If still None, use a default value
        if vehicle_capacity is None:
            total_demand = sum(node.demand for node in nodes)
            num_vehicles = max(1, len(nodes) // 20)  # Rough estimate
            vehicle_capacity = math.ceil(total_demand / num_vehicles * 1.2)

        return nodes, vehicle_capacity

    def generate_all_kellys_instances(self):
        """
        Generate all Kelly's instances and save them to files.
        """
        instances = [
            ("Kelly_n20_k3", 20, 3),
            ("Kelly_n30_k4", 30, 4),
            ("Kelly_n50_k5", 50, 5),
            ("Kelly_n75_k8", 75, 8),
            ("Kelly_n100_k10", 100, 10),
            ("Kelly_n150_k12", 150, 12),
            ("Kelly_n200_k15", 200, 15)
        ]

        for name, num_customers, num_vehicles in instances:
            self.generate_kellys_instance(name, num_customers, num_vehicles)
            print(f"Generated instance: {name}")

    def load_augerat_instance(self, file_path, save_to_data_dir=True):
        """
        Load an Augerat VRP instance from a .vrp file.

        Args:
            file_path: Path to the .vrp file
            save_to_data_dir: Whether to save the instance to the data directory

        Returns:
            nodes: List of Node objects
            vehicle_capacity: Vehicle capacity
        """
        # Extract instance name from file path
        instance_name = os.path.basename(file_path).split('.')[0]

        # Initialize variables
        nodes = []
        vehicle_capacity = None
        dimension = None
        node_coords = {}
        demands = {}
        depot_id = None

        # Parse the .vrp file
        section = None
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Check for section headers
                if line.startswith('CAPACITY'):
                    vehicle_capacity = int(line.split(':')[1].strip())
                    continue
                elif line.startswith('DIMENSION'):
                    dimension = int(line.split(':')[1].strip())
                    continue
                elif line == 'NODE_COORD_SECTION':
                    section = 'coords'
                    continue
                elif line == 'DEMAND_SECTION':
                    section = 'demands'
                    continue
                elif line == 'DEPOT_SECTION':
                    section = 'depot'
                    continue
                elif line == 'EOF':
                    break

                # Parse data based on current section
                if section == 'coords':
                    parts = line.split()
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    node_coords[node_id] = (x, y)
                elif section == 'demands':
                    parts = line.split()
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    demands[node_id] = demand
                elif section == 'depot':
                    if line.strip() != '-1':
                        depot_id = int(line.strip())

        # Create Node objects
        for node_id in range(1, dimension + 1):
            x, y = node_coords[node_id]
            demand = demands[node_id]

            # Adjust node_id to make depot ID 0
            adjusted_id = 0 if node_id == depot_id else node_id

            # Create Node object
            node = Node(adjusted_id, x, y, demand)
            nodes.append(node)

        # Sort nodes to ensure depot is first
        nodes.sort(key=lambda node: node.ID)

        # Calculate number of vehicles from instance name (e.g., A-n32-k5 has 5 vehicles)
        num_vehicles = int(instance_name.split('-k')[1])

        # Save instance to data directory if requested
        if save_to_data_dir:
            self._save_instance(instance_name, nodes, num_vehicles)

            # Save additional info about the Augerat instance
            with open(f'data/{instance_name}_augerat_info.txt', 'w') as f:
                f.write(f"Original file: {file_path}\n")
                f.write(f"Instance type: Augerat\n")
                f.write(f"Number of customers: {len(nodes) - 1}\n")
                f.write(f"Number of vehicles: {num_vehicles}\n")
                f.write(f"Vehicle capacity: {vehicle_capacity}\n")

        return nodes, vehicle_capacity

    def import_all_augerat_instances(self, directory_path):
        """
        Import all Augerat VRP instances from a directory.

        Args:
            directory_path: Path to the directory containing .vrp files

        Returns:
            List of instance names that were imported
        """
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')

        # Get all .vrp files in the directory
        vrp_files = [f for f in os.listdir(directory_path) if f.endswith('.vrp')]

        # Import each instance
        imported_instances = []
        for vrp_file in vrp_files:
            file_path = os.path.join(directory_path, vrp_file)
            instance_name = vrp_file.split('.')[0]

            print(f"Importing Augerat instance: {instance_name}")
            self.load_augerat_instance(file_path)
            imported_instances.append(instance_name)

        return imported_instances

# Example usage
if __name__ == "__main__":
    handler = DataHandler()
    handler.generate_all_kellys_instances()
    # Uncomment to import Augerat instances
    # handler.import_all_augerat_instances('/path/to/augerat/instances')
