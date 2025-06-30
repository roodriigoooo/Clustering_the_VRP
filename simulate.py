import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from itertools import combinations

np.random.seed(42)
depot = np.array([50,50])
zones = [
    np.random.normal(loc=[30,70], scale=5, size(20,2)),
    np.random.normal(loc=[70,30], scale=5, size(20,2))
    np.random.normal(loc=[50,70], scale=5, size(20,2))
]

customers = np.vstack(zones)
demands = np.random.randint(5,26, size=len(customers))
vehicle_capacity = 10

dbscan = DBSCAN(eps=10, min_samples=3)
labels = dbscan.fit_predict(customers)
unique_clusters = set(labels)
unique_clusters.remove(-1)

def compute_savings_matrix(points, depot_idx=0):
    n = len(points)
    savings = []
    for i,j in combinations (range(1,n), 2):
        cost_i = euclidean(points[depot_idx], points[i])
        cost_j = euclidean(points[depot_idx], points[j])
        cost_ij = euclidean(points[i], points[j])
        saving = cost_i + cost_j - cost_ij
        savings.append((i,j, saving))
    return sorted(savings, key=lambda x: x[2], reverse=True)

def clarke_wright(points, demands, capacity):
    n = len(points)
    routes = {i: [0,i,0] for i in range(1,n)}
    route_demands = {i: demands[i] for i in range(1,n)}
    savings = compute_savings_matrix(points)
    for i,j, _ in savings:
        route_i = next((r for r in routes.values() if r[1] == i and r[-2] == i), None)
        route_j = next((r for r in routes.values() if r[1] == j and r[-2] == j), None)
        if route_i is None or route_j is None or route_i == route_j:
            continue
        total_demand = sum(demands[k] for k in route_i[1:-1] + route_j[1:-1])
        if total_demand <= capacity:
            new_route = [0] + route_i[1:-1] + route_j[1:-1] + [0]
            routes = {k:v for k,v in routes.items() if v != route_i and v != route_j}
            routes[new_route[1]] = new_route
    return list(routes.values())

cluster_routes = {}
for cluster_id in unique_clusters:
    cluster_mark = labels == cluster_id
    cluster_points = customers[cluster_mark]
    cluster_demands = demands[cluster_mark]

    points_with_depot = np.vstack([depot, cluster_points])
    demand_with_depot = np.insert(cluster_demands, 0, 0)
    routes = clarke_wright(points_with_depot, demand_with_depot, vehicle_capacity)
    cluster_routes[cluster_id] = (points_with_depot, routes)

colors = ['red', 'blue', 'green', 'orange', 'purple']
plt.figure(figsize=(10,6))
plt.scatter(depot[0], depot[1], c='black', marker='s', label='Depot')
for cluster_id, (points, routes) in cluster_routes.items():
    color = colors[cluster_id % len(colors)]
    plt.scatter(points[1:,0], points[1:,1], c=color, label=f'Cluster {cluster_id}')
    for route in routes:
        route_coords = points[route]
        plt.plot(route_coords[:,0], route_coords[:,1], c=color, alpha=0.7, linewidth=2)

plt.legend()
plt.title('DBSCAN Clustering and Clarke-Wright Routing')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
        
