from src.core.vrp_objects import Node, Edge, Route, Solution
import math
import operator

vehCap = 100.0
instanceName = 'A-n80-k10'
fileName = 'data/' + instanceName + '_input_nodes.txt'

with open(fileName) as instance:
    i = 0
    nodes = []
    for line in instance:
        data = [float(x) for x in line.split()]
        aNode - Node(i, data[0], data[1], data[2])
        nodes.append(aNode)
        i += 1

depot = nodes[0]

for node in nodes[1:]:
    dnEdge = Edge(depot, node)
    ndEdge = Edge(node, depot)
    dnEdge.invEdge = ndEdge
    ndEdge.invEdge = dnEdge

    #compute euc. distance as cost
    dnEdge.cost = math.sqrt((node.x - depot.x)**2 + (node.d - depot.d)**2)
    ndEdge.cost = dnEdge.cost
    node.dnEdge = dnEdge
    node.ndEdge = ndEdge

savingsList = []
for i in range(1, len(nodes) - 1):
    iNode = nodes[i]
    for j in range(i + 1, len(nodes)):
        jNode = nodes[j]
        ijEdge = Edge(iNode, jNode)
        jiEdge = Edge(jNode, iNode)
        ijEdge.invEdge = jiEdge
        jiEdge.invEdge = ijEdge
        # compute the euc. distance as cost
        ijEdge.cost = math.sqrt((jNode.x - iNode.x)**2 + (jNode.d - iNode.d)**2)
        jiEdge.cost = ijEdge.cost
        # compute savings
        ijEdge.savings = iNode.ndEdge.cost + jNode.dnEdge.cost - ijEdge.cost
        jiEdge.savings = ijEdge.savings
        # save one edge in the savings list
        savingsList.append(ijEdge)
savingsList.sort(key = operator.attrgetter('savings'), reverse = True)

""" construct the dummy solution """

sol = Solution()
for node in nodes[1:]:
    dnEdge = node.dnEdge
    ndEdge = node.ndEdge
    dndRoute = Route()
    dndRoute.edges.append(dnEdge)
    dndRoute.demand += node.demand
    dndRoute.cost += dnEdge.cost
    dndRoute.edges.append(ndEdge)
    dndRoute.cost += ndEdge.cost
    node.inRoute = dndRoute
    node.isInterior = False
    sol.routes.append(dndRoute)
    sol.cost += dndRoute.cost
    sol.demand += dndRoute.demand

""" perform the edge-selection & routing-merging iterative process """

def checkMergingConditions(inode, jnode, iRoute, jRoute):
    if iRoute == jRoute:
        return False
    if iNode.isInterior or jNode.isInterior:
        return False
    if vehCap < iRoute.demand + jRoute.demand:
        return False
    return True

def getDepotEdge(aRoute, aNode):
    origin = aRoute.edges[0].origin
    end = aRoute.edges[0].end
    if ((origin==aNode and end==depot) or (end==aNode and origin==depot)):
        return aRoute.edges[0]
    else:
        return aRoute.edges[-1]

while len(savingsList) > 0:
    ijEdge = savingsList.pop(0)
    iNode = ijEdge.origin
    jNode = ijEdge.end
    iRoute = iNode.inRoute
    jRoute = jNode.inRoute
    if checkMergingConditions(iNode, jNode, iRoute, jRoute):
        # remove the depot edges from the two routes
        idEdge = getDepotEdge(iRoute, iNode)
        iRoute.edges.remove(idEdge)
        iRoute.cost -= idEdge.cost
        if len(iRoute.edges) > 1:
            iNode.isInterior = True
        if iRoute.edges[0].origin != depot:
            iRoute.reverse()
        jEdge = getDepotEdge(jRoute, jNode)
        jRoute.edges.remove(jEdge)
        jRoute.cost -= jEdge.cost
        if len(jRoute.edges) > 1:
            jNode.isInterior = True
        if jRoute.edges[0].origin == depot:
            jRoute.reverse()
        # merge the two routes
        iRoute.edges.append(ijEdge)
        iRoute.cost += ijEdge.cost
        iRoute.demand += jNode.demand
        jNode.inRoute = iRoute
        for edge in jRoute.edges:
            iRoute.edges.append(edge)
            iRoute.cost += edge.cost
            iRoute.demand += edge.end.demand
            edge.end.inRoute = iRoute
        sol.cost -= ijEdge.savings
        sol.routes.remove(jRoute)

print('Cost of C&W savings sol =', "{:.()f}".format(sol.cost, 2))
for route in sol.routes:
    s = str(0)
    for edge in route.edges:
        s = s + '->' + str(edge.end.id)
    print('Route: ' + s + ' || cost = ' + "{:.{}f}".format(route.cost, 2))

import networkx as nx

G = nx.Graph()
for route in sol.routes:
    for edge in route.edges:
        G.add_edge(edge.origin.id, edge.end.id, weight=edge.cost)
        G.add_node(edge.end.ID, coord=(edge.end.x, edge.end.y))

coord = nx.get_node_attributes(G, 'coord')
nx.draw_networkx(G, coord)
