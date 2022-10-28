from collections import defaultdict
import pprint


''' This class represents an undirected weighted graph using adjacency list representation '''
class Graph:

    def __init__(self,vertices=0):
        self.V= vertices                # No. of vertices
        self.cost = 0                   # Weight sum of all added edges
        self.graph = defaultdict(list)  # default dictionary to store graph

    def addEdge(self,u,v, weight):
        ''' Function to add an edge to graph '''

        self.cost += weight
        self.graph[u].append((v,weight))
        self.graph[v].append((u,weight))

    
    def delEdge(self, u, v):
        ''' This function removes edge u-v from graph '''

        for index, key in enumerate(self.graph[u]):
            if key == v:
                self.graph[u].pop(index)
        for index, key in enumerate(self.graph[v]):
            if key == u:
                self.graph[v].pop(index)

    def rmvEdge(self, u, v, w):
        ''' This function removes edge u-v-w from graph '''

        for index, key in enumerate(self.graph[u]):
            if key[0] == v and key[1] == w:
                self.graph[u].pop(index)
        for index, key in enumerate(self.graph[v]):
            if key[0] == u  and key[1] == w:
                self.graph[v].pop(index)

 