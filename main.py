from cProfile import label
import sys
import numpy as np
from Graph import Graph
import Chinese_postman
import EulerTour
import networkx as nx
from matplotlib import pyplot as plt
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'resources/')
out = os.path.join(dirname, 'output/')

def initialize_graph():
    '''
                     3
            (0)-----------------(1)
       1  / |                    | \ 1
         /  |                    |  \ 
        (2) | 5                6 |   (3)
         \  |                    |  /
        2 \ |         4          | / 1
            (4)------------------(5)
            
    '''
    graph = Graph (6)
    graph.addEdge(0, 1,3)
    graph.addEdge(0, 2,1)
    graph.addEdge(0, 4,5)
    graph.addEdge(1, 3,1)
    graph.addEdge(1, 5,6)
    graph.addEdge(2, 4,2)
    graph.addEdge(4, 5,4)
    graph.addEdge(5, 3,1)
    '''
    toy = Graph(4)
    toy.addEdge(0, 2,1)
    toy.addEdge(1, 2,1)
    toy.addEdge(1, 3,1)
    toy.addEdge(2, 3,1)

    graph = Graph(5)
    graph.addEdge('0', '1',1)
    graph.addEdge('1', '4',1)
    graph.addEdge('2', '3',1)
    graph.addEdge('2', '4',1)
    graph.addEdge('3', '4',1)
    
    graph = Graph(8)
    graph.addEdge('A', 'B',58)
    graph.addEdge('A', 'B',36)
    graph.addEdge('A', 'C',23)
    graph.addEdge('A', 'D',22)
    graph.addEdge('B', 'C',28)
    graph.addEdge('B', 'F',46)
    graph.addEdge('C', 'F',31)
    graph.addEdge('C', 'E',39)
    graph.addEdge('C', 'D',47)
    graph.addEdge('D', 'E',35)
    graph.addEdge('D', 'G',51)
    graph.addEdge('D', 'G',43)
    graph.addEdge('E', 'G',38)
    graph.addEdge('F', 'G',34)
    graph.addEdge('F', 'H',51)
    graph.addEdge('G', 'H',43)
    
    
    graph = Graph(8)
    graph.addEdge('A', 'B',50)
    graph.addEdge('A', 'D',50)
    graph.addEdge('A', 'C',50)
    graph.addEdge('B', 'E',70)
    graph.addEdge('B', 'D',50)
    graph.addEdge('B', 'F',50)
    graph.addEdge('C', 'D',70)
    graph.addEdge('C', 'G',70)
    graph.addEdge('C', 'H',120)
    graph.addEdge('D', 'F',60)
    graph.addEdge('E', 'F',70)
    graph.addEdge('F', 'H',60)
    graph.addEdge('G', 'H',70)
    
    
    graph = Graph(9)
    graph.addEdge('A', 'B',4)
    graph.addEdge('A', 'H',8)
    graph.addEdge('B', 'C',8)
    graph.addEdge('B', 'H',11)
    graph.addEdge('C', 'D',7)
    graph.addEdge('C', 'F',4)
    graph.addEdge('C', 'I',2)
    graph.addEdge('D', 'E',9)
    graph.addEdge('D', 'G',14)
    graph.addEdge('E', 'F',10)
    graph.addEdge('F', 'G',2)
    graph.addEdge('G', 'H',1)
    graph.addEdge('G', 'I',6)
    graph.addEdge('H', 'I',7)
    
    '''
    return graph
    
def read_graph_txt(path):
    ''' Function that create a graph from an edge list stored in a txt '''

    lines = []
    with open(path) as f:
        lines = f.readlines()

    graph = Graph()
    for line in lines:
        line = line.split()
        graph.addEdge(line[0], line[1], float(line[2]))

    return graph
        


def convert_to_non_weighted_graph(G):
    ''' 
        Convert the undirected weigthed input's graph into a non weighted one
        and return the adj list.
    '''
    non_weightedgraph = {}
    for v in G.graph:
        non_weightedgraph[v] = list(map(lambda x: x[0], G.graph[v]))
    return non_weightedgraph

def convert_to_edgelist(G):
    ''' 
        return the edge list.
    '''

    edge_list = []
    for key, value in G.graph.items():
        for u in value:
            edge_list.append((key,u[0],{"weight": u[1]}))

    return edge_list


def draw_graph(graph, name):
    ''' 
        Simple function that uses networkx and maplotlib to draw the undirected input's graph.
    '''

    G = nx.MultiGraph(convert_to_non_weighted_graph(graph))
    plt.figure(name)
    #plt.subplot(1,2,index)
    pos= nx.spring_layout(G,seed=5)
   
    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos,node_color = 'r', node_size = 200, alpha = 1)
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    for e in G.edges:
        ax.annotate("",
                    xy= pos[e[0]], xycoords='data',
                    xytext= pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                    ),
                                    ),
                    )


    fig1 = plt.gcf()
    plt.axis('off')
    #plt.show()
    fig1.savefig(out + name + "_graph.png", dpi=400)


if __name__ == "__main__":

    if(len(sys.argv) == 2):
        print(f"Path to Graph: {sys.argv[1]}")
        graph = read_graph_txt(filename + sys.argv[1])
    else:
        graph = initialize_graph()

    #print(graph.graph)
    
    #algorithm
    print("===============================")
    dis, modified_graph = Chinese_postman.chinese_postman(graph)
    print('Chinese Postman Distance is:', dis)
    print("===============================")
    print('Chinese Postman Tour is: ')
    EulerTour.print_eulerian_tour(convert_to_non_weighted_graph(modified_graph))

    #Draw graphs
    print("\nDo you want to show the graphs ? y/n")
    to_do = input() 
    if(to_do.lower() == 'y'):
        draw_graph(graph, "original")
        draw_graph(modified_graph, "modified")
        plt.show()
    
    