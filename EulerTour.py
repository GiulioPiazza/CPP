# Fleury's Algorithm implementation
import copy
import os
dirname = os.path.dirname(__file__)
out = os.path.join(dirname, 'output/')

class FleuryException(Exception):
    ''' raise an error when the graph is not eulerian '''

    def __init__(self, message):
        super(FleuryException, self).__init__(message)
        self.message = message


def print_eulerian_tour(graph):
    ''' 
        Start the execution of Fleury's Algorithm 
        and print the Eulerian circuit of the input's graph
    '''
   
    print ('Running Fleury algorithm for modified graph : \n')
    for v in graph:
        print (v, ' => ', graph[v])
    print ('\n')
    output = None
    try:
        output = fleury(graph)
    except FleuryException as message:
        print(message)
    with open(out+"circuit.txt", 'w') as f:
        if output:
            print("Found circuit: ")
            for v in output:
                f.write(f"{v}\n")
                print(v)
       

def is_connected(G):
    """ DFS, check if the graph is connected """
    
    COLOR_WHITE = 'white'
    COLOR_GRAY  = 'gray'
    COLOR_BLACK = 'black'
    
    start_node = list(G)[0]
    color = {}
    iterator = 0
    for v in G:
        color[v] = COLOR_WHITE
    color[start_node] = COLOR_GRAY
    S = [start_node]
    while len(S) != 0:
        u = S.pop()
        for v in G[u]:
            if color[v] == COLOR_WHITE:
                color[v] = COLOR_GRAY
                S.append(v)
            color[u] = COLOR_BLACK
    return list(color.values()).count(COLOR_BLACK) == len(G)


def even_degree_nodes(G):
    """ Return all even degree nodes """

    even_degree_nodes = []
    for u in G:
        if len(G[u]) % 2 == 0:
            even_degree_nodes.append(u)
    return even_degree_nodes


def is_eulerian(even_degree_odes, graph_len):
    """ check if the graph is eulerian """

    return graph_len - len(even_degree_odes) == 0


def convert_graph(G):
    """
    input: {0: [4, 5], 1: [2, 3, 4, 5]}
    Returns: [(0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5)]
    """

    links = []
    for u in G:
        for v in G[u]:
            links.append((u, v))
    return links


def fleury(G):
    '''
       Find all even degree nodes and check if graph is an eulerian one, then procede to find an eulerian trail
       Note: The modified graph of the Chinese Postman Algorithm should be always Eulerian
    '''
    edn = even_degree_nodes(G)
      
    if not is_eulerian(edn, len(G)):
        raise FleuryException('Il Grafo non è Euleriano!')

    # Execution of the Fleury's Algorithm 

    g = copy.copy(G)
    cycle = []
       
    u = edn[0]
    while len(convert_graph(g)) > 0: # fin quando ci sono archi nel grafo
        current_vertex = u
        #print(f"current v {current_vertex}")
        for u in list(g[current_vertex]): #adj list of u
            #print(f"u: {u}")
                
            """remove undirected edge current_vertex -> u"""
            g[current_vertex].remove(u)
            g[u].remove(current_vertex)
            # controlliamo se una volta rimosso l'arco il grafo è connesso
            # se non è connesso abbiamo individuato un ponte
            bridge = not is_connected(g)
            if bridge:
                # quindi lo riaggiungiamo nel grafo
                g[current_vertex].append(u)
                g[u].append(current_vertex)
            else:
                # altrimenti usciamo dal while, aggiungiamo l'arco currentvertex-u a cycle e analizziamo il vertice u
                break
        if bridge:
            # se l'unico vertice raggiungibile da current_vertex è un ponte,
            # lo percorriamo(eliminiamo l'arco currentvertex-u e eliminiamo anche current vertex, dato che abbiamo esplorato tutto le alternative(archi))
            # infine aggiungiamo l'arco currentvertex-u a cycle
            g[current_vertex].remove(u)
            g[u].remove(current_vertex)
            g.pop(current_vertex)
        cycle.append((current_vertex, u))
    return cycle




     